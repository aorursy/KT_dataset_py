!pip install pyenchant pysastrawi
!wget http://archive.ubuntu.com/ubuntu/pool/main/libr/libreoffice-dictionaries/hunspell-id_6.4.3-1_all.deb

!dpkg -i hunspell-id_6.4.3-1_all.deb
!apt update && apt install -y enchant libenchant1c2a hunspell hunspell-en-us libhunspell-1.6-0
import re

import os

import random



import numpy as np

import pandas as pd

import sklearn

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import enchant

import lightgbm as lgbm
!pip freeze > requirements.txt
print('Numpy version:', np.__version__)

print('Pandas version:', pd.__version__)

print('Scikit-Learn version:', sklearn.__version__)

print('Matplotlib version:', matplotlib.__version__)

print('Seaborn version:', sns.__version__)

print('NLTK version:', nltk.__version__)

print('LightGBM version:', lgbm.__version__)
SEED = 42



os.environ['PYTHONHASHSEED']=str(SEED)

random.seed(SEED)

np.random.seed(SEED)
nltk.download('wordnet')
!ls /kaggle/input
X_train = pd.read_parquet('/kaggle/input/shopee-review-cleaned/X_train.parquet', engine='pyarrow')

X_train = X_train['X']



X_test = pd.read_parquet('/kaggle/input/shopee-review-cleaned/X_test.parquet', engine='pyarrow')

X_test = X_test['X']



y_train = pd.read_parquet('/kaggle/input/shopee-review-cleaned/y_train.parquet', engine='pyarrow')

y_train = y_train['y']
rating_count = y_train.value_counts().sort_index().to_list()

total_rating = sum(rating_count)

lowest_rating_count = min(rating_count)

rating_weight = [lowest_rating_count/rc for rc in rating_count]



print(rating_count)

print(total_rating)

print(rating_weight)
rating_weight_dict = {

    1: rating_weight[0],

    2: rating_weight[1],

    3: rating_weight[2],

    4: rating_weight[3],

    5: rating_weight[4],

}
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1,3), analyzer=lambda t:t, min_df=10, sublinear_tf=True)



X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)
print(X_train.shape)

print(X_test.shape)
from sklearn.metrics import classification_report, f1_score, confusion_matrix



def predict(model, X, tweak_proba=False):

    if tweak_proba:

        y = model.predict_proba(X)



        for i in range(len(y)):

            y[i, 0] = y[i, 0] * 1.05 # rating 1

#             y[i, 1] = y[i, 1] * 1.0 # rating 1

#             y[i, 2] = y[i, 2] * 1.0 # rating 1

            y[i, 3] = y[i, 3] * 1.30  # rating 4

            y[i, 4] = y[i, 4] * 1.30  # rating 5



        # +1 because np.argmax range is 0-4, not 1-5

        y = np.argmax(y, axis=1)

        for i in range(len(y)):

            y[i] = y[i] + 1

    else:

        y = model.predict(X)

    return y



def metrics(y_true, y_pred):

    print('F1 Score :', f1_score(y_true, y_pred, average='macro'))

    print(classification_report(y_true, y_pred))



    cm = confusion_matrix(y_true, y_pred)

    cm = pd.DataFrame(cm, range(1, 6), range(1, 6))



    sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d")

    plt.show()
from datetime import datetime



model = lgbm.LGBMClassifier(

    n_estimators=150,

    class_weight=rating_weight_dict,

    boosting_type='dart',

    max_bin=1023,

    max_depth=0,

    num_leaves=255,

    learning_rate=0.03,

    extra_trees=True,

    feature_fraction=0.8

)



START_TIME = datetime.now()

model.fit(X_train, y_train, verbose=3)

END_TIME = datetime.now()



print((END_TIME - START_TIME).seconds)
y_train_pred = predict(model, X_train)

metrics(y_train, y_train_pred)
y_train_pred2 = predict(model, X_train, tweak_proba=True)

metrics(y_train, y_train_pred2)
y_test_pred = predict(model, X_test)



df_submission = pd.concat([pd.Series(list(range(1,60428)), name='review_id', dtype=np.int32), pd.Series(y_test_pred, name='rating')], axis=1)

df_submission.to_csv('submission.csv', index=False)



df_submission
y_test_pred2 = predict(model, X_test, tweak_proba=True)



df_submission2 = pd.concat([pd.Series(list(range(1,60428)), name='review_id', dtype=np.int32), pd.Series(y_test_pred2, name='rating')], axis=1)

df_submission2.to_csv('submission_tweak_proba.csv', index=False)



df_submission2