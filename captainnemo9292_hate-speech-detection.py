#TfidfVectorizer Logistic Regression

import pandas as pd 

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

import numpy as np



train_data = pd.read_csv('../input/hate_speech_data.csv')



X_train,X_test,y_train,y_test = train_test_split(train_data["문장"],train_data["혐오 여부"],random_state=0)

vect = TfidfVectorizer().fit(X_train)

X_train_vectorized = vect.transform(X_train)



model = LogisticRegression()

model.fit(X_train_vectorized, y_train)



feature_names = np.array(vect.get_feature_names())

sorted_tfidf_index=model.coef_[0].argsort()



predictions=model.predict(vect.transform(X_test))



roc_auc_score(y_test, predictions)
sentence="한남충 개돼지 뒤져라 좆팔"

if model.predict(vect.transform([sentence]))==[1]:

    print('혐오 발언')

else:

    print("정상 발언")