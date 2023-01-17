# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/shot_logs.csv')

df.head()
df = df[['LOCATION', 'W', 'FINAL_MARGIN', 'SHOT_NUMBER', 'PERIOD', 'DRIBBLES', 'SHOT_DIST', 'CLOSE_DEF_DIST', 'FGM', 'PTS', 'SHOT_RESULT']]

df.isnull().sum() # We choose a feature which does not include NaN values as possible.
df.head()

print("Total made shot {}".format(df[df['SHOT_RESULT'] == 'made'].shape))

print("Total missed shot {}".format(df[df['SHOT_RESULT'] == 'missed'].shape))
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()



df['LOCATION'] = encoder.fit_transform(df.iloc[:, 0].values)

df['W'] = encoder.fit_transform(df.iloc[:, 1].values)



from sklearn.preprocessing import OneHotEncoder



onehot_encoder = OneHotEncoder(categorical_features=[0, 1])



X = onehot_encoder.fit_transform(df.iloc[:, :-1].values).toarray()

y = encoder.fit_transform(df.iloc[:, -1].values)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline



pipe_svc = Pipeline([('scaler', StandardScaler()), 

                     ('pca', PCA(n_components=10)), 

                     ('svc', SVC(probability=True))])



pipe_svc.fit(X_train, y_train)

print('Test Accuracy: %.3f' % pipe_svc.score(X_test, y_test))
from sklearn.metrics import confusion_matrix



y_pred = pipe_svc.predict(X_test)



confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print(confmat)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_svc,

                         X=X_train,

                         y=y_train,

                         cv=10)



print('Cross validation scores: %s ' % scores)
from sklearn.metrics import roc_curve

from sklearn.cross_validation import StratifiedKFold

from scipy import interp



cv = StratifiedKFold(np.array(y_train), n_folds=3)



import matplotlib.pyplot as plt



fig = plt.figure(figsize=(7, 5))



mean_true_positive_rate = 0.0

mean_false_positive_rate = np.linspace(0, 1, 100)



for i, (train, test) in enumerate(cv):

    probabilities = pipe_svc.fit(X_train[train], 

                                 y_train[train]).predict_proba(X_train[test])

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train[test], 

                                                                    probabilities[:, 1], 

                                                                    pos_label=1)

    



    plt.plot(false_positive_rate, true_positive_rate, label='ROC fold %d' % i)

    

   

plt.xlabel('false positive rate')

plt.ylabel('true positive rate')

plt.xlim([-0.1, 1.1])

plt.ylim([-0.1, 1.1])

plt.show()

    