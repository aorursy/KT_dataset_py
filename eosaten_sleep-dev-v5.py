from sklearn import preprocessing

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pydot

import pandas as pd

import numpy as np

import seaborn as sns

sns.set()



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



import tensorflow as tf

import datetime, os



import sklearn

from sklearn import metrics

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz, DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.exceptions import NotFittedError

from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras import optimizers

from keras.wrappers.scikit_learn import KerasClassifier



from IPython.display import display

%load_ext tensorboard.notebook







from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

import xgboost as xgb

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score,confusion_matrix,classification_report

from sklearn.preprocessing import StandardScaler



from sklearn.preprocessing import normalize

pd.set_option('display.max_columns', None)

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.metrics import mean_squared_error



import pickle
odf=pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv',index_col=0)

odft=pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv',index_col=0)
odf.describe()
odf.shape
odf.dtypes
odf.head(5)
df=odf.replace({'na':-999999})

Xt=odft.replace({'na':-999999})

Xt=Xt.astype(float)

df=df.astype(float)

df['target']=df['target'].astype(int)
sns.countplot(df['target'],label="Count")
X=df.iloc[:,1:]

y=df.iloc[:,0]



#L2 Normalize

Xn=normalize(X)

Xtn=normalize(Xt)
# Confusion Matrix

def confusion_matrix(target, prediction, score=None):

   cm = metrics.confusion_matrix(target, prediction)

   plt.figure(figsize=(4,4))

   sns.heatmap(cm, annot=True,fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')

   plt.ylabel('Act')

   plt.xlabel('Pred')
X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=8)
my_class = ExtraTreesClassifier(random_state=0)

my_class.fit(X_train, y_train)

y_pred= clf.predict(X_test)

print('accuracy: {}'.format(accuracy_score(y_test, y_pred)))

print(f'F1: {f1_score(y_test,y_pred)}')

confusion_matrix(y_test,y_pred)
my_class = AdaBoostClassifier(random_state=0)

my_class.fit(X_train, y_train)

y_pred= my_class.predict(X_test)

print('accuracy: {}'.format(accuracy_score(y_test, y_pred)))

print(f'F1: {f1_score(y_test,y_pred)}')

confusion_matrix(y_test,y_pred)
lg = LogisticRegression(solver='lbfgs', random_state=18)

lg.fit(X_train, y_train)

logistic_prediction = lg.predict(X_test)

score = metrics.accuracy_score(y_test, logistic_prediction)

print(score)

confusion_matrix(y_test,logistic_prediction)
data_dmatrix = xgb.DMatrix(data=Xn,label=y)

xgc = xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.15,

                          learning_rate = 0.1, 

                          max_depth = 20, alpha = 12, n_estimators = 700)

xgc.fit(X,y)
pred_train=xgc.predict(X)

pred_train.sum()
pred_test=xgc.predict(Xt)

pred_test.sum()
yt=pd.DataFrame(pred_test)

yt.index=yt.index+1

yt
test=pd.read_csv('../input/equipfails/equip_failures_test_set.csv',na_values='na')

df= pd.DataFrame()

df['id'] = test['id']

df['target'] = pred_test

df.to_csv('submission2.csv', index=False)
file_name='submision.csv'

yt.to_csv(file_name,index=True)

# from IPython.display import FileLink

# FileLink(file_name)
filename = 'Final_Model.mod'

pickle.dump(xgc, open(filename, 'wb'))