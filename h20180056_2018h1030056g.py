import numpy as np

import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import time

from sklearn.linear_model import LogisticRegression # Logistic regression

from sklearn.naive_bayes import GaussianNB # Naive-Bayes

from sklearn.model_selection import GridSearchCV # Grid Search for model optimization

# The four algorithms that will be used in the stacking

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier # KNN

from sklearn.svm import SVC # Support vector machines

from sklearn.model_selection import train_test_split, cross_val_predict

from sklearn.model_selection import cross_val_score # Cross validation score

from sklearn.ensemble import VotingClassifier # Voting Classifier

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

sns.set_style('whitegrid')

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold

from vecstack import stacking

from sklearn.metrics import accuracy_score

data=pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

data.head()
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
data.drop(['id'], axis=1, inplace=True)

data.head()
data_test=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

data_test.drop(['id'], axis=1, inplace=True)

data_test.head()
data.info()
data.describe()
data['class'].value_counts()
combine = [data, data_test]
X_train=data.drop(['class'], axis=1)

y_train=data['class']

X_test=data_test

# y_test=data_test['class']



X_train.shape, y_train.shape, X_test.shape

models = [

    KNeighborsClassifier(n_neighbors=5,

                        n_jobs=-1),

        

    RandomForestClassifier(random_state=0, n_jobs=-1, 

                           n_estimators=100, max_depth=3),

        

    XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 

                  n_estimators=100, max_depth=3)

]
S_train, S_test = stacking(models,                   

                           X_train, y_train, X_test,   

                           regression=False, 

     

                           mode='oof_pred_bag', 

       

                           needs_proba=False,

         

                           save_dir=None, 

            

                           metric=accuracy_score, 

    

                           n_folds=4, 

                 

                           stratified=True,

            

                           shuffle=True,  

            

                           random_state=0,    

         

                           verbose=2)
model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 

                      n_estimators=100, max_depth=3)

    

model = model.fit(S_train, y_train)

preds = model.predict(S_test)
preds.shape
dd=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')
# sub_df=pd.read_csv('submission.csv')

submission={}

submission['id']= dd.id

submission['class']= preds

submission=pd.DataFrame(submission)

submission
submission.to_csv('mysubmission13.csv',index=False)