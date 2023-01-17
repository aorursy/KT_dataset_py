# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

df.head()
df_dtype_nunique = pd.concat([X_val_test1.dtypes, X_val_test1.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

X = df.iloc[:,0:20]  #independent columns

y = df.iloc[:,-1]    #target column i.e price range

#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
df.describe()
df.info()
df.isnull().any().any()
df.corr()
numerical_features = ['id','chem_0','chem_1','chem_2','chem_3','chem_4','chem_7','attribute']

X = df[numerical_features]

y = df["class"]
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor,RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor, AdaBoostClassifier,GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor

from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier,MLPRegressor

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

from sklearn.metrics import mean_squared_error

from xgboost import XGBClassifier

from sklearn.ensemble import IsolationForest

from mlxtend.classifier import StackingClassifier

clf1 = RandomForestClassifier().fit(X_train,y_train)

clf2 = DecisionTreeClassifier().fit(X_train,y_train)
from sklearn.metrics import accuracy_score  



y_pred_1 = clf1.predict(X_val)

y_pred_2 = clf2.predict(X_val)



acc1 = accuracy_score(y_pred_1,y_val)*100

acc2 = accuracy_score(y_pred_2,y_val)*100



print("Accuracy score of clf1: {}".format(acc1))

print("Accuracy score of clf2: {}".format(acc2))
X_val_test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

X_val_test.head()
X_val_test.isnull().any().any()
numerical_features_test = ['id','chem_0','chem_1','chem_2','chem_3','chem_4','chem_7','attribute']

X_val_test1 = X_val_test[numerical_features_test]
y_pred_3 = clf1.predict(X_val_test1)

X_val_test1['class'] = y_pred_3 
X_val_test1.head()
X_val_test1.info()
X_val_test1.describe()
X_val_test['class'] = y_pred_3 

df1 = X_val_test[['id','class']]

df1.head()
df1.to_csv(r'resultclass9.csv',index=False)
df2 = pd.read_csv('resultclass9.csv')

df2.head(80)
df_dtype_nunique = pd.concat([X_val_test1.dtypes, X_val_test1.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique