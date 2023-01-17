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
df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

df.head()
df.describe()
df.describe(include='object')
df.info()
df.isnull().any().any()
missing_count = df.isnull().sum()

missing_count[missing_count>0].sort_index()
df.fillna(value=df.mean(),inplace=True)

df.head()
df.fillna(value=df.mode().loc[0],inplace=True)

df.head()
df.isnull().any().any()
df.corr()
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



pd.set_option('display.max_columns', 100)



corr = df.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(12, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
numerical_features = ['feature3','feature5','feature6','feature7','feature8','feature10']

categorical_features = ['type']

X = df[numerical_features+categorical_features]

y = df["rating"]
type = {'old':0,'new':1}

X['type'] = X['type'].map(type)

X.head()
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier



clf1 = DecisionTreeClassifier().fit(X_train,y_train)

clf2 = RandomForestClassifier().fit(X_train,y_train)

from sklearn.metrics import accuracy_score  



y_pred_1 = clf1.predict(X_val)

y_pred_2 = clf2.predict(X_val)



acc1 = accuracy_score(y_pred_1,y_val)*100

acc2 = accuracy_score(y_pred_2,y_val)*100



print("Accuracy score of clf1: {}".format(acc1))

print("Accuracy score of clf2: {}".format(acc2))

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



#TODO

clf = RandomForestClassifier()        #Initialize the classifier object



parameters = {'n_estimators':[10,50,100]}    #Dictionary of parameters



scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train



best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf.predict(X_val)        #Same, but use the best estimator



acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))
X_val_test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

X_val_test.head()
X_val_test.isnull().any().any()
X_val_test.fillna(value=df.mean(),inplace=True)

X_val_test.fillna(value=df.mode().loc[0],inplace=True)

X_val_test.head()
X_val_test.isnull().any().any()
type = {'old':0,'new':1}

X_val_test['type'] = X_val_test['type'].map(type)

X_val_test.head()
numerical_features_test = ['feature3','feature5','feature6','feature7','feature8','feature10']

categorical_features_test = ['type']

X_val_test1 = X_val_test[numerical_features_test+categorical_features_test]
y_pred_3 = clf2.predict(X_val_test1)

X_val_test1['rating'] = y_pred_3 
X_val_test1.head()
X_val_test1.info()
X_val_test1.describe()
df_dtype_nunique = pd.concat([X_val_test1.dtypes, X_val_test1.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
X_val_test['rating'] = y_pred_3 

df1 = X_val_test[['id','rating']]

df1.head()
df1.to_csv(r'resultratingnew3.csv',index=False)
df2 = pd.read_csv('resultratingnew3.csv')

df2.head()