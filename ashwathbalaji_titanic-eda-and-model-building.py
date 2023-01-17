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
import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train['Source'] = 'Train'

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test['Source'] = 'Test'

df = pd.concat([df_train,df_test],0)

df.head()
print (df.shape)
df.info()
mr = (df.isna().sum()/len(df))*100

mr.sort_values(ascending=False)
df['Cabin'].fillna('NotAvailable',inplace=True)
df["Cabin_Status"] = df['Cabin'].apply(lambda x : 0 if x=='NotAvailable' else 1)
df.head()
sns.barplot(df['Cabin_Status'] , df['Survived'])

plt.show()
sns.barplot(df['Embarked'], df['Survived'] , hue=df['Sex'])

plt.show()
df.drop(['Cabin','Ticket','Embarked'],1,inplace=True)
df.head()
df['Married']= np.NAN

df['Married'][df['Name'].str.contains('Mr.|Mrs.')] = 1

df['Married'][~(df['Name'].str.contains('Mr.|Mrs.'))] = 0

df['Married']=df['Married'].astype(int)
df.drop('Name',1,inplace=True)
sns.distplot(df['Age'].dropna())
df['Age'].describe()
sns.barplot(df['Married'], df['Age'], hue=df['Pclass'] )

plt.show()
def impute_age(cols):

    married = cols[0]

    pclass = cols[1]

    if married==0 and pclass==1:

        return 32

    elif married==0 and pclass==2:

        return 20

    elif married==0 and pclass==3:

        return 15

    elif married==1 and pclass==1:

        return 42

    elif married==1 and pclass==2:

        return 32

    elif married==1 and pclass==3:

        return 30
df['Age'][df['Age'].isna()] = df[['Married','Pclass']].apply(impute_age,axis=1)
df['Senior_Citizen'] = df['Age'].apply(lambda x : 1 if x>60 else 0)
sns.barplot(df['Senior_Citizen'] , df['Survived'])
mr = (df.isna().sum()/len(df))*100

mr.sort_values(ascending=False)
df.head()
df['Family_members'] = df['SibSp']  + df['Parch']
df['Sex'] = df['Sex'].map({'male':1 , 'female':0})
df.head()
df['Pclass'] = df['Pclass'].astype('object')
dummies = pd.get_dummies(df[['Pclass']],drop_first=True)

dummies.head(2)
df.drop(['Pclass'],1,inplace=True)
df = pd.concat([df,dummies],1)
df.head()
from sklearn.preprocessing import StandardScaler



scale = StandardScaler()

df[['Age','Fare']] = scale.fit_transform(df[['Age','Fare']])
df.head()
df_train = df[df['Source']=='Train']

df_test = df[df['Source']=='Test']
df_train.drop(['PassengerId' , 'Source'],1,inplace=True)

df_test.drop('Source',1,inplace=True)
df_train['Survived'] = df_train['Survived'].astype(int)
X = df_train.drop('Survived',1)

y = df_train['Survived']
from sklearn.model_selection import train_test_split



X_train , X_val , y_train , y_val = train_test_split(X,y,test_size=0.30,random_state=123)
from sklearn.decomposition import PCA

import numpy as np
pca = PCA()

pca = pca.fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.grid(True)

plt.show()
pca_final = PCA(6)

X_train_pca = pca_final.fit_transform(X_train)

X_val_pca = pca_final.transform(X_val)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr = lr.fit(X_train , y_train)
y_pred_lr = lr.predict(X_val)
from sklearn.metrics import classification_report , confusion_matrix



from sklearn.preprocessing import binarize
y_pred_prob_yes = lr.predict_proba(X_val)
y_pred_lr = binarize(y_pred_prob_yes , 5/10)[:,1]
confusion_matrix(y_val,y_pred_lr)
print(classification_report(y_val,y_pred_lr))
from collections import Counter
Counter(y_train)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(3)

knn.fit(X_train,y_train)

y_pred_knn = knn.predict(X_val)

confusion_matrix(y_val,y_pred_knn)

print(classification_report(y_val , y_pred_knn))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
from sklearn.model_selection import GridSearchCV
param_grid = {

    

    'n_estimators':[10,20,30],

    'criterion': ['gini','entropy'],

    'max_depth': [10,15,20,25],

    'min_samples_split' : [5,10,15],

    'min_samples_leaf': [2,5,7],

    'random_state': [42,135,777],

    'class_weight': ['balanced' ,'balanced_subsample']

}
rf_tuned = GridSearchCV(estimator=rf , param_grid=param_grid , n_jobs=-1)
rf_tuned.fit(X_train , y_train)
rf_tuned.best_params_
rff = RandomForestClassifier(**rf_tuned.best_params_)

rff.fit(X_train , y_train)

y_pred_rff = rff.predict(X_val)

print(classification_report(y_val,y_pred_rff))
confusion_matrix(y_val , y_pred_rff)
rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_val)

confusion_matrix(y_val,y_pred_rf)
print(classification_report(y_val , y_pred_rf))
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import binarize
ada_boost = GradientBoostingClassifier()

ada_boost.fit(X_train , y_train)
y_pred_ada = ada_boost.predict(X_val)

y_pred_ada_prob = ada_boost.predict_proba(X_val)
from sklearn.preprocessing import binarize

for i in range(5,9):

    cm2=0

    y_pred_prob_yes=ada_boost.predict_proba(X_val)

    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]

    cm2=confusion_matrix(y_val,y_pred2)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

    
y_pred_ada_prob = binarize(y_pred_ada_prob , 7/10)[:,1]
confusion_matrix(y_val , y_pred_ada_prob)
print(classification_report(y_val,y_pred_ada_prob))
df_test.head(2)
X_test = df_test.drop('Survived',1)
pass_id = X_test['PassengerId']

X_test.drop('PassengerId',1,inplace=True)
X_train.shape
X_test.shape
X_test['Fare'][X_test['Fare'].isna()] = 5
X_train.head(2)
X_test.head(2)
X_test[['Age','Fare']] = scale.transform(X_test[['Age','Fare']])
y_pred_final = ada_boost.predict_log_proba(X_test)

y_pred_final = binarize(y_pred_final , 7/10)[:,1]
y_pred_sub_df = pd.DataFrame(y_pred_final , columns=['Survived'])

y_pred_sub_df['Survived'] = y_pred_sub_df['Survived'].astype(int)

y_pred_sub_df.head(2)
pass_id_df = pd.DataFrame(pass_id , columns=['PassengerId'])

pass_id_df.head(2)
submission_df = pd.concat([pass_id_df , y_pred_sub_df],1)

submission_df.head()
submission_df.shape
submission_df.to_csv('Titanic_Submissions_GB.csv' , index=False)