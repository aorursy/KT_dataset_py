# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/titanic/train.csv")
df.head(2)
df.describe()
df['Survived'].value_counts(normalize=True)
import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go
fig=px.histogram(df['Fare'])

fig.show("notebook")
y1=df.groupby(['Sex']).count()['PassengerId']

fig=px.bar(y1,x=y1.index,y=y1,color=y1.index,title='Sex vs Count')

fig.show("notebook")
y1=df.groupby(['Pclass']).count()['PassengerId']

fig=px.bar(y1,x=y1.index,y=y1,color=y1.index,title='Ticket Class vs Count')

fig.show("notebook")
y1=df.groupby(['SibSp']).count()['PassengerId']

fig=px.bar(y1,x=y1.index,y=y1,color=y1.index,title='# of siblings / spouses')

fig.show("notebook")
y1=df.groupby(['Parch']).count()['PassengerId']

fig=px.bar(y1,x=y1.index,y=y1,color=y1.index,title='# of parents / children')

fig.show("notebook")
fig=px.histogram(df['Age'],title='Distribution of Age')

fig.show("notebook")
y1=df.groupby(['Embarked']).count()['PassengerId']

fig=px.bar(y1,x=y1.index,y=y1,color=y1.index,title='Port of Embarkation')

fig.show("notebook")
df.isnull().sum()
fig=px.box(df['Age'],title="Boxplot of Age")

fig.show('notebook')
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode(),inplace=True)
df.dtypes
cat_columns=['Sex','Embarked']



df_cat=pd.get_dummies(df[cat_columns])



df_copy=pd.concat([df,df_cat],axis=1)

df_copy.drop(cat_columns,inplace=True,axis=1)

df_copy.head(2)
df_copy.columns
from sklearn.model_selection import train_test_split 



X=df_copy.copy()



X.drop(['PassengerId', 'Survived','Name','Ticket','Cabin'],inplace=True,axis=1)



y=df_copy['Survived']
X.head(2)
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=123)

print("Shape of training features:",X_train.shape)

print("Shape of training labels:",y_train.shape)

print("Shape of validation features:",X_val.shape)

print("Shape of validation labels:",y_val.shape)
from sklearn.preprocessing import PowerTransformer

pt=PowerTransformer(standardize=True)



X_train_transformed=pt.fit_transform(X_train)



#not fitting on validation set. 

X_val_transformed=pt.transform(X_val)
X_train_transformed_df=pd.DataFrame(X_train_transformed,columns=X.columns)

X_train_transformed_df.head(2)
from statsmodels.stats.outliers_influence import variance_inflation_factor



#converting to df as vif requires(value, column index)-https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html

X_train_transformed_df.shape

vif=[variance_inflation_factor(X_train_transformed_df.values,i) for i in range(X_train_transformed_df.shape[1])]
vif
X_train_transformed_df.head(2)
X_train_cleaned_df=X_train_transformed_df.drop(['Sex_female','Embarked_S'],axis=1)

vif=[variance_inflation_factor(X_train_cleaned_df.values,i) for i in range(X_train_cleaned_df.shape[1])]
vif
#repeating above steps for validation set as well



X_val_transformed_df=pd.DataFrame(X_val_transformed,columns=X.columns)

X_val_cleaned_df=X_val_transformed_df.drop(['Sex_female','Embarked_S'],axis=1)
X_train_final=np.array(X_train_cleaned_df)

X_val_final=np.array(X_val_cleaned_df)
#using SMOTE with ENN to balance the dataset



from imblearn.combine import SMOTEENN 



smen = SMOTEENN(random_state=12, sampling_strategy = 1.0)

X_train_smen, y_train_smen = smen.fit_sample(X_train_final, y_train)

len(y_train[y_train==0]),len(y_train[y_train==1])
len(y_train_smen[y_train_smen==0]),len(y_train_smen[y_train_smen==1])
from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix 

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
LR=LogisticRegression(random_state=0) 

RF=RandomForestClassifier(random_state=0)

SV=svm.SVC(random_state=0)

XG=XGBClassifier(random_state=0)



LR_cross_val= cross_val_score(LR,X_train_smen,y_train_smen,scoring='accuracy',cv=5)

RF_cross_val= cross_val_score(RF,X_train_smen,y_train_smen,scoring='accuracy',cv=5)

SV_cross_val= cross_val_score(SV,X_train_smen,y_train_smen,scoring='accuracy',cv=5)

XG_cross_val= cross_val_score(XG,X_train_smen,y_train_smen,scoring='accuracy',cv=5)
print("5 fold Cross validation accuracy for logistic Regression:{:.3f}".format(LR_cross_val.mean()))

print("5 fold Cross validation accuracy for Random Forest:{:.3f}".format(RF_cross_val.mean()))

print("5 fold Cross validation accuracy for SVM:{:.3f}".format(SV_cross_val.mean()))

print("5 fold Cross validation accuracy for XGBoost:{:.3f}".format(XG_cross_val.mean()))
#Bayesian optimization over hyper parameters.



from skopt import BayesSearchCV



param_grid= [{'n_estimators': [70,80,90,100], #default is 100

              'max_features': [2,3,4],

              'max_depth':[5,8,10,50,100],

              'min_samples_leaf': [2,3,4,5]

             }]



tuned_RF=BayesSearchCV(RF,param_grid,cv=5,scoring='accuracy',random_state=12,n_jobs=-1)

tuned_RF.fit(X_train_smen,y_train_smen)
#Print tuned RF parameters



print("Best hyperparameters are:",tuned_RF.best_params_)

print("Best accuracy score is:",tuned_RF.best_score_)
#print classification report



from sklearn.metrics import classification_report



final_model=RandomForestClassifier(random_state=0,max_depth=10,

                                   max_features=3,min_samples_leaf=2,n_estimators=100)

final_model.fit(X_train_smen,y_train_smen)

yhat=final_model.predict(X_train_smen)

print(classification_report(y_train_smen,yhat))
yhat=final_model.predict(X_val_final)

print("Accuracy on validation set is:",accuracy_score(y_val,yhat))
print(classification_report(y_val,yhat))
val_scores=cross_val_score(final_model,X_val_final,y_val,cv=5,scoring='accuracy')

print("Average accuracy on validation set:",val_scores.mean())
test_df=pd.read_csv("/kaggle/input/titanic/test.csv")
test_df.isnull().sum()
test_df['Age'].fillna(test_df['Age'].mean(),inplace=True)

test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)



### One hot encoding of categorical variables



cat_columns=['Sex','Embarked']



test_df_cat=pd.get_dummies(test_df[cat_columns])



test_df_copy=pd.concat([test_df,test_df_cat],axis=1)

test_df_copy.drop(cat_columns,inplace=True,axis=1)

test_df_copy.head(2)
X_test=test_df_copy.copy()



X_test.drop(['PassengerId','Name','Ticket','Cabin'],inplace=True,axis=1)
X_test.head(2)
from sklearn.preprocessing import PowerTransformer



#not fitting on validation set. 

X_test_transformed=pt.transform(X_test)
#repeating above steps for validation set as well



X_test_transformed_df=pd.DataFrame(X_test_transformed,columns=X_test.columns)

X_test_cleaned_df=X_test_transformed_df.drop(['Sex_female','Embarked_S'],axis=1)
X_test_final=np.array(X_test_cleaned_df)
predictions=tuned_RF.predict(X_test_final)


final_df=pd.concat([test_df_copy['PassengerId'],pd.DataFrame(predictions,columns=["Survived"])],axis=1)

final_df.head()
final_df.to_csv('submissions.csv', header=True, index=False)