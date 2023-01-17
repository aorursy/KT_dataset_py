# display an image

from IPython.display import Image

Image(url='https://miro.medium.com/max/844/1*MyKDLRda6yHGR_8kgVvckg.png') # Thanks manan-bedi2908 
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_columns' , None)
data = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')

data
data.shape
data.dtypes.value_counts()
data.isnull().sum()
df = data.drop(['RowNumber' , 'CustomerId' , 'Surname'] , axis=1)
# target 

df.Exited.value_counts(normalize=True)
# continuous and categorical variables



var_continuous = df.drop(['Exited' , 'Geography' , 'Gender' , 'HasCrCard' , 'IsActiveMember'] , axis = 1 )

var_categ = df[['Geography' , 'Gender' , 'HasCrCard' , 'IsActiveMember']]
var_continuous
var_categ
# Distributions continuous variables

for col in var_continuous:

    plt.figure()

    sns.distplot(df[col])
#viz categorical variables

for col in var_categ:

    plt.figure()

    df[col].value_counts().plot.pie()
# Dist target/variables



no_churn = df[df['Exited']==0]

churn = df[df['Exited'] == 1]
for col in var_continuous:

    plt.figure()

    sns.distplot(no_churn[col] , label = "negative")

    sns.distplot(churn[col] , label = "positive")

    plt.legend()
# Target / Age

plt.figure(figsize=(20,10))

sns.countplot(x='Age' , hue ='Exited' , data = df)
# Target / categorical variables

pd.crosstab(df['Exited'] , df.Geography)
for col in var_categ:

    plt.figure()

    sns.heatmap(pd.crosstab(df['Exited'] , df[col]) , annot=True)
# Preprocessing - Encoding



df = pd.get_dummies(df , drop_first=True)
plt.figure(figsize=(15,15))

sns.pairplot(df)
df
plt.figure(figsize=(20,10))

sns.heatmap(df.corr() , annot=True)
# Target and features 

X = df.drop(['Exited'] ,  axis=1)

y = df['Exited']

X
y
# train - test - split





from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.2 , random_state = 5)
# Standardizing the Dataset

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
X_train
# Features importance

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(X,y)
print(model.feature_importances_)
feat_importance = pd.Series(model.feature_importances_ , index=X.columns)

feat_importance.nlargest(5).plot(kind='barh') # 

plt.show()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix , f1_score , classification_report

cm = confusion_matrix(y_test,y_pred)

#f1 = f1_score(y_test , y_pred)

print(cm)

print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred)) #1

#print(f1)