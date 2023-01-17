import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.head()
df.Pclass.unique()
df.Parch.unique()
df.Embarked.unique()
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=df,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=df,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=df,palette='rainbow')

#Dalam usaha penyelamatan penumpang kapal titanic, lebih diutamakan wanita dan tidak melihat kelas apa
df['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Age',data=df,palette='rainbow')
plt.figure(figsize=(12,7))

sns.boxplot(x='Pclass',y='Age',data=df,palette='rainbow')
df.groupby('Pclass')['Age'].mean()
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1:

            return 38

        

        elif Pclass == 2:

            return 30

        

        else:

            return 24

        

    else:

        return Age

        
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.drop('Cabin',axis=1,inplace=True)
df.head()
df.dropna(inplace=True)
df.info()
sex = pd.get_dummies(df['Sex'],drop_first=True)

embark = pd.get_dummies(df['Embarked'],drop_first=True)
df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df1 = pd.concat([df,sex,embark],axis=1)
df1.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df1.drop('Survived',axis=1),df1['Survived'],test_size=0.30,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
prediction = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
confusion_matrix(y_test,prediction)
roc_auc_score(y_test,prediction)
print("gini ratio : ", 2*roc_auc_score(y_test,prediction)-1)