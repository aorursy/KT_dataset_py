import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

df=pd.read_csv('/kaggle/input/titanic/train.csv')

df.head()
df.info()
print(df.isnull().sum())
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.title('Empty entries')

plt.show()
df['Embarked'].value_counts()
df.loc[df['Embarked'].isnull()==True,'Embarked']='S'
df.drop('Cabin', axis=1, inplace=True)
sns.countplot(df['Survived'])

print('Approximately {:.1f}% of total passengers survived'.format(100*df[df['Survived']==1]['Survived'].sum()/len(df)))
fig,ax=plt.subplots(1,1,figsize=(10,5))

sns.heatmap(df.corr(),annot=True,ax=ax,cmap='summer_r')
f,ax=plt.subplots(2,figsize=(10,12))



sns.countplot('Sex', hue='Survived', data=df, ax=ax[0])

ax[0].set_title('Survival with respect to sex')



sns.countplot(x='Embarked', data=df, hue='Survived', ax=ax[1])

ax[1].set_title('Survival with respect to Embarked')



plt.show()
len(df['Name'].unique())
df['Name'].head(10)
import re



def titleExtractor(string):

    return re.findall(r'\w+\.', string)[0][:-1]



df['Title']=df['Name'].apply(titleExtractor)



df['Title'].value_counts()
df['Title'].replace(

    ['Countess','Don', 'Mme','Ms', 'Jonkheer', 'rs., L', 'Lady', 'Sir', 'Capt', 'Major', 'Mlle', 'Col', 'Dr','Rev'],

    ['Other']*14,

    inplace=True   )

df['Title'].value_counts()
df.drop('Name', axis=1, inplace=True)

df['Sex'].replace(['male', 'female'], [0,1], inplace=True)

df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

df['Title'].replace(list(df['Title'].unique()), [0,1,2,3,4], inplace=True)

df.head()
print(len(df['Ticket'].unique()))

df['Ticket'].head(20)
df.drop('Ticket', axis=1, inplace=True)

df.head()
fig,ax=plt.subplots(1,1,figsize=(10,5))

sns.heatmap(df.corr(),annot=True,ax=ax,cmap='summer_r')
df.groupby('Pclass')['Age'].median()
def fill_age(cols):

    age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return age

df['Age'] = df[['Age','Pclass']].apply(fill_age,axis=1)
df.isnull().sum()
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb



x_train, x_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1),

                                                    df['Survived'], 

                                                    test_size=0.33, 

                                                    random_state=1)
#Random forest 

rfc=RandomForestClassifier(n_estimators=500)

rfc.fit(x_train, y_train)

rfc_predictions=rfc.predict(x_test)



print(classification_report(y_test, rfc_predictions))
xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=500,

                            learning_rate=0.05)



xgboost.fit(x_train, y_train)

xgb_predictions= xgboost.predict(x_test)



print(classification_report(y_test, xgb_predictions))
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA







scaler = StandardScaler()

scaler.fit(df)

scaled_data = scaler.transform(df)



pca = PCA(n_components=3)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)



df_pca=pd.DataFrame(df['Survived'], columns=['Survived'])

df_pca['pca1']=x_pca[:,0]

df_pca['pca2']=x_pca[:,1]

df_pca['pca3']=x_pca[:,2]
plt.figure(figsize=(8,6))

plt.scatter(x_pca[:,0],x_pca[:,1],c=df['Survived'],cmap='plasma')

plt.xlabel('First principal component')

plt.ylabel('Second Principal Component')

plt.show()
df_comp = pd.DataFrame(pca.components_,columns=df.columns)

plt.figure(figsize=(12,6))

sns.heatmap(df_comp,cmap='summer_r',annot=True)

plt.show()
x_train, x_test, y_train, y_test = train_test_split(df_pca.drop('Survived', axis=1),

                                                    df_pca['Survived'], 

                                                    test_size=0.33, random_state=1)
xgboost_pca = xgb.XGBClassifier(max_depth=5,

                                n_estimators=500, 

                                learning_rate=0.05)



xgboost_pca.fit(x_train, y_train)

xgb_prediction_pca = xgboost_pca.predict(x_test)



print(classification_report(y_test, xgb_prediction_pca))
rfc_pca=RandomForestClassifier(n_estimators=500)



rfc_pca=rfc_pca.fit(x_train, y_train)



predictions_pca=rfc_pca.predict(x_test)



print(classification_report(y_test, predictions_pca))