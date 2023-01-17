import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

sns.set_palette('summer_r')
import warnings
warnings.filterwarnings('ignore')
data1 = pd.read_csv('../input/my-dataset/train.csv')
data2 = pd.read_csv('../input/my-dataset/test.csv')
x_train = data1.drop(['PassengerId','Survived','Name','Ticket'],axis=1)
y_train = data1.Survived
x_test = data2.drop(['PassengerId','Name','Ticket'],axis=1)

# print(data1.head())
# print(data2.head())
x_train['Cabin'] = x_train['Cabin'].fillna('X')
x_test['Cabin'] = x_test['Cabin'].fillna('X')
new_cabin = pd.Series(x_train['Cabin'].str.slice(0,1))
i=0

x_train['Cabin'] = new_cabin.values

new_cabin = pd.Series(x_test['Cabin'].str.slice(0,1))
x_test['Cabin'] = new_cabin.values
print(data1.isnull().sum(),data2.isnull().sum())


from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder
le = LabelEncoder()
x_train['Embarked'] = x_train['Embarked'].fillna('$')
x_train['Embarked'] = le.fit_transform(x_train['Embarked'])
x_train['Cabin'] = le.fit_transform(x_train['Cabin'])
imr = Imputer(missing_values=8,strategy='median',axis=0,copy=False)
x_train[['Cabin']]=imr.fit_transform(x_train[['Cabin']])
imr.set_params(missing_values = np.nan,strategy='mean')
x_train[['Age']] = imr.fit_transform(x_train[['Age']])
imr.set_params(missing_values=3,strategy='most_frequent')
x_train[['Embarked']] =  imr.fit_transform(x_train[['Embarked']])
ohe = OneHotEncoder(categorical_features=[1])
x_train['Sex'] = le.fit_transform(x_train['Sex'])

print(x_train.head())
fig,ax1 = plt.subplots(figsize=(10,10))
sns.heatmap(data=x_train.corr(),annot=True,fmt='.1f',linewidths=.1)

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,10))
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_train_scaled = pd.DataFrame(data=x_train_scaled)
x_train_scaled.columns = ['Pclass','Sex','Age','SibSp','Parch','Ticket','Cabin','Embarked']
temp_data = pd.concat((y_train,x_train_scaled.loc[:,:]),axis=1)
temp_data = pd.melt(temp_data,id_vars='Survived',value_name='value',var_name='features')
sns.boxplot(data=temp_data,x='features',y='value',hue='Survived')
plt.figure(figsize=(12,12))
sns.violinplot(data=temp_data,x='features',y='value',hue='Survived',split=True)
x_test['Embarked'] = x_test['Embarked'].fillna('$')
x_test['Embarked'] = le.fit_transform(x_test['Embarked'])
x_test['Cabin'] = le.fit_transform(x_test['Cabin'])
imr = Imputer(missing_values=8,strategy='median',axis=0,copy=False)
x_test[['Cabin']]=imr.fit_transform(x_test[['Cabin']])
imr.set_params(missing_values = np.nan,strategy='mean')
x_test[['Age']] = imr.fit_transform(x_test[['Age']])
imr.set_params(missing_values=3,strategy='most_frequent')
x_test[['Embarked']] =  imr.fit_transform(x_test[['Embarked']])
imr.set_params(missing_values = np.nan)
x_test[['Fare']] = imr.fit_transform(x_test[['Fare']])
ohe = OneHotEncoder(categorical_features=[1])
x_test['Sex'] = le.fit_transform(x_test['Sex'])


tes_Scaler = sc.fit_transform(x_test)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(x_train_scaled)
x_train_new = pca.transform(x_train_scaled)
pca.fit(tes_Scaler)
x_test_new = pca.transform(tes_Scaler)
blah = x_train_new
from sklearn.linear_model import LogisticRegression
x_train_new = pd.DataFrame(data=x_train_new,copy=False)
x_train_new.columns = ['A','B']
x_train_new = pd.concat((y_train,x_train_new),axis=1)
sns.set_palette('muted')
sns.lmplot(x='A',y='B',data=x_train_new,hue='Survived')
from sklearn.metrics import accuracy_score
lr = LogisticRegression()

lr.fit(x_train_scaled,y_train)
pred = lr.predict(tes_Scaler)
print(pred)


a = []

final_data= pd.DataFrame(data=pred,columns=['Survived'])
final_data.to_csv('output2.csv')



