import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
%matplotlib inline
train = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
train.head()
train.info()
plt.figure(figsize=(5,5))
sns.heatmap(train.isnull())
train.dropna(axis=0, inplace=True)
# função para arrumar a versão
def get_android_ver(x):
    if x == "Varies with device":
        return 4           # 4 it's median
    else:
        palavra =  re.findall(r'[^a-zA-Z]',x)
        palavra = "".join(palavra)
        return float(palavra[:3])
train['Android Ver'] = train['Android Ver'].apply(lambda x: get_android_ver(x))
train['Android Ver'].hist()
print(train['Android Ver'].median())
# funções para obter o tamanho do app
def get_string(x):
    if (len(x)) >5:
        return x
    elif (len(x)) <=5:
        string = re.findall(r'[a-zA-Z]',x)
        return "".join(string)
    
def get_number(x):
    number =  re.findall(r'[^a-zA-Z]',x)
    if len(x) > 5:
        return 0   # analisar essa parte
    else:
        return float("".join(number))
train['Strings'] = train['Size'].apply(lambda x: get_string(x))
train['Numbers'] = train['Size'].apply(lambda x: float(get_number(x)))
plt.title('Strings in dataset')
sns.countplot(train['Strings'])
# train['Numbers'].median()
# transformar tudo de kb para mb
def get_size(cols):
    letra = cols[0]
    numero = cols[1]
    if letra == 'k':
        tamanho = numero /1024
        return tamanho
    elif letra == 'M':
        tamanho = numero /1 
        return tamanho
    else:
        return "none"
train['Size'] = train[['Strings','Numbers']].apply(get_size, axis=1)
mediansize = train[train['Size']!='none']
sns.distplot(mediansize['Size'])
def get_size2(x):
    if x == "none":
        return 22.97
    else:
        return x
train['Size'] = train['Size'].apply(lambda x: get_size2(x))
# arrumar a coluna installs
def get_installs(x):
    x = x.replace(",","")
    x = x.replace("+","")
    return int(x)
train['Installs'] = train['Installs'].apply(lambda x: int(get_installs(x)))
train['Last Updated'] = train['Last Updated'].apply(lambda x: "".join(l for l in x if l not in string.punctuation))
train['Last Updated'] = train['Last Updated'].apply(lambda x: x.split())
train['Last Updated'].head()
train['Ano'] = train['Last Updated'][0][2]
train['Mês'] = train['Last Updated'][0][0]
months = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
          'August':8, 'September':9, 'October':10, 'November':11,'December':12 }
train['Mês'] = train['Mês'].apply(lambda x: months.get(x))
train['Ano'] =train['Ano'].astype(int)
train['Mês'] =train['Mês'].astype(int)

train['Reviews'] = train['Reviews'].astype(int)
def get_price(x):
    numero = re.findall('[^$]', x)
    return "".join(numero)
train['Price'] = train['Price'].apply(lambda x: get_price(x))
train['Price'] = train['Price'].astype(float)
train['Rating'].hist(bins=30, by=train['Type'])
train[['Type','Rating']].groupby('Type').mean()
category = train[['Category','Rating']].groupby('Category').mean().sort_values(ascending=False,by='Rating')
plt.figure(figsize=(15,15))
plt.title("Rating by Category")
sns.barplot(x=category['Rating'], y=category.index)
train.info()
train.drop(['Strings','Numbers','Current Ver','Last Updated'],axis=1, inplace=True)
objectcol = list(train.select_dtypes(include='object').columns)
objectcol
badcol = []
goodcol = []
for col in objectcol:
    nunique = train[col].nunique()
    print(f'{col}: {nunique}')
    if nunique > 10:
        badcol.append(col)
    else:
        goodcol.append(col)
print(badcol)
print(goodcol)
typedummie = pd.get_dummies(train['Type'])
train = pd.concat([train,typedummie],axis=1)
ratingdummie = pd.get_dummies(train['Content Rating'])
train = pd.concat([train,ratingdummie],axis=1)
train.drop(goodcol, axis=1, inplace=True)
train.drop(['App','Genres','Category'], axis=1, inplace=True)
train.head()
train.shape
y = train['Rating']
x = train.drop('Rating',axis=1)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
xtrain=sc_X.fit_transform(xtrain)
xtest=sc_X.transform(xtest)
from sklearn.tree import DecisionTreeRegressor
def score_dataset(xtrain,xtest,ytrain,ytest,n_estimators):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(xtrain,ytrain)
    pred = model.predict(xtest)
    return mean_absolute_error(ytest,pred)
index = []
score = []
for n in [100,200,300,400,500]:
    index.append(n)
    score.append(score_dataset(xtrain,xtest,ytrain,ytest,n,))
plt.figure(figsize=(7,7))
plt.title('Mean Absolut Error')
sns.lineplot(x=index,y=score)
plt.xlabel('n_estimators')
plt.ylabel('mae')
model = RandomForestRegressor(n_estimators=400, random_state=42)
model.fit(xtrain,ytrain)
predict = model.predict(xtest)
mean_absolute_error(ytest,predict)
print(f' Mean Absolut Error: {mean_absolute_error(ytest,predict)}')
print(f' Mean Squared Error: {mean_squared_error(ytest, predict)}')
print(f' SQRT  {np.sqrt(mean_squared_error(ytest, predict))}')
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
my_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                              ('model', RandomForestRegressor(n_estimators=400,
                                                              random_state=42))
                             ])
scores = -1 * cross_val_score(my_pipeline, x, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
index = np.arange(1,6)
plt.figure(figsize=(7,7))
plt.title('Mean Absolut Error')
sns.lineplot(x=index,y=scores)
plt.xlabel('cv')
plt.ylabel('mae')
print(scores.mean())