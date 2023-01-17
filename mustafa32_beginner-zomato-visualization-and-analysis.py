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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score,r2_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB 
df = pd.read_csv("/kaggle/input/zomato-bangalore-restaurants/zomato.csv")
df.head()
df.drop(['url','phone','dish_liked','menu_item','address'],axis=1,inplace=True)

df.isnull().sum()
df.dropna(how='any',inplace=True)
df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'},inplace=True)

df = df[df.rate!='NEW']

df = df[df.rate!='-']

remove_slash = lambda x:x.replace('/5','') if type(x) == np.str else x

df.rate = df.rate.apply(remove_slash)





df.cost = df.cost.apply(lambda x:x.replace(',','') if type(x) == np.str else x)
plt.figure(figsize=(15,7))

sns.countplot(df['online_order'])

plt.title("Restaurants delivering online or not")
plt.figure(figsize=(15,7))

sns.countplot(df['book_table'])

plt.title("Restaurants Book Table or not")


plt.rcParams['figure.figsize'] = (13, 9)

Y = pd.crosstab(df['rate'], df['book_table'])

Y.div(Y.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])

plt.title('table booking vs rate', fontweight = 30, fontsize = 20)

plt.legend(loc="upper right")

plt.show()
g = sns.countplot(df['city']).set_xticklabels(sns.countplot(df['city']).get_xticklabels(), rotation=90, ha="right")
loc_plt = pd.crosstab(df['city'],df['rate'])

loc_plt.plot(kind='bar',stacked=True)

plt.title('Location - Rating',fontsize=15,fontweight='bold')

plt.ylabel('Location',fontsize=10,fontweight='bold')

plt.xlabel('Rating',fontsize=10,fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold');

plt.legend().remove();
plt.figure(figsize=(15,8))

plt.title("Restaurant Type in Banglore")

sns.countplot(df['rest_type']).set_xticklabels(sns.countplot(df['rest_type']).get_xticklabels(),rotation=90)
loc_plt = pd.crosstab(df['rate'],df['rest_type'])

loc_plt.plot(kind='bar',stacked=True)

plt.title('Location - Rating',fontsize=15,fontweight='bold')

plt.ylabel('Location',fontsize=10,fontweight='bold')

plt.xlabel('Rating',fontsize=10,fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold');

plt.legend().remove();
type_rate = pd.crosstab(df['rate'],df['type'])

type_rate.plot.bar(stacked=True)

# sns.countplot(type_rate)
sns.countplot(df['cost']).set_xticklabels(sns.countplot(df['cost']).get_xticklabels(),rotation=90)
plt.figure(figsize=(20,7))

sns.countplot(df['location']).set_xticklabels(sns.countplot(df['location']).get_xticklabels(),rotation=90)

plt.xlabel("Locations")

plt.ylabel("Frequency")

plt.title("No of Resturatent in Location")
plt.figure(figsize=(20,7))

chains = df['name'].value_counts()[:20]

sns.barplot(x=chains,y=chains.index)

plt.title("Most Number of Chains")
mostpopular = df.loc[df.rate.value_counts()[:20]]

mostpopular.dropna(inplace=True)

mostpopular.head()
g = sns.scatterplot(df['cost'],df['rate'])
dfx = df

dfx = dfx.drop(['name','reviews_list'],axis=1)

dfx.head()
enc = LabelEncoder()

dfx.online_order = enc.fit_transform(dfx.online_order)

dfx.book_table = enc.fit_transform(dfx.book_table)

dfx.location = enc.fit_transform(dfx.location)

dfx.rest_type = enc.fit_transform(dfx.rest_type)

dfx.cuisines = enc.fit_transform(dfx.cuisines)

dfx.type = enc.fit_transform(dfx.type)

dfx.city = enc.fit_transform(dfx.city)

dfx = dfx.dropna()

dfx.head()
X = dfx.drop(['rate'],axis=1)

y = dfx['rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
ref=DecisionTreeRegressor(min_samples_leaf=.0001)

ref.fit(X_train,y_train)

y_predict=ref.predict(X_test)

r2_score(y_test,y_predict)
ref = RandomForestRegressor(n_estimators=500,random_state=329,min_samples_leaf=.0001)

ref.fit(X_train,y_train)

y_predict = ref.predict(X_test)

r2_score(y_predict,y_test)
from sklearn.ensemble import  ExtraTreesRegressor

import joblib

ETree=ExtraTreesRegressor(n_estimators = 100)

ETree.fit(X_train,y_train)

y_predict=ETree.predict(X_test)

from sklearn.metrics import r2_score

r2_score(y_test,y_predict)

filename = 'zomato_prediction_rate.sav'

joblib.dump(ETree, filename)
gnb = GaussianNB() 

gnb.fit(X_train, y_train) 

y_pred = gnb.predict(X_test) 

accuracy_score(y_test, y_pred)
knn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1)    

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(accuracy_score(y_test,y_pred))
X_train.head()
X_train['city'].unique()