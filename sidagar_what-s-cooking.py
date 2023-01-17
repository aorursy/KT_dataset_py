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
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sb
zip = ZipFile('/kaggle/input/whats-cooking/train.json.zip','r')
zip.extractall()
zip = ZipFile('/kaggle/input/whats-cooking/test.json.zip','r')
zip.extractall()
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
feed = pd.read_json('/kaggle/working/train.json')
feed
ingre=set()

for i in feed['ingredients']:
    for j in i:
        if j in ingre:
            pass
        else:
            ingre.add(j)
print ("Number of Ingredients :" + str(len(ingre)))

cus= set()
for i in feed['cuisine']:
    if i in cus:
        pass
    else:
        cus.add(i)
print ("Number of Cuisines :"+str(len(cus)))
feed.drop('id',axis = 1).groupby('cuisine').count().plot(kind='bar')
ingre = sorted(ingre)
ingre
columns = list(ingre)

for i in columns:
    tem =[ ]
    for j in feed['ingredients']:
        if i in j:
            tem.append(1)
        else:
            tem.append(0)
    feed[i] = tem
feed.head()
feed.head()
feed.columns
feed = feed.drop(['id','ingredients'],axis = 1)
feed.head()
df_train_x = feed.drop('cuisine',axis = 1)
df_train_y = feed[['cuisine']]
a=[]
cuisine = list(cus)
for i in df_train_y.cuisine:
    a.append(cuisine.index(i))
df_train_y['cuisine'] = a
df_train_y.describe()
df_train_y.describe
x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.15, random_state=42)
acc=[]
gauss = GaussianNB().fit(x_train,y_train)
prediction = gauss.predict(x_test)
y_test['output'] = prediction

accuracy = 0
y_test['equal'] = np.where(y_test['cuisine']==y_test['output'],1,0)
for i in y_test.equal:
    accuracy +=i
accuracy = (accuracy/5967)*100
acc.append((accuracy))
dt = DecisionTreeRegressor().fit(x_train,y_train)
prediction = dt.predict(x_test)
y_test['output'] = prediction

accuracy = 0
y_test['equal'] = np.where(y_test['cuisine']==y_test['output'],1,0)
for i in y_test.equal:
    accuracy +=i
accuracy = (accuracy/5967)*100
acc.append((accuracy))
rf = RandomForestRegressor(n_estimators = 10).fit(x_train,y_train)
prediction = rf.predict(x_test)
y_test['output'] = prediction

accuracy = 0
y_test['equal'] = np.where(y_test['cuisine']==y_test['output'],1,0)
for i in y_test.equal:
    accuracy +=i
accuracy = (accuracy/5967)*100
acc.append((accuracy))
acc
y_test.describe
dt = DecisionTreeRegressor().fit(df_train_x,df_train_y)
test = pd.read_json('/kaggle/working/test.json')
columns = list(ingre)

for i in columns:
    tem =[]
    for j in test['ingredients']:
        if i in j:
            tem.append(1)
        else:
            tem.append(0)
    test[i] = tem
test.head()
test_output = test[['id']]
test.columns
test = test.drop(['id','ingredients'],axis=1)
test.columns
predictions = dt.predict(test)

test_output['cuisine'] = predictions

test_output.describe()
cuisines = []
test_output.describe
for i in test_output.cuisine:
    cuisines.append(cuisine[int(i)])
test_output['cuisine'] = cuisines 
test_output.describe
test_output.to_csv('submission.csv',index=False)
