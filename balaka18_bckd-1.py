# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
from math import sqrt as s
import pandas as pd
import matplotlib as ml
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Sklearn libraries
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,LabelEncoder
import statsmodels.api as sm
from sklearn.metrics import r2_score,mean_squared_error

# Serializing
import pickle as pkl

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/quikr_car - quikr_car.csv')
print("Data shape : ",data.shape)
print("\nInformation :")
print(data.info())
print("\nDescription :")
print(data.describe(),"\n")
data.head()
# data['company_new'] = np.nan
names,models = list(data.name.values),list(data.company.values)
model_types = list(data.company.unique())
erronous = ['I','selling','URJENT','Used','Sale','very','i','2012','7','9','well','all','scratch','urgent','sell','Any',7,9,'']
# lower = list(map(lambda x : x.lower(), models))
filtered = list(map(lambda x : x.lower(),list(filter(lambda y : y not in erronous,model_types))))
splt,models_new = [],[]
for entry in names:
    splt.append(entry.split())
flag = 0
for i in range(len(splt)):
    if models[i] in erronous:
        for word in splt[i]:
            if word.lower() in filtered or word in filtered:
                models_new.append(word)
                break
            else:
                flag = flag + 1
                continue
        if flag == len(splt[i]):
            models_new.append('BrandX')
    else:
        models_new.append(models[i])
for _ in range(6):
    item = filtered[np.random.randint(0,len(filtered))]
    models_new.append(item)

'''print(len(filtered))
print(len(models_new))'''

data['company_new'] = models_new
data.drop(columns=['company'],axis=1,inplace=True)
data
pattern = r'[1-3][0-9]{3}'
yrs = list(data.year.values)
numbers_no = list(filter(lambda n : len(re.findall(pattern, n))==0, yrs))
numbers_yes = list(filter(lambda n : re.findall(pattern, n), yrs))
for i in range(len(yrs)):
    if yrs[i] in numbers_no:
        yrs[i] = int(numbers_yes[np.random.randint(0,len(numbers_yes))])
    else:
        yrs[i] = int(yrs[i])

data['year_new'] = yrs
data.drop(columns=['year'],axis=1,inplace=True)
data
# Price
price_actual = list(filter(lambda s : len(s.split())==1, list(data.Price.values)))
price_no = list(filter(lambda s : len(s.split())!=1, list(data.Price.values)))

for i in range(len(price_actual)):
    price_actual[i] = int(price_actual[i].replace(',',''))

for i in range(len(price_no)):
    price_no[i] = price_actual[np.random.randint(0,len(price_actual))]

price_actual.extend(price_no)

# Distance covered
dist = list(data.kms_driven.values)
filter1 = list(filter(lambda t1 : type(t1)==float, dist))
filter2 = list(filter(lambda t1 : type(t1)!=float and len(t1.split())==1, dist))
store = filter2
filter3 = list(filter(lambda t1 : type(t1)!=float and len(t1.split())!=1, dist))
for i in range(len(filter3)):
    try:
        filter3[i] = int((filter3[i].replace(' kms','')).replace(',',''))
    except:
        pass
filter3
for j in range(len(filter1)):
    filter1[j] = filter3[np.random.randint(0,len(filter3))]
for k in range(len(filter2)):
    filter2[k] = filter3[np.random.randint(0,len(filter3))]
filter3.extend(filter1)
filter3.extend(filter2)


data['price'] = price_actual
data['new_dist'] = filter3
data.drop(columns=['Price','kms_driven'],axis=1,inplace=True)
data
types = ['Petrol','Diesel','LPG']
data.fuel_type.fillna(types[np.random.randint(0,len(types))],inplace=True)
data
fd,cd = {},{}
le = LabelEncoder()
data['fuel_type'] = le.fit_transform(data['fuel_type'])
fuel = le.inverse_transform(data['fuel_type'])
for i in range(len(fuel)):
    fd[fuel[i]] = data['fuel_type'].values[i]
data['company_new'] = le.fit_transform(data['company_new'])
company = le.inverse_transform(data['company_new'])
for i in range(len(company)):
    cd[company[i]] = data['company_new'].values[i]
data_orig = data
data_orig
ohe = pd.get_dummies(data,columns=['fuel_type','company_new'],drop_first=True)
ohe.head(12)
# Getting the values for the form.
pkl.dump(fd,open('fuel2.pkl','wb'))
pkl.dump(cd,open('company2.pkl','wb'))
# Initializing fig and axes
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111,projection='3d')
# Scatter plot
ax.scatter(data.company_new.values, data.new_dist.values, data.price.values, zdir='z', s = 180, c = 'red', depthshade=True)
ax.set_xlabel('COMPANY')
ax.set_ylabel('DISTANCE')
ax.set_zlabel('PRICE')
ax.set_title('3D REPRESENTATION OF THE DATA')
plt.show()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111,projection='3d')
# Scatter plot
ax.scatter(data.year_new.values, data.new_dist.values, data.price.values, zdir='z', s = 180, c = 'red', depthshade=True)
ax.set_xlabel('YEAR')
ax.set_ylabel('DISTANCE')
ax.set_zlabel('PRICE')
ax.set_title('3D REPRESENTATION OF THE DATA')
plt.show()
plt.scatter(data.year_new,data.price,c='red')
plt.show()
plt.scatter(data.price,data.new_dist,c='red')
plt.show()
sns.distplot(data.price)
plt.show()
Y = np.array(data.iloc[:,2].values)
X = np.array(data.drop(columns=['price'],axis=1).iloc[:,1:].values)

# Train test split
trainx,testx,trainy,testy = train_test_split(X,Y,test_size = 0.2, random_state = 12)
print("TRAINING PHASE : trainx.shape = {} , trainy.shape = {}".format(trainx.shape,trainy.shape))
print("TESTING PHASE : testx.shape = {} , testy.shape = {}".format(testx.shape,testy.shape))
Y2 = np.array(ohe.iloc[:,2].values)
X2 = np.array(ohe.drop(columns=['price'],axis=1).iloc[:,1:].values)

# Train test split
trainx2,testx2,trainy2,testy2 = train_test_split(X2,Y2,test_size = 0.2, random_state = 12)
print("TRAINING PHASE : trainx.shape = {} , trainy.shape = {}".format(trainx2.shape,trainy2.shape))
print("TESTING PHASE : testx.shape = {} , testy.shape = {}".format(testx2.shape,testy2.shape))
scaler = StandardScaler()
trainx = scaler.fit_transform(trainx)
testx = scaler.fit_transform(testx)

# Linear Regression
lr = LinearRegression()
lr.fit(trainx, trainy)
y_pred1 = lr.predict(testx)
r2_score(testy,y_pred1)
scaler = StandardScaler()
trainx2 = scaler.fit_transform(trainx2)
testx2 = scaler.fit_transform(testx2)

# Linear Regression
lr2 = LinearRegression()
'''lr.fit(trainx, trainy)
y_pred1 = lr.predict(testx)'''
r2_score(testy2,y_pred1)
cross_val_score(lr2,X2,Y2,cv=45)
# Lasso Regression
lasso = Lasso(alpha=0.3)
cross_val_score(lasso,X,Y,cv=10)
# MLPRegressor
mlp = MLPRegressor()
param_grid = {'hidden_layer_sizes': [i for i in range(2,20)],
              'activation': ['relu'],
              'solver': ['adam'],
              'learning_rate': ['constant'],
              'learning_rate_init': [0.01],
              'power_t': [0.5],
              'alpha': [0.0001],
              'max_iter': [1000],
              'early_stopping': [True],
              'warm_start': [False]}
mlp_GS = GridSearchCV(mlp, param_grid=param_grid, 
                   cv=10, verbose=True, pre_dispatch='2*n_jobs')
mlp_GS.fit(trainx,trainy)
y_pred2 = mlp_GS.predict(testx)
r2_score(testy,y_pred2)
pkl.dump(lr,open('linreg2.pkl','wb'))
pkl.dump(mlp_GS,open('mlp3.pkl','wb'))
