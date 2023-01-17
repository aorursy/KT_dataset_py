import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

color=sns.color_palette()
data=pd.read_csv('../input/BlackFriday.csv')

data.head()
fig = plt.figure(figsize=(8,5))

sns.distplot(data.Purchase)

plt.title('Purchase Distribution')
print("skew",data.Purchase.skew(),"kurt",data.Purchase.kurt())
data.describe(include = ['object', 'integer', 'float'])
sns.heatmap(data.isnull(), cmap= 'Blues')
data.User_ID.plot.hist(bins=70)
df = pd.DataFrame(data = data.User_ID.value_counts())

df=df.reset_index()

df.columns = ['users', 'Purchase_History']

data = data.merge(df, left_on = 'User_ID', right_on = 'users')
data.head()
del data['User_ID']

del data['users']
del data['Product_ID']
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.boxplot(x='Gender',y='Purchase',data=data,palette='Set3',ax=ax[0])

ax[0].set_title("F -> Female , M -> Male",size=12)

data.Gender.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True, explode=[0.1,0],cmap='Blues')

ax[1].set_title("Total")
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.boxplot(x='Age',data=data,y='Purchase',palette='Set2',ax=ax[0])

ax[0].set_title('Purchase v/s Age',size=12)

data.Age.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Oranges')
fig,ax=plt.subplots(2,1,figsize=(15,12))

sns.boxplot(x='Occupation',data=data,y='Purchase',palette='Set1',ax=ax[0])

ax[0].set_title('Purchase v/s Occupation')

data.Occupation.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Reds')
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.boxplot(x='City_Category',data=data,y='Purchase',palette='Set3',ax=ax[0])

ax[0].set_title('Purchase v/s City Category', size=12)

data.City_Category.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Greens', 

                                           explode=[0.05,0.05,0.05])
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.boxplot(x='Stay_In_Current_City_Years',data=data,y='Purchase',palette='Set3',ax=ax[0])

ax[0].set_title('Purchase v/s No. of years stayed')

data.Stay_In_Current_City_Years.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Greys')
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.boxplot(x='Marital_Status',data=data,y='Purchase',palette='Set3',ax=ax[0])

data.Marital_Status.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',explode=[0.05,0.05],shadow=True, cmap='Blues')
fig,ax=plt.subplots(1,2,figsize=(15,6))

sns.boxplot(y='Purchase',data=data,x='Product_Category_1',palette='Set2',ax=ax[0])

sns.countplot(x= 'Product_Category_1',ax=ax[1], data= data)
fig,ax=plt.subplots(1,2,figsize=(15,6))

sns.boxplot(y='Purchase',data=data,x='Product_Category_2',palette='Set2',ax=ax[0])

sns.countplot(x= 'Product_Category_2',ax=ax[1], data= data)
fig,ax=plt.subplots(1,2,figsize=(15,6))

sns.boxplot(y='Purchase',data=data,x='Product_Category_3',palette='Set2',ax=ax[0])

sns.countplot(x= 'Product_Category_3',ax=ax[1], data= data)

del data['Product_Category_3']

data.Product_Category_2.fillna(0, inplace=True)
def outliers(df):

    q1= pd.DataFrame(df.quantile(0.25))

    q3= pd.DataFrame(df.quantile(0.75))

    iqr = pd.DataFrame(q3[0.75] - q1[0.25])

    iqr['lower'] = q1[0.25] - 1.5 * iqr [0]

    iqr['upper'] = q3[0.75] + 1.5 * iqr [0]

    return(np.where(df > iqr['upper']) or (df < iqr['lower']))
x = outliers(pd.DataFrame(data.Purchase))

data = data.drop(x[0])
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.boxplot(x='Gender',y='Purchase',data=data,palette='Set3',ax=ax[0])

ax[0].set_title("F -> Female , M -> Male",size=12)

sns.boxplot(x='Age',data=data,y='Purchase',palette='Set2',ax=ax[1])

ax[1].set_title('Purchase v/s Age',size=12)
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.boxplot(x='City_Category',data=data,y='Purchase',palette='Set1',ax=ax[0])

sns.boxplot(x='Stay_In_Current_City_Years',data=data,y='Purchase',palette='Set2',ax=ax[1])
fig=plt.figure(figsize=(12,8))

sns.heatmap(data.corr(), annot= True, cmap='Blues')
data.info()
data.Product_Category_1=data.Product_Category_1.astype('category')

data.Marital_Status=data.Marital_Status.astype('category')

data.Occupation=data.Occupation.astype('category')

data.Product_Category_2=data.Product_Category_2.astype('category')
data_label=data['Purchase']

del data['Purchase']

data_label=pd.DataFrame(data_label)
data=pd.get_dummies(data,drop_first=True)

data.head()
from sklearn.preprocessing import MinMaxScaler

data_scaled=MinMaxScaler().fit_transform(data)

data_scaled=pd.DataFrame(data=data_scaled, columns=data.columns)
data_scaled.head()
data_scaled.shape
from sklearn.decomposition import PCA

variance_ratio = []

for i in range(5,65,5):

    pca=PCA(n_components = i)

    pca.fit_transform(data_scaled)

    variance_ratio = np.append(variance_ratio,np.sum(pca.explained_variance_ratio_))
df =pd.Series(data = variance_ratio, index = range(5,65,5))

df.plot.bar(figsize=(8,6))
pca=PCA(n_components = 40, whiten = False, random_state=876)

data_scaled = pd.DataFrame(pca.fit_transform(data_scaled), index= data_scaled.index)
data_scaled.head()
from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest = train_test_split(data_scaled, data_label, test_size=0.30,random_state=54368)
from sklearn.linear_model import SGDRegressor

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
def CrossVal(dataX,dataY,mode,cv=3):

    score=cross_val_score(mode,dataX , dataY, cv=cv, scoring='neg_mean_squared_error')

    return(np.sqrt(np.mean((-score))))
sgd=SGDRegressor(random_state=324,penalty= "l1", alpha=0.4)

score_sgd=CrossVal(Xtrain,Ytrain,sgd)

print("RMSE is : ",score_sgd)
lr=LinearRegression(n_jobs=-1)

score_lr=CrossVal(Xtrain,Ytrain,lr)

print("RMSE is : ",score_sgd)
dtc=DecisionTreeRegressor(random_state=42234)

score_dtc=CrossVal(Xtrain,Ytrain,dtc)

print("RMSE is : ",score_dtc)
rf=RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=487987)

score_rf= CrossVal(Xtrain,Ytrain,rf)

print('RMSE is:',score_rf)
etc=ExtraTreesRegressor(n_estimators=10, n_jobs=-1, random_state=3141)

score_etc= CrossVal(Xtrain,Ytrain,etc)

print('RMSE is:',score_etc)
model_accuracy = pd.Series(data=[score_sgd, score_lr, score_dtc, score_rf, score_etc], 

                           index=['Stochastic GD','linear Regression','decision tree', 'Random Forest',

                            'Extra Tree'])

fig= plt.figure(figsize=(8,8))

model_accuracy.sort_values(ascending= False).plot.barh()

plt.title('MODEL RMSE SCORE')