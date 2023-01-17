# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Melbourne_housing_extra_data.csv")
df.head(5)
df.shape
df.isnull().sum()
sns.countplot(df['Method'])
sns.regplot(x='Rooms',y='Price',data=df)
sns.boxplot(x='Rooms',y='Price',data=df)
df['Price'].skew()
plt.plot(df['Price'])
plt.figure(figsize=(13,5))

sns.boxplot(x='CouncilArea',y='Price',data=df)
plt.figure(figsize=(12,5))

sns.countplot(df['CouncilArea'])
sns.countplot(df['Type'])
df['Price']=df['Price'].fillna(0)
df_test=df[df.Price == 0]
df_test.shape
#df = df.price[df.price > 0]

#Drop column where price is zero

df = df.drop(df[df.Price == 0.0].index)
df.head(5)
df.isnull().sum()
cluster=['Rooms','Type','Bedroom2','Bathroom','Landsize','Car','BuildingArea','YearBuilt','Price']

price=df['Price']
km=df[cluster]
from sklearn import model_selection, preprocessing

for c in km.columns:

    if km[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(km[c].values)) 

        km[c] = lbl.transform(list(km[c].values))

        #x_train.drop(c,axis=1,inplace=True)
km.head(3)
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)

model.fit(km)

x = model.fit_predict(km)

km["Cluster"]= x
#km=km.sort(['Cluster'])

km.head(5)
df['Cluster']=km['Cluster']
df.head(5)
sns.countplot(df['Cluster'])
df=df.replace({'Unavailable': None }, regex=True)
df.BuildingArea=pd.to_numeric(df.BuildingArea)
build_1=np.mean(df.BuildingArea[df.Cluster ==0 ])

build_2=np.mean(df.BuildingArea[df.Cluster ==1 ])

build_3=np.mean(df.BuildingArea[df.Cluster ==2 ])
corr=df.corr()

corr = (corr)

plt.figure(figsize=(5,5))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws=



{'size': 15},

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')
df.BuildingArea[df.Cluster ==0]=df.BuildingArea[df.Cluster ==0].fillna(build_1)

df.BuildingArea[df.Cluster ==1]=df.BuildingArea[df.Cluster ==1].fillna(build_2)

df.BuildingArea[df.Cluster ==2]=df.BuildingArea[df.Cluster ==2].fillna(build_3)
#df['BuildingArea']=df.BuildingArea[df.Cluster ==0].fillna(build_1)

#df['BuildingArea']=df.BuildingArea[df.Cluster ==1].fillna(build_2)

#df['BuildingArea']=df.BuildingArea[df.Cluster ==2].fillna(build_3)
df.Landsize=pd.to_numeric(df.Landsize)
corr=df.corr()

corr = (corr)

plt.figure(figsize=(5,5))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws=



{'size': 15},

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')
df.Landsize=pd.to_numeric(df.Landsize)

land_1=np.mean(df.Landsize[df.Cluster ==0 ])

land_2=np.mean(df.Landsize[df.Cluster ==1 ])

land_3=np.mean(df.Landsize[df.Cluster ==2 ])





df.Landsize[df.Cluster ==0]=df.Landsize[df.Cluster ==0].fillna(land_1)

df.Landsize[df.Cluster ==1]=df.Landsize[df.Cluster ==1].fillna(land_2)

df.Landsize[df.Cluster ==2]=df.Landsize[df.Cluster ==2].fillna(land_3)
corr=df.corr()

corr = (corr)

plt.figure(figsize=(14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws=



{'size': 15},

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')
df.head(5)
df.Bedroom2=pd.to_numeric(df.Bedroom2)

df.Bathroom=pd.to_numeric(df.Bathroom)

df.Car=pd.to_numeric(df.Car)
df.dtypes
df.head(5)
df.groupby('CouncilArea')['CouncilArea'].count()
plt.figure(figsize=(10,5))

sns.countplot(df['CouncilArea'])
dic={'Banyule':1,'Bayside':2 ,'Boroondara':3 ,'Brimbank':4 ,'Darebin':5,         

'Glen Eira':6,      

'Hobsons Bay':7,      

'Hume':8,              

'Kingston':9,         

'Manningham':10,       

'Maribyrnong':11,      

'Melbourne':12,        

'Monash':13,           

'Moonee Valley':14,    

'Moreland':15,         

'Port Phillip':16,     

'Stonnington':17,      

'Whitehorse':18,       

'Yarra':19}   
df['CouncilArea']=df.CouncilArea.map(dic)
corr=df.corr()

corr = (corr)

plt.figure(figsize=(14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws=



{'size': 15},

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')
df=df.fillna(0)
df.YearBuilt=pd.to_numeric(df.YearBuilt)
df=df.drop(['Lattitude','Longtitude'],axis=1)
df.isnull().sum()
from sklearn import model_selection, preprocessing

for c in df.columns:

    if df[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df[c].values)) 

        df[c] = lbl.transform(list(df[c].values))
df.head(3)
#Train-Test split

from sklearn.model_selection import train_test_split

label = df.pop('Price')

label=np.log(label)

data_train, data_test, label_train, label_test = train_test_split(df, label, test_size = 0.2, random_state = 500)
import xgboost as xgb

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}
dtrain = xgb.DMatrix(data_train, label_train)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)
num_boost_rounds = len(cv_output)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
fig, ax = plt.subplots(1, 1, figsize=(8, 13))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
dtest=xgb.DMatrix(data_test)
y_predict = np.exp2(model.predict(dtest))
Actual_Price=np.exp2(label_test)
out = pd.DataFrame({'Actual_Price': Actual_Price, 'predict_Price': y_predict,'Diff' :(Actual_Price-y_predict)})

out[['Actual_Price','predict_Price','Diff']].head(10)
sns.regplot(out['predict_Price'],out['Diff'])
sns.regplot(out['Actual_Price'],out['Diff'])