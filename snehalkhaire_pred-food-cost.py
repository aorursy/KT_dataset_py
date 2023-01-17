# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sub_file=pd.read_excel("/kaggle/input/pred-of-food-cost/Sample_submission.xlsx")

sub_file.head()
train=pd.read_excel("/kaggle/input/pred-of-food-cost/Data_Train.xlsx")

train.head()
test=pd.read_excel("/kaggle/input/pred-of-food-cost/Data_Test.xlsx")

test.head()
train.isnull().mean()
test.isnull().mean()
train.describe()
train["CITY"] = train["CITY"].fillna('NOTFOUND')

test["CITY"] = test["CITY"].fillna('NOTFOUND')

train["LOCALITY"] = train["LOCALITY"].fillna('NOTFOUND')

test["LOCALITY"] = test["LOCALITY"].fillna('NOTFOUND')

train["RATING"] = train["RATING"].replace({'NEW': '3.7','-':'3.7',np.nan:'0.0'})

test["RATING"] = test["RATING"].replace({'NEW': '3.7','-':'3.7',np.nan:'0.0'})

train["VOTES"] = train["VOTES"].fillna('0.0 Votes')

test["VOTES"] = test["VOTES"].fillna('0.0 Votes')
train.isnull().mean()
test.isnull().mean()
train["CITY"].unique()
train["VOTES"]=train["VOTES"].map(lambda x:x.split(' ')[0])

test["VOTES"]=test["VOTES"].map(lambda x:x.split(' ')[0])
train["RATING"].unique()
train['RATING']=train['RATING'].astype(np.float64)

train['RATINGS']=pd.cut(train['RATING'],4,labels=['0','1','2','3']).astype(np.object)

test["RATING"]=test["RATING"].astype(np.float64)

test["RATINGS"]=pd.cut(test["RATING"],4,labels=['0','1','2','3']).astype(np.object)
train["SERVINGS"]=train["TIME"].apply(lambda x:2 if len(x.split(','))>1 else 1)
test["SERVINGS"]=test["TIME"].apply(lambda x:2 if len(x.split(','))>1 else 1)
train["CUISINES"].unique()
#For the values of TIME which are closed

import re

def extract_closed(time):

    a = re.findall('Closed \(.*?\)', time)

    if a != []:

        return a[0]

    else:

        return 'NA'



train['CLOSED'] = train['TIME'].apply(extract_closed)

test['CLOSED'] = test['TIME'].apply(extract_closed)





train['TIME'] = train['TIME'].str.replace(r'Closed \(.*?\)','')

test['TIME'] = test['TIME'].str.replace(r'Closed \(.*?\)','')
# Import label encoder 

#from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

#label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels in column 'CUISINES'. 

#train['CUISINES']= label_encoder.fit_transform(train['CUISINES']) 

#test['CUISINES']= label_encoder.fit_transform(test['CUISINES'])   

#test['CUISINES'].unique()
test.head()
train.head()
calc_mean = train.groupby(['CITY'], axis=0).agg({'RATING': 'mean'}).reset_index()

calc_mean.columns = ['CITY','CITY_MEAN_RATING']

train = train.merge(calc_mean, on=['CITY'],how='left')



calc_mean = train.groupby(['LOCALITY'], axis=0).agg({'RATING': 'mean'}).reset_index()

calc_mean.columns = ['LOCALITY','LOCALITY_MEAN_RATING']

train = train.merge(calc_mean, on=['LOCALITY'],how='left')
train.head()
train.isnull().mean()


DROP_COLS= (['TITLE', 'CUISINES', 'CITY', 'LOCALITY', 'TIME','CLOSED'])
train.drop(DROP_COLS,axis=1).drop(["COST"],axis=1).values
train.drop(DROP_COLS,axis=1).drop(["COST"],axis=1).columns
X = train.drop(DROP_COLS,axis=1).drop(["COST"],axis=1).values

y = train["COST"].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=101)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,Ridge,SGDRegressor

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,VotingRegressor
'''from sklearn.ensemble import GradientBoostingRegressor



gbr=GradientBoostingRegressor( loss = 'huber',learning_rate=0.001,n_estimators=350, max_depth=6

                              ,subsample=1,

                              verbose=False,random_state=126)   # Leaderboard SCORE :  0.8364249755816828 @ RS =126 ,n_estimators=350, max_depth=6



gbr.fit(X_train,y_train)



y_pred_gbr = sc_X.inverse_transform(gbr.predict(X_test))'''
regression_models = ['LinearRegression','ElasticNet','Lasso','Ridge','SGDRegressor','SVR',

                    'DecisionTreeRegressor','RandomForestRegressor','AdaBoostRegressor']
mse = []

rmse = []

mae = []

models = []

estimators = []
for reg_model in regression_models:

    

    model = eval(reg_model)()

    

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    

    models.append(type(model).__name__)

    estimators.append((type(model).__name__,model))

    

    mse.append(mean_squared_error(y_test,y_pred))

    rmse.append(mean_squared_error(y_test,y_pred)**0.5)

    mae.append(mean_absolute_error(y_test,y_pred))
model_dict = {"Models":models,

             "MSE":mse,

             "RMSE":rmse,

             "MAE":mae}
model_df = pd.DataFrame(model_dict)

model_df
model_df["Inverse_Weights"] = model_df['RMSE'].map(lambda x: np.log(1.0/x))

model_df
vr = VotingRegressor(estimators=estimators,weights=model_df.Inverse_Weights.values)
vr.fit(X_train,y_train)
y_pred = vr.predict(X_test)
models.append("GradientBoostingRegressor")

mse.append(mean_squared_error(y_test,y_pred))

rmse.append(mean_squared_error(y_test,y_pred)**0.5)

mae.append(mean_absolute_error(y_test,y_pred))
sub_file = pd.DataFrame(y_pred,columns=["COST"])

writer = pd.ExcelWriter('Output.xlsx', engine='xlsxwriter')

sub_file.to_excel(writer,sheet_name='Sheet1', index=False)

writer.save()

sub_file.head()