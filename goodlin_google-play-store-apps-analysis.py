# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import re

import seaborn as sns

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
reviews=pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")

apps_infor=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

apps_infor.head()
apps_infor.drop_duplicates(inplace=True)

apps_infor.drop([10472],inplace=True)

apps_infor['Reviews']=apps_infor["Reviews"].astype("int32")

apps_infor['Last Updated']=apps_infor['Last Updated'].astype("datetime64")

apps_infor[["App","Category","Installs","Type","Price","Content Rating","Genres","Current Ver","Android Ver"]]=apps_infor[["App","Category","Installs","Type","Price","Content Rating","Genres","Current Ver","Android Ver"]].astype("string")
apps_infor.isna().sum()
# drop rows with na at columns "Type","Current Ver", "Android Ver" 

apps_infor.dropna(axis=0,inplace=True,subset=['Android Ver','Current Ver','Type'])
# fill rating na with the mean rating of category the apps in

def fill_rating(c):

    c_mean=apps_infor.loc[apps_infor["Category"]==c]["Rating"].mean()

    return c_mean



apps_infor.loc[apps_infor["Rating"].isna(),"Rating"]=apps_infor.loc[apps_infor["Rating"].isna(),"Category"].apply(fill_rating)
# transfrom Size to float: 1. unit k/1024->M; 2. unit M->M; Varies with device->-1

apps_infor["Size"].replace("Varies with device",-1,inplace=True)

apps_infor.loc[apps_infor["Size"].str[-1]=="k","Size"]=(apps_infor.loc[apps_infor["Size"].str[-1]=="k","Size"].str[:-1].astype("float64")/1024)

apps_infor.loc[apps_infor["Size"].str[-1]=="M","Size"]=apps_infor.loc[apps_infor["Size"].str[-1]=="M","Size"].str[:-1].astype("float64")

apps_infor["Size"]=apps_infor["Size"].astype("float64")
# transfrom Installs to int32 

apps_infor.loc[apps_infor["Installs"].str[-1]=='+',"Installs"]=apps_infor.loc[apps_infor["Installs"].str[-1]=='+',"Installs"].str[:-1]

apps_infor["Installs"]=apps_infor["Installs"].str.replace(",",'').astype("int32")

#transfrom Price to float64

apps_infor["Price"]=apps_infor["Price"].str.replace("$",'').astype("float32")

apps_infor.drop(["Type"],axis=1,inplace=True)
apps_infor["Category"]=apps_infor["Category"].str.lower()

apps_infor["Genres"]=apps_infor["Genres"].str.lower()

apps_infor=pd.concat([apps_infor, apps_infor['Genres'].str.split(';', expand=True)], axis=1).drop(["Genres"],axis=1)

apps_infor.rename(columns={0:'Genres0',1:'Genres1'},inplace=True)
#Transform "Content Rating" to int32

#Only 3 apps satisfy  "Content Rating"=="Adults only 18+" ,it's resonable to classify it together with "Mature 17+"

apps_infor["Content Rating"].replace({'Everyone':0, 'Teen':13, 'Everyone 10+':10, 'Mature 17+':17, 'Adults only 18+':17,

 'Unrated':0},inplace=True)

apps_infor["Content Rating"]=apps_infor["Content Rating"].astype("int32")
apps_infor.loc[apps_infor["App"]=="ROBLOX"]
columns=apps_infor.columns

columns=columns.drop(["Reviews"])

apps_infor.drop_duplicates(subset=columns.values.tolist(),keep="first",inplace=True)

apps_infor.reset_index(drop=True,inplace=True)
# groupby category

age_list=[0,10,13,17]

age_infor={}

for age in age_list:

    age_infor[age]=apps_infor.loc[apps_infor["Content Rating"]==age].groupby("Category")[["Installs","Rating"]].agg(["count","sum","mean","min","max","std"]).sort_values(by=("Installs","sum"),ascending=False)
age_infor[0].head()
plt.figure(figsize=(8,8))

age_infor[0]["Installs"]["count"].plot.pie(fontsize=8)
age_infor[10].head()
plt.figure(figsize=(8,8))

age_infor[10]["Installs"]["count"].plot.pie(fontsize=8)
age_infor[13].head()
plt.figure(figsize=(8,8))

age_infor[13]["Installs"]["count"].plot.pie(fontsize=8)
age_infor[17].head()
plt.figure(figsize=(8,8))

age_infor[17]["Installs"]["count"].plot.pie(fontsize=8)
apps_infor.groupby("Category")[["Installs","Rating"]].agg(["count","sum","mean","min","max","std"]).sort_values(by=("Rating","mean"),ascending=False)
Cat_gen=apps_infor[["Category","Genres0","Genres1"]].groupby("Category")

for cat,group in Cat_gen:

    cat=cat.replace('_and_',' & ')

    #print(cat,end='->')

    count_0=group["Genres0"].fillna("None").value_counts()

    count_1=group["Genres1"].fillna("None").value_counts()

    print_list=''

    for index,row in count_0.iteritems():

        if index!=cat and index!='None':

            print_list+=index

            print_list+=' '

            print_list+=str(row)

            print_list+=' '

          #  print(index,end=' ')

           # print(row,end=' ')

   # print('|',end='')

    print_list+='|'

    for index,row in count_1.iteritems():

        if index!=cat and index!='None':

            print_list+=index

            print_list+=' '

            print_list+=str(row)

            print_list+=' '

          #  print(index,end=' ')

          #  print(row,end=' ')

    if len(print_list)>1:

        print(cat,sum(count_1),end='->')

        print(print_list)

        print('\n')

columns=columns.drop(["Category"])

apps_infor[apps_infor.duplicated(subset=columns,keep=False)].groupby("Category").size()
apps_infor.groupby(pd.Grouper(key="Last Updated", freq="M"))["App"].agg(["count"]).plot()

plt.ylabel('APPs Num')

plt.xlabel('Last Unpdated(Year-Month)')
apps_infor.groupby(pd.Grouper(key="Last Updated", freq="M"))["Installs"].agg(["mean","count"]).query("count>100")["mean"].plot()

plt.ylabel('Installs Num Per App')

plt.xlabel('Last Updated(Year-Month)')
apps_infor.groupby(pd.Grouper(key="Last Updated", freq="M"))["Rating"].agg(["mean","count"]).query("count>100")["mean"].plot()

plt.ylabel('Rating Per App')

plt.xlabel('Last Updated(Year-Month)')
apps_infor["Last Updated_dis"]=((apps_infor["Last Updated"].max()-apps_infor["Last Updated"]).dt.days).astype("int")
apps_infor.groupby(apps_infor["Price"]>0)[["Price"]].agg(["count"]).plot(kind="bar")

plt.legend()

plt.xticks([False,True],["Free","Paid"],rotation=90)

apps_infor.loc[apps_infor["Price"]>0].groupby((apps_infor["Price"]).astype("int")+1)[["Price"]].agg(["count"]).plot(kind="bar")

plt.legend()

apps_infor.groupby((apps_infor["Price"]/10).astype("int"))[["Rating"]].agg(["mean"]).plot(kind="bar")

plt.ylim([3.5,5])

plt.xticks(rotation=45)

plt.legend()

apps_infor.groupby(apps_infor["Price"]>0)[["Rating"]].agg(["mean"]).plot(kind="bar")

plt.xticks([False,True],["Free","Paid"],rotation=90)

plt.ylim([4,4.5])

plt.legend()
apps_infor["Reviews"].apply(lambda x:np.log10(x+1)).plot(kind="kde")

plt.xlabel("Reviews,base=10")
apps_infor["Reviews_log"]=apps_infor["Reviews"].apply(lambda x:np.log(x+1))

apps_infor["Installs_log"]=apps_infor["Installs"].apply(lambda x:np.log(x+1))
plt.figure(figsize=(10,8))

sns.heatmap(apps_infor.corr(),annot=True)
sns.pairplot(apps_infor,diag_kind="hist")
from sklearn.model_selection import learning_curve, train_test_split,GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor





#trasform Category into dummy variables

feature=apps_infor[["Category","Size","Installs","Reviews","Installs_log","Price","Content Rating","Last Updated_dis","Reviews_log","Android Ver"]]

feature=feature.merge(feature["Category"].str.get_dummies(),left_index=True,right_index=True)

feature.drop(["Category"],axis=1,inplace=True)

feature=feature.merge(feature["Android Ver"].str.get_dummies(),left_index=True,right_index=True)

feature.drop(["Android Ver"],axis=1,inplace=True)



feature=feature.values



target=apps_infor["Rating"].values





# 20% data for test, the rest for trainning.

train_x,test_x,train_y,test_y = train_test_split(feature ,target, test_size=0.20, random_state = 1)

    

# build all kinds of regressors

regressors = [   

    DecisionTreeRegressor(random_state = 1, criterion = 'mse'),

    RandomForestRegressor(random_state = 1, criterion = 'mse'),

    KNeighborsRegressor(metric = 'minkowski'),

]

# regressor name

regressor_names = [

            'decisiontreeregressor',

            'randomforestregressor',

            'kneighborsregressor',

]

# regressor paramters

regressor_param_grid = [

            {'decisiontreeregressor__max_depth':range(1,6)},

            {'randomforestregressor__n_estimators':[98]} ,# I tried range(1,100) and find 98 is the best.

            {'kneighborsregressor__n_neighbors':range(95,100)},

]

 

# GridSearchCV for choosing a good parameter

def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score='neg_root_mean_squared_error'):

    response = {}

    gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, scoring = score)

    # find the best parameters

    search = gridsearch.fit(train_x, train_y)

    print("GridSearch Best Parameter：", search.best_params_)

    print("GridSearch Smallest Error： %0.4lf" %(-search.best_score_))

    predict_y = gridsearch.predict(test_x)

    print("mean squared error %0.4lf" %mean_squared_error(test_y, predict_y))

    response['predict_y'] = predict_y

    response['mean_squared_error'] = mean_squared_error(test_y,predict_y)

    return response

 

for model, model_name, model_param_grid in zip(regressors, regressor_names, regressor_param_grid):

    pipeline = Pipeline([

            ('scaler', StandardScaler()),

            (model_name, model)

    ])

    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid , score ='neg_root_mean_squared_error'



)
# use the best regressor and best parameters for prediction

pipeline = Pipeline([

            ('scaler', StandardScaler()),

            ('DT',DecisionTreeRegressor(random_state = 1, criterion = 'mse',max_depth=5))

    ])
y_predict=(pipeline.fit(feature,target)).predict(feature)

apps_infor["Rating_pre"]=y_predict

apps_infor["Rating_pre"].plot(kind="kde",c="g")

apps_infor["Rating"].plot(kind="kde",c="b")

plt.xstick("Rating Value")