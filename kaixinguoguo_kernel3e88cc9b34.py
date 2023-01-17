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

from collections import Counter  
train_data=pd.read_csv("../input/train.csv")

test_data=pd.read_csv("../input/test.csv")
#需要转换的列表dict_columns

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
#定义转换的函数

def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x:{} if pd.isnull(x) else eval(x) )

    return df

train_data = text_to_dict(train_data)

test_data = text_to_dict(test_data)
#belongs_to_collection,0/1转换

train_data['belongs_to_collection']=train_data['belongs_to_collection'].map(lambda x:len(x) if x!='{}' else 0)

test_data['belongs_to_collection']=test_data['belongs_to_collection'].map(lambda x:len(x) if x!='{}' else 0)
#genres列处理

#1 类型数量(genres_num)

#训练集处理

train_data['genres_num'] = train_data['genres'].map(lambda x:len(x) if x!={} else 0)

#测试集处理

test_data['genres_num'] = test_data['genres'].map(lambda x:len(x) if x!={} else 0)
#2 名字连接(genres_all)

#训练集处理

train_data['genres_all'] = train_data['genres'].map(lambda x:','.join([i['name'] for i in x]))

#测试集处理

test_data['genres_all'] = test_data['genres'].map(lambda x:','.join([i['name'] for i in x]))
#3类型列

list_genres = list(train_data['genres'].map(lambda x:[i['name'] for i in x]))

list_genres_count = []

for i in list_genres:

    for j in i:

        list_genres_count.append(j)
#取前19名的类型名

list_genres_count=list(set(list_genres_count)-set(["TV Movie"]))
#为电影类型创建相关列,存在为1否则为0

#训练集处理

for i in list(list_genres_count):

    train_data['genres_'+i] = train_data['genres_all'].map(lambda x:1 if i in x else 0)

#测试集处理

for i in list(list_genres_count):

    test_data['genres_'+i] = test_data['genres_all'].map(lambda x:1 if i in x else 0)
#homepage列处理

#训练集处理

train_data['has_homepage'] = train_data['homepage'].map(lambda x:0 if pd.isnull(x) else 1)

#测试集处理

test_data['has_homepage'] = test_data['homepage'].map(lambda x:0 if pd.isnull(x) else 1)
#original_language列处理

list_orginal=train_data["original_language"].value_counts().reset_index()
list_orginal=list(list_orginal.loc[:10,"index"])
#训练集处理

for i in list_orginal:

    train_data['orginal_language_' + i] = train_data['original_language'].map(lambda x:1 if x==i else 0 )

#测试集处理

for i in list_orginal:

    test_data['orginal_language_' + i] = test_data['original_language'].map(lambda x:1 if x==i else 0 )
#production_companies列处理

#训练集处理

train_data['pr_companies_num'] = train_data['production_companies'].map(lambda x:len(x) if x!={} else 0)

#测试集处理

test_data['pr_companies_num'] = test_data['production_companies'].map(lambda x:len(x) if x!={} else 0)
#测试集处理

train_data['companies_all'] = train_data['production_companies'].map(lambda x:','.join([i['name'] for i in x]))

#测试集处理

test_data['companies_all'] = test_data['production_companies'].map(lambda x:','.join([i['name'] for i in x])) 
list_companies = list(train_data['production_companies'].map(lambda x:[i['name'] for i in x]))

list_companies_count = []

for i in list_companies:

    for j in i:

        list_companies_count.append(j)
#取前10名

companies_list=[]

companies_top = Counter(list_companies_count).most_common(10)

for i in companies_top:

    companies_list.append(i[0])
#训练集处理

for i in companies_list:

    train_data['companies_' + i] = train_data['companies_all'].map(lambda x:1 if i in x else 0 )

#测试集处理

for i in companies_list:

    test_data['companies_' + i] = test_data['companies_all'].map(lambda x:1 if i in x else 0 )
#production_countries列处理

#训练集处理

train_data['pr_countries_num'] = train_data['production_countries'].map(lambda x:len(x) if x!={} else 0)

#测试集处理

test_data['pr_countries_num'] = test_data['production_countries'].map(lambda x:len(x) if x!={} else 0)
#测试集处理

train_data['countries_all'] = train_data['production_countries'].map(lambda x:','.join([i['iso_3166_1'] for i in x]))

#测试集处理

test_data['countries_all'] = test_data['production_countries'].map(lambda x:','.join([i['iso_3166_1'] for i in x]))
list_countries = list(train_data['production_countries'].map(lambda x:[i['iso_3166_1'] for i in x]))

list_countries_count = []

for i in list_countries:

    for j in i:

        list_countries_count.append(j)
countries_list=[]

countries_top = Counter(list_countries_count).most_common(10)

for i in countries_top:

    countries_list.append(i[0])
#训练集处理

for i in countries_list:

    train_data['pr_countries_' + i] = train_data['countries_all'].map(lambda x:1 if i in x else 0 )

#测试集处理

for i in countries_list:

    test_data['pr_countries_' + i] = test_data['countries_all'].map(lambda x:1 if i in x else 0 )
#spoken_languages列

#训练集处理

train_data['spoken_languages_num'] = train_data['spoken_languages'].map(lambda x:len(x) if x!={} else 0)

#测试集处理

test_data['spoken_languages_num'] = test_data['spoken_languages'].map(lambda x:len(x) if x!={} else 0)
#测试集处理

train_data['spoken_languages_all'] = train_data['spoken_languages'].map(lambda x:','.join([i['iso_639_1'] for i in x]))

#测试集处理

test_data['spoken_languages_all'] = test_data['spoken_languages'].map(lambda x:','.join([i['iso_639_1'] for i in x]))
list_spoken_languages = list(train_data['spoken_languages'].map(lambda x:[i['iso_639_1'] for i in x]))

list_spoken_languages_count = []

for i in list_spoken_languages:

    for j in i:

        list_spoken_languages_count.append(j)
spoken_list=[]

spoken_top = Counter(list_countries_count).most_common(10)

for i in spoken_top:

    spoken_list.append(i[0])
#训练集处理

for i in spoken_list:

    train_data['spoken_languages_' + i] = train_data['spoken_languages_all'].map(lambda x:1 if i in x else 0 )

#测试集处理

for i in spoken_list:

    test_data['spoken_languages_' + i] = test_data['spoken_languages_all'].map(lambda x:1 if i in x else 0 )
#Keywords列

#训练集处理

train_data['Keywords_num'] = train_data['Keywords'].map(lambda x:len(x) if x!={} else 0)

#测试集处理

test_data['Keywords_num'] = test_data['Keywords'].map(lambda x:len(x) if x!={} else 0)
#训练集处理

train_data['Keywords_all'] = train_data['Keywords'].map(lambda x:','.join([i['name'] for i in x]))

#测试集处理

test_data['Keywords_all'] = test_data['Keywords'].map(lambda x:','.join([i['name'] for i in x]))
list_Keywords = list(train_data['Keywords'].map(lambda x:[i['name'] for i in x]))

list_Keywords_count = []

for i in list_Keywords:

    for j in i:

        list_Keywords_count.append(j)
Keywords_list=[]

Keywords_top = Counter(list_Keywords_count).most_common(10)

for i in Keywords_top:

    Keywords_list.append(i[0])
#训练集处理

for i in Keywords_list:

    train_data['Keywords_'+i] = train_data['Keywords_all'].map(lambda x:1 if i in x else 0)

#测试集处理

for i in Keywords_list:

    test_data['Keywords_'+i] = test_data['Keywords_all'].map(lambda x:1 if i in x else 0)
#cast列

#训练集处理

train_data['cast_num'] = train_data['cast'].map(lambda x:len(x) if x!={} else 0)

#测试集处理

test_data['cast_num'] = test_data['cast'].map(lambda x:len(x) if x!={} else 0)
#训练集处理

train_data['cast_all'] = train_data['cast'].map(lambda x:','.join([i['name'] for i in x]))

#测试集处理

test_data['cast_all'] = test_data['cast'].map(lambda x:','.join([i['name'] for i in x]))
list_cast = list(train_data['cast'].map(lambda x:[i['name'] for i in x]))

list_cast_count = []

for i in list_cast:

    for j in i:

        list_cast_count.append(j)
cast_list=[]

cast_top = Counter(list_cast_count).most_common(10)

for i in cast_top:

    cast_list.append(i[0])
for i in cast_list:

    train_data['cast_'+i] = train_data['cast_all'].map(lambda x:1 if i in x else 0)

for i in cast_list:

    test_data['cast_'+i] = test_data['cast_all'].map(lambda x:1 if i in x else 0)
#gender_type看是男性大于女性还是其他

#训练集

for i in range(train_data.shape[0]):

    male=0

    female=0

    for j in range(len(train_data.cast[i])):

        if train_data.cast[i][j]["gender"]==2:

            male+=1

        elif train_data.cast[i][j]["gender"]==1:

            female+=1

    if male>female:

        train_data.loc[i,"gender_type"]=2

    elif male<female:

        train_data.loc[i,"gender_type"]=1

    else:

        train_data.loc[i,"gender_type"]=0

#测试集

for i in range(test_data.shape[0]):

    male=0

    female=0

    for j in range(len(test_data.cast[i])):

        if test_data.cast[i][j]["gender"]==2:

            male+=1

        elif test_data.cast[i][j]["gender"]==1:

            female+=1

    if male>female:

        test_data.loc[i,"gender_type"]=2

    elif male<female:

        test_data.loc[i,"gender_type"]=1

    else:

        test_data.loc[i,"gender_type"]=0  
#crew列处理

#由于电影制作部门都差不多不做处理

#训练集处理

train_data['crew_num'] = train_data['crew'].map(lambda x:len(x) if x!={} else 0)

#测试集处理

test_data['crew_num'] = test_data['crew'].map(lambda x:len(x) if x!={} else 0)
#提取导演,优秀导演更容易获得高的收入

#训练集

for i in range(train_data.shape[0]):

    for j in range(len(train_data.crew[i])):

        if train_data.crew[i][j]["job"]=="Director":

            train_data.loc[i,"Director"]=train_data.crew[i][j]["name"]

            break

#测试集

for i in range(test_data.shape[0]):

    for j in range(len(test_data.crew[i])):

        if test_data.crew[i][j]["job"]=="Director":

            test_data.loc[i,"Director"]=test_data.crew[i][j]["name"]

            break
Director_list=train_data.Director.value_counts().reset_index()
Director_top=list(Director_list.loc[Director_list.Director>10,"index"])
for i in Director_top:

    train_data['Director_'+i] = train_data['Director'].map(lambda x:1 if i==x else 0)

for i in Director_top:

    test_data['Director_'+i] = test_data['Director'].map(lambda x:1 if i==x else 0)
#tagline列处理

train_data['has_tagline'] = train_data['tagline'].map(lambda x:0 if pd.isna(x) else 1)

test_data['has_tagline'] = test_data['tagline'].map(lambda x:0 if pd.isna(x) else 1)
#status列

train_data['has_Released'] = train_data['status'].map(lambda x:1 if x=='Released' else 0)

test_data['has_Released'] = test_data['status'].map(lambda x:1 if x=='Released' else 0)
time_df=train_data[["id","release_date"]]
#release_date列

#单独提取时间列进行处理,把时间列转换为时间戳形式

time_df=train_data[["id","release_date"]]

time_df.loc[:,"time"]=np.array([pd.Timestamp(t) for t in np.array(time_df.loc[:,"release_date"])])
#提取上映的年份,月份,日,季度,星期几,新增5列

#训练集

time_df.loc[:,"year"]=time_df.time.apply(lambda x:x.year)

time_df.loc[:,"month"]=time_df.time.apply(lambda x:x.month)

time_df.loc[:,"day"]=time_df.time.apply(lambda x:x.day)

time_df.loc[:,"quarter"]=time_df.time.apply(lambda x:x.quarter)

time_df.loc[:,"dayofweek"]=time_df.time.apply(lambda x:x.dayofweek)

#测试集

time_df1=test_data[["id","release_date"]]

time_df1.loc[:,"time"]=np.array([pd.Timestamp(t) for t in np.array(time_df1.release_date)])

time_df1.loc[:,"year"]=time_df1.time.apply(lambda x:x.year)

time_df1.loc[:,"month"]=time_df1.time.apply(lambda x:x.month)

time_df1.loc[:,"day"]=time_df1.time.apply(lambda x:x.day)

time_df1.loc[:,"quarter"]=time_df1.time.apply(lambda x:x.quarter)

time_df1.loc[:,"dayofweek"]=time_df1.time.apply(lambda x:x.dayofweek)
#检查发现时间有大于2019年的,大于2019年应减去100

time_df.year=time_df.year.apply(lambda x:x-100 if x>2019 else x)

time_df1.year=time_df1.year.apply(lambda x:x-100 if x>2019 else x)
timetype=["year","month","day","quarter","dayofweek"]
train_data=pd.concat([train_data,time_df[timetype]],axis=1)

test_data=pd.concat([test_data,time_df1[timetype]],axis=1)
## runtime存在缺失值,众数填充

#训练集处理

train_data['runtime'] = train_data['runtime'].fillna(train_data['runtime'].mode()[0])

#测试集处理

test_data['runtime'] = test_data['runtime'].fillna(test_data['runtime'].mode()[0])
#对缺失值填充

#其中某一部电影的上映时间缺失,尝试搜索填充Jails, Hospitals & Hip-Hop(2001-3-20)

test_data.loc[828,["year","month","day","quarter","dayofweek"]]=[2001,3,20,1,2]
train_drop_col = ['id','genres','homepage','imdb_id','overview','original_language','original_title',

           'poster_path','production_companies','production_countries','release_date','spoken_languages',

           'status','title','Keywords','cast','crew','genres_all','companies_all','countries_all',

                 'spoken_languages_all','Keywords_all','cast_all','tagline','Director']



test_drop_col = ['id','genres','homepage','imdb_id','overview','original_language','original_title',

           'poster_path','production_companies','production_countries','release_date','spoken_languages',

           'status','title','Keywords','cast','crew','genres_all','companies_all','countries_all',

                 'spoken_languages_all','Keywords_all','cast_all','tagline','Director']
train_data.drop(train_drop_col,axis=1,inplace=True)

test_data.drop(test_drop_col,axis=1,inplace=True)
## 对年份做分箱处理,分5箱,KbinsDiscretizer

from sklearn.preprocessing import KBinsDiscretizer

KB=KBinsDiscretizer(n_bins=5,encode="ordinal",strategy="uniform")

train_data["year"]=KB.fit_transform(train_data["year"].values.reshape(-1,1)).ravel()

test_data["year"]=KB.fit_transform(test_data["year"].values.reshape(-1,1)).ravel()
Y=train_data.loc[:,"revenue"]

X=train_data.loc[:,train_data.columns!="revenue"]

from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.3,random_state=420)

from xgboost.sklearn import XGBRegressor
clf=XGBRegressor(max_depth=3)

clf.fit(Xtrain,Ytrain)

clf.score(Xtest,Ytest)
import xgboost as xgb

dtrain=xgb.DMatrix(Xtrain,label=Ytrain)

dtest=xgb.DMatrix(Xtest,label=Ytest)

params={'booster':'gbtree',

'objective': 'reg:linear',

'eval_metric': 'rmse',

'max_depth':8,

'gamma':0,  

'lambda':2, 

'subsample':0.7,  

'colsample_bytree':0.8,   

'min_child_weight':3,   

'eta': 0.2,  #

'nthread':8, 

'silent':1}  

watchlist = [(dtrain,'train')]

bst=xgb.train(params,dtrain,num_boost_round=20,evals=watchlist)
bst=xgb.train(params,dtrain,num_boost_round=20,evals=watchlist)
#交叉验证

bst_cv1 = xgb.cv(params,dtrain,num_boost_round=50,nfold=5,seed=0)

bst_cv1
from sklearn.model_selection import GridSearchCV
#忽略警告

import warnings

warnings.filterwarnings("ignore")
#使用网格搜索调参,第一次探索max_depth

param_grid = {'max_depth':list(range(3,8))}

grid = GridSearchCV(XGBRegressor(num_boost_round=100),

                   param_grid=param_grid,cv=5)

grid.fit(Xtrain,Ytrain)
grid.best_params_
#使用网格搜索调参,第二次探索min_child_weight

param_grid = {'min_child_weight':list(range(1,10))}

grid = GridSearchCV(XGBRegressor(num_boost_round=100,max_depth=5),

                   param_grid=param_grid,cv=5)

grid.fit(Xtrain,Ytrain)
grid.best_params_
##使用网格搜索调参,第三次探索gamma

param_grid ={'gamma':[0.1*i for i in range(1,10)]}

grid = GridSearchCV(XGBRegressor(num_boost_round=100,max_depth=5,min_child_weight=1),

                   param_grid=param_grid,cv=5)

grid.fit(Xtrain,Ytrain)
grid.best_params_
##使用网格搜索调参,第四次探索subsample

param_grid ={'subsample': [0.6, 0.7, 0.8, 0.9]}

grid = GridSearchCV(XGBRegressor(num_boost_round=100,max_depth=5,min_child_weight=1,gamma=0,subsample=0.6),

                   param_grid=param_grid,cv=5)

grid.fit(Xtrain,Ytrain)
grid.best_params_
##使用网格搜索调参,第五次探索reg_lambda

param_grid ={'reg_lambda': [0.05, 0.1, 1, 2, 3]}

grid = GridSearchCV(XGBRegressor(num_boost_round=100,max_depth=5,min_child_weight=1,gamma=0,subsample=0.6),

                   param_grid=param_grid,cv=5)

grid.fit(Xtrain,Ytrain)
grid.best_params_
##使用网格搜索调参,第六次探索learning_rate

param_grid ={'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}

grid = GridSearchCV(XGBRegressor(num_boost_round=100,max_depth=5,min_child_weight=1,gamma=0,subsample=0.6,reg_lambda=3),

                   param_grid=param_grid,cv=5)

grid.fit(Xtrain,Ytrain)
grid.best_params_
##使用网格搜索调参,第七次探索learning_rate

param_grid ={'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}

grid = GridSearchCV(XGBRegressor(num_boost_round=100,max_depth=5,min_child_weight=1,gamma=0,subsample=0.6,reg_lambda=3,learning_rate=0.1),

                   param_grid=param_grid,cv=5)

grid.fit(Xtrain,Ytrain)
grid.best_params_
XGBRegressor(num_boost_round=100,max_depth=5,min_child_weight=1,gamma=0,subsample=0.6,reg_lambda=3,learning_rate=0.1)
import xgboost as xgb

dtrain=xgb.DMatrix(Xtrain,label=Ytrain)

dtest=xgb.DMatrix(test_data)

params={'booster':'gbtree',

'objective': 'reg:linear',

'eval_metric': 'rmse',

'max_depth':5,

'gamma':0,  

'lambda':3, 

'subsample':0.6,  

'colsample_bytree':0.6,   

'min_child_weight':3,   

'eta': 0.1,

'nthread':4,  

'silent':1} 

watchlist = [(dtrain,'train')]

bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)
#进行预测

ypred=bst.predict(dtest)