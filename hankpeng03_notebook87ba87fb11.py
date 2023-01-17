# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/data.csv")
print('The columns')

print(df.columns)

print('The data type of each columns')

print(df.dtypes)
print(Counter(df['shot_made_flag'].isnull()))
# remove the rows which value of column 'shot_made_flag' is null

df = df[df['shot_made_flag'].notnull()]

print(df.shape)
# Label the columns

from sklearn.preprocessing import LabelEncoder

def Label(df,col_names):

    return_df = df.copy()

    if not isinstance(col_names,list):

        print('col_names input wrong!')

        return None

    for col_name in col_names:

        return_df[col_name] = LabelEncoder().fit(return_df[col_name]).transform(return_df[col_name])

    return(return_df)

def sparse(df,col_names):

    '''

    convert dataframe to sparse dataframe

    '''

    return_df = pd.DataFrame()

    if not isinstance(col_names,list):

        print('col_names input wrong!')

        return None

    for col_name in col_names:

        values_list = list(set(df[col_name]))

        temp=[]

        for value in values_list:

            temp.append(np.array(df[col_name] == value,dtype='int'))

        temp = np.vstack(temp).T

        values_list = [col_name+'_'+value for value in values_list]

        temp_df = pd.DataFrame(data=temp,columns=values_list)

        return_df = pd.concat([return_df,temp_df],axis=1)

    return(return_df)

col_names=['action_type',

 'combined_shot_type',

 'season',

 'shot_type',

 'shot_zone_area',

 'shot_zone_basic',

 'shot_zone_range',

 'team_name',

 'matchup',

 'opponent']

#sparse the data

sparse_df = sparse(df,col_names)

df.drop(col_names,axis=1,inplace=True)

sparse_df.reset_index(inplace=True)

df.reset_index(inplace=True)

df=pd.concat([df,sparse_df],axis=1,ignore_index=False)
#从字段‘game_date’中提取年（year）、月（month）、日（day）

df['game_date'] = pd.to_datetime(df['game_date'])

df['year'] = df['game_date'].dt.year

df['month'] = df['game_date'].dt.month

df['day'] = df['game_date'].dt.day

df.drop(['game_date'],axis=1,inplace=True) #删除字段‘game_date’t = sparse(df,col_names)
#df_bak = df.copy()

#df = df_bak.copy()

df.drop(['index'],axis=1,inplace=True)
def train_valid_test_split(X,y,valid_size=0.2,test_size=0.2):

    #拆分训练集、验证集和测试集

    from sklearn.model_selection import  train_test_split

    train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=test_size) 

    train_X,valid_X,train_y,valid_y = train_test_split(train_X,train_y,test_size=valid_size)

    return(train_X,valid_X,test_X,train_y,valid_y,test_y)



y = df['shot_made_flag']

X = df.drop(['shot_made_flag'],axis=1)

train_X,valid_X,test_X,train_y,valid_y,test_y = train_valid_test_split(X,y)
#SVM

from sklearn.svm import SVC

from sklearn.model_selection import  GridSearchCV

params = {'C':[0.7],'kernel':['rbf'],'shrinking':[True]}

svc=SVC(kernel='rbf',C=0.7,shrinking=True)

svc.fit(X,y)

#grid_search = GridSearchCV(estimator=svc,param_grid=params,n_jobs=-1,cv=5)

#grid_search.fit(X,y)

#svc_best=grid_search.best_estimator_

#print(grid_search.best_params_)
#RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

params = {'criterion':['gini'],'max_features':[100],'min_samples_split':[100]}

rf=RandomForestClassifier(n_estimators=50,n_jobs=-1,criterion='gini',max_features=100,min_samples_split=50)

rf.fit(X,y)

#grid_search = GridSearchCV(estimator=rf,param_grid=params,cv=5)

#grid_search.fit(X,y)

#rf_best = grid_search.best_estimator_

#print(grid_search.best_params_)
#GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier

params = {'loss':['exponential'],'learning_rate':[0.3],'max_depth':[4],'max_features':[80]}

gbrc = GradientBoostingClassifier(n_estimators=100,loss='exponential',learning_rate=0.3,max_depth=4,max_features=80)

gbrc.fit(X,y)

#grid_search = GridSearchCV(estimator=gbrc,param_grid=params,cv=5)

#grid_search.fit(X,y)

#gbrc_best = grid_search.best_estimator_

#print(grid_search.best_params_)
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score

eclf=VotingClassifier(estimators=[('svc_best',svc_best),('rf_best',rf_best),('gbrc_best',gbrc_best)],voting='hard')

score = cross_val_score(eclf,X[:5000],y[:5000],cv=5,scoring='accuracy')
score