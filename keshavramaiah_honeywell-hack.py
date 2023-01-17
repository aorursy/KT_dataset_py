import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers.core import Dense,Activation,Dropout

from keras.layers import LSTM

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier





from keras.callbacks import EarlyStopping

import math
df=pd.read_csv('../input/pick_time_warehouse_train.csv')
df.shape
df.columns


df['Pick_Time'].plot.box(grid=True)

df.shape
def remove_outlier(df_in, col_name):

    q1 = df_in[col_name].quantile(0.25)

    q3 = df_in[col_name].quantile(0.75)

    iqr = q3-q1 #Interquartile range

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]

    return df_out



def remove_outlier_cat(df_in, name):

    fl = {}

    fh = {}

    for nn in np.unique(df_in[name]):

        q1 = df_in[df_in[name]==nn]['Pick_Time'].quantile(0.25)

        q3 = df_in[df_in[name]==nn]['Pick_Time'].quantile(0.75)

        iqr = q3-q1 #Interquartile range

        fence_low  = q1-1.5*iqr

        fence_high = q3+1.5*iqr

        fl[nn] = fence_low

        fh[nn] = fence_high

    df_out = df_in.loc[(df_in['Pick_Time'] > fl[int(df_in[name])]) & (df_in['Pick_Time'] < fh[int(df_in[name])])]

    return df_out
for i in range(10):

    df = remove_outlier(df, 'Pick_Time')

df.isnull().sum()
df['total_quantity_picked_by_user']=df['total_quantity_picked_by_user'].fillna(0)
del df['sl.no']
fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(df.corr(),annot=True, linewidths=.5,ax=ax)
#del df['last_station_served_by_user']

#del df['Cube_of_item']

#del df['volume_of_items']

#del df['volume_of_items_present_in_container']

sns.boxplot(x='Actual_Quantity', y='Pick_Time', data=df)
df = remove_outlier_cat(df, 'Actual_Quantity')
df
sns.boxplot(x='Actual_Quantity', y='Pick_Time', data=df)
sns.scatterplot(x='Actual_Quantity',y='Pick_Time',data=df)
df.columns
df['day'].nunique()
sns.countplot(df['total_quantity_of_items_in_container'])
df['total_quantity_of_items_in_container'].value_counts()
hr=[]

minutes=[]

seconds=[]

for i in range(len(df['Start_Time_of_Picking'])):

    hr.append(int(df.iloc[i]['Start_Time_of_Picking'].split(':')[0]))

    a, b = df.iloc[i]['Start_Time_of_Picking'].split(':')[1].split('.')

    minutes.append(int(a))

    seconds.append(int(b))
#df['hr']=hr

df['min']=minutes

df['sec']=seconds
del df['Start_Time_of_Picking']
#sns.countplot(df['total_quantity_picked_by_user'])

df=pd.concat([df,pd.get_dummies(df['SKU'], drop_first=True, prefix=1)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['User'], drop_first=True, prefix=2)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['number_of_container_conveyor'], drop_first=True, prefix=3)], axis=1,sort=False)

df=pd.concat([df,pd.get_dummies(df['last_station_served_by_user'], drop_first=True, prefix=3)], axis=1,sort=False)

df=df.drop(['SKU','User','number_of_container_conveyor','last_station_served_by_user'],axis=1)
target=df.Pick_Time

del df['Pick_Time']
X_train, X_test, y_train, y_test = train_test_split(df,target,test_size = 0.30,random_state=0)


X_train, X_test, y_train, y_test = train_test_split(df,target,test_size = 0.30, shuffle=False)

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor

from sklearn.metrics import mean_squared_error

import numpy as np

import warnings

import keras

warnings.filterwarnings("ignore")

model= CatBoostRegressor(logging_level='Silent')

model.fit(X_train, y_train)

predictions = model.predict(X_test)



print('Lasso', (np.sqrt(mean_squared_error((y_test), (predictions)))))

classifiers = [['DecisionTree :',DecisionTreeRegressor()],

               ['RandomForest :',RandomForestRegressor()],

               ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],

               ['AdaBoostClassifier :', AdaBoostRegressor()],

               ['GradientBoostingClassifier: ', GradientBoostingRegressor()],

               ['Xgboost: ', XGBRegressor()],

               ['CatBoost: ', CatBoostRegressor(logging_level='Silent')],

               ['Lasso: ', Lasso()],

               ['Ridge: ', Ridge()],

               ['BayesianRidge: ', BayesianRidge()],

               ['ElasticNet: ', ElasticNet()],

               ['HuberRegressor: ', HuberRegressor()]]



print("Accuracy Results...")





for name,classifier in classifiers:

    classifier = classifier

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))

    
import pandas

import numpy as np

import catboost as cb

from sklearn.model_selection import KFold

from itertools import product,chain

from sortedcontainers import SortedList

import copy

import collections

import numpy as np

from itertools import product,chain

import pandas

from sklearn.model_selection import KFold

import catboost as cb



class paramsearch:

    def __init__(self,pdict):    

        self.pdict = {}

        # if something is not passed in as a sequence, make it a sequence with 1 element

        #   don't treat strings as sequences

        for a,b in pdict.items():

            if isinstance(b, collections.Sequence) and not isinstance(b, str): self.pdict[a] = b

            else: self.pdict[a] = [b]

        # our results are a sorted list, so the best score is always the final element

        self.results = SortedList()       

                    

    def grid_search(self,keys=None):

        # do grid search on only the keys listed. If none provided, do all

        if keys==None: keylist = self.pdict.keys()

        else: keylist = keys

 

        listoflists = [] # this will be list of lists of key,value pairs

        for key in keylist: listoflists.append([(key,i) for i in self.pdict[key]])

        for p in product(*listoflists):

            # do any changes to the current best parameter set

            if len(self.results)>0: template = self.results[-1][1]

            else: template = {a:b[0] for a,b in self.pdict.items()}

            # if our updates are the same as current best, don't bother

            if self.equaldict(dict(p),template): continue

            # take the current best and update just the ones to change

            yield self.overwritedict(dict(p),template)

                              

    def equaldict(self,a,b):

        for key in a.keys(): 

            if a[key] != b[key]: return False

        return True            

                              

    def overwritedict(self,new,old):

        old = copy.deepcopy(old)

        for key in new.keys(): old[key] = new[key]

        return old            

    

    # save a (score,params) pair to results. Since 'results' is a sorted list,

    #   the best score is always the final element. A small amount of noise is added

    #   because sorted lists don't like it when two scores are exactly the same    

    def register_result(self,result,params):

        self.results.add((result+np.random.randn()*1e-10,params))    

        

    def bestscore(self):

        return self.results[-1][0]

        

    def bestparam(self):

        return self.results[-1][1]

        



params = {'depth':[3,1,2,6,4,5,7,8,9,10],

          'iterations':[250,100,500,1000],

          'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 

          'l2_leaf_reg':[3,1,5,10,100],

          'border_count':[32,5,10,20,50,100,200],

          'ctr_border_count':[50,5,10,20,100,200],

          'thread_count':4}



def crossvaltest(params,train_set,train_label,cat_dims,n_splits=3):

    kf = KFold(n_splits=n_splits,shuffle=True) 

    res = []

    a=1000000000000

    for train_index, test_index in kf.split(train_set):

        train = train_set.iloc[train_index,:]

        test = train_set.iloc[test_index,:]



        labels = train_label.ix[train_index]

        test_labels = train_label.ix[test_index]



        clf = cb.CatBoostRegressor(**params)

        clf.fit(train, np.ravel(labels), cat_features=cat_dims)

        a=min(a,np.sqrt(mean_squared_error(y_test, predictions)))

    return a 

def catboost_param_tune(params,train_set,train_label,cat_dims=None,n_splits=3):

    ps = paramsearch(params)

    for prms in chain(ps.grid_search(['border_count']),

                      ps.grid_search(['ctr_border_count']),

                      ps.grid_search(['l2_leaf_reg']),

                      ps.grid_search(['iterations','learning_rate']),

                      ps.grid_search(['depth'])):

        res = crossvaltest(prms,train_set,train_label,cat_dims,n_splits)

        ps.register_result(res,prms)

        print(res,prms,s,'best:',ps.bestscore(),ps.bestparam())

    return ps.bestparam()



bestparams = catboost_param_tune(params,X_train,y_train,cat_dims)


