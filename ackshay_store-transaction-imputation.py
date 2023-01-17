# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

ideal=pd.read_csv('../input/ackshay-store/Hackathon_Ideal_Data.csv')

working=pd.read_csv('../input/ackshay-store/Hackathon_Working_Data.csv')

ideal=ideal[~((ideal['QTY']==0)&(ideal['VALUE']!=0))]



all_cats=[]

all_cats.extend(ideal['GRP'].values.tolist()+working['GRP'].values.tolist())

all_cats=list(set(all_cats))





ideal_new=ideal.groupby(['MONTH','STORECODE','GRP'],as_index=False)['QTY','VALUE'].sum()

ideal_new



ideal_new_extend=[]

for store in ideal_new['STORECODE'].unique().tolist():

    for m in ideal_new['MONTH'].unique().tolist():

        

        

        fn=[item for item in all_cats if item not in ideal_new[(ideal_new['STORECODE']==store)&(ideal_new['MONTH']==m)]['GRP'].values.tolist()]

        

        for cat in fn:

            ideal_new_extend.append([m,store,cat,0,0])

        

ideal_new=ideal_new.append(pd.DataFrame(ideal_new_extend,columns=['MONTH','STORECODE','GRP','QTY','VALUE']))

full_final=[]

for index,row in ideal_new.iterrows():

    

#     each_value=int(row['VALUE']/31)

    if row['QTY']==0:

        each_qty=0

        for i in range(1,32):

            full_final.append([row['MONTH'],i,row['STORECODE'],row['GRP'],each_qty])

    else:

        if row['QTY']<=31:

            for i in range(1,32):

                if i<=row['QTY']:

                    full_final.append([row['MONTH'],i,row['STORECODE'],row['GRP'],1])

                else:

                    full_final.append([row['MONTH'],i,row['STORECODE'],row['GRP'],0])

        else:

            each_qty=int(row['QTY']/31)

            final_rows=[]

            for i in range(1,32):

                if i==31:

                    final_rows.append([row['MONTH'],i,row['STORECODE'],row['GRP'],each_qty])

                    remaining=row['QTY']-sum([each_qty]*31)

                    if remaining<=31:

                        for i in range(1,32):

                            if i<=remaining:

                                final_rows.append([row['MONTH'],i,row['STORECODE'],row['GRP'],1])

                            else:

                                final_rows.append([row['MONTH'],i,row['STORECODE'],row['GRP'],0])

                    else:

                        each_qty=int(remaining/31)

                        for i in range(1,32):

                            if i==31:

                                final_rows.append([row['MONTH'],i,row['STORECODE'],row['GRP'],remaining-sum([each_qty]*30)])

                            else:

                                final_rows.append([row['MONTH'],i,row['STORECODE'],row['GRP'],each_qty])



                else:

                    final_rows.append([row['MONTH'],i,row['STORECODE'],row['GRP'],each_qty])

            full_final.extend(pd.DataFrame(final_rows,columns=['MONTH','DAY','STORECODE','GRP','QTY']).groupby(['MONTH','DAY','STORECODE','GRP'],as_index=False)['QTY'].sum().values.tolist())

final_ideal=pd.DataFrame(full_final,columns=['MONTH','DAY','STORECODE','GRP','QTY'])


ideal=final_ideal

working=working.drop(['BILL_ID','BILL_AMT','SGRP','CMP','MBRD','BRD','SSGRP'],axis=1)

working.QTY = working.QTY.astype(int)

working=working[working['PRICE']*working['QTY']==working['VALUE']].drop(['VALUE','PRICE'],axis=1)

working=working.groupby(['MONTH','STORECODE','GRP','DAY'],as_index=False)['QTY'].sum()

from sklearn.model_selection import cross_val_score,cross_val_predict, GridSearchCV

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_validate

import xgboost as xgb

from xgboost.sklearn import XGBRegressor



def xg_model(X, y):

# Perform Grid-Search

    xgb1 = XGBRegressor()

    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['reg:linear'],

              'learning_rate': [.03, 0.05, .07], #so called `eta` value

              'max_depth': [3,4,5, 6, 7],

              'min_child_weight': [4],

              'silent': [1],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [20,100,200,500]}



    gsc = GridSearchCV(

        estimator=xgb1,

        param_grid=parameters,

        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    

    grid_result = gsc.fit(X, y)

    best_params = grid_result.best_params_

    print(best_params)

    
id_w=ideal.append(working)

id_w['DAY'] = id_w['DAY'].astype(object)





xg=XGBRegressor(max_depth=6, n_estimators=100)

xg.fit(pd.get_dummies(id_w.drop(['QTY'],axis=1)),id_w['QTY'])
scores = cross_validate(xg, pd.get_dummies(id_w.drop(['QTY'],axis=1)),id_w['QTY'], cv=10,scoring=['neg_mean_squared_error','neg_mean_absolute_error'],return_train_score=False)

round(scores['test_neg_mean_absolute_error'].mean())
import pandas as pd

import numpy as np

ideal=pd.read_csv('../input/ackshay-store/Hackathon_Ideal_Data.csv')

working=pd.read_csv('../input/ackshay-store/Hackathon_Working_Data.csv')

ideal=ideal[~((ideal['QTY']==0)&(ideal['VALUE']!=0))]

full_final=[]

for index,row in ideal.iterrows():

    

#     each_value=int(row['VALUE']/31)

    if row['QTY']==0:

        each_qty=0

        price=0

        for i in range(1,32):

            full_final.append([row['MONTH'],i,row['STORECODE'],row['GRP'],row['BRD'],price,each_qty])

    else:

        price=row['VALUE']/row['QTY']

        if row['QTY']<=31:

            for i in range(1,32):

                if i<=row['QTY']:

                    full_final.append([row['MONTH'],i,row['STORECODE'],row['GRP'],row['BRD'],price,1])

                else:

                    full_final.append([row['MONTH'],i,row['STORECODE'],row['GRP'],row['BRD'],price,0])

        else:

            each_qty=int(row['QTY']/31)

            final_rows=[]

            for i in range(1,32):

                if i==31:

                    final_rows.append([row['MONTH'],i,row['STORECODE'],row['GRP'],row['BRD'],price,each_qty])

                    remaining=row['QTY']-sum([each_qty]*31)

                    if remaining<=31:

                        for i in range(1,32):

                            if i<=remaining:

                                final_rows.append([row['MONTH'],i,row['STORECODE'],row['GRP'],row['BRD'],price,1])

                            else:

                                final_rows.append([row['MONTH'],i,row['STORECODE'],row['GRP'],row['BRD'],price,0])

                    else:

                        each_qty=int(remaining/31)

                        for i in range(1,32):

                            if i==31:

                                final_rows.append([row['MONTH'],i,row['STORECODE'],row['GRP'],row['BRD'],price,remaining-sum([each_qty]*30)])

                            else:

                                final_rows.append([row['MONTH'],i,row['STORECODE'],row['GRP'],row['BRD'],price,each_qty])



                else:

                    final_rows.append([row['MONTH'],i,row['STORECODE'],row['GRP'],row['BRD'],price,each_qty])

            full_final.extend(pd.DataFrame(final_rows,columns=['MONTH','DAY','STORECODE','GRP','BRD','PRICE','QTY']).groupby(['MONTH','DAY','STORECODE','GRP','BRD','PRICE'],as_index=False)['QTY'].sum().values.tolist())

final_ideal=pd.DataFrame(full_final,columns=['MONTH','DAY','STORECODE','GRP','BRD','PRICE','QTY'])



ideal=final_ideal

working=working.drop(['BILL_ID','BILL_AMT','SGRP','CMP','MBRD','SSGRP'],axis=1)

working.QTY = working.QTY.astype(int)

working=working[working['PRICE']*working['QTY']==working['VALUE']].drop(['VALUE'],axis=1)

working=working.groupby(['MONTH','STORECODE','GRP','DAY','BRD','PRICE'],as_index=False)['QTY'].sum()



working=working[['MONTH','DAY','STORECODE','GRP','QTY','BRD','PRICE']]

id_w=pd.concat([ideal,working])

id_w['DAY'] = id_w['DAY'].astype(object)

id_w=id_w[id_w['QTY']!=0]





import xgboost as xgb

from xgboost.sklearn import XGBRegressor

from sklearn.ensemble import RandomForestRegressor 



xgb=XGBRegressor(max_depth= 6, n_estimators= 200)

scores = cross_validate(xgb, pd.get_dummies(id_w.drop(['QTY'],axis=1)),id_w['QTY'], cv=5,scoring=['neg_mean_squared_error','neg_mean_absolute_error'],return_train_score=False)

print(scores['test_neg_mean_absolute_error'].mean())