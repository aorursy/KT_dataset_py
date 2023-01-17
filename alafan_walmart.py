# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_squared_error





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMRegressor



import seaborn as sns

from matplotlib import  pyplot as plt

%matplotlib inline
#reading the data

folder = '/kaggle/input/m5-forecasting-accuracy/'

calendar = pd.read_csv(folder+'calendar.csv')

sell_prices = pd.read_csv(folder+'sell_prices.csv')

stv = pd.read_csv(folder+'sales_train_validation.csv')
calendar.head(5)
calendar.describe()
stv.head()
print(

'''

Total amount of id: {5}

Unique:

item_id: {0}

dept_id: {1}

cat_id:  {2}

store_id:{3}

state_id:{4}

'''.format(

    stv.item_id.nunique(),

    stv.dept_id.nunique(),

    stv.cat_id.nunique(),

    stv.store_id.nunique(),

    stv.state_id.nunique(),

    len(stv)

)

)
sell_prices.head().T
cat_col = ['item_id','dept_id','cat_id','store_id','state_id']

index_col = cat_col+['id']
le = LabelEncoder()

for i in cat_col: 

    stv[i+'_label'] = le.fit_transform(stv[i])

#train

indx_col_label =['id','item_id_label','dept_id_label','cat_id_label','store_id_label',

             'state_id_label']

X = pd.DataFrame(

    columns=['id','item_id_label','dept_id_label','cat_id_label','store_id_label',

             'state_id_label','day','target'])



learning_days = 84

period = 28

for day in range(1914-learning_days,1914):

    #print(day)

    cols_to = ['target']

    cols_from = ['d_'+str(day)]

    for i in range(1,4):

            #print(i)

        if day>=i*period:

            cols_from.append('d_'+str(day-period*i))

            cols_to.append('prev'+str(i)) 

    cols_to+=indx_col_label

    cols_from+=indx_col_label

    tmp = stv.loc[:,cols_from]

    tmp.columns = cols_to

    tmp.loc[:,'day'] = day

    

    X = X.append(tmp)
X.head()
X[X.target>=170]
ftr = [ 'item_id_label', 'dept_id_label', 'cat_id_label', 'store_id_label', 'state_id_label','day','prev1','prev2','prev3']
for i in ftr:

    X[i] = X[i].astype(int)

X['target'] = X['target'].astype(int)
sns.pairplot(X[ftr+['target']][::300])
X[X.target>=100].item_id_label.value_counts()
fig, ax = plt.subplots(1,1,figsize = (15,5))

X[::100].plot.scatter(x = 'prev1',y = 'target',ax = ax,alpha = .1)

X[::100].plot.scatter(x = 'prev2',y = 'prev1',ax = ax,alpha = .1, color = 'g')

X[::100].plot.scatter(x = 'prev3',y = 'prev2',ax = ax,alpha = .1, color = 'r')

X['prev1'] = X['prev1'].clip(0,75)

X['prev2'] = X['prev2'].clip(0,75)

X['prev3'] = X['prev3'].clip(0,75)

X['target'] = X['target'].clip(0,75)
X[X.item_id_label.isin([702,1198])][::100][['prev1','prev2','prev3','target']].plot.box()
lgbm = LGBMRegressor(max_depth = 7, objective='rmse')
split_points = [X.day.min(),int(X.day.mean()),X.day.max()] 
split_points
X_train = X[(X.day>=split_points[0])&(X.day<split_points[1])][ftr]

X_test = X[(X.day>=split_points[1])&(X.day<split_points[2])][ftr]

y_train = X[(X.day>=split_points[0])&(X.day<split_points[1])][['target']]

y_test = X[(X.day>=split_points[1])&(X.day<split_points[2])][['target']]
lgbm.fit(X_train,y_train.target)

val_pred = lgbm.predict(X_test)

y_train.loc[:,'prev'] = y_train.target.shift(1)
(mean_squared_error(val_pred,y_test.target)/mean_squared_error(y_train[1:]['target'],y_train[1:]['prev']))**0.5
(mean_squared_error(y_test,val_pred))**0.5
#validation

test = pd.DataFrame(columns=['id','item_id_label','dept_id_label','cat_id_label','store_id_label',

             'state_id_label','day','prev1','prev2','prev3'])

for day in range(1914,1942):

    #tmp = stv.loc[:,indx_col_label]

    tmp = X[X.day==day-28].loc[:,indx_col_label+['target','prev1','prev2']]

    tmp.rename(columns = {'target':'prev1','prev1':'prev2','prev2':'prev3'})

    tmp.loc[:,'day'] = day



    test = test.append(tmp)

lgbm.fit(X[ftr],X['target'])
test = test.fillna(0)

for i in ftr:

    test[i] = test[i].astype(int)

pred = lgbm.predict(test[ftr])

res = test[['id','day']]

res['target'] = pred

base = 1914

r = stv[['id']]



for i in range(base,base+28):

    col = 'F'+str(i-base+1)

    r = r.merge(res[res.day==i][['id','target']], on = 'id',how = 'left')

    r = r.rename(columns = {'target':col})
X.head()
#evaluation

evaluation = pd.DataFrame(columns=['id','item_id_label','dept_id_label','cat_id_label','store_id_label',

             'state_id_label','day','prev1','prev2','prev3'])

for day in range(1942,1970):

    tmp = X[X.day==day-56].loc[:,indx_col_label+['prev1','prev2']]

    tmp.rename(columns = {'prev1':'prev2','prev2':'prev3'})

    tmp.loc[:,'prev1'] = res[res.day==day]['target']

    tmp.loc[:,'day'] = day



    test = test.append(tmp)

    evaluation = evaluation.append(tmp)



evaluation = evaluation.fillna(0)

for i in ftr:

    evaluation[i] = evaluation[i].astype(int)
evaluation['id'] = evaluation['id'].apply(lambda x: x.replace('validation','evaluation'))
pred_ev = lgbm.predict(evaluation[ftr])
evaluation.head()
base = 1942

evaluation['target']=pred_ev

r2 = pd.DataFrame(columns = ['id'])

r2['id'] =stv['item_id']+'_'+stv['store_id']+'_'+'evaluation'

for i in range(base,base+28):

    col = 'F'+str(i-base+1)

    r2 = r2.merge(evaluation[evaluation.day==i][['id','target']], on = 'id',how = 'left')

    r2 = r2.rename(columns = {'target':col})
r.append(r2).to_csv('submission.csv',index = False)