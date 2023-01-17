# import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score, GridSearchCV

import datetime as dt



plt.rcParams["figure.figsize"] = (15,5)

data = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv',parse_dates =['date'],index_col=['date'])

data.head()
data.info()
data['sales'].hist(bins = 20)

mean = data['sales'].mean()

median = np.median(data['sales'])

minimum = data.sales.min()

maximum = data.sales.max()



mean,median,minimum,maximum
logged = np.log1p(data['sales'])

logged.hist(bins = 20)

logged.mean(),logged.median()
def smape(actual,predict,islog=True):

    if islog == True:

        actual = np.exp(actual) - 1

        predict = np.exp(predict) -1

        

    return 100*np.mean(2*np.abs(actual-predict)/(np.abs(actual)+np.abs(predict)))

    

smape_score = make_scorer(smape,greater_is_better=False)



def evaluate_model(df,features):

    all_X = df[features]

    all_y = np.log1p(df['sales'])

    

    tree = DecisionTreeRegressor(random_state = 1)

    scores = cross_val_score(tree,all_X,all_y,scoring=smape_score,cv=5)

    avg_score = -(scores.mean())  # avoid negative sign which caused due to make_scorer

    

    return avg_score
df = data.groupby(['date','item'])['sales'].sum().unstack()

df.plot(figsize=(15,10))

plt.ylabel('Total sale')

plt.title('itemwise sale')
df = data.groupby(['date','store'])['sales'].sum().unstack()

df.plot(figsize=(15,10))

plt.ylabel('sales')

plt.title('Total sale store wise')
features = list(data.columns)

features.remove('sales')

print('SMAPE with available features: {:.4f}'.format(evaluate_model(data,features)))
data['month'] = data.index.month

data['year'] = data.index.year

data['dow'] = data.index.dayofweek

data['day'] = data.index.day

data['quarter'] = data.index.quarter

data['week'] = data.index.week



data.head()
sns.boxplot(x='month',y='sales',hue = 'year', data=data)
sns.boxplot(x='week',y='sales',hue ='year', data=data)
sns.boxplot(x='dow',y='sales',hue ='year', data=data)
sns.boxplot(x='day',y='sales',hue ='year', data=data)
sns.boxplot(x='quarter',y='sales',hue ='year', data=data)
features = list(data.columns)

features.remove('sales')

print('SMAPE with Basic feature engineering: {:.4f}'.format(evaluate_model(data,features)))
store_item_df = pd.pivot_table(data, index='item', values='sales', columns='store',margins=True, aggfunc=np.mean)

store_item_df.head()
sns.heatmap(store_item_df)
fig,(ax1,ax2) = plt.subplots(1,2, figsize = (15,5))

store_item_df['All'].sort_values().plot.bar(ax=ax1, title ='itemwise average sale' )

store_item_df.loc['All',:].sort_values().plot.bar(ax=ax2, title = 'storewise average sale')
i = store_item_df['All'].sort_values().index

c = store_item_df.loc['All',:].sort_values().index

store_item_df = store_item_df[c]

store_item_df = store_item_df.reindex(i)



store_item_df.drop('All',axis=1,inplace=True)

store_item_df.drop('All',axis=0,inplace = True)



store_item_df.head()

plt.figure(figsize=(10,10))

sns.heatmap(store_item_df, square=True)
# Prepare dataframe to encode store item interaction

encode_df = pd.DataFrame(np.arange(1,501,1).reshape((50,10)))

encode_df.columns = store_item_df.columns

encode_df.index = store_item_df.index

def encode_feature(row):

    r = row['item']

    c = row['store']

    return encode_df.loc[r,c]



data['store_item'] = data.apply(encode_feature,axis=1)



data.head()
features = list(data.columns)

features.remove('sales')



print('SMAPE with feature engineering step 3: {:.4f}'.format(evaluate_model(data,features)))
data['m_yr'] = data['year'] + data['month']/100

data['week_frac'] = data['week'] + data['dow']/100

data.head()
drop_col = ['sales'] # 'year','day','month','dow','week','store','item',



features = list(data.columns)

for i in drop_col:

    features.remove(i)

    

evaluate_model(data,features)
h = {'criterion' : ['mse','friedman_mse'],

                   'min_samples_leaf': [1,3,5],

                   'min_samples_split': [2,4,6]

                  }





dtr = DecisionTreeRegressor(random_state=1)

grid = GridSearchCV(dtr, param_grid=h, scoring=smape_score, cv=5, verbose=10, n_jobs =-1)



all_X = data[features]

all_y = np.log1p(data['sales'])

grid.fit(all_X,all_y)

        

pred = grid.predict(data[features])

pred = [max(0,p) for p in pred]



error = smape(all_y,pred,islog=True)



print('SMAPE on Last Step: {:.4f}'.format(error))





def transform_feature(df):

    

    # Apply basic feature engineering

    df['month'] = df.index.month

    df['year'] = df.index.year  

    df['dow'] = df.index.dayofweek

    df['day'] = df.index.day

    df['quarter'] = df.index.quarter

    df['week'] = df.index.week

    

    # Apply advance feature engineering

    df['store_item'] = df.apply(encode_feature,axis=1)

    df['m_yr'] = df['year']+df['month']/100

    df['week_frac'] = df['week']+df['dow']/100

    

    return df

    

    
holdout = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv',parse_dates =['date'],index_col=['date'])

ids= holdout['id']

holdout = transform_feature(holdout)



pred_h = grid.predict(holdout[features])

pred_h = np.exp(pred_h)-1

pred_h = [max(0,p) for p in pred_h]



submission_df = {'id':ids, 'sales':pred_h}

submission =pd.DataFrame(submission_df)



submission.to_csv('submission.csv',index=False)


