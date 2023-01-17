# Import data manipulation libraries
import pandas as pd
import numpy as np
import datetime

# Visualization libaries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.linear_model import SGDRegressor
import xgboost as xgb
# Import training and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combine = [train, test]

print(train.head(3))
print(test.head(3))
# Define column date as datatype date and define new date features
for dataset in combine:
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['year'] = dataset.date.dt.year
    dataset['month'] = dataset.date.dt.month
    dataset['day'] = dataset.date.dt.day
    dataset['dayofyear'] = dataset.date.dt.dayofyear
    dataset['dayofweek'] = dataset.date.dt.dayofweek
    dataset['weekofyear'] = dataset.date.dt.weekofyear
    
    # Additional date features
    dataset['log_dayofyear'] = np.log(dataset['dayofyear'])
    dataset['day_power_year'] = np.log((np.log(dataset['dayofyear'] + 1)) ** (dataset['year'] - 2000))
    dataset['day_week_power_year'] = np.log(np.log(dataset['dayofyear'] + 1) * (np.log(dataset['weekofyear'] + 1)) ** (dataset['year'] - 2000))
    
    # Drop date
    dataset.drop('date', axis=1, inplace=True)
    
train.head()

# Add features, such as average sales pr. day, average sales pr. month, rolling mean 90 periods
def add_avg(x):
    x['daily_avg']=x.groupby(['item','store','dayofweek'])['sales'].transform('mean')
    x['monthly_avg']=x.groupby(['item','store','month'])['sales'].transform('mean')
    return x
train = add_avg(train).dropna()

daily_avg = train.groupby(['item','store','dayofweek'])['sales'].mean().reset_index()
monthly_avg = train.groupby(['item','store','month'])['sales'].mean().reset_index()

def merge(x,y,col,col_name):
    x =pd.merge(x, y, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False,validate=None)
    x=x.rename(columns={'sales':col_name})
    return x

test = merge(test, daily_avg,['item','store','dayofweek'],'daily_avg')
test = merge(test, monthly_avg,['item','store','month'],'monthly_avg')
# Adding rolling mean feature to train
df = train.groupby(['item'])['sales'].rolling(10).mean().reset_index().drop('level_1', axis=1)
train['rolling_mean'] = df['sales']

# Adding last 3 months of rolling mean from training to test data 
# (Doing this and shifting rolling mean 3 months in training data)
rolling_mean_test = train.groupby(['item','store'])['rolling_mean'].tail(90).copy().reset_index().drop('index', axis=1)
test['rolling_mean'] = rolling_mean_test

# Shifting rolling mean 3 months
train['rolling_mean'] = train.groupby(['item'])['rolling_mean'].shift(90)
train.tail()
combine = [train, test]

for dataset in combine:
    dataset['item_times_rolling_mean'] = dataset['item'] * dataset['rolling_mean']
    dataset['store_times_rolling_mean'] = dataset['store'] * dataset['rolling_mean']
    dataset['dayofyear_times_rolling_mean'] = dataset['dayofyear'] * dataset['rolling_mean']
    
train.columns,test.columns
# Let's check how the features correlate
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
# Seems like dayofyear and weekofyear has high internal correlation and correlates highly with month, so let's drop those.
# All average/mean features also correlated heavily. Since Monthly Average correlates most with sales, we keep this.
combine = [train, test]

for dataset in combine:
    dataset.drop(['dayofyear', 
                  'weekofyear',
                  'daily_avg',
                  'day',
                  'month',
                  'item',
                  'store',
                  'day_week_power_year',
                  'log_dayofyear',
                  #'monthly_avg',
                  'dayofyear_times_rolling_mean',
                  #'store_times_rolling_mean',
                  #'rolling_mean',
                  'item_times_rolling_mean'],
                  #'year'],
                  #'dayofweek',
                  #'day_power_year'], 
                 axis=1, 
                 inplace=True)
    
# Let's check the correlation with all dropped features
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

train.tail()
# Feature scaling for faster algorithm optimum
temp_sales = train['sales']
temp_id = test['id']
train = (train - train.mean()) / train.std()
test = (test - test.mean()) / test.std()
train['sales'] = temp_sales
test['id'] = temp_id

train.head()
# Let's prepare the training and data set
x_train = train.drop('sales', axis=1).dropna()
y_train = train['sales']
test.sort_values(by=['id'], inplace=True)
x_test = test.drop('id', axis=1)

x_pred = test.drop('id', axis=1)
df = train

train.tail()
# Linear Regression Model
#regr = LinearRegression()
#regr.fit(x_train, y_train)
#prediction = regr.predict(x_test)

# Decision Tree (Best performer so far)
# max_depth = 100, min_samples_leaf = 10
#clf = tree.DecisionTreeRegressor()
#clf.fit(x_train, y_train)
#prediction = clf.predict(x_test)

# SGD Stochastic Gradient Descent Regressor
#combine = [x_train, x_test]

#for dataset in combine:
#    dataset['day_power_year'] = (dataset['day_power_year'] - dataset['day_power_year'].mean()) / dataset['day_power_year'].std()
#    dataset['monthly_avg'] = (dataset['monthly_avg'] - dataset['monthly_avg'].mean()) / dataset['monthly_avg'].std()
#sgd = SGDRegressor(max_iter = 1000, alpha = .0003, learning_rate='constant', verbose=1)
#sgd.fit(x_train, y_train)
#prediction = sgd.predict(x_test)

# Let's run XGBoost and predict those sales!
x_train,x_test,y_train,y_test = train_test_split(df.drop('sales',axis=1),df.pop('sales'),random_state=123,test_size=0.2)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'mae'}
                    ,dtrain=matrix_train,num_boost_round=500, 
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
y_pred = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)
# Add to submission

submission = pd.DataFrame({
        "id": test['id'],
        "sales": y_pred.round()
})

submission.to_csv('sub500.csv',index=False)
submission.head()

