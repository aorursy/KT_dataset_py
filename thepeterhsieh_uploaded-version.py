# import 
from IPython import get_ipython
from IPython.display import display
import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
fInput = "../input/aia-dt4-fraudcard"
# check the data source 
from subprocess import check_output
print(check_output(["ls", fInput]).decode("utf8"))
# load the dataset as test & train
train = pd.read_csv(fInput + "/train.csv")
test = pd.read_csv(fInput + "/test.csv")
train.head(5)
# raw shape and type
display(f'train data type: {type(train)}', f'test data type: {type(test)}')

# store the shapes
shapes = {'train':[], 'test':[], 'prep':[]}

# define a func to store shapes along the processes
def record(train, test, process='initial'):
    shapes['train'].append(train.shape)
    shapes['test'].append(test.shape)
    shapes['prep'].append(process)
    
    print(shapes)

record(train, test, process='initial')
# pop the columns = 'Id'

print('the first train column is userId: ', end='')
print(train.columns[0] == 'user_id')
print('the first test column is userId: ', end='')
print(test.columns[0] == 'user_id')

## saveand pop the 'Id' columns for further use 
train_id = train.pop('user_id') # might be useless
test_id = test.pop('user_id') # this is for the submission 

# check if the popping is executed
# since the prediction is not hinged on it
print('after popping, train: ' , end='') 
print(not('user_id' in train.columns))
print('after popping, test: ' , end='') 
print(not('user_id' in test.columns))

# record after popping 'user_id'
record(train, test, process='pop_userID')
# pop the columns = 'device_id'

print('the first train column is device_id: ', end='')
print(train.columns[0] == 'device_id')
print('the first test column is device_id: ', end='')
print(test.columns[0] == 'device_id')

## saveand pop the 'deviceId' columns for further use 
train_id = train.pop('device_id') # might be useless
test_id = test.pop('device_id') # this is for the submission 

# check if the popping is executed
# since the prediction is not hinged on it
print('after popping, train: ' , end='') 
print(not('device_id' in train.columns))
print('after popping, test: ' , end='') 
print(not('device_id' in test.columns))

# record after popping 'deviceId'
record(train, test, process='pop_deviceID')
train.info()
# search for the original numerical entries
## 'age', 'sex', 'purchase_value'
display(train.describe().columns)
train.describe()
# imbalance binary class
train['class'].value_counts()
def calc_prevalence(y):
    return (sum(y)/len(y))

display(calc_prevalence(train['class'].values))
# almost 10% of cards have been frauded

display(train['class'].sum())
def relation(feature='sex', plot='pie'):
    
    relation = train.groupby(['class'])[feature].value_counts()
    display(relation)
    
    if plot == 'pie':
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 10))
        ax0.pie(relation[0], labels=relation[0].index, autopct='%1.1f%%')
        ax0.legend('0')
        ax1.pie(relation[1], labels=relation[1].index, autopct='%1.1f%%')
        ax1.legend('1')

    if plot == 'bar':
        fig, ax = plt.subplots(2, 1, figsize=(15, 5))

        index = relation[0].index
        relation_ = pd.DataFrame(relation[0])
        relation_['index'] = index
        relation_ = relation_.sort_values('index')
        relation_.drop('index', inplace=True, axis=1)
        relation_.plot.bar(ax=ax[0])
        ax[0].legend('0')
        

        index = relation[1].index
        relation_ = pd.DataFrame(relation[1])
        relation_['index'] = index
        relation_ = relation_.sort_values('index')
        relation_.drop('index', inplace=True, axis=1)
        relation_.plot.bar(ax=ax[1])
        ax[1].legend('1')

relation(feature='age', plot='bar')
relation(feature='purchase_value', plot='bar')
relation(feature='sex', plot='pie')
relation(feature='browser', plot='pie')
relation(feature='source', plot='pie')
# country with nan    
relation(feature='country', plot='pie')
test_time = test[['purchase_time', 'signup_time']].copy()

# target relation with times
test_time['purchase_time'] = pd.to_datetime(test_time['purchase_time'])
test_time['signup_time'] = pd.to_datetime(test_time['signup_time'])
test_time['diff_time'] = test_time['purchase_time'] - test_time['signup_time']
display(test_time.head(5))

# check if signup_time is later than the purchase_time 
display("test: signup time later than the purchase time")
display((test_time['diff_time'] <= pd.Timedelta(0)).sum())
train_time = train[['purchase_time', 'signup_time', 'class']].copy()

# to_datetime
train_time['purchase_time'] = pd.to_datetime(train_time['purchase_time'])
train_time['signup_time'] = pd.to_datetime(train_time['signup_time'])
# train_time['diff_time'] = train_time['purchase_time'] - train_time['signup_time']

# delta_hour and delta_day
train_time['diff_hour'] = (train_time['purchase_time'] - train_time['signup_time']).dt.total_seconds()/(60*60)
train_time['diff_day'] = (train_time['purchase_time'] - train_time['signup_time']).dt.total_seconds()/(60*60*24)

train_time.head(5)
# check if signup_time is later than the purchase_time
display("train: signup time later than the purchase time")
display((train_time['diff_hour'] <= 0).sum())
# the odd time and its class 
display("the fraud classes")
display(train_time.loc[(train_time['diff_hour'] <= 0)]['class'].sum())
display(train_time.loc[(train_time['diff_hour'] <= 0)])
# check those are not fraud
display("the non fraud classes")
display(len(train_time.loc[(train_time['diff_hour'] <= 0) & (train_time['class'] == 0)]))
display(train_time.loc[(train_time['diff_hour'] <= 0) & (train_time['class'] == 0)])
# check the signup time is later than the purchase time 1 hour or more 
display("any signup time is later than the purchase time 1 hour or more?")
display(len(train_time.loc[(train_time['diff_hour'] <= -1)]))

display("if the signup time is later than the purchase time, it is highly likely that it is a fraud.")
# check out the distribution of the fraud time 
fraud = train_time.loc[(train_time['class'] == 1)]
display(fraud.min(), fraud.max())
plt.hist(fraud['diff_day'].values, bins=120, density=False)
plt.show()
# check out the distribution of the fraud time 
non_fraud = train_time.loc[(train_time['class'] == 0)]
display(non_fraud.min(), non_fraud.max())
plt.hist(non_fraud['diff_day'].values, bins=120, density=False)
plt.show()
fig, ax = plt.subplots()

# fraud
fraud_and_time = []
span = range(-2, 24, 1)

for i in span:
    fraud_and_time.append(len(fraud.loc[(fraud['diff_hour'] <= (i + 1)) & (i < fraud['diff_hour'])]))
    
print(fraud_and_time)
ax.bar(span, fraud_and_time, color=(1, 0, 0, 0.5), label='fraud')


# non-fraud
non_fraud_and_time = []

for i in span:
    non_fraud_and_time.append(len(non_fraud.loc[(non_fraud['diff_hour'] <= (i + 1)) & (i < non_fraud['diff_hour'])]))
    
print(non_fraud_and_time)
ax.bar(span, non_fraud_and_time, color=(0,0,1,0.5), label='nonfraud')

ax.legend()
fig, ax = plt.subplots()

# fraud
fraud_and_time = []
span = range(1, 2880, 1)

for i in span:
    fraud_and_time.append(len(fraud.loc[(fraud['diff_hour'] <= (i + 1)) & (i < fraud['diff_hour'])]))
    
#print(fraud_and_time)
ax.bar(span, fraud_and_time, color=(1, 0, 0, 0.5), label='fraud')
ax.legend()
fig, ax = plt.subplots()

# non-fraud
non_fraud_and_time = []

for i in span:
    non_fraud_and_time.append(len(non_fraud.loc[(non_fraud['diff_hour'] <= (i + 1)) & (i < non_fraud['diff_hour'])]))
    
#print(non_fraud_and_time)
ax.bar(span, non_fraud_and_time, color=(0,0,1,0.5), label='nonfraud')

ax.legend()
fig, ax = plt.subplots()

# fraud
fraud_and_time = []
span = range(-2, 120, 1)

for i in span:
    fraud_and_time.append(len(fraud.loc[(fraud['diff_day'] <= (i + 1)) & (i < fraud['diff_day'])]))
    
print(fraud_and_time)
ax.bar(span, fraud_and_time, color=(1, 0, 0, 0.5), label='fraud')


# non-fraud
non_fraud_and_time = []

for i in span:
    non_fraud_and_time.append(len(non_fraud.loc[(non_fraud['diff_day'] <= (i + 1)) & (i < non_fraud['diff_day'])]))
    
print(non_fraud_and_time)
ax.bar(span, non_fraud_and_time, color=(0,0,1,0.5), label='nonfraud')

ax.legend()
# pop the target: class from training dataframe
target = train.pop('class')
# target.head()
record(train, test, process='pop_class')
# concate train and test data
all_Data = pd.concat((train, test), axis=0, ignore_index=True)
# all_Data.head(5)
def onehot_concat(data=all_Data, train=train, feature='browser', joint=True):
    # count the unique browser of all_data
    values = pd.DataFrame(data[feature].value_counts().values, columns=['count']) 
    values[feature] = data[feature].value_counts().index

    # the pie chart
    ax1 = plt.subplot(1, 2, 1)
    plt.pie(values['count'], labels=values[feature], autopct='%1.1f%%')
    ax1.set_title('allData')

    # count the unique browser of train
    values = pd.DataFrame(train[feature].value_counts().values, columns=['count']) 
    values[feature] = train[feature].value_counts().index

    # the pie chart
    ax2 = plt.subplot(1, 2, 2)
    plt.pie(values['count'], labels=values[feature], autopct='%1.1f%%')
    ax2.set_title('trainData')
    
    # one-hot 'browser'
    hot = preprocessing.OneHotEncoder(sparse=False, dtype=np.int64)
    arr = np.array(data[feature]).reshape(-1, 1)
    hot.fit(arr)
    arr = pd.DataFrame(hot.transform(arr), columns=list(hot.categories_[0]))
    
    if joint:
        # drop the 'browser', and concate with the one-hot browser
        data = data.drop(feature, axis=1)
        data = pd.concat((data, arr), axis=1)
    
    return data
all_Data = onehot_concat(data=all_Data, train=train, feature='browser', joint=True)
all_Data.head(5)
all_Data = onehot_concat(data=all_Data, train=train, feature='source', joint=True)
all_Data.head(5)
# purchase time 
all_Data['purchase_time'] = pd.to_datetime(all_Data['purchase_time'])
all_Data['purchase_month'] = all_Data['purchase_time'].dt.month
all_Data['purchase_day'] = all_Data['purchase_time'].dt.day
all_Data['purchase_hour'] = all_Data['purchase_time'].dt.hour
all_Data['purchase_dayofweek'] = all_Data['purchase_time'].dt.dayofweek
#display(all_Data['purchase_min'])

# signup time
all_Data['signup_time'] = pd.to_datetime(all_Data['signup_time'])
all_Data['signup_month'] = all_Data['signup_time'].dt.month
all_Data['signup_day'] = all_Data['signup_time'].dt.day
all_Data['signup_hour'] = all_Data['signup_time'].dt.hour
all_Data['signup_dayofweek'] = all_Data['signup_time'].dt.dayofweek

# delta
all_Data['diff_hour'] = (all_Data['purchase_time'] - all_Data['signup_time']).dt.total_seconds()/(60*60)
all_Data['diff_day'] = (all_Data['purchase_time'] - all_Data['signup_time']).dt.total_seconds()/(60*60*24)
all_Data.info()
# country 
## missing value
country = all_Data['country'].value_counts(ascending=False)
_countryweight = dict(country.apply(lambda x: int(round(x/country.sum()*1000))))

countryweight = [ country for country, weight in _countryweight.items() for i in range(weight)]
#display(countryweight)
# random.choice(countryweight)

# fill in with random weight 
all_Data['country'] = all_Data['country'].apply(lambda x: random.choice(countryweight) if x is np.nan else x)
all_Data.info()
# Normalize age, purchase_value, diff_hour, diff_day
norm = ['age', 'purchase_value', 'diff_hour', 'diff_day']

for i in norm:
    u = all_Data[i]
    u = (u - u.min(axis=0))/(u.max(axis=0) - u.min(axis=0) + 1e-7)
    all_Data[i] = u
    
# or using sklearn.preprocessing.StandardScaler
# store the shapes
shapes_all = {'all':[], 'prep':[]}

# define a func to store shapes along the processes
def record_all(all_Data, process=''):
    shapes_all['all'].append(all_Data.shape)
    shapes_all['prep'].append(process)
    
    print(shapes_all)

record_all(all_Data, process='initial')
# drop the datetime datatype
all_Data.drop(['signup_time', 'purchase_time'], axis=1, inplace=True)

record_all(all_Data, process='drop datetime')
# drop the sex
# all_Data.drop(['sex'], axis=1, inplace=True)

# record_all(all_Data, process='drop sex')
# drop the country
all_Data.drop(['country'], axis=1, inplace=True)

record_all(all_Data, process='drop country')
# drop the browsers
#all_Data.drop(['Chrome', 'FireFox', 'IE', 'Opera', 'Safari'], axis=1, inplace=True)

#record_all(all_Data, process='drop browsers')
# drop the source
#all_Data.drop(['Ads', 'Direct', 'SEO'], axis=1, inplace=True)

#record_all(all_Data, process='drop sources')
all_Data.info()
# slice the all_Data
train = all_Data[0:108800].copy()
test = all_Data[108800:].copy()
test.index = range(27200)

record(train, test, 'after feature selecting')
display(train.head(5), test.head(5))
## train_test_split, split train data into train and valid
x_train, x_valid, y_train, y_valid = train_test_split(train.values,
                                                      target.values,
                                                      test_size=0.2,
                                                      random_state=24,
                                                      stratify=target.values)

print('x_train shape:', x_train.shape, '\ny_train shape:', y_train.shape)
print('x_valid shape:', x_valid.shape, '\ny_valid shape:', y_valid.shape)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(#max_depth = 5, 
                            n_estimators=1000, 
                            criterion='entropy',
                            random_state = 24)
from sklearn import metrics

rf.fit(x_train, y_train)

pred = rf.predict(x_valid)
#pred = np.log(pred)
print('train evaluation :')
print('R2 score:{:.2f}'.format(metrics.r2_score(y_valid, pred)))
print('RMSE:{:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_valid, pred))))
pred = rf.predict(test.values)
from sklearn.linear_model import Lasso,Ridge

# Lasso
model_Lasso = Lasso(alpha = 0.001)
model_Lasso.fit(x_train,y_train)
pred = model_Lasso.predict(x_valid)
#pred = np.log(pred)
print('train evaluation :')
print('R2 score:{:.2f}'.format(metrics.r2_score(y_valid, pred)))
print('RMSE:{:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_valid, pred))))
# Ridge
model_Ridge = Ridge(alpha = 0.1)
model_Ridge.fit(x_train,y_train)
pred = model_Ridge.predict(x_valid)
#pred = np.log(pred)
print('train evaluation :')
print('R2 score:{:.2f}'.format(metrics.r2_score(y_valid, pred)))
print('RMSE:{:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_valid, pred))))
from sklearn import svm

model_SVM = svm.LinearSVC(class_weight='balanced')
model_SVM.fit(x_train, y_train
              #, sample_weight={1:2}
             )
pred = model_SVM.predict(x_valid)
#pred = np.log(pred)
print('train evaluation :')
print('R2 score:{:.2f}'.format(metrics.r2_score(y_valid, pred)))
print('RMSE:{:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_valid, pred))))
from sklearn.ensemble import GradientBoostingClassifier

GBoost = GradientBoostingClassifier(n_estimators=1000, 
                                    learning_rate=0.1,
                                    #max_depth=5, 
                                    #max_features='sqrt',
                                    #min_samples_leaf=15, 
                                    #min_samples_split=10, 
                                    loss='deviance', 
                                    random_state =24)

GBoost.fit(x_train, y_train)
pred = GBoost.predict(x_valid)
#pred = np.log(pred)
print('train evaluation :')
print('R2 score:{:.2f}'.format(metrics.r2_score(y_valid, pred)))
print('RMSE:{:.2f}'.format(np.sqrt(metrics.mean_squared_error(y_valid, pred))))
sub = pd.read_csv('./input/sampleSubmission.csv')
test_submission = pd.DataFrame({'user_id':sub['user_id'], 'class': pred})
test_submission.to_csv('./test_submission.csv', index = False)
