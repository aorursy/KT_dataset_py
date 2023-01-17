import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import os



seed=37
train = pd.read_csv('../input/train.csv',parse_dates=['datetime'])

print(train.shape)

train.head()
test = pd.read_csv('../input/test.csv', parse_dates=['datetime'])

print(test.shape)

train.head()
train['datetime-year'] = train['datetime'].dt.year

train['datetime-month'] = train['datetime'].dt.month

train['datetime-day'] = train['datetime'].dt.day

train['datetime-hour'] = train['datetime'].dt.hour

train['datetime-minute'] = train['datetime'].dt.minute

train['datetime-second'] = train['datetime'].dt.second



print(train.shape)

train.head()
fig, ((axis1, axis2, axis3), (axis4, axis5, axis6)) = plt.subplots(nrows=2, ncols=3, figsize= (18,8))



print('List of Years =' + str(train['datetime-year'].unique())) # unique()로 범주형 data에서 유일한 값만 출력

print('List of Month =' + str(train['datetime-month'].unique()))

print('List of Day =' + str(train['datetime-day'].unique()))

print('List of hours =' + str(train['datetime-hour'].unique()))

print('List of minutes =' + str(train['datetime-minute'].unique()))

print('List of seconds =' + str(train['datetime-second'].unique()))



sns.countplot(data=train, x='datetime-year', ax=axis1)

sns.countplot(data=train, x='datetime-month', ax=axis2)

sns.countplot(data=train, x='datetime-day', ax=axis3)

sns.countplot(data=train, x='datetime-hour', ax=axis4)

sns.countplot(data=train, x='datetime-minute', ax=axis5)

sns.countplot(data=train, x='datetime-second', ax=axis6)

# countplot() 는 x축을 기준으로 dataframe에서 몇개의 row(value)를 가지고 있는지 나타내준다.
fig, (axis1,axis2,axis3) = plt.subplots(1,3 , figsize=(18,4))



sns.barplot(data=train, x='datetime-year', y='count' ,ax= axis1)

sns.barplot(data=train, x='datetime-month', y='count', ax= axis2)

sns.barplot(data=train, x='datetime-hour', y='count',ax= axis3)



# barplot은 기본적인 bar 그래프이다. x값(범주형)에 대해 y값(연속형) 데이터의 mean을 출력해준다.
fig, (axis1, axis2) = plt.subplots(1,2 ,figsize=(18,4))



sns.barplot(data=train, x='datetime-year', y='count', ax=axis1)

sns.barplot(data=train, x='datetime-month', y='count', ax=axis2)



train['datetime-year_month'] = train['datetime'].apply(lambda dt: str(dt.year) + "-" + str(dt.month))

# apply()는 train['datetime']의 각 row(value) 하나씩에 함수를 적용시켜준다.

# datetime field 전체에 str()를 적용해주면 이상한 모양으로 문자열을 변형 시키기 때문에 apply로 각 value마다 str()을 적용시킨다,



fig, axis3 = plt.subplots(figsize=(18,4))



sns.barplot(data=train, x='datetime-year_month', y='count', ax=axis3)
def bin_hour(hour):

    if hour<=7:

        return "others"

    elif hour <= 12:

        return "morning"

    elif hour <= 17:

        return "afternoon"

    elif hour <= 22:

        return "night"

    else:

        return "others"



train['datetime-bin_hour'] = train['datetime-hour'].apply(bin_hour)



print(train.shape)

train.head()
fig, (axis1,axis2) = plt.subplots(2,1, figsize=(24,12))



sns.pointplot(data=train, x='datetime-month', y='count', hue='datetime-bin_hour', ax=axis1)

sns.pointplot(data=train, x='datetime-year_month', y='count', hue='datetime-bin_hour', ax=axis2)
days = {0: 'Monday', 1: 'Tuesday', 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}



train['datetime-dayofweek'] = train['datetime'].dt.dayofweek

train['datetime-dayofweek'] = train['datetime-dayofweek'].apply(lambda day: days[day])

# 숫자형 data인 dt.dayofweek를 문자열로 변경시키는 방법 

# Dictionary 자료형을 이용하기!!!!!!!!!



figures, (ax1, ax2, ax3) = plt.subplots(nrows=3)

figures.set_size_inches(18, 12)



sns.pointplot(data=train, x="datetime-hour", y="count", ax=ax1)

sns.pointplot(data=train, x="datetime-hour", y="count", hue="workingday", ax=ax2)

sns.pointplot(data=train, x="datetime-hour", y="count", hue="datetime-dayofweek", ax=ax3)
fig, (axis1, axis2, axis3) = plt.subplots(1,3, figsize=(18,4))



train['log_count'] = np.log(train['count'])

train['log_count_plus'] = np.log(train['count']+1)



sns.distplot(train['count'], ax=axis1)

sns.distplot(train['log_count'], ax=axis2)

sns.distplot(train['log_count_plus'], ax=axis3)
train = pd.read_csv('../input/train.csv', parse_dates=['datetime'])

test = pd.read_csv('../input/test.csv', parse_dates=['datetime'])
combi = pd.concat([train, test])



print(combi.shape)

combi.head()
combi['datetime-year'] = combi['datetime'].dt.year

combi['datetime-month'] = combi['datetime'].dt.month

combi['datetime-day'] = combi['datetime'].dt.day

combi['datetime-hour'] = combi['datetime'].dt.hour

combi['datetime-minute'] = combi['datetime'].dt.minute

combi['datetime-second'] = combi['datetime'].dt.second



print(combi.shape)

combi[['datetime','datetime-year','datetime-month','datetime-day','datetime-hour','datetime-minute','datetime-second']].head()
combi['datetime-year_month'] = combi['datetime'].apply(lambda dt: str(dt.year) + '-' + str(dt.month))



year_month = pd.get_dummies(combi['datetime-year_month'], prefix="datetime-year_month").astype(np.bool)

print(year_month.shape)
combi2 = combi.copy()
combi = pd.concat([combi2, year_month], axis=1)

print(combi.shape)

combi.head()
dayofweek = combi['datetime'].dt.dayofweek

dayofweek = pd.get_dummies(dayofweek, prefix='datetime-dayofweek').astype(np.bool)
combi2= combi.copy()
combi = pd.concat([combi2, dayofweek], axis=1)

print(combi.shape)

combi.head()
def bin_hour(hour):

    if hour <= 7:

        return 0

    elif hour <= 15:

        return 1

    elif hour <= 21:

        return 2

    else:

        return 0

    

    

combi['datetime-bin_hour'] = combi['datetime-hour'].apply(bin_hour)

bin_hour_dummies = pd.get_dummies(combi['datetime-bin_hour'], prefix='datetime-hour_bin').astype(np.bool)

print(bin_hour_dummies.shape)
combi2 = combi.copy()
combi = pd.concat([combi2, bin_hour_dummies],axis=1)

print(combi.shape)

combi.head()
train = combi[combi['count'].notnull()]



print(train.shape)

train.head()
test = combi[combi['count'].isnull()]



print(test.shape)

test.head()
bin_hour_dummies.columns
feature_names = train.columns

feature_names = list(feature_names)



remove_content =['casual','registered','count','datetime-year', 'datetime-month','datetime',

                 'datetime-day', 'datetime-minute', 'datetime-second', 'datetime-year_month',

                'datetime-bin_hour']

for x in remove_content:

    feature_names.remove(x)

    

label_name= 'count'
X_train = train[feature_names]



print(X_train.shape)

X_train.head()
y_train = np.log(train[label_name]+1)



print(y_train.shape)

y_train.head()
X_test = test[feature_names]



print(X_test.shape)

X_test.head()
from sklearn.metrics import make_scorer



def rmsle(predict, actual):

    predict = np.array(predict)

    actual = np.array(actual)

    

    log_predict = predict + 1

    log_actual = actual + 1

    

    difference = log_predict - log_actual

    difference = np.square(difference) # 제곱함수

    difference = np.mean(difference) # 전체합 / N = mean

    

    score = np.sqrt(difference)

    

    return score



rmsle_score = make_scorer(rmsle)

rmsle_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.cross_validation import cross_val_score



# make model

model = RandomForestRegressor(random_state=seed)



# score

score = cross_val_score(model, X_train, y_train,scoring=rmsle_score, cv=20).mean()



print('score is {score: .5f}' .format(score=score))
from sklearn.ensemble import RandomForestRegressor

from sklearn.cross_validation import cross_val_score



n_estimators = 300

max_depth_list = [1,3, 10, 30, 50, 100]

max_features_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]



grid_parameters_list = []



for max_depth in max_depth_list:

    for max_features in max_features_list:

        model = RandomForestRegressor(n_estimators=n_estimators,

                                     max_depth=max_depth,

                                     max_features=max_features,

                                     n_jobs=-1, # 자동으로 가지치기를 하지않는다(?)

                                     random_state = seed)

        

        score = cross_val_score(model, X_train, y_train, scoring=rmsle_score, cv=20).mean()

        

        print('n_estimators = {0}, max_depth = {1:2}, max_features ={2}, Score(RMSLE) ={3: .6f}' \

             .format(n_estimators,max_depth,max_features, score))

        

        parameter = {

            'n_estimators' : n_estimators,

            'max_depth' : max_depth,

            'max_features' : max_features,

            'score' : score

        }

        

        grid_parameters_list.append(parameter)

        

grid_parameters_list = pd.DataFrame.from_dict(grid_parameters_list)

# list를 dataFrame형태로 변경한다. 이때 기준은 dict로 사용 왜냐하면 list의 각 요소는 dict형태이기 때문에

grid_parameters_list.sort_values('score', ascending=True, inplace=True)

# dataFrame의 정렬에서는 sort_values을 사용하여 'score' field를 기준으로 정렬



grid_parameters_list.head(10)
from sklearn.ensemble import RandomForestRegressor

from sklearn.cross_validation import cross_val_score



num_epoch = 100

coarse_parameters_list = []



n_estimators = 300



for epoch in range(num_epoch):

    np.random.seed(epoch)

    max_depth = int(np.random.uniform(1, 100))

    

    np.random.seed(epoch)

    max_features = np.random.uniform(0.1, 1.0)



    model = RandomForestRegressor(n_estimators=n_estimators,

                                  max_depth=max_depth,

                                  max_features=max_features,

                                  random_state=seed,

                                  n_jobs=-1)



    score = cross_val_score(model, X_train, y_train, scoring=rmsle_score, cv=20).mean()



    print("epoch = {0:2}, n_estimators = {1}, max_depth = {2:2}, max_features = {3:.6f}, Score(RMSLE) = {4:.5f}"\

          .format(epoch, n_estimators, max_depth, max_features, score))



    parameters = {

        'n_estimators': n_estimators,

        'max_depth': max_depth,

        'max_features': max_features,

        'score': score,

    }



    coarse_parameters_list.append(parameters)

    

coarse_parameters_list = pd.DataFrame.from_dict(coarse_parameters_list)

coarse_parameters_list.sort_values('score', ascending=True, inplace=True)



coarse_parameters_list.head(10)
# Coarse search 결과, 다음의 범위 안에 optimal parameter가 있다는 사실을 찾을 수 있다.

minimum_max_depth = 30

maximum_max_depth = 60



minimum_max_features = 0.4

maximum_max_features = 0.6
from sklearn.ensemble import RandomForestRegressor

from sklearn.cross_validation import cross_val_score



num_epoch = 100

finer_parameters_list = []



n_estimators = 300



for epoch in range(num_epoch):

    np.random.seed(epoch)

    max_depth = int(np.random.uniform(minimum_max_depth, maximum_max_depth))



    np.random.seed(epoch)

    max_features = np.random.uniform(minimum_max_features, maximum_max_features)

    

    model = RandomForestRegressor(n_estimators=n_estimators,

                                  max_depth=max_depth,

                                  max_features=max_features,

                                  random_state=seed,

                                  n_jobs=-1)



    score = cross_val_score(model, X_train, y_train, scoring=rmsle_score, cv=20).mean()



    print("epoch = {0:2}, n_estimators = {1}, max_depth = {2:2}, max_features = {3:.6f}, Score(RMSLE) = {4:.5f}"\

          .format(epoch, n_estimators, max_depth, max_features, score))



    parameters = {

        'n_estimators': n_estimators,

        'max_depth': max_depth,

        'max_features': max_features,

        'score': score,

    }



    finer_parameters_list.append(parameters)



finer_parameters_list = pd.DataFrame.from_dict(finer_parameters_list)

finer_parameters_list.sort_values('score', ascending=True, inplace=True)



finer_parameters_list.head(10)
from sklearn.ensemble import RandomForestRegressor



#optimal_hyperparameters = finer_parameters_list.iloc[0]



n_estimators = 300

#max_depth = optimal_hyperparameters["max_depth"]

max_depth = 46

# max_features = optimal_hyperparameters["max_features"]

max_features = 0.509763



model = RandomForestRegressor(n_estimators=n_estimators,

                              max_depth=max_depth,

                              max_features=max_features,

                              random_state=seed,

                              n_jobs=-1)



model
from sklearn.cross_validation import cross_val_score



%time score = cross_val_score(model, X_train, y_train, scoring=rmsle_score, cv=20).mean()



print("Score = {score:.5f}".format(score=score))
model.fit(X_train, y_train)



predictions = model.predict(X_test)



## 가우시안 분포에 맞게 수정한 label(=count)를 원래대로 돌려논다.

predictions = np.exp(predictions)-1



print(predictions.shape)

predictions[:3]