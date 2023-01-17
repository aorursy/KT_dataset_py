#importing libraries

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import PoissonRegressor

from sklearn.pipeline import Pipeline

from sklearn import metrics
#fetching data

df = pd.read_csv(r"/kaggle/input/award-competition/competition_awards_data.csv",  sep = ',', header= 0 )

df.head()
# number of observations: 200

df.shape
# checking NA

# there are no missing values in the dataset

df.isnull().values.any()
# plotting awards agains Math score

fig, ax = plt.subplots(figsize=(20,8))

plt.grid()

ax.set_ylabel("Awards")                                

ax.set_xlabel("Math Score")

ax.scatter( df['Math Score'],df.Awards)

plt.show()
from sklearn.model_selection import train_test_split

train,test=train_test_split(df, train_size = .8,random_state =1)
print(len(train))

print(len(test))
# defining X and y for model training and test

X_train = train['Math Score'].values.reshape(-1, 1)

y_train = train.Awards



X_train.shape,y_train.shape
X_test = test['Math Score'].values.reshape(-1, 1)

y_test = test.Awards

X_test.shape,y_test.shape
# Doing a polynomial regression: Comparing linear, quadratic and cubic fits

# Pipeline helps you associate two models or objects to be built sequentially with each other, 

# in this case, PoissonRegressor() is the only object



pipeline = Pipeline([('model', PoissonRegressor())])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

r2_test = metrics.r2_score(y_test, y_pred)



r2_test
# training performance

y_pred_train = pipeline.predict(X_train)

r2_train = metrics.r2_score(y_train, y_pred_train)

r2_train
# plot predictions and actual values against Math score



fig, ax = plt.subplots(figsize=(20,8))

plt.grid()

ax.set_xlabel("Math Score")                                

ax.set_ylabel("Awards")

# train data in blue

ax.scatter(X_train, y_train,color='blue',label="Original Train Data")

ax.plot(X_train, y_pred_train, '.', color='green',label="Predicted Train Data")

# test data

ax.scatter(X_test, y_test,color='black',label="Original Test Data")

ax.plot(X_test, y_pred, '*', color='red',label="Predicted Test Data")

ax.legend()

plt.show()
eval = pd.DataFrame({'y_pred': [round(y, 0) for y in y_pred], 'y': y_test}).reset_index()

eval.head()
print('     Frequency table')

eval.groupby(['y', 'y_pred']).agg('count').reset_index().pivot(index='y', columns='y_pred', values='index').fillna(0)