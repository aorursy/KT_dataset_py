import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'



from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
#df = pd.read_csv('../input/satisfaction/x23.txt', header=None)
# reading every line
df = open('../input/satisfaction/x23.txt', 'r')
lines = df.readlines()

# deleting /n in the end of the lines
for index, line in enumerate(lines):
    lines[index] = line.strip()
lines
lines = lines[-30:]
for i in range(len(lines)):
    lines[i] = [int(x) for x in lines[i].split()]

columns = ['id','A1','A2','A3','A4','A5','A6','B']
df = pd.DataFrame(lines, columns=columns)
df.drop(['id'], axis = 1, inplace=True)

# some scaling our data
df_train, df_test = train_test_split(df, test_size=.3, random_state=17)
#rescale features

scaler = MinMaxScaler()
cols = df_train.columns
df_train[cols] = scaler.fit_transform(df_train[cols])
df_test[cols] = scaler.fit_transform(df_test[cols])

df[cols] = scaler.fit_transform(df[cols])
df
print(type(df_train))
y_train = df_train['B'].astype('double').values.reshape((-1,1)) #making label
X_train = df_train.drop('B', axis=1)

y_test = df_test['B'].astype('double').values.reshape((-1,1)) #making label
X_test = df_test.drop('B', axis=1)
df_train
X_test
y_train
corr_matr = df.corr()
fig, ax = plt.subplots(figsize =(12, 10))
sns.heatmap(corr_matr, annot = True, cmap='viridis',linewidths=.5,ax=ax)
plt.show()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.3) # splitting our test on two parts
linreg = LinearRegression()
linreg.fit(X_train, y_train) 
print(linreg.coef_) # prints weights for every predictor

print('Results with linear regression: {0}'.format(linreg.score(X_test, y_test)))  # prints R^2 score

#R^2 (.score(X, y)) is defined as (1 - u/v); u=((y_true - y_pred) ** 2).sum(), v=((y_true - y.mean()) ** 2).sum()
#bigger score is better
type(X)
# num of max features 
len(X_train.columns)
linreg =LinearRegression()
linreg.fit(X_train, y_train)

rfe=RFE(linreg,n_features_to_select=4)
rfe=rfe.fit(X_train,y_train)
# tuples of (feature name, whether selected, ranking)
# note that the 'rank' is >1 for not selected features
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
# predictions
y_pred =rfe.predict(X_test)

#now check some score
r2 =r2_score(y_test, y_pred)
print(r2)
#try with another value of RFE
lm =LinearRegression()
lm.fit(X_train, y_train)

rfe=RFE(lm, n_features_to_select=4)
rfe=rfe.fit(X_train,y_train)

#predict prices of X_test
y_pred=rfe.predict(X_test)
r2=r2_score(y_test,y_pred)
print(r2)
folds = KFold(n_splits=5,shuffle=True,random_state=100)
scores=cross_val_score(lm,df[df.columns[:-1]],df[df.columns[-1:]].values,scoring='r2',cv=folds)
print(scores)
scores.mean()
# can tune other metrics, like MSE e.g.
folds = KFold(n_splits=5,shuffle=True,random_state=100)
scores=cross_val_score(lm,df[df.columns[:-1]],df[df.columns[-1:]].values,scoring='neg_mean_squared_error',cv=folds)
print(scores)
scores.mean()
# num of features in X_train 
len(X_train.columns)
# step1
folds = KFold(n_splits=5,shuffle=True,random_state=100)
# step2
hyper_param_str = 'n_features_to_select'
hyper_params = [{hyper_param_str:list(range(1,7))}]

# step3.1
linreg =LinearRegression()
linreg.fit(X_train, y_train)
rfe=RFE(linreg)

# step3.2
model_cv=GridSearchCV(estimator=rfe,
                      param_grid=hyper_params,
                      scoring='r2',
                      cv=folds,
                      verbose=1,
                     return_train_score=True)
# fit the model
model_cv.fit(df[df.columns[:-1]],df[df.columns[-1:]].values.ravel())
df[df.columns[len(df.columns)-1]].values
df[df.columns[:-1]]
# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results
best_param = model_cv.best_params_[hyper_param_str]
#plotting cv results
plt.figure(figsize=(16,6))
plt.plot(cv_results['param_n_features_to_select'],cv_results['mean_test_score'])
plt.plot(cv_results['param_n_features_to_select'],cv_results['mean_train_score'])
plt.xlabel('number of features')
plt.ylabel('scores')
plt.title('Optimal Features Number')
plt.legend(['test_score', 'train_score'], loc='upper left')
# now we're choosing best num of features 
num_of_features= best_param
linreg=LinearRegression()
linreg.fit(X_train,y_train)

rfe=RFE(linreg,n_features_to_select=num_of_features)
rfe.fit(X_train,y_train)

y_pred_rfe = rfe.predict(X_test)
y_pred_linreg=linreg.predict(X_test)
print('r2 score is: {0} and {1}'.format(r2_score(y_test, y_pred_rfe),r2_score(y_test, y_pred_linreg)))
print(linreg.coef_)
X_train_arr = X_train.values
X_test_arr = X_test.values

X_train_arr = X_train_arr[:, np.newaxis, 2]
X_test_arr = X_test_arr[:, np.newaxis, 2]

print(X_test_arr.shape)

X_train_arr = X_train_arr[len(y_train):]
#X_test_arr = X_test_arr[:-len(y_test)]

print(X_test_arr.shape)
plt.scatter(X_test_arr, y_test,color='black')
plt.plot(X_test_arr, y_pred_rfe)