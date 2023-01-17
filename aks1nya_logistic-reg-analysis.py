import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn.linear_model
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
M = train['Id'].count() # размер обучающей выборки
train.head()

train.describe()
train.pivot_table(index = 'Id', values = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']).corr()

plt.rcParams["figure.figsize"] = (20, 10)
plt.figure()
plt.subplot(211)
plt.scatter(train['Horizontal_Distance_To_Hydrology'].values, \
         train['Vertical_Distance_To_Hydrology'].values, c = train['Cover_Type'], s = 2)
plt.title = 'train'

plt.subplot(212)
plt.plot(test['Horizontal_Distance_To_Hydrology'].values, \
         test['Vertical_Distance_To_Hydrology'].values, 'go', markersize = 2)
plt.title = 'test'
threshold = int(M * 0.6) # 60% обучающей выборки - новый train, 40% - cross-validation
train_new = train[:threshold]
cv = train[threshold:]
print (M)
print (train_new.shape)
print (cv.shape)
train['Elevation'] = (train['Elevation'] - train['Elevation'].mean())/train['Elevation'].std()
train['Horizontal_Distance_To_Hydrology'] = (train['Horizontal_Distance_To_Hydrology'] - train['Horizontal_Distance_To_Hydrology'].mean())/train['Horizontal_Distance_To_Hydrology'].std()
train['Vertical_Distance_To_Hydrology'] = (train['Vertical_Distance_To_Hydrology'] - train['Vertical_Distance_To_Hydrology'].mean())/train['Vertical_Distance_To_Hydrology'].std()
train['Horizontal_Distance_To_Roadways'] = (train['Horizontal_Distance_To_Roadways'] - train['Horizontal_Distance_To_Roadways'].mean())/train['Horizontal_Distance_To_Roadways'].std()
train['Aspect'] = (train['Aspect'] - train['Aspect'].mean())/train['Aspect'].std()
train['Slope'] = (train['Slope'] - train['Slope'].mean())/train['Slope'].std()
train['Hillshade_9am'] = (train['Hillshade_9am'] - train['Hillshade_9am'].mean())/train['Hillshade_9am'].std()
train['Hillshade_Noon'] = (train['Hillshade_Noon'] - train['Hillshade_Noon'].mean())/train['Hillshade_Noon'].std()
train['Hillshade_3pm'] = (train['Hillshade_3pm'] - train['Hillshade_3pm'].mean())/train['Hillshade_3pm'].std()
'''logistic_reg = sklearn.linear_model.LogisticRegression(random_state = 1, \
                                                  solver = 'lbfgs', \
                                                  tol = 1e-4, \
                                                  max_iter = 500, \
                                                  n_jobs = -1)
                                                    '''
X = train_new.drop('Cover_Type', axis = 1)
y = train_new['Cover_Type'].values
#logistic_reg.fit(X, y)
#pred = logistic_reg.predict(cv.drop('Cover_Type', axis = 1))
#print (logistic_reg.score(X, y))
#print (logistic_reg.score(cv.drop('Cover_Type', axis = 1), cv['Cover_Type'].values))
log_reg = sklearn.linear_model.LogisticRegression(random_state = 1, \
                                                  solver = 'lbfgs', \
                                                  tol = 1e-4, \
                                                  max_iter = 1500, \
                                                  n_jobs = -1)
X = train.drop('Cover_Type', axis = 1)
y = train['Cover_Type'].values
log_reg.fit(X, y)
print (log_reg.score(X, y))
print (log_reg.score(cv.drop('Cover_Type', axis = 1), cv['Cover_Type'].values))
pred = log_reg.predict(test)
pred_df = pd.DataFrame({'Cover_Type' : pred}, test['Id'])
pred_df.head()
pred_df.to_csv('output.csv')
