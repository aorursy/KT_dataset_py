from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as pl
%matplotlib inline

fifa = pd.read_csv("../input/fifa19cleaned/data.csv", index_col = 0)
df= fifa
df.head(5)
df.shape
df.dropna(how= 'any', inplace= True)
df.Wage = df['Wage'].str.replace('€', '')
df.Value = df['Value'].str.replace('€', '')
df
df.Value = (df.Value.replace(r'[KM]+', '', regex=True).astype(float) *  df.Value.str.extract(r'[\d\.]+([KM]+)', expand=False)
           .fillna(1)
          .replace(['K','M'], [10**3, 10**6]).astype(int))
df.Wage = (df.Wage.replace(r'[KM]+', '', regex=True).astype(float) *  df.Wage.str.extract(r'[\d\.]+([KM]+)', expand=False)
           .fillna(1)
          .replace(['K','M'], [10**3, 10**6]).astype(int))
df
print(df[df.Position == 'GK'])

df.drop(df[df.Position == 'GK'].index, axis = 0, inplace = True)
df.reset_index(drop = True)
df.describe()
df.info()
scaler = MinMaxScaler()
print(scaler.fit(df[['Wage', 'Value']]))
df[['Wage', 'Value']] = scaler.transform(df[['Wage', 'Value']])
df

fig, ax = pl.subplots(figsize=(25,25)) 
sns.heatmap(df.corr(), annot=True, ax=ax, cmap='BrBG').set(
    title = 'Feature Correlation', xlabel = 'Columns', ylabel = 'Columns')
pl.show()

# Counting the frequencies of the categorical attributes.

count_position=df['Position'].value_counts()
count_club=df['Club'].value_counts()
count_nation=df['Nationality'].value_counts()


count_position
count_nation
count_club

count_position.plot.bar(figsize = (20,10),width = 0.7)

count_nation.plot.bar(figsize=(100,50))
count_club.plot.bar(figsize= (100,50))
fig = pl.figure(figsize=(100, 50))
df.groupby('Position')['Wage'].mean().sort_values().plot(kind='bar', color='coral')
pl.title('Wage v/s Positioin', fontsize= 65)
pl.xlabel("Position", fontsize= 65)
pl.ylabel('Wage', fontsize= 65)
pl.show()
fig = pl.figure(figsize=(100, 50))
df.groupby('Nationality')['Wage'].mean().sort_values().plot(kind='bar', color='coral')
pl.title('Wage v/s Nationality', fontsize= 65)
pl.xlabel("Nationality", fontsize= 65)
pl.ylabel('Wage', fontsize= 65)
pl.show()
fig = pl.figure(figsize=(100, 50))
df.groupby('Club')['Wage'].mean().sort_values().plot(kind='bar', color='coral')
pl.title('Wage v/s Club', fontsize= 65)
pl.xlabel("Club", fontsize= 65)
pl.ylabel('Wage' , fontsize= 65)
pl.show()


fig = pl.figure(figsize=(100, 50))
df.groupby('Position')['Value'].mean().sort_values().plot(kind='bar', color='coral')
pl.title('Value v/s Positioin', fontsize= 65)
pl.xlabel("Position", size= 65)
pl.ylabel('Value', size= 65)
pl.show()


fig = pl.figure(figsize=(100, 50))
df.groupby('Nationality')['Value'].mean().sort_values().plot(kind='bar', color='coral')
pl.title('Value v/s Nationality', fontsize= 65)
pl.xlabel("Position", fontsize= 6)
pl.ylabel('Value', fontsize= 65)
pl.show()

fig = pl.figure(figsize=(100, 50))
df.groupby('Club')['Value'].mean().sort_values().plot(kind='bar', color='coral')
pl.title('Value v/s Club', fontsize = 65)
pl.xlabel("Club" ,fontsize = 65)
pl.ylabel('Value', fontsize= 65)

pl.show()
new = df.filter(['Club','Position','Nationality'], axis=1)
new

# Creating a copy of data Frame

fd =df
fd2 =df

N = pd.get_dummies(df['Nationality'])
C = pd.get_dummies(df['Club'])
P = pd.get_dummies(df['Position'])
P.head(5)
# Merging the categorical value to dataframe.


fd =df.merge(P,left_index=True, right_index=True)
fd.head(5)
# Seperating the Target variable{WAGE}  from Dataframe

y = df['Wage']
fd = fd.drop(columns =['Wage', 'Nationality', 'Club', 'Position']) 
x = fd

# Seperating the Target variable{VALUE}  from Dataframe

y2 = df['Value']
fd2 = fd2.drop(columns =['Value', 'Nationality', 'Club', 'Position']) 
x2 = fd2


x.info()
# Splitting to test and Training dataset with regards to Wage.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)

x_train.head(5)
from sklearn.model_selection import train_test_split

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=33)

x2_train.head(5)

def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

w = standRegres(x_train,y_train)
print(w)


# computing the predictions {Wage} using regression coefficients:
xMat=np.mat(x_train)
yMat=np.mat(y_train)
yHat = xMat*w

yHat = yHat.A.ravel()
print(yHat)

# Computing the errors.
errors = abs(yHat - y_train)

#Dot product of the error gives Sum of squared errors.

tot_errors = np.dot(errors,errors)
print("\n Sum of Squared Errors WAGE: ",tot_errors)

# Computing RMSE
rmse_train = np.sqrt(tot_errors/len(yHat))
print("\n RMSE of Training Data WAGE: ", rmse_train)
# Ploting predicted values against the actual values on the training data.

import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure(figsize=(12, 6))
plt.plot(yHat, y_train,'bo', markersize=3)
plt.plot([0,1],[0,1], 'g-')
plt.xlabel('Predicted Wage')
plt.ylabel('Actual Wage')
pl.grid()
plt.show()




# Regression Coefficients on Wage: 

for i in range(len(x_train.columns)):
    print(x_train.columns[i], w[i])
w2 = standRegres(x2_train,y2_train)
print(w2)
# computing the predictions {Wage} using regression coefficients:
xMat=np.mat(x2_train)
yMat=np.mat(y2_train)
yHat = xMat*w2

yHat = yHat.A.ravel()
print(yHat)

# Computing the errors.
errors = abs(yHat - y2_train)

#Dot product of the error gives Sum of squared errors.

tot_errors = np.dot(errors,errors)
print("\n Sum of Squared Errors VALUE: ",tot_errors)

# Computing RMSE
rmse_train = np.sqrt(tot_errors/len(yHat))
print("\n RMSE of Training Data VALUE: ", rmse_train)
# Ploting predicted values against the actual values on the training data.

import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure(figsize=(12, 6))
plt.plot(yHat, y2_train,'bo', markersize=3)
plt.plot([0,1],[0,1], 'g-')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.grid()
plt.show()
# Regression Coefficients on Wage: 

for i in range(len(x2_train.columns)):
    print(x2_train.columns[i], w[i])
wf = w.A.ravel()
wf2 = w2.A.ravel()
# K fold cross validation
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=32, shuffle=True)

# Crossvalidation with Wage:

xval_err = 0
print("Wage:\n")

for train, test in kfold.split(x_train):
    x_fold_train = x_train[x_train.index.isin(train)]
    y_fold_train= y_train[y_train.index.isin(train)]

    w = standRegres(x_fold_train,y_fold_train)

    xMat=np.mat(x_fold_train)
    yMat=np.mat(y_fold_train)
    yHat = xMat*w
    yHat = yHat.A.ravel()
    
    error = abs(yHat - y_fold_train)
    tot_errors = np.dot(error,error)
    
    rmse = np.sqrt(np.dot(error,error)/len(x_fold_train))
    print(rmse)
    xval_err += rmse
xval_err = 0
print("Value:\n")
for train, test in kfold.split(x2_train):
    x2_fold_train = x2_train[x2_train.index.isin(train)]
    y2_fold_train= y2_train[y2_train.index.isin(train)]

    w = standRegres(x2_fold_train,y2_fold_train)

    xMat=np.mat(x2_fold_train)
    yMat=np.mat(y2_fold_train)
    yHat = xMat*w
    yHat = yHat.A.ravel()
    
    error = abs(yHat - y2_fold_train)
    tot_errors = np.dot(error,error)
    
    rmse = np.sqrt(np.dot(error,error)/len(x2_fold_train))
    print(rmse)
    xval_err += rmse
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn import feature_selection
from sklearn.model_selection import cross_val_score, cross_validate


linreg = LinearRegression()

percentile = range(5, 101, 5)
results = []
print("Wage:\n")
for i in percentile:
    feature_select = feature_selection.SelectPercentile(feature_selection.f_regression, percentile=i)
    x_train_feature_select = feature_select.fit_transform(x_train, y_train)
    scores = cross_val_score(linreg, x_train_feature_select, y_train, cv=5)
    print(i , scores.mean())
    results = np.append(results, scores.mean())
    
optimal_percentile_ind = np.where(results == results.max())[0][0]
print('\n')
print("Optimal Percintile:",optimal_percentile_ind)

optimal_percentile_index = np.where(results == results.max())[0][0]
print("Optimal percentile of features:",percentile[optimal_percentile_index])
optimal_num_features = int(percentile[optimal_percentile_index]*len(fd.columns)/100)
print("Number of optimal features:",optimal_num_features)

# Graph of percentile of features VS. cross-validation scores

import pylab as pl
pl.rcParams["figure.figsize"] = (7,7)
pl.figure()

pl.xlabel("Percentage of features selected")
pl.ylabel("Cross validation accuracy")
pl.plot(percentile,results)
f_scores = []

for i in range(len(fd.columns.values)):
    if feature_select.get_support()[i]:
        t = (fd.columns.values[i],feature_select.scores_[i])
        f_scores.append(t)
        
f_scores.sort(key=lambda x:x[1], reverse=True)

for i in range(len(f_scores)):
    print(i+1 ,f_scores[i][0] ,f_scores[i][1])
fs = feature_selection.SelectPercentile(feature_selection.f_regression, percentile=optimal_percentile_ind)
x_train_fs = fs.fit_transform(x_train, y_train)
score = cross_val_score(linreg, x_train_fs, y_train, cv=5, scoring='neg_mean_absolute_error')
score = [abs(i) for i in scores]
linreg.fit(x_train, y_train)


print("Score:",score)
print("\n")
print("Cross-validation mean Wage:",np.mean(score))
# Computing errors on all test instances Wage
p = linreg.predict(x_test)

# Printing MAE RMSE and r2 score.
print('MAE Wage:', metrics.mean_absolute_error(y_test,p ))
print('RMSE Wage:', np.sqrt(metrics.mean_squared_error(y_test, p)))
print('R2_Score Wage: ', metrics.r2_score(y_test, p))
# Ploting predicted values of Wage against the actual values on the training data.

fig = plt.figure(figsize=(12, 6))
pl.plot(p, y_test,'bo', markersize=5)
pl.plot([0,1],[0,1], 'g-')
pl.xlabel('Predicted')
pl.ylabel('Actual')
pl.grid()
pl.show()
linreg = LinearRegression()

percentile2 = range(5, 101, 5)
results2 = []
print("Value:\n")
for i in percentile2:
    feature_select = feature_selection.SelectPercentile(feature_selection.f_regression, percentile=i)
    x2_train_feature_select = feature_select.fit_transform(x2_train, y2_train)
    scores = cross_val_score(linreg, x2_train_feature_select, y2_train, cv=5)
    print(i , scores.mean())
    results2 = np.append(results2, scores.mean())
    
optimal_percentile_ind = np.where(results == results.max())[0][0]
print('\n')
print("Optimal Percintile:",optimal_percentile_ind)

optimal_percentile_index = np.where(results == results.max())[0][0]
print("Optimal percentile of features:",percentile2[optimal_percentile_index])
optimal_num_features = int(percentile2[optimal_percentile_index]*len(fd.columns)/100)
print("Number of optimal features:",optimal_num_features)
# Grapg of percentile of features VS. cross-validation scores

import pylab as pl
pl.rcParams["figure.figsize"] = (7,7)
pl.figure()

pl.xlabel("Percentage of features selected")
pl.ylabel("Cross validation accuracy")
pl.plot(percentile2,results2)
fs = feature_selection.SelectPercentile(feature_selection.f_regression, percentile=optimal_percentile_ind)
x2_train_fs = fs.fit_transform(x2_train, y2_train)
score = cross_val_score(linreg, x2_train_fs, y2_train, cv=5, scoring='neg_mean_absolute_error')
score = [abs(i) for i in scores]
linreg.fit(x2_train, y2_train)


print("Score:",score)
print("\n")
print("Cross-validation mean:",np.mean(score))
# Computing errors on all test instances Value
p = linreg.predict(x2_test)

# Printing MAE RMSE and r2 score.
print('MAE Value:', metrics.mean_absolute_error(y2_test,p ))
print('RMSE Value:', np.sqrt(metrics.mean_squared_error(y2_test, p)))
print('R2_Score Value: ', metrics.r2_score(y2_test, p))
# Ploting predicted values of Value against the actual values on the training data. 

fig = plt.figure(figsize=(12, 6))
pl.plot(p, y2_test,'bo', markersize=5)
pl.plot([0,1],[0,1], 'g-')
pl.xlabel('Predicted')
pl.ylabel('Actual')
pl.grid()
pl.show()
def calculate_params(m, n, clf, param_values, param_name, K):
    
    # Convert input to Numpy arrays
    X = np.array(m)
    y = np.array(n)

    # initialize training and testing score arrays with zeros
    train_scores = np.zeros(len(param_values))
    test_scores = np.zeros(len(param_values))
    
    # iterate over the different parameter values
    for i, param_value in enumerate(param_values):
        
        # set classifier parameters
        clf.set_params(**{param_name:param_value})
        
        # initialize the K scores obtained for each fold
        k_train_scores = np.zeros(K)
        k_test_scores = np.zeros(K)
        
        # create KFold cross validation
        cv = KFold(n_splits=K, shuffle=True, random_state=0)
        
        # iterate over the K folds
        j = 0
        for train, test in cv.split(X):
            # fit the classifier in the corresponding fold
            # and obtain the corresponding accuracy scores on train and test sets
            clf.fit(X[train], y[train])
            k_train_scores[j] = clf.score(X[train], y[train])
            k_test_scores[j] = clf.score(X[test], y[test])
            j += 1
            
        # store the mean of the K fold scores
        train_scores[i] = np.mean(k_train_scores)
        test_scores[i] = np.mean(k_test_scores)
        print(param_name,param_value,train_scores[i])

    # plot the training and testing scores in a log scale
    plt.plot(param_values, train_scores, label='Train', alpha=0.4, lw=2, c='b')
    plt.plot(param_values, test_scores, label='X-Val', alpha=0.4, lw=2, c='g')
    plt.legend(loc=7)
    plt.xlabel(param_name + " values")
    plt.ylabel("Mean cross validation accuracy")

    # return the training and testing scores on each parameter value
    return train_scores, test_scores



# Ridge

print('Wage:\n')
alpha = np.linspace(.01,20,50)
ridge = Ridge(alpha=alpha)

train_scores, test_scores = calculate_params(x_train, y_train, ridge, alpha, 'alpha', 10)

print(np.argmax(train_scores))
print(np.argmax(test_scores))


# Creating linear regression with best ridge coefficient
ridge = Ridge(alpha=0.01)

# Training model using the train  data-set
ridge.fit(x_train,y_train)


# Computing RMSE 
p = ridge.predict(x_test)
error = p-y_test
tot_error = np.dot(error,error)
RMSE_test = np.sqrt(tot_error/len(p))

print('RMSE of test Dataset Wage:',RMSE_test)
# Ridge
print('Value:\n')
alpha2 = np.linspace(.01,20,50)
ridge2 = Ridge(alpha=alpha)

train_scores2, test_scores2 = calculate_params(x2_train, y2_train, ridge2, alpha2, 'alpha', 10)
print(np.argmax(train_scores2))
print(np.argmax(test_scores2))


# Creating linear regression with best ridge coefficient
ridge = Ridge(alpha=0.01)

# Training model using the train  data-set
ridge.fit(x2_train,y2_train)


# Computing RMSE 
p = ridge.predict(x2_test)
error = p-y2_test
tot_error = np.dot(error,error)
RMSE_test = np.sqrt(tot_error/len(p))

print('RMSE of test Dataset Value:',RMSE_test)
# Lasso Wage
alpha = np.linspace(.0001,.04,50)
lasso = Lasso(alpha=alpha)

train_scores, test_scores = calculate_params(x_train, y_train, lasso, alpha, 'alpha', 10)
print(np.argmax(train_scores))
print(np.argmax(test_scores))


# Creating linear regression with best ridge coefficient
lasso = Lasso(alpha=test_scores[0])

# Training model using the train  data-set
lasso.fit(x_train,y_train)


# Computing RMSE 
p = lasso.predict(x_test)
error = p-y_test
tot_error = np.dot(error,error)
RMSE_test = np.sqrt(tot_error/len(p))

print('RMSE of test Dataset Wage: ',RMSE_test)




# Lasso on Value
alpha2 = np.linspace(.0001,.04,50)
lasso2 = Lasso(alpha=alpha)

train_scores2, test_scores2 = calculate_params(x2_train, y2_train, lasso2, alpha2, 'alpha', 10)
print(np.argmax(train_scores2))
print(np.argmax(test_scores2))


# Creating linear regression with best ridge coefficient
lasso = Lasso(alpha=test_scores[0])

# Training model using the train  data-set
lasso.fit(x2_train,y2_train)


# Computing RMSE 
p = lasso.predict(x2_test)
error = p-y2_test
tot_error = np.dot(error,error)
RMSE_test = np.sqrt(tot_error/len(p))

print('RMSE of test Dataset Value:',RMSE_test)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


scaler = StandardScaler()
scaler.fit(x_train)
x_train_sgd = scaler.transform(x_train)

sgdreg = SGDRegressor()

parameters = {
    'penalty': ['l2','l1'],
    'alpha': np.linspace(.0001, 1, 100),
}

grid_search = GridSearchCV(sgdreg, parameters, verbose=1, cv=5)
grid_search.fit(x_train_sgd, y_train)

params = grid_search.best_params_
print(params, grid_search.best_score_)


sgdreg = SGDRegressor(penalty=params["penalty"], alpha=params["alpha"])
sgdreg.fit(x_train_sgd,y_train)

scaler.fit(x_test)
x_test_sgd = scaler.transform(x_test)

# Computing RMSE 
p = sgdreg.predict(x_test_sgd)
error = p-y_test
tot_error = np.dot(error,error)
RMSE_test = np.sqrt(tot_error/len(p))
print("\nRMSE mean test of SGD:",np.mean(RMSE_test))

fig = plt.figure(figsize=(12, 6))
pl.plot(p, y_test,'bo', markersize=5)
pl.plot([0,1],[0,1], 'g-')
pl.xlabel('Predicted')
pl.ylabel('Actual')
pl.grid()
pl.show()
sgdreg = SGDRegressor(penalty='elasticnet')
l1_ratio = np.linspace(0, 1, 100)

train_scores, test_scores = calculate_params(x_train_sgd, y_train, sgdreg, l1_ratio, 'l1_ratio', 10)
l1 = l1_ratio[np.argmax(test_scores)]
print(l1)


sgdreg = SGDRegressor(penalty="elasticnet", l1_ratio=l1)
sgdreg.fit(x_train_sgd,y_train)

scaler.fit(x_test)
x_test_sgd = scaler.transform(x_test)

# Computing RMSE
p = sgdreg.predict(x_test_sgd)
error = p-y_test
tot_error = np.dot(error,error)
RMSE_test = np.sqrt(tot_error/len(p))
print("\nRMSE mean test on SGD:",np.mean(RMSE_test))

fig = plt.figure(figsize=(12, 6))
pl.plot(p, y_test,'bo', markersize=5)
pl.plot([0,1],[0,1], 'g-')
pl.xlabel('Predicted')
pl.ylabel('Actual')
pl.grid()
pl.show()
scaler2 = StandardScaler()
scaler.fit(x2_train)
x2_train_sgd = scaler.transform(x2_train)

sgdreg = SGDRegressor()

parameters = {
    'penalty': ['l2','l1'],
    'alpha': np.linspace(.0001, 1, 100),
}

grid_search = GridSearchCV(sgdreg, parameters, verbose=1, cv=5)
grid_search.fit(x2_train_sgd, y2_train)

params = grid_search.best_params_
print(params, grid_search.best_score_)


sgdreg = SGDRegressor(penalty=params["penalty"], alpha=params["alpha"])
sgdreg.fit(x2_train_sgd,y2_train)

scaler.fit(x2_test)
x2_test_sgd = scaler.transform(x2_test)

# Computing RMSE 
p = sgdreg.predict(x2_test_sgd)
error = p-y2_test
tot_error = np.dot(error,error)
RMSE_test = np.sqrt(tot_error/len(p))
print("\nRMSE mean test of SGD:",np.mean(RMSE_test))


fig = plt.figure(figsize=(12, 6))
pl.plot(p, y2_test,'bo', markersize=5)
pl.plot([0,1],[0,1], 'g-')
pl.xlabel('Predicted')
pl.ylabel('Actual')
pl.grid()
pl.show()
sgdreg2 = SGDRegressor(penalty='elasticnet')
l1_ratio2 = np.linspace(0, 1, 100)

train_scores2, test_scores2 = calculate_params(x2_train_sgd, y2_train, sgdreg2, l1_ratio2, 'l1_ratio', 10)
l1 = l1_ratio2[np.argmax(test_scores2)]
print(l1)


sgdreg = SGDRegressor(penalty="elasticnet", l1_ratio=l1)
sgdreg.fit(x2_train_sgd,y2_train)

scaler.fit(x2_test)
x2_test_sgd = scaler.transform(x2_test)

# Computing RMSE
p = sgdreg.predict(x2_test_sgd)
error = p-y2_test
tot_error = np.dot(error,error)
RMSE_test = np.sqrt(tot_error/len(p))
print("\nRMSE mean test on SGD:",np.mean(RMSE_test))

fig = plt.figure(figsize=(12, 6))
pl.plot(p, y2_test,'bo', markersize=5)
pl.plot([0,1],[0,1], 'g-')
pl.xlabel('Predicted')
pl.ylabel('Actual')
pl.grid()
pl.show()
svm = SVR(kernel='rbf')
svm.fit(x_train,y_train)
svm_pred = svm.predict(x_test)

print('SVM Performance on Wage:')

print('\nall features, No scaling:')
print('MAE:', metrics.mean_absolute_error(y_test, svm_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svm_pred)))
print('R2_Score: ', metrics.r2_score(y_test, svm_pred))
fig = plt.figure(figsize=(12, 6))
pl.plot( y_test,svm_pred,'bo', markersize=5)
pl.plot([0,1],[0,1], 'g-')
pl.xlabel('Predicted')
pl.ylabel('Actual')
pl.grid()
pl.show()
param_grid = {'C': [1, 10, 100], 'gamma': [0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVR(),param_grid,refit=True,verbose=3)

grid.fit(x_train,y_train)

grid.best_params_

grid.best_estimator_

grid_predictions = grid.predict(x_test)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, grid_predictions)))
print('R2_Score: ', metrics.r2_score(y_test, grid_predictions))

fig = plt.figure(figsize=(12, 6))
pl.plot( y_test,grid_predictions,'bo', markersize=5)
pl.plot([0,1],[0,1], 'g-')
pl.xlabel('Predicted')
pl.ylabel('Actual')
pl.grid()
pl.show()
svm2 = SVR(kernel='rbf')
svm2.fit(x2_train,y2_train)
svm2_pred = svm2.predict(x2_test)

print('SVM Performance on Value:')

print('\nall features, No scaling:')
print('MAE Value:', metrics.mean_absolute_error(y_test, svm_pred))
print('RMSE value:', np.sqrt(metrics.mean_squared_error(y_test, svm_pred)))
print('R2_Score value : ', metrics.r2_score(y_test, svm_pred))
fig = plt.figure(figsize=(12, 6))
pl.plot( y2_test,svm2_pred,'bo', markersize=5)
pl.plot([0,1],[0,1], 'g-')
pl.xlabel('Predicted')
pl.ylabel('Actual')
pl.grid()
pl.show()
param_grid2 = {'C': [1, 10, 100], 'gamma': [0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVR(),param_grid2,refit=True,verbose=3)

grid.fit(x2_train,y2_train)

grid.best_params_

grid.best_estimator_

grid_predictions2 = grid.predict(x2_test)

print('RMSE Value:', np.sqrt(metrics.mean_squared_error(y2_test, grid_predictions2)))
print('R2_Score value: ', metrics.r2_score(y2_test, grid_predictions2))

fig = plt.figure(figsize=(12, 6))
pl.plot( y2_test,grid_predictions2,'bo', markersize=5)
pl.plot([0,1],[0,1], 'g-')
pl.xlabel('Predicted')
pl.ylabel('Actual')
pl.grid()
pl.show()
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

rf = RandomForestRegressor(random_state=101, n_estimators=200)
rf.fit(x_train, y_train)

rf_pred = rf.predict(x_test)

print('Random Forest Performance:')

print('\nall features, No scaling:')
print('MAE:', metrics.mean_absolute_error(y_test, rf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf_pred)))
print('R2_Score: ', metrics.r2_score(y_test, rf_pred))


fig = plt.figure(figsize=(12, 6))
pl.plot( y_test,rf_pred,'bo', markersize=5)
pl.plot([0,1],[0,1], 'g-')
pl.xlabel('True Wage')
pl.ylabel('Random Forest prediction Performance')
pl.grid()
pl.show()
rf2 = RandomForestRegressor(random_state=101, n_estimators=200)
rf2.fit(x2_train, y2_train)

rf2_pred = rf2.predict(x2_test)

print('Random Forest Performance:')

print('\nall features, No scaling:')
print('MAE:', metrics.mean_absolute_error(y2_test, rf2_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y2_test, rf2_pred)))
print('R2_Score: ', metrics.r2_score(y2_test, rf2_pred))


fig = plt.figure(figsize=(12, 6))
pl.plot( y2_test,rf2_pred,'bo', markersize=5)
pl.plot([0,1],[0,1], 'g-')
pl.xlabel('True Wage')
pl.ylabel('Random Forest prediction Performance')
pl.grid()
pl.show()