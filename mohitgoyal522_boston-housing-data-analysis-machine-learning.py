#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split as tts

from sklearn.linear_model import LinearRegression as lr

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score as cvs
#importing dataset

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

dataset = pd.read_csv('/kaggle/input/boston-house-prices/housing.csv', delimiter=r'\s+', names=column_names)
#Top 5 rows of dataset

dataset.head()
#Shape of dataset (rows, columns)

dataset.shape
#describing the dataste to see distribution of data

dataset.describe()
#removing variables 'ZN' and 'CHAS' form data

dataset = dataset.drop(['ZN', 'CHAS'], axis=1)
dataset.isnull().sum()
#Plotting boxplots to see if there are any outliers in our data (considering data betwen 25th and 75th percentile as non outlier)

fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15, 5))

ax = ax.flatten()

index = 0

for i in dataset.columns:

  sns.boxplot(y=i, data=dataset, ax=ax[index])

  index +=1

plt.tight_layout(pad=0.4)

plt.show()
#checking percentage/ amount of outliers

for i in dataset.columns:

  dataset.sort_values(by=i, ascending=True, na_position='last')

  q1, q3 = np.nanpercentile(dataset[i], [25,75])

  iqr = q3-q1

  lower_bound = q1-(1.5*iqr)

  upper_bound = q3+(1.5*iqr)

  outlier_data = dataset[i][(dataset[i] < lower_bound) | (dataset[i] > upper_bound)] #creating a series of outlier data

  perc = (outlier_data.count()/dataset[i].count())*100

  print('Outliers in %s is %.2f%% with count %.f' %(i, perc, outlier_data.count()))

  #----------------------code below is for comming sections----------------------

  if i == 'B':

    outlierDataB_index = outlier_data.index

    outlierDataB_LB = dataset[i][(dataset[i] < lower_bound)]

    outlierDataB_UB = dataset[i][(dataset[i] > upper_bound)]

  elif i == 'CRIM':

    outlierDataCRIM_index = outlier_data.index

    outlierDataCRIM_LB = dataset[i][(dataset[i] < lower_bound)]

    outlierDataCRIM_UB = dataset[i][(dataset[i] > upper_bound)]

  elif i == 'MEDV':

    lowerBoundMEDV = lower_bound

    upperBoundMEDV = upper_bound
dataset2 = dataset.copy() # I copied the data in another variable just for an ease of coding, but this is not required
#removing extreme outliers form B and CRIM (removing those observations)

removed=[]

outlierDataB_LB.sort_values(ascending=True, inplace=True)

outlierDataB_UB.sort_values(ascending=False, inplace=True)

counter=1

for i in outlierDataB_LB.index:

  if counter<=19:

    dataset2.drop(index=i, inplace=True)

    counter+=1

    removed.append(i)

for i in outlierDataB_UB.index:

  if counter<=38:

    dataset2.drop(index=i, inplace=True)

    counter+=1

    removed.append(i)

for i in outlierDataB_LB.index:

  if counter<=38 and i not in removed:

    dataset2.drop(index=i, inplace=True)

    counter+=1

    removed.append(i)





outlierDataCRIM_LB.sort_values(ascending=True, inplace=True)

outlierDataCRIM_UB.sort_values(ascending=False, inplace=True)

counter=1

for i in outlierDataCRIM_LB.index:

  if counter<=16 and i not in removed:

    dataset2.drop(index=i, inplace=True)

    counter+=1

    removed.append(i)

for i in outlierDataCRIM_UB.index:

  if counter<=33 and i not in removed:

    dataset2.drop(index=i, inplace=True)

    counter+=1

    removed.append(i)

for i in outlierDataCRIM_LB.index:

  if counter<=33 and i not in removed:

    dataset2.drop(index=i, inplace=True)

    counter+=1

    removed.append(i)



dataset2.shape
dataset3 = dataset2.copy() # I copied the data in another variable just for an ease of coding, but this is not required
#replacing remaning outliers by mean

for i in dataset.columns:

  dataset.sort_values(by=i, ascending=True, na_position='last')

  q1, q3 = np.nanpercentile(dataset[i], [25,75])

  iqr = q3-q1

  lower_bound = q1-(1.5*iqr)

  upper_bound = q3+(1.5*iqr)

  mean = dataset3[i].mean()

  if i != 'MEDV':

    dataset3.loc[dataset3[i] < lower_bound, [i]] = mean

    dataset3.loc[dataset3[i] > upper_bound, [i]] = mean

  else:

    dataset3.loc[dataset3[i] < lower_bound, [i]] = mean

    dataset3.loc[dataset3[i] > upper_bound, [i]] = 50
dataset3.describe()
#independent variable(X) and dependent variable(Y)

X = dataset3.iloc[:, :-1]

Y = dataset3.iloc[:, 11]
#Feature selection using P-Value/ Backward elimination

def BackwardElimination(sl, w):

    for i in range(0, len(w.columns)):

        regressor_OLS = sm.OLS(endog=Y, exog=w).fit()

        max_pvalue = max(regressor_OLS.pvalues)

        pvalues = regressor_OLS.pvalues

        if max_pvalue > SL:

            index_max_pvalue = pvalues[pvalues==max_pvalue].index

            w = w.drop(index_max_pvalue, axis = 1) #delete the valriable for that p value

    return w,pvalues,index_max_pvalue



SL = 0.05

ones = np.ones((435,1))  #adding a columns of ones to X as it is required by statsmodels library

W = X

W.insert(0, 'Constant', ones, True)

W_optimal = W.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]]



W_optimal,pvalues,index_max_pvalue = BackwardElimination(SL, W_optimal)

X = W_optimal.drop('Constant', axis=1)
#remaning variabls after backward elimination

X.columns
#Ploting heatmap using pearson correlation among independent variables

plt.figure(figsize=(8, 8))

ax = sns.heatmap(X.corr(method='pearson').abs(), annot=True, square=True)

plt.show()
#dropping TAX and NOX

X.drop('TAX', axis=1, inplace=True)

X.drop('NOX', axis=1, inplace=True)



#remaning columns after removing multicollinearity

X.columns
#now checking correlation of each variable with MEDV by pearson method and dropping the one with least correlation with MEDV

for i in X.columns:

  corr, _ = pearsonr(X[i], Y)

  print(i,corr)
X.drop(['DIS', 'RAD'], axis=1, inplace=True)
#remaning variables/ features that can predict the MEDV most

X.columns
#spliting data into traning set and test set

X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2, random_state=0)
linear = lr()

linear.fit(X_train, Y_train)

Y_pred = linear.predict(X_test)

Y_compare_linear = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

Y_compare_linear.head() #displaying the comparision btween actual and predicted values of MEDV
polyRegressor = PolynomialFeatures(degree=3)

X_train_poly = polyRegressor.fit_transform(X_train)

X_test_poly = polyRegressor.fit_transform(X_test)

poly = lr()

poly.fit(X_train_poly, Y_train)

Y_pred = poly.predict(X_test_poly)

Y_compare_poly = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

Y_compare_poly.head() #displaying the comparision btween actual and predicted values of MEDV
svr = SVR(kernel= 'poly', gamma='scale')

svr.fit(X_train,Y_train)

Y_pred = svr.predict(X_test)

Y_compare_svr = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

Y_compare_svr.head() #displaying the comparision btween actual and predicted values of MEDV
rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train,Y_train)

Y_pred = rf.predict(X_test)

Y_compare_randomforrest = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

Y_compare_randomforrest.head() #displaying the comparision btween actual and predicted values of MEDV
knn = KNeighborsRegressor(n_neighbors=13)

knn.fit(X_train,Y_train)

Y_pred = knn.predict(X_test)

Y_compare_knn = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

Y_compare_knn.head() #displaying the comparision btween actual and predicted values of MEDV
fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(25, 4))

ax = ax.flatten()

Y_compare_linear.head(10).plot(kind='bar', title='Linear Regression', grid=True, ax=ax[0])

Y_compare_poly.head(10).plot(kind='bar', title='Polynomial Regression', grid=True, ax=ax[1])

Y_compare_svr.head(10).plot(kind='bar', title='Support Vector Regression', grid=True, ax=ax[2])

Y_compare_randomforrest.head(10).plot(kind='bar', title='Random Forrest Regression', grid=True, ax=ax[3])

Y_compare_knn.head(10).plot(kind='bar', title='KNN Regression', grid=True, ax=ax[4])

plt.show()
print('According to R squared scorring method we got below scores for out machine learning models:')

modelNames = ['Linear', 'Polynomial', 'Support Vector', 'Random Forrest', 'K-Nearest Neighbour']

modelRegressors = [linear, poly, svr, rf, knn]

models = pd.DataFrame({'modelNames' : modelNames, 'modelRegressors' : modelRegressors})

counter=0

score=[]

for i in models['modelRegressors']:

  if i is poly:

    accuracy = cvs(i, X_train_poly, Y_train, scoring='r2', cv=5)

    print('Accuracy of %s Regression model is %.2f' %(models.iloc[counter,0],accuracy.mean()))

    score.append(accuracy.mean())

  else:

    accuracy = cvs(i, X_train, Y_train, scoring='r2', cv=5)

    print('Accuracy of %s Regression model is %.2f' %(models.iloc[counter,0],accuracy.mean()))

    score.append(accuracy.mean())

  counter+=1
pd.DataFrame({'Model Name' : modelNames,'Score' : score}).sort_values(by='Score', ascending=True).plot(x=0, y=1, kind='bar', figsize=(15,5), title='Comparison of R2 scores of differnt models', )

plt.show()