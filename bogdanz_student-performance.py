# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

data.head()
data.info()
for col in data.columns:

    print(col)

    print(data[col].unique())
from sklearn.model_selection import train_test_split



data, test = train_test_split(data, test_size = 0.1, random_state =3)
import matplotlib.pyplot as plt



train_col = []

for col in data.columns:

    if type(data[col][0]) == str:

        train_col.append(col)



for i in range(len(train_col)):

    plt.figure()

    data[train_col[i]].value_counts().plot(kind = 'bar')
import seaborn as sns



col_num = ['math score', 'reading score', 'writing score']

cols = ['blue', 'green', 'orange']



for i in range(3):

    plt.figure()

    sns.distplot(data[col_num[i]], color = cols[i])

    x0, x1 = plt.xlim()

    y0, y1 = plt.ylim()

    plt.text(x=x0 + 2, y=y1 - 0.002, s="Skewness: " + str(data[col_num[i]].skew()), color = 'xkcd:poo brown')

    plt.text(x=x0 + 2, y=y1 - 0.004, s="Kurtosis: " + str(data[col_num[i]].kurt()), color = 'xkcd:dried blood')

    
plt.figure(figsize = (10, 10))

corr = sns.heatmap(data.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
one_hot = pd.get_dummies(data[train_col])

data = data.join(one_hot)

data.head()
one_hot_test = pd.get_dummies(test[train_col])

test = test.join(one_hot_test)

test.head()
plt.figure(figsize = (12, 12))

corr2 = sns.heatmap(data.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.figure(figsize = (18, 8))



plt.subplot(131)

ax1 = plt.subplot(1,3,1)

pd.crosstab(data['gender'], data['race/ethnicity']).plot(kind='bar', ax=ax1)

ax2 = plt.subplot(1,3,2)

pd.crosstab(data['lunch'], data['race/ethnicity']).plot(kind='bar', ax=ax2)

ax3 = plt.subplot(1,3,3)

pd.crosstab(data['test preparation course'], data['race/ethnicity']).plot(kind='bar', ax=ax3)
plt.figure(figsize = (18, 8))



plt.subplot(131)

ax1 = plt.subplot(1,3,1)

pd.crosstab(data['gender'], data['parental level of education']).plot(kind='bar', ax=ax1)

ax2 = plt.subplot(1,3,2)

pd.crosstab(data['lunch'], data['parental level of education']).plot(kind='bar', ax=ax2)

ax3 = plt.subplot(1,3,3)

pd.crosstab(data['test preparation course'], data['parental level of education']).plot(kind='bar', ax=ax3)
plt.rcParams['figure.figsize'] = (15, 9)

pd.crosstab(data['race/ethnicity'], data['parental level of education']).plot(kind='bar')
plt.figure(figsize = (18, 8))



plt.subplot(131)

ax1 = plt.subplot(1,3,1)

pd.crosstab(data['race/ethnicity'], data['gender']).plot(kind='bar', ax=ax1)

ax2 = plt.subplot(1,3,2)

pd.crosstab(data['race/ethnicity'], data['lunch']).plot(kind='bar', ax=ax2)

ax3 = plt.subplot(1,3,3)

pd.crosstab(data['race/ethnicity'], data['test preparation course']).plot(kind='bar', ax=ax3)



plt.figure(figsize = (18, 8))



plt.subplot(131)

ax1 = plt.subplot(1,3,1)

x = pd.crosstab(data['race/ethnicity'], data['gender'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind='bar', ax=ax1)

ax2 = plt.subplot(1,3,2)

x = pd.crosstab(data['race/ethnicity'], data['lunch'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind='bar', ax=ax2)

ax3 = plt.subplot(1,3,3)

x = pd.crosstab(data['race/ethnicity'], data['test preparation course'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind='bar', ax=ax3)
plt.figure(figsize = (18, 8))



plt.subplot(131)

ax1 = plt.subplot(1,3,1)

pd.crosstab(data['parental level of education'], data['gender']).plot(kind='bar', ax=ax1)

ax2 = plt.subplot(1,3,2)

pd.crosstab(data['parental level of education'], data['lunch']).plot(kind='bar', ax=ax2)

ax3 = plt.subplot(1,3,3)

pd.crosstab(data['parental level of education'], data['test preparation course']).plot(kind='bar', ax=ax3)



plt.figure(figsize = (18, 8))



plt.subplot(131)

ax1 = plt.subplot(1,3,1)

x = pd.crosstab(data['parental level of education'], data['gender'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind='bar', ax=ax1)

ax2 = plt.subplot(1,3,2)

x = pd.crosstab(data['parental level of education'], data['lunch'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind='bar', ax=ax2)

ax3 = plt.subplot(1,3,3)

x = pd.crosstab(data['parental level of education'], data['test preparation course'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind='bar', ax=ax3)
grouped = data[['math score','parental level of education']].iloc[:len(data)].groupby('parental level of education')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))

grouped = data[['math score', 'gender']].iloc[:len(data)].groupby('gender')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))

grouped = data[['math score', 'lunch']].iloc[:len(data)].groupby('lunch')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))

grouped = data[['math score', 'test preparation course']].iloc[:len(data)].groupby('test preparation course')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))

grouped = data[['math score', 'race/ethnicity']].iloc[:len(data)].groupby('race/ethnicity')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))
grouped = data[['reading score','parental level of education']].iloc[:len(data)].groupby('parental level of education')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))

grouped = data[['reading score', 'gender']].iloc[:len(data)].groupby('gender')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))

grouped = data[['reading score', 'lunch']].iloc[:len(data)].groupby('lunch')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))

grouped = data[['reading score', 'test preparation course']].iloc[:len(data)].groupby('test preparation course')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))

grouped = data[['reading score', 'race/ethnicity']].iloc[:len(data)].groupby('race/ethnicity')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))
grouped = data[['writing score','parental level of education']].iloc[:len(data)].groupby('parental level of education')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))

grouped = data[['writing score', 'gender']].iloc[:len(data)].groupby('gender')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))

grouped = data[['writing score', 'lunch']].iloc[:len(data)].groupby('lunch')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))

grouped = data[['writing score', 'test preparation course']].iloc[:len(data)].groupby('test preparation course')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))

grouped = data[['writing score', 'race/ethnicity']].iloc[:len(data)].groupby('race/ethnicity')

print(grouped.agg(['min', 'max', 'mean', 'median', 'std']))
plt.figure(figsize = (18, 15))

plt.subplot(325)



eth = data['race/ethnicity'].unique()



for k in range(1, len(eth)+1):

    ax = plt.subplot(3,2,k)

    data_tmp = data[data['race/ethnicity'] == eth[k-1]]

    sns.distplot(data_tmp['math score'])

    x0, x1 = plt.xlim()

    y0, y1 = plt.ylim()

    plt.text(x=x0 + 2, y=y1 - 0.002, s="Skewness: " + str(data_tmp['math score'].skew()), color = 'xkcd:poo brown')

    plt.text(x=x0 + 2, y=y1 - 0.004, s="Kurtosis: " + str(data_tmp['math score'].kurt()), color = 'xkcd:dried blood')

    plt.title(eth[k-1])
plt.figure(figsize = (18, 15))

plt.subplot(325)



eth = data['parental level of education'].unique()



for k in range(1, len(eth)+1):

    ax = plt.subplot(3,2,k)

    data_tmp = data[data['parental level of education'] == eth[k-1]]

    sns.distplot(data_tmp['math score'])

    x0, x1 = plt.xlim()

    y0, y1 = plt.ylim()

    plt.text(x=x0 + 2, y=y1 - 0.002, s="Skewness: " + str(data_tmp['math score'].skew()), color = 'xkcd:poo brown')

    plt.text(x=x0 + 2, y=y1 - 0.004, s="Kurtosis: " + str(data_tmp['math score'].kurt()), color = 'xkcd:dried blood')

    plt.title(eth[k-1])
plt.figure(figsize = (18, 15))

plt.subplot(325)



eth = data['race/ethnicity'].unique()



for k in range(1, len(eth)+1):

    ax = plt.subplot(3,2,k)

    data_tmp = data[data['race/ethnicity'] == eth[k-1]]

    sns.distplot(data_tmp['reading score'])

    x0, x1 = plt.xlim()

    y0, y1 = plt.ylim()

    plt.text(x=x0 + 2, y=y1 - 0.002, s="Skewness: " + str(data_tmp['reading score'].skew()), color = 'xkcd:poo brown')

    plt.text(x=x0 + 2, y=y1 - 0.004, s="Kurtosis: " + str(data_tmp['reading score'].kurt()), color = 'xkcd:dried blood')

    plt.title(eth[k-1])

    

plt.figure(figsize = (18, 15))

plt.subplot(325)



eth = data['parental level of education'].unique()



for k in range(1, len(eth)+1):

    ax = plt.subplot(3,2,k)

    data_tmp = data[data['parental level of education'] == eth[k-1]]

    sns.distplot(data_tmp['reading score'])

    x0, x1 = plt.xlim()

    y0, y1 = plt.ylim()

    plt.text(x=x0 + 2, y=y1 - 0.002, s="Skewness: " + str(data_tmp['reading score'].skew()), color = 'xkcd:poo brown')

    plt.text(x=x0 + 2, y=y1 - 0.004, s="Kurtosis: " + str(data_tmp['reading score'].kurt()), color = 'xkcd:dried blood')

    plt.title(eth[k-1])
plt.figure(figsize = (18, 15))

plt.subplot(325)



eth = data['race/ethnicity'].unique()



for k in range(1, len(eth)+1):

    ax = plt.subplot(3,2,k)

    data_tmp = data[data['race/ethnicity'] == eth[k-1]]

    sns.distplot(data_tmp['writing score'])

    x0, x1 = plt.xlim()

    y0, y1 = plt.ylim()

    plt.text(x=x0 + 2, y=y1 - 0.002, s="Skewness: " + str(data_tmp['writing score'].skew()), color = 'xkcd:poo brown')

    plt.text(x=x0 + 2, y=y1 - 0.004, s="Kurtosis: " + str(data_tmp['writing score'].kurt()), color = 'xkcd:dried blood')

    plt.title(eth[k-1])

    

plt.figure(figsize = (18, 15))

plt.subplot(325)



eth = data['parental level of education'].unique()



for k in range(1, len(eth)+1):

    ax = plt.subplot(3,2,k)

    data_tmp = data[data['parental level of education'] == eth[k-1]]

    sns.distplot(data_tmp['writing score'])

    x0, x1 = plt.xlim()

    y0, y1 = plt.ylim()

    plt.text(x=x0 + 2, y=y1 - 0.002, s="Skewness: " + str(data_tmp['writing score'].skew()), color = 'xkcd:poo brown')

    plt.text(x=x0 + 2, y=y1 - 0.004, s="Kurtosis: " + str(data_tmp['writing score'].kurt()), color = 'xkcd:dried blood')

    plt.title(eth[k-1])
def data_for_math(X):

    return (X[X['math score'] > 20]['math score'], X[X['math score'] > 20].drop(['math score', 'reading score', 'writing score'], axis = 1))



def data_for_reading(X):

    return (X[(X['race/ethnicity_group E'] == 0) | ((X['race/ethnicity_group E'] == 1) & (X['reading score'] > 40))]['reading score'], X[(X['race/ethnicity_group E'] == 0) | ((X['race/ethnicity_group E'] == 1) & (X['reading score'] > 40))].drop(['math score', 'reading score', 'writing score'], axis = 1))



def data_for_writing(X):

    return (X[(X['race/ethnicity_group E'] == 0) | ((X['race/ethnicity_group E'] == 1) & (X['writing score'] > 30))]['writing score'], X[(X['race/ethnicity_group E'] == 0) | ((X['race/ethnicity_group E'] == 1) & (X['writing score'] > 30))].drop(['math score', 'reading score', 'writing score'], axis = 1))
X = data.drop(['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'gender_female', 'lunch_standard', 'test preparation course_completed'], axis = 1)



col_num = ['math score', 'reading score', 'writing score']



from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

lr = LinearRegression(normalize = True)

#print(cross_val_score(lr, X, data[col_num[i]], cv=5, scoring = 'neg_mean_squared_error').mean())

print(cross_val_score(lr, data_for_math(X)[1], data_for_math(X)[0], cv=5, scoring = 'neg_mean_squared_error').mean())

print(cross_val_score(lr, data_for_reading(X)[1], data_for_reading(X)[0], cv=5, scoring = 'neg_mean_squared_error').mean())

print(cross_val_score(lr, data_for_writing(X)[1], data_for_writing(X)[0], cv=5, scoring = 'neg_mean_squared_error').mean())

#-179.42219401041666

#-170.67588758680554

#-159.81538024902343 - scores if we do not remove variables with correlation -1 - as we can see, they cause larger errors.



#-178.22145182291666

#-169.28279527452256

#-158.72642145368786 - scores for lr if we do not remove outliers



#-171.16810390479515

#-167.92170391667636

#-157.68080284227577 - scores if we remove outliers
# preparing test data

X_test = test.drop(['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'math score', 'reading score', 'writing score', 'gender_female', 'lunch_standard', 'test preparation course_completed'], axis = 1)
col_num = ['math score', 'reading score', 'writing score']

cols = ['blue', 'green', 'orange']

col_train = []

for col in X.columns:

    if type(X[col][0]) != str and (col not in col_num):

        col_train.append(col)

        



for i in range(3):

    d = {}

    for col in col_train:

        d[col] = abs(data[col_num[i]].corr(X[col]))

        

    plt.figure()

    d_s = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}

    plt.bar(list(d_s.keys()), list(d_s.values()), color = cols[i])

    plt.xticks(rotation=90)

    
from sklearn.model_selection import KFold

from sklearn.decomposition import PCA

from sklearn import metrics



def error(n_pca, score, model):

    kf = KFold(n_splits = 5) # 5-fold cross-validation

    #y = data['math score'] #first just do for math score

    

    if score == 'math score':

        (y_tmp, X_tmp) = data_for_math(X)

    elif score == 'reading score':

        (y_tmp, X_tmp) = data_for_reading(X)

    else:

        (y_tmp, X_tmp) = data_for_writing(X)



    error = 0



    for train_index, test_index in kf.split(X_tmp):

        X_train, X_test = X_tmp.iloc[list(train_index), :], X_tmp.iloc[list(test_index), :]

        y_train, y_test = y_tmp.iloc[list(train_index)], y_tmp.iloc[list(test_index)]

        pca = PCA(n_components = n_pca, whiten = True)

        train_red = pca.fit_transform(X_train)

        test_red = pca.transform(X_test)

    

        model.fit(train_red, y_train)

        y_pred = model.predict(test_red)

        err = metrics.mean_squared_error(y_pred, y_test)

        error += err

    



    return error/5
errs = [error(i, 'math score', LinearRegression(normalize = True)) for i in range(1, 15)]

plt.plot(range(1, 15), errs)

print(min(errs), errs.index(min(errs)))

errs = [error(i, 'reading score', LinearRegression(normalize = True)) for i in range(1, 15)]

plt.plot(range(1, 15), errs)

print(min(errs), errs.index(min(errs)))

errs = [error(i, 'writing score', LinearRegression(normalize = True)) for i in range(1, 15)]

plt.plot(range(1, 15), errs)

print(min(errs), errs.index(min(errs)))
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(random_state = 3)

print(cross_val_score(rf, data_for_math(X)[1], data_for_math(X)[0], cv=5, scoring = 'neg_mean_squared_error').mean())

print(cross_val_score(rf, data_for_reading(X)[1], data_for_reading(X)[0], cv=5, scoring = 'neg_mean_squared_error').mean())

print(cross_val_score(rf, data_for_writing(X)[1], data_for_writing(X)[0], cv=5, scoring = 'neg_mean_squared_error').mean())
from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.preprocessing import RobustScaler



pipeline_rf = Pipeline(

                    [ 

                     ('rf', RandomForestRegressor(random_state = 3))

                     

])



parameters = {}

parameters['rf__min_samples_split'] = [38, 40, 42] 

parameters['rf__max_depth'] = [2, 4, 6, 8, None] 

parameters['rf__n_estimators'] = [10, 25, 50, 100] 



CV = GridSearchCV(pipeline_rf, parameters, scoring = 'neg_mean_squared_error', n_jobs= 4, cv = 5)

CV.fit(data_for_math(X)[1], data_for_math(X)[0])   



print('Best score and parameter combination = ')



print(CV.best_score_)    

print(CV.best_params_) 
CV = GridSearchCV(pipeline_rf, parameters, scoring = 'neg_mean_squared_error', n_jobs= 4, cv = 5)

CV.fit(data_for_reading(X)[1], data_for_reading(X)[0])   



print('Best score and parameter combination = ')



print(CV.best_score_)    

print(CV.best_params_) 
CV = GridSearchCV(pipeline_rf, parameters, scoring = 'neg_mean_squared_error', n_jobs= 4, cv = 5)

CV.fit(data_for_writing(X)[1], data_for_writing(X)[0])   



print('Best score and parameter combination = ')



print(CV.best_score_)    

print(CV.best_params_) 
err_min = 10000

params = [0,0,0,0]

depth = [4,5,6]

samples = [20, 30, 40]

n_est = [40, 50, 60]

n_pca = [i for i in range(6, 12)]





# Finding the optimal parameters

#for d in depth:

#    for s in samples:

#        for n in n_est:

#            for n_p in n_pca:

#                model = RandomForestRegressor(max_depth = d, min_samples_split = s, n_estimators = n)

#                if error(n_p, 'math score', model) < err_min:

#                    err_min = error(n_p, 'math score', model)

#                    params = [n_p, s, d, n]

#                    

#    print("tried depth " + str(d))

                    

#print("Min error: " + str(err_min))

#print("parameters: " + str(params))



#Min error: 185.30627310373836

#parameters: [11, 40, 4, 60]
from sklearn.svm import SVR



pipeline_svr = Pipeline(

                    [ 

                     ('svr', SVR())

                     

])



parameters = {}

parameters['svr__kernel'] = ['rbf', 'poly', 'sigmoid', 'linear'] 

parameters['svr__C'] = [0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]



CV = GridSearchCV(pipeline_svr, parameters, scoring = 'neg_mean_squared_error', n_jobs= 4, cv = 5)

CV.fit(data_for_math(X)[1], data_for_math(X)[0])   



print('Best score and parameter combination = ')



print(CV.best_score_)    

print(CV.best_params_) 
CV = GridSearchCV(pipeline_svr, parameters, scoring = 'neg_mean_squared_error', n_jobs= 4, cv = 5)

CV.fit(data_for_reading(X)[1], data_for_reading(X)[0])   



print('Best score and parameter combination = ')



print(CV.best_score_)    

print(CV.best_params_) 
CV = GridSearchCV(pipeline_svr, parameters, scoring = 'neg_mean_squared_error', n_jobs= 4, cv = 5)

CV.fit(data_for_writing(X)[1], data_for_writing(X)[0])   



print('Best score and parameter combination = ')



print(CV.best_score_)    

print(CV.best_params_) 
err_min = 10000

params = [0,0]

kernels = ['rbf', 'poly', 'sigmoid', 'linear'] 

Cs = [0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

n_pca = [i for i in range(6, 15)]



for k in kernels:

    for c in Cs:

        for n in n_pca:

            model = SVR(kernel = k, C = c)

            if error(n, 'math score', model) < err_min:

                err_min = error(n, 'math score', model)

                params = [k, c, n]

                

    print("tried " + str(k))

                    

print("Min error: " + str(err_min))

print("parameters: " + str(params))
err_min = 10000

params = [0,0]



for k in kernels:

    for c in Cs:

        for n in n_pca:

            model = SVR(kernel = k, C = c)

            if error(n, 'reading score', model) < err_min:

                err_min = error(n, 'reading score', model)

                params = [k, c, n]

                

    print("tried " + str(k))

                    

print("Min error: " + str(err_min))

print("parameters: " + str(params))
err_min = 10000

params = [0,0]



for k in kernels:

    for c in Cs:

        for n in n_pca:

            model = SVR(kernel = k, C = c)

            if error(n, 'writing score', model) < err_min:

                err_min = error(n, 'writing score', model)

                params = [k, c, n]

                

    print("tried " + str(k))

                    

print("Min error: " + str(err_min))

print("parameters: " + str(params))
print("Linear regression")



lr = LinearRegression(normalize = True)



lr.fit(data_for_math(X)[1], data_for_math(X)[0])

y_pred = lr.predict(X_test)

print("MSE for math score " + str(metrics.mean_squared_error(y_pred, test['math score'])))

plt.plot(test['math score'], y_pred, 'o')

plt.xlabel("Actual score")

plt.ylabel("Predicted score")

plt.ylim([0, 100])



lr.fit(data_for_reading(X)[1], data_for_reading(X)[0])

y_pred = lr.predict(X_test)

print("MSE for reading score " + str(metrics.mean_squared_error(y_pred, test['reading score'])))

plt.plot(test['reading score'], y_pred, 'o')



lr.fit(data_for_writing(X)[1], data_for_writing(X)[0])

y_pred = lr.predict(X_test)

print("MSE for writing score " + str(metrics.mean_squared_error(y_pred, test['writing score'])))

plt.plot(test['writing score'], y_pred, 'o')



plt.plot(test['math score'], test['math score'], 'r')



plt.legend(('math', 'reading', 'writing'))
print("Linear regression on principal components")



lr = LinearRegression(normalize = True)

pca = PCA(n_components = 12, whiten = True)



y_tmp, X_tmp = data_for_math(X)

X_tr = pca.fit_transform(X_tmp)

test_set = pca.transform(X_test)

lr.fit(X_tr, y_tmp)

y_pred = lr.predict(test_set)

print("MSE for math score " + str(metrics.mean_squared_error(y_pred, test['math score'])))

plt.plot(test['math score'], y_pred, 'o')

plt.xlabel("Actual score")

plt.ylabel("Predicted score")

plt.ylim([0, 100])



y_tmp, X_tmp = data_for_reading(X)

X_tr = pca.fit_transform(X_tmp)

test_set = pca.transform(X_test)

lr.fit(X_tr, y_tmp)

y_pred = lr.predict(test_set)

print("MSE for math score " + str(metrics.mean_squared_error(y_pred, test['reading score'])))

plt.plot(test['math score'], y_pred, 'o')



y_tmp, X_tmp = data_for_writing(X)

X_tr = pca.fit_transform(X_tmp)

test_set = pca.transform(X_test)

lr.fit(X_tr, y_tmp)

y_pred = lr.predict(test_set)

print("MSE for math score " + str(metrics.mean_squared_error(y_pred, test['writing score'])))

plt.plot(test['math score'], y_pred, 'o')



plt.plot(test['math score'], test['math score'], 'r')



plt.legend(('math', 'reading', 'writing'))
print("SVR")



svr = SVR(kernel = 'linear', C = 0.5)



svr.fit(data_for_math(X)[1], data_for_math(X)[0])

y_pred = svr.predict(X_test)

print("MSE for math score " + str(metrics.mean_squared_error(y_pred, test['math score'])))

plt.plot(test['math score'], y_pred, 'o')

plt.xlabel("Actual score")

plt.ylabel("Predicted score")

plt.ylim([0, 100])



svr.fit(data_for_reading(X)[1], data_for_reading(X)[0])

y_pred = svr.predict(X_test)

print("MSE for reading score " + str(metrics.mean_squared_error(y_pred, test['reading score'])))

plt.plot(test['reading score'], y_pred, 'o')



svr.fit(data_for_writing(X)[1], data_for_writing(X)[0])

y_pred = svr.predict(X_test)

print("MSE for writing score " + str(metrics.mean_squared_error(y_pred, test['writing score'])))

plt.plot(test['writing score'], y_pred, 'o')



plt.plot(test['math score'], test['math score'], 'r')



plt.legend(('math', 'reading', 'writing'))
print("SVR on principal components")



svr = SVR(kernel = 'sigmoid', C = 2.0)

pca = PCA(n_components = 12, whiten = True)



y_tmp, X_tmp = data_for_math(X)

X_tr = pca.fit_transform(X_tmp)

test_set = pca.transform(X_test)

svr.fit(X_tr, y_tmp)

y_pred = svr.predict(test_set)

print("MSE for math score " + str(metrics.mean_squared_error(y_pred, test['math score'])))

plt.plot(test['math score'], y_pred, 'o')

plt.xlabel("Actual score")

plt.ylabel("Predicted score")

plt.ylim([0, 100])



y_tmp, X_tmp = data_for_reading(X)

X_tr = pca.fit_transform(X_tmp)

test_set = pca.transform(X_test)

svr.fit(X_tr, y_tmp)

y_pred = svr.predict(test_set)

print("MSE for math score " + str(metrics.mean_squared_error(y_pred, test['reading score'])))

plt.plot(test['math score'], y_pred, 'o')



y_tmp, X_tmp = data_for_writing(X)

X_tr = pca.fit_transform(X_tmp)

test_set = pca.transform(X_test)

svr.fit(X_tr, y_tmp)

y_pred = svr.predict(test_set)

print("MSE for math score " + str(metrics.mean_squared_error(y_pred, test['writing score'])))

plt.plot(test['math score'], y_pred, 'o')



plt.plot(test['math score'], test['math score'], 'r')



plt.legend(('math', 'reading', 'writing'))
r_e = {'group A':0, 'group B':1, 'group C':2, 'group D':3, 'group E':4}

p_e = {'high school':0, 'some high school':1, 'some college':2, 'associate\'s degree':3, 'bachelor\'s degree':4, 'master\'s degree':5}
data_ord = pd.DataFrame()



data_ord['r_e'] = data['race/ethnicity'].apply(lambda row: r_e[row])

data_ord['p_e'] = data['parental level of education'].apply(lambda row: p_e[row])

data_ord[['gender_male', 'lunch_free/reduced', 'test preparation course_none', 'math score', 'reading score', 'writing score']] = data[['gender_male', 'lunch_free/reduced', 'test preparation course_none', 'math score', 'reading score', 'writing score']]

data_ord

def data_for_math_ord(X):

    return (X[X['math score'] > 20]['math score'], X[X['math score'] > 20].drop(['math score', 'reading score', 'writing score'], axis = 1))



def data_for_reading_ord(X):

    return (X[(X['r_e'] != 4) | ((X['r_e'] == 4) & (X['reading score'] > 40))]['reading score'], X[(X['r_e'] != 4) | ((X['r_e'] == 4) & (X['reading score'] > 40))].drop(['math score', 'reading score', 'writing score'], axis = 1))



def data_for_writing_ord(X):

    return (X[(X['r_e'] != 4) | ((X['r_e'] == 4) & (X['writing score'] > 30))]['writing score'], X[(X['r_e'] != 4) | ((X['r_e'] == 4) & (X['writing score'] > 30))].drop(['math score', 'reading score', 'writing score'], axis = 1))
lr = LinearRegression(normalize = True)

print(cross_val_score(lr, data_for_math_ord(data_ord)[1], data_for_math_ord(data_ord)[0], cv=5, scoring = 'neg_mean_squared_error').mean())

print(cross_val_score(lr, data_for_reading_ord(data_ord)[1], data_for_reading_ord(data_ord)[0], cv=5, scoring = 'neg_mean_squared_error').mean())

print(cross_val_score(lr, data_for_writing_ord(data_ord)[1], data_for_writing_ord(data_ord)[0], cv=5, scoring = 'neg_mean_squared_error').mean())
plt.figure(figsize = (12, 12))

corr_ord = sns.heatmap(data_ord.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
rf = RandomForestRegressor(random_state = 3, max_depth = 5, min_samples_split = 30, n_estimators = 50)

print(cross_val_score(rf, data_for_math_ord(data_ord)[1], data_for_math_ord(data_ord)[0], cv=5, scoring = 'neg_mean_squared_error').mean())

print(cross_val_score(rf, data_for_reading_ord(data_ord)[1], data_for_reading_ord(data_ord)[0], cv=5, scoring = 'neg_mean_squared_error').mean())

print(cross_val_score(rf, data_for_writing_ord(data_ord)[1], data_for_writing_ord(data_ord)[0], cv=5, scoring = 'neg_mean_squared_error').mean())
svr = SVR(kernel = 'linear', C = 0.5)

print(cross_val_score(svr, data_for_math_ord(data_ord)[1], data_for_math_ord(data_ord)[0], cv=5, scoring = 'neg_mean_squared_error').mean())

print(cross_val_score(svr, data_for_reading_ord(data_ord)[1], data_for_reading_ord(data_ord)[0], cv=5, scoring = 'neg_mean_squared_error').mean())

print(cross_val_score(svr, data_for_writing_ord(data_ord)[1], data_for_writing_ord(data_ord)[0], cv=5, scoring = 'neg_mean_squared_error').mean())
test_ord = pd.DataFrame()



test_ord['r_e'] = test['race/ethnicity'].apply(lambda row: r_e[row])

test_ord['p_e'] = test['parental level of education'].apply(lambda row: p_e[row])

test_ord[['gender_male', 'lunch_free/reduced', 'test preparation course_none']] = test[['gender_male', 'lunch_free/reduced', 'test preparation course_none']]
print("Linear regression")



lr = LinearRegression(normalize = True)



lr.fit(data_for_math_ord(data_ord)[1], data_for_math_ord(data_ord)[0])

y_pred = lr.predict(test_ord)

print("MSE for math score " + str(metrics.mean_squared_error(y_pred, test['math score'])))

plt.plot(test['math score'], y_pred, 'o')

plt.xlabel("Actual score")

plt.ylabel("Predicted score")

plt.ylim([0, 100])



lr.fit(data_for_reading_ord(data_ord)[1], data_for_reading_ord(data_ord)[0])

y_pred = lr.predict(test_ord)

print("MSE for reading score " + str(metrics.mean_squared_error(y_pred, test['reading score'])))

plt.plot(test['reading score'], y_pred, 'o')



lr.fit(data_for_writing_ord(data_ord)[1], data_for_writing_ord(data_ord)[0])

y_pred = lr.predict(test_ord)

print("MSE for writing score " + str(metrics.mean_squared_error(y_pred, test['writing score'])))

plt.plot(test['writing score'], y_pred, 'o')



plt.plot(test['math score'], test['math score'], 'r')



plt.legend(('math', 'reading', 'writing'))
print("SVR")



svr = SVR(kernel = 'linear', C = 0.5)



svr.fit(data_for_math_ord(data_ord)[1], data_for_math_ord(data_ord)[0])

y_pred = svr.predict(test_ord)

print("MSE for math score " + str(metrics.mean_squared_error(y_pred, test['math score'])))

plt.plot(test['math score'], y_pred, 'o')

plt.xlabel("Actual score")

plt.ylabel("Predicted score")

plt.ylim([0, 100])



svr.fit(data_for_reading_ord(data_ord)[1], data_for_reading_ord(data_ord)[0])

y_pred = svr.predict(test_ord)

print("MSE for reading score " + str(metrics.mean_squared_error(y_pred, test['reading score'])))

plt.plot(test['reading score'], y_pred, 'o')



svr.fit(data_for_writing_ord(data_ord)[1], data_for_writing_ord(data_ord)[0])

y_pred = svr.predict(test_ord)

print("MSE for writing score " + str(metrics.mean_squared_error(y_pred, test['writing score'])))

plt.plot(test['writing score'], y_pred, 'o')



plt.plot(test['math score'], test['math score'], 'r')



plt.legend(('math', 'reading', 'writing'))