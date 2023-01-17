# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy import array
#df1 = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv', delimiter=',', nrows = nRowsRead)

df1 = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv', delimiter=',')

df1.dataframeName = 'master.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')

df1.head(5)
print(df1.describe())

print("----------------------------")

print(df1.dtypes)

print(df1.groupby('HDI for year').size())

print("----------------------------")

#Deletion of unnecessary columns

print(df1.columns.values)

df1 = df1.drop("suicides_no", axis=1)

df1 = df1.drop("country-year", axis=1)

df1 = df1.drop(" gdp_for_year ($) ", axis=1)

print("df1.shape after column deletion= ",df1.shape)

print(df1.dtypes)



#Check missing data

print("Missing data----------------------")

missing_val_count_by_column = (df1.isnull().sum())

print(missing_val_count_by_column)

print("HDI for year= ")

print(df1['HDI for year'].head(4))

print("HDI for year mean= ",df1['HDI for year'].mean())

print("HDI for year min= ",df1['HDI for year'].min())

print("HDI for year max= ",df1['HDI for year'].max())



print("Data after filling NaN----------------------") #filling NaN with mean value



df1['HDI for year']=df1['HDI for year'].fillna(df1['HDI for year'].mean())

print("HDI for year= ")

print(df1['HDI for year'].head(4))

missing_val_count_by_column2 = (df1.isnull().sum())

print(missing_val_count_by_column2)
# Categorical encoding 

df1["country"] = df1["country"].astype('category')  #Change type from object to category

df1["sex"] = df1["sex"].astype('category')

df1["age"] = df1["age"].astype('category')

df1["generation"] = df1["generation"].astype('category')

print(df1.dtypes)



df1["country_cat"] = df1["country"].cat.codes #Append new column to 

df1["sex_cat"] = df1["sex"].cat.codes

df1["age_cat"] = df1["age"].cat.codes

df1["generation_cat"] = df1["generation"].cat.codes



print("df1.shape after adding categorical column= ",df1.shape)

print(df1.head(5))

print("unique values country_cat = ",df1["country_cat"].nunique())

print("unique values sex_cat = ",df1["sex_cat"].nunique())

print("unique values age_cat = ",df1["age_cat"].nunique())

print("unique values generation_cat = ",df1["generation_cat"].nunique())





#Check relationship of each features



df1.plot(kind="scatter", x="population", y="suicides/100k pop")

df1.plot(kind="scatter", x="country_cat", y="suicides/100k pop")

df1.plot(kind="scatter", x="age_cat", y="suicides/100k pop")

df1.plot(kind="scatter", x="generation_cat", y="suicides/100k pop")

df1.plot(kind="scatter", x="year", y="suicides/100k pop")

df1.plot(kind="scatter", x="HDI for year", y="suicides/100k pop")

df1.plot(kind="scatter", x="gdp_per_capita ($)", y="suicides/100k pop")

df1.plot(kind="scatter", x="sex_cat", y="suicides/100k pop")



'''

#fails to create automatic loops

print(df1.columns[1])

print(df1.columns.dtype)



a=df1.columns.to_numpy()

print("a[0:3]====================")

print(a[0:3])



#str_obj=repr(a[0])

#str_obj=str(a[0])

#astr="country"

b = df1.columns.map(str)

#b = df1.columns.astype(str)

print(b)

print(b[0]=="country")





for i in range (df1.shape[1]):

    #print(df1.columns[i])

    #str=df1.columns[i]

    str="population"

    #str=str(a[i])

    df1.plot(kind="scatter", x=str, y="suicides/100k pop")





    

'''

import seaborn as sns



correlation_matrix = df1.corr()

print(correlation_matrix)



sns.heatmap(correlation_matrix) #most features are unrelated to suicides rate except sex_cat
#X & y separation

X=df1.drop("suicides/100k pop", axis=1)

X=X.drop("country", axis=1)

X=X.drop("sex", axis=1)

X=X.drop("age", axis=1)

X=X.drop("generation", axis=1)



y=df1["suicides/100k pop"]



print("X.shape= ",X.shape)

print(X.head(3))

print(X.dtypes)

print("------------------------------------------")

print("y.shape= ",y.shape)

print(y.head(3))

print(y.dtypes)
# Split training & test

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

print("X_train.shape= ",X_train.shape)

print("X_test.shape= ",X_test.shape)

print("y_train.shape= ",y_train.shape)

print("y_test.shape= ",y_test.shape)
from sklearn.preprocessing import StandardScaler



print(X_train.head(3))



scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)   # train only on X_train not the whole set

print("X after StandardScaler-------------------------")

print(X_train[:3,:])



X_test = scaler.transform (X_test)  # only transform
# Training & Evaluation



# LinearRegression (vanilla)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



classifier_linreg =  LinearRegression()

classifier_linreg.fit(X_train, y_train)



y_pred = classifier_linreg.predict(X_test)



print ("Linreg (vanilla)-----------------------------")

print("Training error: " + str(mean_squared_error(y_train, classifier_linreg.predict(X_train))))

print("Test error: " + str(mean_squared_error(y_test, classifier_linreg.predict(X_test))))

print ("Use CV, score= -----------------------------")

CV_score=cross_val_score(classifier_linreg, X_train, y_train,scoring="neg_mean_squared_error", cv=4)

print(CV_score)





# LinearRegression (poly)

trafo = PolynomialFeatures(4)      #set degree of polynomials = 4

X_train_poly = trafo.fit_transform(X_train)

X_test_poly = trafo.fit_transform(X_test)

print ("")

classifier_linreg_poly = LinearRegression()

classifier_linreg_poly.fit(X_train_poly, y_train)

print ("Linreg (polynomials)-----------------------------")

print("Training error Poly: " + str(mean_squared_error(y_train, classifier_linreg_poly.predict(X_train_poly))))

print("Test error Poly: " + str(mean_squared_error(y_test, classifier_linreg_poly.predict(X_test_poly))))

print ("Use CV, score= -----------------------------")

CV_score=cross_val_score(classifier_linreg_poly, X_train_poly, y_train,scoring="neg_mean_squared_error", cv=4)

print(CV_score)

#Test Error is getting worse if we increase the degree of polynomials
# LinearRegression (vanilla) for only 1 feature



print ("Linreg (vanilla) for only 1 feature----------------------------")



feature_idx=[]



for i in range(X_train.shape[1]):    

    feature_idx.append(i)



for i in feature_idx:

    X_train_01 =X_train[:,i]

    X_test_01 =X_test[:,i]



    classifier_linreg_01 =  LinearRegression()

    classifier_linreg_01.fit(X_train_01.reshape(-1, 1), y_train)



    y_pred = classifier_linreg_01.predict(X_test.reshape(-1, 1))



    print("Use feature no",i)

    print("   Training error: " + str(mean_squared_error(y_train, classifier_linreg_01.predict(X_train_01.reshape(-1, 1)))))

    print("   Test error: " + str(mean_squared_error(y_test, classifier_linreg_01.predict(X_test_01.reshape(-1, 1)))))

# DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score



tree_reg = DecisionTreeRegressor()

tree_reg. fit(X_train, y_train)



y_pred = tree_reg. predict(X_test)

tree_mse = mean_squared_error(y_test, y_pred)

tree_rmse = np.sqrt(tree_mse)

print("tree_rmse= ",tree_rmse)

print("y_train.mean= ",y_train.mean())



print("----------Use CV-------------------")

scores = cross_val_score(tree_reg, X_train, y_train,scoring="neg_mean_squared_error", cv=4)

rmse_scores = np.sqrt(-scores)

print("Scores: ", scores)

print("Mean: ", scores.mean())

print("Standard deviation: ", scores.std())

# RandomForestRegressor

from sklearn. ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor()

forest_reg.fit(X_train, y_train)



y_pred = forest_reg. predict(X_test)

tree_mse = mean_squared_error(y_test, y_pred)

tree_rmse = np.sqrt(tree_mse)

print("tree_rmse= ",tree_rmse)

print("y_train.mean= ",y_train.mean())



print("----------Use CV-------------------")

scores = cross_val_score(tree_reg, X_train, y_train,scoring="neg_mean_squared_error", cv=4)

rmse_scores = np.sqrt(-scores)

print("Scores: ", scores)

print("Mean: ", scores.mean())

print("Standard deviation: ", scores.std())

'''

model.fit(X_train, y_train)

print("training error: " + str(model.evaluate(X_train, y_train, verbose=0)))

print("test error: " + str(model.evaluate(X_test, y_test, verbose=0)))



#accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, is_neural_net=True)

print ("CNN accuracy",  accuracy)

'''
