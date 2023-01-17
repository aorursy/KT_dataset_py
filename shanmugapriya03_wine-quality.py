#Import all the necessary modules
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import scipy.stats
import os
print(os.listdir("../input"))

wine_data = pd.read_csv('../input/winequality-red.csv')
wine_data.info()
# checking whether column has any value other than numeric value
wine_data[~wine_data.applymap(np.isreal).all(1)]
# checking for nan
wine_data.isin([np.nan]).any()
# grouping data based on quality
# since the problem explains wine quality range from 0 to 6 as 'bad' & quality range from 7 to 10 as 'good'
wine_data.groupby(["quality"]).count()
wine_data.head(10)
# since i am going to build a decission tree to predict the wine quality as good or bad 
# I am replacing the vaules
# less than 7 and above 0 with '0' 
# greater than 6 and less than 11 with '1'

wine_data['quality']=wine_data['quality'].replace(3,0)
wine_data['quality']=wine_data['quality'].replace(4,0)
wine_data['quality']=wine_data['quality'].replace(5,0)
wine_data['quality']=wine_data['quality'].replace(6,0)
wine_data['quality']=wine_data['quality'].replace(7,1)
wine_data['quality']=wine_data['quality'].replace(8,1)
# after replacing the quality col with '0' and '1' inferring the count of data based on good(1) and bad(0) quality
# as we can see we have more data for bad quality wine and less data for good quality wine
# so the model ability to predict bad quality wine will be better than to predict good quality wine
wine_data.groupby(["quality"]).count()
wine_data.columns
wine_data.dtypes
wine_data.shape
wine_data.describe().transpose()
# positive values denotes more data is distributed around the tails [chlorides,residual sugar,sulphates]
# negative value denotes less data is distributed around the tails [citric acid]
wine_data.kurtosis(numeric_only=True)
print(scipy.stats.mstats.normaltest(wine_data['chlorides']))
print(scipy.stats.kurtosis(wine_data['chlorides']))
print(scipy.stats.kurtosistest(wine_data['chlorides']))
# positive skew denotes right tail is longer
# negative skew denotes left tail is longer
wine_data.skew(numeric_only=True)
print(scipy.stats.skew(wine_data['chlorides']))
print(scipy.stats.skewtest(wine_data['chlorides']))
sns.boxplot(x='chlorides',data=wine_data,orient='h')
# as you can see it is right skewed
sns.boxplot(x='residual sugar',data=wine_data,orient='h')
# as you can see it is right skewed
print(scipy.stats.mstats.normaltest(wine_data['citric acid']))
print(scipy.stats.kurtosis(wine_data['citric acid']))
print(scipy.stats.kurtosistest(wine_data['citric acid']))
print(scipy.stats.skew(wine_data['citric acid']))
print(scipy.stats.skewtest(wine_data['citric acid']))
sns.boxplot(x='citric acid',data=wine_data,orient='h')
# kurtosis gives a negative value which means less data is distributed around the data
sns.boxplot(x='fixed acidity',data=wine_data,orient='h')
sns.boxplot(x='pH',data=wine_data,orient='h')
sns.pairplot(wine_data, hue="quality",diag_kind="kde")
# Attributes which look normally distributed (density, pH).
# Some of the attributes look like they may have an exponential distribution (residual sugar, chlorides ,etc).
# finding the correlation with each and every feature

# as you can see 'alcohol' is less highly correlated with the 'quality' of the wine when compared to other fetures
# 'volatile acidity , chlorides , free sulfur dioxide , total sulfur dioxide , density & pH' are weakly correlated 
# with 'quality' of the wine

wine_data.corr()
X = wine_data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
Y = wine_data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7)
dt_model = DecisionTreeClassifier(criterion="entropy")
dt_model.fit(X_train,y_train)
print(dt_model.score(X_train , y_train))
print(dt_model.score(X_test , y_test))
y_predict = dt_model.predict(X_test)
print(metrics.confusion_matrix(y_test, y_predict))
from IPython.display import Image  
from sklearn import tree
from os import system

Credit_Tree_File = open('wine_tree.dot','w')
dot_data = tree.export_graphviz(dt_model, out_file=Credit_Tree_File, feature_names = list(X_train))

Credit_Tree_File.close()

print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = X_train.columns))
system("dot -Tpng wine_tree.dot -o wine_tree.png")
Image("wine_tree.png")
dt_model = DecisionTreeClassifier(criterion = 'entropy',max_depth = 9)
dt_model.fit(X_train, y_train)
Credit_Tree_File = open('wine_tree_regularized.dot','w')

dot_data = tree.export_graphviz(dt_model, out_file=Credit_Tree_File, feature_names = list(X_train))

Credit_Tree_File.close()

system("dot -Tpng wine_tree_regularized.dot -o wine_tree_regularized.png")
Image("wine_tree_regularized.png")
print(dt_model.score(X_train , y_train))
print(dt_model.score(X_test , y_test))