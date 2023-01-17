# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/database.csv")
df.head()
df.describe()
df.loc[df["birth_year"] == "530s", "birth_year"] = 530
df.loc[df["birth_year"] == "1237?", "birth_year"] = 1237
df = df.drop(df.index[df.loc[:,"birth_year"] == "Unknown"])
df.loc[:, "birth_year"] = pd.to_numeric(df.loc[:,"birth_year"])
df.info()
df.count()
df["country"].fillna(value="Unknown", inplace=True)
df["continent"].fillna(value="Unknown", inplace = True)
df["latitude"].fillna(value=0, inplace=True)
df["longitude"].fillna(value=0, inplace=True)

df.loc[:, "birth_year"] = pd.to_numeric(df.loc[:,"birth_year"])
df.info()
from ggplot import *
ggplot(df, aes(x='birth_year',y='historical_popularity_index',color='domain'))+\
    geom_point(alpha='0.5')+\
    ggtitle("Historical Popularity by Birthyear")+ylab("Historical Popularity Index")+xlab("Year of Birth")+\
    coord_flip()+\
    theme_bw()
    #scale_x_continuous(breaks=(-3500,2005,100))+\
    #scale_y_continuous(breaks=(0,40,2))

from ggplot import *
ggplot(df, aes(x='birth_year',y='historical_popularity_index',color='sex'))+\
    geom_point(alpha='0.5')+\
    ggtitle("Historical Popularity by Birthyear")+ylab("Historical Popularity Index")+xlab("Year of Birth")+\
    coord_flip()+\
    theme_bw()
ggplot(df, aes(x='birth_year',y='historical_popularity_index',color='continent'))+\
    geom_point(alpha='0.5')+\
    ggtitle("Historical Popularity by Birthyear")+ylab("Historical Popularity Index")+xlab("Year of Birth")+\
    coord_flip()+\
    theme_bw()
# creating dummy variables for the columns that were objects
data_dummies = pd.get_dummies(df[['sex','country','continent','occupation','industry','domain']])
#add numerical columns and drop "article_id(column 0)  & city (column 4) & state (column 5)
pan = df.drop(df.columns[[0, 4, 5]], axis=1)
pan = pd.concat([pan, data_dummies], axis=1)
pan.head()
#Dropping original columns converted to dummy variarables 

pan2 = pan.drop(pan.columns[[0,1,3,4,7,8,9]], axis=1)
pan2.head()
corr_matrix=pan2.corr()
#what columns are correlated to the popularity index
corr_matrix['historical_popularity_index'].sort_values(ascending=False)
#calculate rmse for predictions
def get_rmse(y_test, predictions, model, data):
    from sklearn.metrics import mean_squared_error
    lin_mse = mean_squared_error(y_test, predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("------")
    print("Model: "+model)
    print("Data:  "+data)
    print("MSE:  {}".format(lin_mse))
    print("RMSE: {}".format(lin_rmse))
    lin_rmse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = pan2.drop('historical_popularity_index', axis = 1)
y = pan2['historical_popularity_index']

model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
#predict on test data
predictions = model.predict(X_test)
#predict on train data
predictions_train = model.predict(X_train)
get_rmse(y_test, predictions, "Linear Regression", "Test split")
get_rmse(y_train, predictions_train, "Linear Regression", "Train split")
#plotting predictions and data
import matplotlib.pyplot as plt
plt.hist(predictions, alpha=0.5)
plt.hist(y_test, alpha=0.5)

#plt.hist(np.subtract(predictions,y_test), alpha=0.7)
#Decision Tree
from sklearn.tree import DecisionTreeRegressor

model_tree = DecisionTreeRegressor()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_tree.fit(X_train, y_train)

#predict on test data
predictions_tree = model_tree.predict(X_test)
#predict on train data - overfitted..
predictions_tree_train = model_tree.predict(X_train)
get_rmse(y_test, predictions_tree, "Decision Tree Regression", "Test split")
get_rmse(y_train, predictions_tree_train, "Decision Tree Regression", "Train split")
import matplotlib.pyplot as plt
plt.hist(predictions,alpha=0.5,label='Logistic Regression')
plt.hist(predictions_tree,alpha=0.5,label='Decission Tree Regressor')
plt.hist(y_test,alpha=0.5,label='True Story')
plt.title("Linear Models - Predict Historical popularity index")
plt.legend()
#random forest
from sklearn.ensemble import RandomForestRegressor
model_forest = RandomForestRegressor()
model_forest.fit(X_train, y_train)
#predict on test data
predictions_forest = model_forest.predict(X_test)
#predict on train data - overfitted..
predictions_forest_train = model_forest.predict(X_train)
get_rmse(y_test, predictions_forest, "Random Forrest Regression", "Test split")
get_rmse(y_train, predictions_forest_train, "Random Forrest Regression", "Train split")
import matplotlib.pyplot as plt
plt.figure(figsize=(20,12))
plt.hist(predictions,alpha=0.5,label='Logistic Regression')
plt.hist(predictions_tree,alpha=0.5,label='Decission Tree Regressor')
plt.hist(predictions_forest,alpha=0.5,label='Random Forest Regressor')
plt.hist(y_test,alpha=0.5,label='True Story')
plt.title("Linear Models - Predict Historical popularity index")
plt.legend()
