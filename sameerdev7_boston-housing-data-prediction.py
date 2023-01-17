# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#importng libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #data visualization
from matplotlib import pyplot as plt #data visualization
%matplotlib inline
sns.set_style("darkgrid")

import warnings
warnings.filterwarnings("ignore")


#importing data and segregating columns using the 'names' parameter
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
training = pd.read_csv("../input/housing.csv",header=None, delimiter=r"\s+", names=column_names)

#displaying the dataset
training.head(10)


#describing the dataset
training.describe()
#data visualization
sns.barplot(palette="Blues_d", x="RAD", y="MEDV", data=training)
plt.title("Index of accessibility to radial highways vs Median value of owner-occupied homes")
plt.show()
#dividing age into various groups for better visialization of data
data = [training]
for dataset in data:
    dataset['GROUPAGE'] = dataset['AGE'].astype(int)
    dataset.loc[ dataset['GROUPAGE'] <= 10, 'GROUPAGE'] = 0
    dataset.loc[(dataset['GROUPAGE'] > 10) & (dataset['GROUPAGE'] <= 20), 'GROUPAGE'] = 1
    dataset.loc[(dataset['GROUPAGE'] > 20) & (dataset['GROUPAGE'] <= 30), 'GROUPAGE'] = 2
    dataset.loc[(dataset['GROUPAGE'] > 30) & (dataset['GROUPAGE'] <= 40), 'GROUPAGE'] = 3
    dataset.loc[(dataset['GROUPAGE'] > 40) & (dataset['GROUPAGE'] <= 50), 'GROUPAGE'] = 4
    dataset.loc[(dataset['GROUPAGE'] > 50) & (dataset['GROUPAGE'] <= 60), 'GROUPAGE'] = 5
    dataset.loc[(dataset['GROUPAGE'] > 60) & (dataset['GROUPAGE'] <= 70), 'GROUPAGE'] = 6
    dataset.loc[(dataset['GROUPAGE'] > 70) & (dataset['GROUPAGE'] <= 80), 'GROUPAGE'] = 7
    dataset.loc[(dataset['GROUPAGE'] > 80) & (dataset['GROUPAGE'] <= 90), 'GROUPAGE'] = 8
    dataset.loc[ dataset['GROUPAGE'] > 90, 'GROUPAGE'] = 9

# let's see how it's distributed train_df['Age'].value_counts()
training['GROUPAGE'].value_counts()
sns.swarmplot(x="GROUPAGE", y="MEDV", data=training)
plt.title("Proportion of owner-occupied units built prior to 1940 vs Median value of owner-occupied homes")
plt.show()
sns.barplot(x="GROUPAGE", y="MEDV", data=training)
plt.title("Proportion of owner-occupied units built prior to 1940 vs Median value of owner-occupied homes")
plt.show()
#dividing LSTAT into various groups for better visialization of data
data = [training]
for dataset in data:
    dataset['GROUPLSTAT'] = dataset['LSTAT'].astype(int)
    dataset.loc[ dataset['GROUPLSTAT'] <= 5, 'GROUPLSTAT'] = 0
    dataset.loc[(dataset['GROUPLSTAT'] > 5) & (dataset['GROUPLSTAT'] <= 10), 'GROUPLSTAT'] = 1
    dataset.loc[(dataset['GROUPLSTAT'] > 10) & (dataset['GROUPLSTAT'] <= 15), 'GROUPLSTAT'] = 2
    dataset.loc[(dataset['GROUPLSTAT'] > 15) & (dataset['GROUPLSTAT'] <= 20), 'GROUPLSTAT'] = 3
    dataset.loc[(dataset['GROUPLSTAT'] > 20) & (dataset['GROUPLSTAT'] <= 25), 'GROUPLSTAT'] = 4
    dataset.loc[(dataset['GROUPLSTAT'] > 25) & (dataset['GROUPLSTAT'] <= 30), 'GROUPLSTAT'] = 5
    dataset.loc[(dataset['GROUPLSTAT'] > 30) & (dataset['GROUPLSTAT'] <= 35), 'GROUPLSTAT'] = 6
    dataset.loc[ dataset['GROUPLSTAT'] > 35, 'GROUPLSTAT'] = 6

# let's see how it's distributed train_df['Age'].value_counts()
training['GROUPLSTAT'].value_counts()
#swarmplot
sns.swarmplot(x="GROUPLSTAT",y="MEDV", data=training)
plt.title("Percentage of lower status of the population vs Median value of owner-occupied homes")
plt.show()
#From this data we can see that the Lower is the LSTAT, the Higher is the Median Value of the House
sns.barplot(x="GROUPLSTAT", y="MEDV", data=training)
plt.title("Proportion of owner-occupied units built prior to 1940 vs Median value of owner-occupied homes")
plt.show()
#Proportion of non-retail business acres per town vs Median value of owner-occupied homes
sns.jointplot(x="INDUS", y="MEDV", data=training)
plt.show()
#Average number of rooms per dwelling vs Median value of owner-occupied homes
sns.jointplot(x="RM", y="MEDV", data=training)
plt.show()
#dropping the newly created columns GROUPAGE and GROUPLSTAT
training.drop(labels = ["GROUPAGE","GROUPLSTAT"], axis = 1, inplace = True)
#displaying the dataset
training.head(10)
#seperating features and target
x=training.iloc[:,:-1].values
y=training.iloc[:,-1].values
#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split,cross_val_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=500, criterion='mae',min_samples_leaf=1,min_samples_split=4)
regressor.fit(x_train,y_train)
#checking the score of the model
regressor.score(x_test,y_test)
#predicting the test set result
y_pred=regressor.predict(x_test)
# checking the Mean Absolute Error value for accuracy
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test,y_pred)
mae
# checking the Root Mean Squared Error value for accuracy
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
np.sqrt(mse)
#checking the cross val score of the model
results = cross_val_score(regressor, x_train, y_train, cv=10, n_jobs=-1)
results.mean()
#we can use GridSearchCV to find the optimal parameters for our model. Uncomment to use this section
"""from sklearn.model_selection import GridSearchCV
param_grid = { "criterion" : ["mse", "mae"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [25,50,75,100, 250,500]}
clf = GridSearchCV(estimator=regressor, param_grid=param_grid, n_jobs=-1)
clf.fit(x_train, y_train)
clf.best_params_"""

#result of running GridSearchCV
"""{'criterion': 'mae',
 'min_samples_leaf': 1,
 'min_samples_split': 4,
 'n_estimators': 500}"""