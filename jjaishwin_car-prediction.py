# problem statement: to predict the selling price of car based on feauters based on dataset 

import pandas as pd

import numpy as np

df = pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")

df.head()
df.shape
##to find how many catergorical features are there in unique 

##this can be done in a loop function as well

print(df['Seller_Type'].unique())

print(df['Transmission'].unique())

print(df['Owner'].unique())

print(df['Fuel_Type'].unique())
##to check missing or null values

df.isnull().sum()

#count of null values is zero
df.describe()
df.columns
#how old is the car? we use (present year - selling year) price

# to create a new derived feature, use  



final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',

       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.head()
#creating a new year column

final_dataset['Current_Year']=2020
final_dataset.head()
#to calculate the number of years

final_dataset['no_year']=final_dataset['Current_Year'] - final_dataset['Year']
final_dataset.head()
#dropping the year column and inplace is true because the operation has to be permanent

final_dataset.drop(['Year'],axis =1,inplace = True)
final_dataset.head()
final_dataset.drop(['Current_Year'],axis =1,inplace = True)
final_dataset.head()
#get_dummies

#dummy variable trap

# converting categorical features to onehotencoder

# drop_first : this is done so that first column with dummy variable has to be deleted

final_dataset = pd.get_dummies(final_dataset,drop_first=True)

final_dataset.head()
#correlation

final_dataset.corr()
import seaborn as sns

sns.pairplot(final_dataset)
#plot heat map 



import matplotlib.pyplot as plt

%matplotlib inline



corrmat=final_dataset.corr() 

top_corr_features=corrmat.index 

plt.figure(figsize=(20,20)) 

g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
final_dataset.head()
#independent and dependant features

x=final_dataset.iloc[:,1:]

y=final_dataset.iloc[:,0]
x.head
y.head()
### feature importance

from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()

model.fit(x,y)
print(model.feature_importances_)
#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_,index = x.columns)

feat_importances.nlargest(5).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#test size is taken as 20%
x_train.shape
x_train
from sklearn.ensemble import RandomForestRegressor

rf_random = RandomForestRegressor ()
###Hyperparameters

n_estimators=[int(X) for X in np.linspace(start = 100, stop = 1200 , num =12)]

print(n_estimators)
from sklearn.model_selection import RandomizedSearchCV

#Hyperparameterized tuning is done using RandomizedSearchCV

#Used to find best parameters  
#Randomized Search CV



# Number of trees in random forest

n_estimators=[int(X) for X in np.linspace(start = 100, stop = 1200 , num =12)]

# Number of features to consider at every split

max_features = ['auto','sqrt']

# Maximum number of levels in tree

max_depth = [int(X) for X in np.linspace(5,30,num =6)]

# max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split =[2,5,10,15,100]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1,2,5,10]
# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



print(random_grid)
# Initalizing random forest regressor

# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(x_train,y_train)
rf_random.best_params_
rf_random.best_score_
predictions=rf_random.predict(x_test)
predictions
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
import pickle

# open a file, where you ant to store the data

file = open('random_forest_regression_model.pkl', 'wb')



# dump information to that file

pickle.dump(rf_random, file)