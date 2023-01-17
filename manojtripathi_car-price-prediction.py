# importing some important libraries like 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv("../input/car-data/car data.csv") # reading the dataset from the local location ("car data.csv")

df.head() # it will display first 5 rows/ records , for more records you can pass no of rows in head().
df.shape # it will display no of rows and no of columns in the dataset here 301 rows and 9 columns.
print(df['Seller_Type'].unique()) # trying to figure that how many unique features / columns are in that columns

print(df['Transmission'].unique()) # in seller_type their are 2 , in Transmission their are also 2 and in Owner their are 3 unique columns.

print(df['Owner'].unique())

print(df['Fuel_Type'].unique())
# checking missing or null vlaues in dataset for all the columns. 

df.isnull().sum() # by luckely their is no null values here , if it comes then we should try to handel that.
df.describe() # describe() display some information like count , mean, std, min, 25%

# 50%,75%,max NOTE : describe() ONLY SHOW THE INFROMATION ABOUT NUMERICAL VALUES.
df.columns # to show the no of columns in the dataset. we can also see the column name by df.head(0) but use this when their are less no of columns

            #, otherwise df.head(0) will display first five and last five columns. 
# So, here i am removing car name column from the dataset we can use slicing or drop function to do this but,

# here i am doing with the simple method , why i am doing this is because the 

# car name might not be that much of important.



final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',

       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset.head() # after removing the car name again checking the dataset.
final_dataset['Current_Year']=2020 # here adding a new column which if current year and assigning with value 2020 we'll use it later .
final_dataset.head() # checking if the column current year is added or not.
#here i am subtracting the values of current year column with the values of year column to get the total no of year (means the age of car like something)

# and then assigning those values to a column no year and adding that column to the final dataset.



final_dataset['no_year']=final_dataset['Current_Year']-final_dataset['Year']
final_dataset.head() # again checking the final dataset if the changes are happen or not.
# As now we have no_year column which tells about the age of car so, i think we don't require year and current year columns

# because from then we have calculated the no year column. This is always a good practice to reduce no of columns as much you can.



final_dataset.drop(['Year'],axis=1,inplace=True) # dropeing year and current year from dataset axis 1 is used for 

final_dataset.drop(['Current_Year'],axis=1,inplace=True) # column wise operations and inplace true is used to make permanent changes in dataset.

final_dataset.head() # displaying top 5 records.
# Now here to convert our categorical columns into integer values because as you all know ML model only take integer values ,

# so for doing this i am using one hot encoding so, as here very less no of categorical columns are for that we use .get_dummies()

# Take ex of fuel type their are 3 categories petrol diesel and cng so, it'll convert them in 0 1 from.

# what drop_frist=True do is ex fuel type is cng as cng is droped by function so if diesel and pertrol values are 0 so by-default 

# model will understand that cng value is 1. so, if model is able to understad that ,then we can drop 1st column.



final_dataset=pd.get_dummies(final_dataset,drop_first=True)

final_dataset.head() # displaying first 5 records.
final_dataset.corr() # finding co-relations, it shows how one feature is co-related with others.
sns.pairplot(final_dataset) # displaying above co-relation in graphical format, it will difacult to understand in this form we will make it easy by using heat map after that.
# displaying the above pair plot in heatmap form because heat map are easier to read and understadn and also give more infromation.

# more the darker side higher the co-relation between those columns and for light color their is less co-realtion in the columns.

# we can understad it by seeing the indector bar in right side of heat map and values with +ive and -ive sign also.

corrmat=final_dataset.corr()

top_corr_features=corrmat.index

plt.figure(figsize=(20,20))

g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="vlag") # annot=true for displaying the values inside it and cmap is color type of that map.
final_dataset.head() # checking again the final dataset.
# Splitting the columsn in independent and dependent features

# dependent feature is one for which the model will predict the values it is also called traget variable.

#  independent feature are those on the basis of them model will predict the value for our dependent feature or target column.



X=final_dataset.iloc[:,1:] # using slicing we are saying like select all rows from dataset but select column from index pos 1 

#  (means we are not adding the selling_ price as it is target column for which model will predict the values) till last.



y=final_dataset.iloc[:,0] # here selecting our dependent feature or column or target column all rows but that column whose

# index value is 0 which is nothing other than selling_price.
X.head() # checking all the independent columns in X 
y.head() # checking our dependent column in y which is selling_price.
# Feature importance for our ML model...

# here finding which feature is how much important for our ML model for doing that we have

# extra tree regressor.
from sklearn.ensemble import ExtraTreesRegressor

model=ExtraTreesRegressor() # creating object of regressor

model.fit(X,y) # fitting the X and y in model to check the importance of all the columns.
print(model.feature_importances_) # here is the importance of all the columns.

print("higest important feature value ",max(model.feature_importances_))

print("least important feature value ",min(model.feature_importances_))
pd.DataFrame(X.columns,model.feature_importances_) # let's try to print above in table form with column so, it will be much clear.
# here, ploting the bar graph to get better Visualization of above records means importance of all features.



feat_importance=pd.Series(model.feature_importances_,index=X.columns)

feat_importance.nlargest(8).plot(kind='barh') # nlargest means how many bins higest value bins you want here i have 8 columns so

#, i given as 8 you may give like top 5 if you want to see the top 5 important colums in the X.
# spliting the data in train and test split

# here we are splitting data like for training 80% and for testing 20%...

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=11) # Random_state controls the shuffling applied to the data before applying the split.It will freeze the training and testing set of data.
# printing the shape of all the train test split.

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Impliting the random forest ,if we are using RF then we do't have to scale these values because RF useses DT and usually in DT

# scaling is not required..
from sklearn.ensemble import RandomForestRegressor

rf_random=RandomForestRegressor()
# Hyperparameters in this we will use no of combinations using n_estimatorrs.

import numpy as np

n_estimators=[int(x)for x in np.linspace(start=100,stop=1200,num=12)]

print(n_estimators)
# Performing hyper parameter tuning with Randomized search CV.. it helps us to find the best parameter considering max , min , depth.

#Randomized Search CV

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]# Number of features to consider at every split



max_features = ['auto', 'sqrt']# Maximum number of levels in tree



max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]# max_depth.append(None)  Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100] # Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



print(random_grid) #it will select best parameter out of these..
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train) # it will take time because it is performing cross validation and picking all other estimators and checking for it and applying cross validation in that.

rf_random.best_params_
predictions=rf_random.predict(X_test) # here performing prediction for the x test data.

predictions
# for comparing the above predictions we'll use distplot here what i am doing is substracting predictions values with y-test data.

# so, here our graph shows a normal distribution (which is also called bell curve distribution)

sns.distplot(y_test-predictions)

# the bell curve shows that the model is giving good results.
# if we plot above data it shows linear(we can put a linear line we can see if we apply a linear line here, 

# so, maximum of data points will be covered by that liner line.)

plt.scatter(y_test,predictions)
from sklearn.metrics import r2_score # checking the accuracy with with Adjusted R2 

r2_score(y_test,predictions)
# this is to generate a pickle file of this complete model which will further use in model deployement.

import pickle 

file = open('random_forest_regression_model.pkl', 'wb')

pickle.dump(rf_random, file)# .dump is used to dump the information of model to that file specially(classifier)...