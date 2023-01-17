#Team Members:
#Arpit Gupta : Roll no 6
#Rohit Soman :  Roll no 9
#Krushika Khanna : Roll no 25
#Aishwarya Bote : Roll no 46

#Import packages
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Load Files
#String - File Path
# sep - 
df = pd.read_csv('C:/Users/aishw/ML with Python/Python/Trainmart.csv',sep=',',delimiter=None,keep_default_na=True,skipinitialspace=False)
#Show the data
df.head(15)
#Shows the information
df.info()
#Describes the data
df.describe()
#Data Cleaning

#Counting Nulls
print('columns with null')
df.isnull().sum()
#Replacing Outlet_size with the mode of column
df['Outlet_Size'] = np.where(df['Outlet_Size'].isnull(),df['Outlet_Size'].mode(),df['Outlet_Size'])
#Replacing errronoeus values
df['Item_Fat_Content'] = np.where(df['Item_Fat_Content']=='low fat','Low Fat',df['Item_Fat_Content'])
df['Item_Fat_Content'] = np.where(df['Item_Fat_Content']=='LF','Low Fat',df['Item_Fat_Content'])
df['Item_Fat_Content'] = np.where(df['Item_Fat_Content']=='reg','Regular',df['Item_Fat_Content'])
#Dropping null values temporarily to check for modality of the column Item_Weights
df['Item_Weight'] = df['Item_Weight'].notnull()
#Distribution plot to check modality of column Item_Weight
sns.distplot(df.Item_Weight)
plt.show()

#Here we see that Item_Weight has two means i.e., the data is bi-modal.
from sklearn import linear_model
import statsmodels.api as sm
#Encoding the String values to categorical so as to build regression model
df["Outlet_Type"] = df["Outlet_Type"].astype('category')
df["Outlet_Type"] = df["Outlet_Type"].cat.codes

df["Outlet_Size"] = df["Outlet_Size"].astype('category')
df["Outlet_Size"] = df["Outlet_Size"].cat.codes

df["Outlet_Location_Type"] = df["Outlet_Location_Type"].astype('category')
df["Outlet_Location_Type"] = df["Outlet_Location_Type"].cat.codes

df["Item_Fat_Content"] = df["Item_Fat_Content"].astype('category')
df["Item_Fat_Content"] = df["Item_Fat_Content"].cat.codes

df["Item_Type"] = df["Item_Type"].astype('category')
df["Item_Type"] = df["Item_Type"].cat.codes
df.head()
#We now build a regression model to impute the null values in Item_Weight.
#In this model, Item_Weight is our dependent variable and rest all are our independent variables.
X = df[['Outlet_Size','Outlet_Location_Type','Outlet_Type', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP']]
Y = df['Item_Weight']
regr = linear_model.LinearRegression()
regr.fit(X, Y)
#Re-reading data so that null values can be imputed
df_toImpute = pd.read_csv('C:/Users/aishw/ML with Python/Python/Trainmart.csv',sep=',',delimiter=None,keep_default_na=True,skipinitialspace=False)

#Cleaning the data for null values
df_toImpute['Outlet_Size'] = np.where(df_toImpute['Outlet_Size'].isnull(),df_toImpute['Outlet_Size'].mode(),df_toImpute['Outlet_Size'])
#Replacing errronoeus values
df_toImpute['Item_Fat_Content'] = np.where(df_toImpute['Item_Fat_Content']=='low fat','Low Fat',df_toImpute['Item_Fat_Content'])
df_toImpute['Item_Fat_Content'] = np.where(df_toImpute['Item_Fat_Content']=='LF','Low Fat',df_toImpute['Item_Fat_Content'])
df_toImpute['Item_Fat_Content'] = np.where(df_toImpute['Item_Fat_Content']=='reg','Regular',df_toImpute['Item_Fat_Content'])
#Encoding Again
df_toImpute["Outlet_Type"] = df_toImpute["Outlet_Type"].astype('category')
df_toImpute["Outlet_Type"] = df_toImpute["Outlet_Type"].cat.codes

df_toImpute["Outlet_Size"] = df_toImpute["Outlet_Size"].astype('category')
df_toImpute["Outlet_Size"] = df_toImpute["Outlet_Size"].cat.codes

df_toImpute["Outlet_Location_Type"] = df_toImpute["Outlet_Location_Type"].astype('category')
df_toImpute["Outlet_Location_Type"] = df_toImpute["Outlet_Location_Type"].cat.codes

df_toImpute["Item_Fat_Content"] = df_toImpute["Item_Fat_Content"].astype('category')
df_toImpute["Item_Fat_Content"] = df_toImpute["Item_Fat_Content"].cat.codes

df_toImpute["Item_Type"] = df_toImpute["Item_Type"].astype('category')
df_toImpute["Item_Type"] = df_toImpute["Item_Type"].cat.codes

#Printing the data
df_toImpute.head()
X_pred = df_toImpute[['Outlet_Size','Outlet_Location_Type','Outlet_Type', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP']]
Y_pred = regr.predict(X_pred)
Y_pred
#Replacing the values of Item_Weight with the predicted values from reegression values
df_toImpute['Item_Weight'] = Y_pred
#Rechecking null values
#Counting Nulls
print('columns with null')
df_toImpute.isnull().sum()
# check relation with corelation - heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df_toImpute.corr(), annot=True,fmt='.2g', cmap = 'RdYlGn',linecolor='white',square=True, linewidth=0.30)
plt.show()
#Here we are grouping the outlet sales according to the item's fat content.
#In the dataset, Item_Fat_content has two values represented by categorical variables as follows:
# 0: Low Fat
# 1: Regular

grouped1=df_toImpute.groupby(['Item_Fat_Content'])['Item_Outlet_Sales'].mean()
#grouped1.sort_values(ascending=False, inplace=True)
print(grouped1)

#Bar Chart
grouped1[:5].plot.bar()
plt.show()
#Here, we are grouping item's outlet sale according to the various items types.
#Item types are here encoded as below:
# 0: Baking Goods
# 1: Breads
# 2: Breakfasts and so on.

grouped2=pd.DataFrame(df_toImpute.groupby(['Item_Type'])['Item_Outlet_Sales'].mean(), index=None)
#grouped2.sort(ascending=False, inplace=True)
print(grouped2)
#Printing the Correlation table
df_toImpute.corr()
#Printing the distribution plot for Outlet sales
sns.distplot(df_toImpute.Item_Outlet_Sales)
plt.show()

#The plot shows that the outlet sales are right-skewed.
#A right-skewed distribution has a long right tail, as seen here. 
#Right-skewed distributions are also called positive-skew distributions.
#That’s because there is a long tail in the positive direction on the number line. 
##The mean is also to the right of the peak.
#Here, we are grouping the item's outlet sales according to the location type.
#This data has 3 location types which are encoded as shown:
# 0: Tier 2
# 1: Tier 3
# 2: Tier 1

grouped4=df_toImpute.groupby(['Outlet_Location_Type'])['Item_Outlet_Sales'].mean()
grouped4.sort_values(ascending=False, inplace=True)
print(grouped4)
#Feature Engineering: A process of constructing new features from existing data to train a machine learning model.
#Feature engineering are of two types: Transformations & Aggregations.

#Here we are doing 'Transformation' type of feature engineering which is a transformation thar acts on a single dataframe
#by creating new features out of one or more of the existing columns.

#For this data we are making a new column called Outlet_Age which is calculated using the Outlet's establishment year.
from datetime import datetime
establish_year = df_toImpute['Outlet_Establishment_Year']
currentYear = datetime.now().year
outlet_age = (currentYear - establish_year) 
df_toImpute['Outlet_Age'] = outlet_age
df_toImpute.head()
#Linear Regression Model
#To predict the item's sale, we build a regression model

from statsmodels.formula.api import ols
reg_model = ols("Item_Outlet_Sales ~ Outlet_Age + Outlet_Size + Outlet_Location_Type + Outlet_Type + Item_Weight + Item_Fat_Content + Item_Visibility + Item_Type + Item_MRP",df_toImpute).fit()
#Regression Model Summary
reg_model.summary()

#The R-squared value here is 0.508 which means only 50.8% of variation in Outlet sales is explained by other independent factors
#To further improve the model, we eliminate the factors whose p-values > 0.05.
#This means for the improved model, we drop Outlet_Age,Item_Fat_Content & Item_Type
#Linear Regression - Improved Model
from statsmodels.formula.api import ols
reg_model_imp = ols("Item_Outlet_Sales ~  Outlet_Size + Outlet_Location_Type + Outlet_Type + Item_Visibility + Item_MRP",df_toImpute).fit()
reg_model_imp.summary()
from sklearn import linear_model
import statsmodels.api as sm
X = df_toImpute[['Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Visibility','Item_MRP']]
Y = df_toImpute['Item_Outlet_Sales']
regr = linear_model.LinearRegression()
regr.fit(X, Y)
#Printing the Intercept and coefficients of the model.
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
print('Variance score: %.2f' % regr.score(X, Y))
#Predictions for test data set

#Reading Test data
df_test = pd.read_csv('C:/Users/aishw/ML with Python/Python/Testmart.csv',sep=',',delimiter=None,keep_default_na=True,skipinitialspace=False)
df_test.head()
#Data Cleaning Test data

#Data Cleaning

#Counting Nulls
print('columns with null')
df_test.isnull().sum()
#Dropping Null Values
df_test.dropna()
#Replacing errronoeus values
df_test['Item_Fat_Content'] = np.where(df_test['Item_Fat_Content']=='low fat','Low Fat',df_test['Item_Fat_Content'])
df_test['Item_Fat_Content'] = np.where(df_test['Item_Fat_Content']=='LF','Low Fat',df_test['Item_Fat_Content'])
df_test['Item_Fat_Content'] = np.where(df_test['Item_Fat_Content']=='reg','Regular',df_test['Item_Fat_Content'])
#Encoding the test data
df_test["Outlet_Type"] = df_test["Outlet_Type"].astype('category')
df_test["Outlet_Type"] = df_test["Outlet_Type"].cat.codes

df_test["Outlet_Size"] = df_test["Outlet_Size"].astype('category')
df_test["Outlet_Size"] = df_test["Outlet_Size"].cat.codes

df_test["Outlet_Location_Type"] = df_test["Outlet_Location_Type"].astype('category')
df_test["Outlet_Location_Type"] = df_test["Outlet_Location_Type"].cat.codes

df_test["Item_Fat_Content"] = df_test["Item_Fat_Content"].astype('category')
df_test["Item_Fat_Content"] = df_test["Item_Fat_Content"].cat.codes

df_test["Item_Type"] = df_test["Item_Type"].astype('category')
df_test["Item_Type"] = df_test["Item_Type"].cat.codes
df.head()
X_pred = df_test[['Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Visibility','Item_MRP']]
Y_pred = regr.predict(X_pred)
#Predictions
Y_pred
#To check the accuracy in multiple linear regression , Use the variance.
#If variance score is near about the 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_pred, Y_pred))
# Now we use logistic regression to predict optimum Outlet type for every item
# logistic regression summary
import statsmodels.formula.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train = df_toImpute[['Outlet_Size','Outlet_Location_Type','Item_Fat_Content','Item_Type']]

y_train = df_toImpute['Outlet_Type']

model = linear_model.LogisticRegression()

model = model.fit (X_train,y_train)
# Use score method to get accuracy of model
score = model.score(X_train,y_train)
print(score)
#Creating hyperparameter search space

# Creating regularization penalty space
penalty = ['l1', 'l2']

# Creating regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Creating hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
# Create grid search using 5-fold cross validation to check for over fit
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(model, hyperparameters, cv=5, verbose=1)
#Conduct Grid Search
# Fit grid search
best_model = clf.fit(X_train,y_train)
# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
# Predict target variable
best_model.predict(X_train)
#Printing the best model score
score_best = best_model.score(X_train,y_train)
print(score_best)
#Predictions for test data set using the model built by logistic regression
X_pred = df_test[['Outlet_Size','Outlet_Location_Type','Item_Fat_Content','Item_Type']]
predicted_logreg = best_model.predict(X_pred)
#Print predictions
predicted_logreg
#Tuning hyperparameters using Random Search
# Creating regularization penalty space
from scipy.stats import uniform
penalty = ['l1', 'l2']

# Creating regularization hyperparameter distribution using uniform distribution
C = uniform(loc=0, scale=4)

# Creating hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
# Creating randomized search 5-fold cross validation and 100 iterations
from sklearn.model_selection import RandomizedSearchCV
clf = RandomizedSearchCV(model, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)
# Fit randomized search
best_model_RS = clf.fit(X_train,y_train)
# View hyperparameter values of best model
print('Best Penalty:', best_model_RS.best_estimator_.get_params()['penalty'])
print('Best C:', best_model_RS.best_estimator_.get_params()['C'])
# Predict target vector
Y_train_Predicted = best_model_RS.predict(X_train)
#Printing the best model score 
#We compared two different methods of hyper parameter using grid search and random search
score_best_RS = best_model_RS.score(X_train,y_train)
print(score_best_RS)
#Using the model built on train to predict outlet type on test data set 
X_pred = df_test[['Outlet_Size','Outlet_Location_Type','Item_Fat_Content','Item_Type']]
predicted_logreg_RS = best_model_RS.predict(X_pred)
predicted_logreg_RS
#Printing confusion matrix for predicted and train values
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, Y_train_Predicted)

#Printing accuracy score
metrics.accuracy_score(y_train, Y_train_Predicted)