import pandas as pd #we use this to load, read and transform the dataset

import numpy as np #we use this for statistical analysis

import matplotlib.pyplot as plt #we use this to visualize the dataset

import seaborn as sns #we use this to make countplots

import sklearn.metrics as sklm #This is to test the models
#here we load the data

data = pd.read_csv('/kaggle/input/credit-card-data/Fn-UseC_-Marketing-Customer-Value-Analysis.csv')



#and immediately I would like to see how this dataset looks like

data.head()
#now let's look closer at the dataset we got

data.info()
data.shape
data.describe()
data.describe(include='O')
#Let's see what the options are in the text columns with two or three options (the objects)

print('Response: '+ str(data['Response'].unique()))

print('Coverage: '+ str(data['Coverage'].unique()))

print('Education: '+ str(data['Education'].unique()))

print('Employment Status: '+ str(data['EmploymentStatus'].unique()))

print('Gender: ' + str(data['Gender'].unique()))

print('Location Code: ' + str(data['Location Code'].unique()))

print('Married: ' + str(data['Marital Status'].unique()))

print('Policy Type: ' + str(data['Policy Type'].unique()))

print('Vehicle Size: ' + str(data['Vehicle Size'].unique()))
#As this is a numeric, thus continous number, I will use a scatterplot to see if there is a pattern. 

plt.hist(data['Customer Lifetime Value'], bins = 10)

plt.title("Customer Lifetime Value") #Assign title 

plt.xlabel("Value") #Assign x label 

plt.ylabel("Customers") #Assign y label 

plt.show()
plt.boxplot(data['Customer Lifetime Value'])
#We see that there are some great outliers here. 

#let's look closer to these outliers over 50000

outliers = data[data['Customer Lifetime Value'] > 50000]

outliers.head(25)
outliers.info()
#let's look in what columns there are missing values 

data.isnull().sum().sort_values(ascending = False)
#First we drop the customer column, as this is a unique identifier and will bias the model

data = data.drop(labels = ['Customer'], axis = 1)
#let's load the required packages

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# Let's transform the categorical variables to continous variables

column_names = ['Response', 'Coverage', 'Education', 

                     'Effective To Date', 'EmploymentStatus', 

                     'Gender', 'Location Code', 'Marital Status',

                     'Policy Type', 'Policy', 'Renew Offer Type',

                     'Sales Channel', 'Vehicle Class', 'Vehicle Size', 'State']



for col in column_names:

    data[col] = le.fit_transform(data[col])

    

data.head()
data.dtypes
data['Customer Lifetime Value'] = data['Customer Lifetime Value'].astype(int)

data['Total Claim Amount'] = data['Total Claim Amount'].astype(int)

#First we need to split the dataset in the y-column (the target) and the components (X), the independent columns. 

#This is needed as we need to use the X columns to predict the y in the model. 



y = data['Customer Lifetime Value'] #the column we want to predict 

X = data.drop(labels = ['Customer Lifetime Value'], axis = 1)  #independent columns 

 
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k='all')

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Name of the column','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
#get correlations of each features in dataset

corrmat = data.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,10))



#plot heat map

g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



#First try with the 5 most important features

X_5 = data[['Total Claim Amount', 'Monthly Premium Auto', 'Income', 'Coverage', 'Months Since Policy Inception']] #independent columns chosen 

y = data['Customer Lifetime Value']    #target column 



#I want to withhold 30 % of the trainset to perform the tests

X_train, X_test, y_train, y_test= train_test_split(X_5,y, test_size=0.3 , random_state = 25)
print('Shape of X_train is: ', X_train.shape)

print('Shape of X_test is: ', X_test.shape)

print('Shape of Y_train is: ', y_train.shape)

print('Shape of y_test is: ', y_test.shape)
#To check the model, I want to build a check:

import math

def print_metrics(y_true, y_predicted, n_parameters):

    ## First compute R^2 and the adjusted R^2

    r2 = sklm.r2_score(y_true, y_predicted)

    r2_adj = r2 - (n_parameters - 1)/(y_true.shape[0] - n_parameters) * (1 - r2)

    

    ## Print the usual metrics and the R^2 values

    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))

    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))

    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))

    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))

    print('R^2                    = ' + str(r2))

    print('Adjusted R^2           = ' + str(r2_adj))

   

# Linear regression model

model_5 = LinearRegression() 

model_5.fit(X_train, y_train)
Predictions = model_5.predict(X_test)

print_metrics(y_test, Predictions, 5)
#I want to withhold 30 % of the trainset to perform the tests

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3 , random_state = 25)



print('Shape of X_train is: ', X_train.shape)

print('Shape of X_test is: ', X_test.shape)

print('Shape of Y_train is: ', y_train.shape)

print('Shape of y_test is: ', y_test.shape)
# Linear regression model

model = LinearRegression() 

model.fit(X_train, y_train)
Predictions = model.predict(X_test)

print_metrics(y_test, Predictions, 22)
#to see the CLV data as is (without having the extremes removed)

data.hist('Customer Lifetime Value', bins = 10)

plt.show()
#Chech the skewness, if p < 0.05 it is skewed

clv = data['Customer Lifetime Value']

from scipy.stats import shapiro

shapiro(clv)[1]
#as this does not work, let's continue with the log function

log_clv = np.log(clv)

import seaborn as sns

sns.distplot(log_clv)
#it is slightly improved regarding the skewness. Let's try Box Cox now

from scipy.stats import boxcox

boxcox_clv = boxcox(clv)[0]

sns.distplot(boxcox_clv)
#I want to withhold 30 % of the trainset to perform the tests

X_train, X_test, y_train, y_test= train_test_split(X_5,boxcox_clv, test_size=0.3 , random_state = 25)
model_5.fit(X_train, y_train)
Predictions_box = model_5.predict(X_test)

print_metrics(y_test, Predictions_box, 5)