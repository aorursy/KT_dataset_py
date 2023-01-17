# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fancyimpute import KNN

from scipy.stats import chi2_contingency

from random import randrange, uniform

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

% matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_excel("../input/Absenteeism_at_work_Project.xls")
df.shape
df.head()
df.columns
type(df.ID)
df.dtypes
df['ID'] = df['ID'].astype('category')



df['Reason for absence'] = df['Reason for absence'].replace(0,20)

df['Reason for absence'] = df['Reason for absence'].astype('category')



df['Month of absence'] = df['Month of absence'].replace(0,np.nan)

df['Month of absence'] = df['Month of absence'].astype('category')



df['Day of the week']  = df['Day of the week'].astype('category')

df['Seasons'] = df['Seasons'].astype('category')

df['Disciplinary failure'] = df['Disciplinary failure'].astype('category')

df['Education'] = df['Education'].astype('category')

df['Son'] = df['Son'].astype('category')

df['Social drinker'] = df['Social drinker'].astype('category')

df['Social smoker'] = df['Social smoker'].astype('category')

df['Pet'] = df['Pet'].astype('category')
df.dtypes
#making copy of reordered data

ordered_data = df.copy()
#separating continous and categrocal variables

continuous_variables = ["Transportation expense", "Distance from Residence to Work", 

                          "Service time" , "Age" , "Work load Average/day " ,

                          "Hit target", "Weight" , "Height", "Body mass index",

                          "Absenteeism time in hours"

                        ]



categorical_variables = [ "ID", "Reason for absence", "Month of absence", "Day of the week",

                           "Seasons", "Disciplinary failure", "Education", "Son",                

                           "Social drinker",  "Social smoker", "Pet"

                          ]
#craeating separate dataframe with misssing valuse

missing_val = pd.DataFrame(df.isnull().sum())

missing_val = missing_val.reset_index()

missing_val = missing_val.rename(columns = {'index' :'Variables',0:'missing_perc'})

missing_val
missing_val['missing_perc'] = (missing_val['missing_perc']/len(df))*100

missing_val = missing_val.sort_values('missing_perc', ascending=False).reset_index(drop = True)

missing_val.to_csv("missing_val.csv", index=False)

#Actual Value = 29

#Mean = 26

#Median = 25

#KNN = 27



#print(df['Body mass index'].iloc[9])

#df['Body mass index'].iloc[9] = np.nan
#Mean

#df['Body mass index'] = df['Body mass index'].fillna(df['Body mass index'].median())



#Median

#df['Body mass index'] = df['Body mass index'].fillna(df['Body mass index'].median())



#KNN

df = pd.DataFrame(KNN(k = 5).fit_transform(df), columns = df.columns)
df.isnull().sum()

#df.columns
#Rounding the values of categorical variables



for i in categorical_variables:

    df.loc[:,i] = df.loc[:,i].round()

    df.loc[:,i] = df.loc[:,i].astype('category')
sns.set_style("whitegrid")

sns.factorplot(data=df, x='Reason for absence', kind= 'count',size=3,aspect=2)

sns.factorplot(data=df, x='Seasons', kind= 'count',size=3,aspect=2)

sns.factorplot(data=df, x='Education', kind= 'count',size=3,aspect=2)

sns.factorplot(data=df, x='Disciplinary failure', kind= 'count',size=3,aspect=2)
df.columns

#Checking Outliers in  data using boxplot

sns.boxplot(data=df[['Hit target','Age','Service time','Transportation expense',]])

fig=plt.gcf()

fig.set_size_inches(8,8)



#Checking Outliers in  data using boxplot

sns.boxplot(data=df[['Absenteeism time in hours','Body mass index','Height','Weight',]])

fig=plt.gcf()

fig.set_size_inches(8,8)
sns.boxplot(data=df[['Work load Average/day ','Distance from Residence to Work',]])

fig=plt.gcf()

fig.set_size_inches(8,8)
#Detecting outliers using boxplot and replacing with NA

for i in continuous_variables:

    q75, q25 = np.percentile(df[i],[75,25])

    

    # Calculating Interquartile range

    iqr = q75 - q25

    

    #calculating upper and lower fences

    minimum = q25 - (iqr*1.5)

    maximum = q75 + (iqr*1.5)

    

    #Replace all the outliers with NA

    df.loc[df[i]<minimum,i] = np.nan

    df.loc[df[i]>maximum,i] = np.nan

    

#Check for missing values

df.isnull().sum()
#Impute missing values withb knn

df = pd.DataFrame(KNN(k=3).fit_transform(df), columns = df.columns)
#Check for missing values after applying KNN

df.isnull().sum()
#checking once again for outliers in the data after applying KNN 

sns.boxplot(data=df[['Absenteeism time in hours','Body mass index','Height','Weight',]])

fig=plt.gcf()

fig.set_size_inches(8,8)
#checking once again for outliers in the data after applying KNN 

sns.boxplot(data=df[['Hit target','Age','Service time','Transportation expense',]])

fig=plt.gcf()

fig.set_size_inches(8,8)
#checking once again for outliers in the data after applying KNN 

sns.boxplot(data=df[['Work load Average/day ','Distance from Residence to Work',]])

fig=plt.gcf()

fig.set_size_inches(8,8)
##Correlation analysis

#Correlation plot

df_cor = df.loc[:,continuous_variables]
#Check for multicollinearity using corelation graph

#Set the width and hieght of the plot

f, ax = plt.subplots(figsize=(10,10))



#Generate correlation matrix

cor_mat = df_cor.corr()



#Plot using seaborn library

sns.heatmap(cor_mat, mask=np.zeros_like(cor_mat, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)

plt.plot()
#Getting copy of the data

df_old = df.copy()
df_old.shape
#Variable reduction

df_new = df.drop(['Body mass index'], axis = 1)
df_new.shape
#Check for Columns

df_new.columns
continuous_variables
#Updating columns in Continous_variable

continuous_variables.remove('Body mass index')

continuous_variables.remove('Absenteeism time in hours')
continuous_variables
#Make a copy of cleaned data

df_cleaned_data = df_new.copy()

#df_new = df_cleaned_data.copy()

df_cleaned_data.shape
df_new.shape
#Normality Check

for i in continuous_variables:

    plt.hist(df_new[i],bins='auto')

    plt.title("Checking Distribution for Variable "+str(i))

    plt.ylabel("Density")

    plt.xlabel(i)

    plt.show()
#Normalization of continous variables

for i in continuous_variables:

    

    df_new[i] = (df_new[i] - min(df_new[i]))/(max(df_new[i]) - min(df_new[i]))

df_new['Age'].describe()
#Dummy Variable creation for categorical variables

df_new = pd.get_dummies(data = df_new,columns=categorical_variables)
df_new.shape
#create a copy of dataframe

df_new_dummies = df_new.copy()
df_new_dummies.shape
#Splitting data into train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df_new.iloc[:,df_new.columns != 'Absenteeism time in hours' ],df_new.iloc[:, 8],test_size = 0.20, random_state = 1) 
#------------------Decision Tree------------------------#

#Importing libraries for Decision tree regressor

from sklearn.tree import DecisionTreeRegressor



#create a model decision tree using DecisionTreeRegressor

model_DT = DecisionTreeRegressor(random_state = 1).fit(X_train,Y_train)



#Predict for the test data

predictions_DT = model_DT.predict(X_test)



#Create separate dataframe for actual and predicted data

df_new_dt_pred = pd.DataFrame({'actual':Y_test,'predicted':predictions_DT})



print(df_new_dt_pred.head())



#Function to create to RMSE

def RMSE(y_actual,y_predicted):

    rmse = np.sqrt(mean_squared_error(y_actual,y_predicted))

    return rmse

#Calculate RMSE and R-Squared Value

print("RMSE: "+str(RMSE(Y_test, predictions_DT)))

print("R-Squared Value: "+str(r2_score(Y_test, predictions_DT)))
#--------------------Random Forest------------------------#

#Impoorting libraries for Random Forest

from sklearn.ensemble import RandomForestRegressor



#create a model Random forest using RandomForestRegressor

model_RF = RandomForestRegressor(n_estimators = 500, random_state = 1).fit(X_train,Y_train)



#predict for test data

predictions_RF = model_RF.predict(X_test)



#craete a dataframe for actual and predicted data

df_new_rf_pred = pd.DataFrame({'actual':Y_test,'predicted':predictions_RF})

print(df_new_rf_pred.head())



#calculate RMSE and RSquared values

print("RMSE: "+str(RMSE(Y_test, predictions_RF)))

print("R-Squared Value: "+str(r2_score(Y_test, predictions_RF)))
#-----------------------------Linear Regression----------------------------#

#Import libraries for Linear Regression

from sklearn.linear_model import LinearRegression



#Create model Linear Regression using LinearRegression

model_LR = LinearRegression().fit(X_train,Y_train)



#Predict for the test cases

predictions_LR = model_LR.predict(X_test)



#Create a separate dataframee for the actual and predicted data

df_new_lr_pred = pd.DataFrame({'actual':Y_test,'predicted':predictions_LR})



print(df_new_lr_pred.head())



#Calculate RMSE and RSquared values

print("RMSE: "+str(RMSE(Y_test, predictions_LR)))

print("R-Squared Value: "+str(r2_score(Y_test, predictions_LR)))

#Get a target variable

target_variable = df_new['Absenteeism time in hours']

df_new.shape
#import library for PCA

from sklearn.decomposition import PCA



#Converting data into numpy array

X = df_new.values



pca = PCA(n_components = 115)

pca.fit(X)



#Proportion of variance

var = pca.explained_variance_ratio_



#Calculate Screen plot

var1 = np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100)



#Draw the plot

plt.plot(var1)

plt.show()
#Selecting 45 Components since it explains almost 95+ % data variance

pca = PCA(n_components=45)



#Fitting the selected components to the data

pca.fit(X)



#Splitting data into train and test

X_train, X_test, Y_train, Y_test = train_test_split(X,target_variable, test_size=0.2, random_state = 1)

#------------------Decision Tree------------------------#

#Importing libraries for Decision tree regressor

from sklearn.tree import DecisionTreeRegressor



#create a model decision tree using DecisionTreeRegressor

model_DTP = DecisionTreeRegressor(random_state = 1).fit(X_train,Y_train)



#Predict for the test data

predictions_DTP = model_DTP.predict(X_test)



#Create separate dataframe for actual and predicted data

df_new_dtp_pred = pd.DataFrame({'actual':Y_test,'predicted':predictions_DTP})



print(df_new_dtp_pred.head())



#Function to create to RMSE

def RMSE(y_actual,y_predicted):

    rmse = np.sqrt(mean_squared_error(y_actual,y_predicted))

    return rmse



#Calculate RMSE and R-Squared Value

print("RMSE: "+str(RMSE(Y_test, predictions_DTP)))

print("R-Squared Value: "+str(r2_score(Y_test, predictions_DTP)))
#--------------------Random Forest------------------------#

#Impoorting libraries for Random Forest

from sklearn.ensemble import RandomForestRegressor



#create a model Random forest using RandomForestRegressor

model_RFP = RandomForestRegressor(n_estimators = 500, random_state = 1).fit(X_train,Y_train)



#predict for test data

predictions_RFP = model_RFP.predict(X_test)



#craete a dataframe for actual and predicted data

df_new_rfp_pred = pd.DataFrame({'actual':Y_test,'predicted':predictions_RFP})

print(df_new_rfp_pred.head())



#calculate RMSE and RSquared values

print("RMSE: "+str(RMSE(Y_test, predictions_RFP)))

print("R-Squared Value: "+str(r2_score(Y_test, predictions_RFP)))
#-----------------------------Linear Regression----------------------------#

#Import libraries for Linear Regression

from sklearn.linear_model import LinearRegression



#Create model Linear Regression using LinearRegression

model_LRP = LinearRegression().fit(X_train,Y_train)



#Predict for the test cases

predictions_LRP = model_LRP.predict(X_test)



#Create a separate dataframee for the actual and predicted data

df_new_lrp_pred = pd.DataFrame({'actual':Y_test,'predicted':predictions_LRP})



print(df_new_lrp_pred.head())



#Calculate RMSE and RSquared values

print("RMSE: "+str(RMSE(Y_test, predictions_LRP)))

print("R-Squared Value: "+str(r2_score(Y_test, predictions_LRP)))
