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
#importing usual libraries

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#importing dataset csv to pandas dataframe



automobile = pd.read_csv("/kaggle/input/CarPrice_Assignment.csv")

automobile.head()
#checking number of rows and columns



automobile.shape
#checking dtypes and null values of columns



automobile.info()
#checking summary of numeric variables



automobile.describe()
#checking number of columns of each data type for general EDA



automobile.dtypes.value_counts()
#cleaning Car Name to keep only brand(company) name and remove model names 



automobile['CarName']=automobile['CarName'].apply(lambda x:x.split(' ', 1)[0])

automobile.rename(columns = {'CarName':'companyname'}, inplace = True)

automobile.head()
#checking unique values in company name column



automobile.companyname.unique()
#counting number of unique company names



automobile.companyname.nunique()
# Fixing values in company name



automobile.companyname = automobile.companyname.str.lower()



def replace_name(a,b):

    automobile.companyname.replace(a,b,inplace=True)



replace_name('maxda','mazda')

replace_name('porcshce','porsche')

replace_name('toyouta','toyota')

replace_name('vokswagen','volkswagen')

replace_name('vw','volkswagen')



automobile.companyname.unique()
#counting number of unique company names



automobile.companyname.nunique()
#plotting count of company names



plt.figure(figsize=(30, 8))

plt1=sns.countplot(x=automobile.companyname, data=automobile, order= automobile.companyname.value_counts().index)

plt.title('Company Wise Popularity', size=14)

plt1.set_xlabel('Car company', fontsize=14)

plt1.set_ylabel('Frequency of Car Body', fontsize=14)

plt1.set_xticklabels(plt1.get_xticklabels(),rotation=360, size=14)

plt.show()
#plotting company wise average price of car



plt.figure(figsize=(30, 6))



df = pd.DataFrame(automobile.groupby(['companyname'])['price'].mean().sort_values())

df=df.reset_index(drop=False)

plt1=sns.barplot(x="companyname", y="price", data=df)

plt1.set_title('Car Range vs Average Price', size=14)

plt1.set_xlabel('Car company', fontsize=14)

plt1.set_ylabel('Price', fontsize=14)

plt1.set_xticklabels(plt1.get_xticklabels(),rotation=360, size=14)

plt.show()
#Binning the Car Companies based on avg prices of each Company.



def replace_values(a,b):

    automobile.companyname.replace(a,b,inplace=True)



replace_values('chevrolet','Low_End')

replace_values('dodge','Low_End')

replace_values('plymouth','Low_End')

replace_values('honda','Low_End')

replace_values('subaru','Low_End')

replace_values('isuzu','Low_End')

replace_values('mitsubishi','Budget')

replace_values('renault','Budget')

replace_values('toyota','Budget')

replace_values('volkswagen','Budget')

replace_values('nissan','Budget')

replace_values('mazda','Budget')

replace_values('saab','Medium')

replace_values('peugeot','Medium')

replace_values('alfa-romero','Medium')

replace_values('mercury','Medium')

replace_values('audi','Medium')

replace_values('volvo','Medium')

replace_values('bmw','High_End')

replace_values('porsche','High_End')

replace_values('buick','High_End')

replace_values('jaguar','High_End')



automobile.rename(columns = {'companyname':'segment'}, inplace = True)

automobile.head()
## FUNCTION TO PLOT CHARTS



def plot_charts(var1, var2):

    plt.figure(figsize=(15, 10))   

    plt.subplot(2,2,1)

    plt.title('Histogram of '+ var1)

    sns.countplot(automobile[var1], palette=("husl"))

    plt1.set(xlabel = '%var1', ylabel='Frequency of'+ '%s'%var1)

    

    plt.subplot(2,2,2)

    plt.title(var1+' vs Price')

    sns.boxplot(x=automobile[var1], y=automobile.price, palette=("husl"))

    

    plt.subplot(2,2,3)

    plt.title('Histogram of '+ var2)

    sns.countplot(automobile[var2], palette=("husl"))

    plt1.set(xlabel = '%var2', ylabel='Frequency of'+ '%s'%var2)

    

    plt.subplot(2,2,4)

    plt.title(var1+' vs Price')

    sns.boxplot(x=automobile[var2], y=automobile.price, palette=("husl"))

    

    plt.show()   
plot_charts('symboling', 'fueltype')
plot_charts('aspiration', 'doornumber')
plot_charts('drivewheel', 'carbody')
plot_charts('enginelocation', 'enginetype')
plot_charts('cylindernumber', 'fuelsystem')
#checking distribution and spread of car price



plt.figure(figsize=(20,6))



plt.subplot(1,2,1)

plt.title('Car Price Distribution Plot')

sns.distplot(automobile.price)



plt.subplot(1,2,2)

plt.title('Car Price Spread')

sns.boxplot(y=automobile.price)



plt.show()
# checking numeric columns



automobile.select_dtypes(include=['float64','int64']).columns
#function to plot scatter plot numeric variables with price



def pp(x,y):

    sns.pairplot(automobile, x_vars=[x,y], y_vars='price',height=4, aspect=1, kind='scatter')

    plt.show()



pp('carlength', 'carwidth')

pp('carwidth', 'curbweight')
#function to plot scatter plot numeric variables with price



def pp(x,y,z):

    sns.pairplot(automobile, x_vars=[x,y,z], y_vars='price',height=4, aspect=1, kind='scatter')

    plt.show()



pp('wheelbase', 'compressionratio', 'enginesize')

pp('boreratio', 'horsepower', 'peakrpm')

pp('stroke', 'highwaympg', 'citympg')
#converting cylinder number to numeric and replacing values



def replace_values(a,b):

    automobile.cylindernumber.replace(a,b,inplace=True)



replace_values('four','4')

replace_values('six','6')

replace_values('five','5')

replace_values('three','3')

replace_values('twelve','12')

replace_values('two','2')

replace_values('eight','8')



automobile.cylindernumber=automobile.cylindernumber.astype('int')
automobile.symboling.unique()
#converting symboling to categorical because the numeric values imply weight



def replace_values(a,b):

    automobile.symboling.replace(a,b,inplace=True)



replace_values(3,'Very_Risky')

replace_values(2,'Moderately_Risky')

replace_values(1,'Neutral')

replace_values(0,'Safe')

replace_values(-1,'Moderately_Safe')

replace_values(-2,'Very_Safe')
# Converting variables with 2 values to 1 and 0



automobile['fueltype'] = automobile['fueltype'].map({'gas': 1, 'diesel': 0})

automobile['aspiration'] = automobile['aspiration'].map({'std': 1, 'turbo': 0})

automobile['doornumber'] = automobile['doornumber'].map({'two': 1, 'four': 0})

automobile['enginelocation'] = automobile['enginelocation'].map({'front': 1, 'rear': 0})
#dropping card_Id because it has all unique values



automobile.drop(['car_ID'], axis =1, inplace = True)
#numeric variables



num_vars=automobile.select_dtypes(include=['float64','int64']).columns
# plotting heatmap to check correlation amongst variables



plt.figure(figsize = (20,10))  

sns.heatmap(automobile[num_vars].corr(),cmap="YlGnBu",annot = True)
#dropping variables which are highly correlated to other variables



automobile.drop(['compressionratio','carwidth','curbweight','wheelbase','citympg'], axis =1, inplace = True)
automobile.head()
#getting dummies for categorical variables



df = pd.get_dummies(automobile)

df.head()
#checking column names for dummy variables



df.columns
# importing necessary libraries and functions



from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively



df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
# for scaling



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables



num_vars = ['fueltype', 'aspiration', 'doornumber', 'enginelocation', 'enginesize','horsepower', 

            'peakrpm', 'highwaympg', 'carlength', 'carheight', 'boreratio', 'stroke', 'price']





df_train[num_vars] = scaler.fit_transform(df_train[num_vars])



df_train.head()
#dividing into x and y sets where y has the variable we have to predict



y_train = df_train.pop('price')

X_train = df_train
# Importing RFE and LinearRegression



from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 10)             # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
#checking RFE columns

col = X_train.columns[rfe.support_]

col
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
#function for checking VIF



def checkVIF(X):

    vif = pd.DataFrame()

    vif['variable'] = X.columns    

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return(vif)
# building MODEL #1



lm = sm.OLS(y_train,X_train_rfe).fit() # fitting the model

print(lm.summary()) # model summary
#dropping constant to calculate VIF



X_train_rfe.drop('const', axis = 1, inplace=True)
#checking VIF



checkVIF(X_train_rfe)
#dopping boreratio because it has the highest p-value and also high VIF. It is also something which is difficult to explain to management



X_train_new = X_train_rfe.drop(["boreratio"], axis = 1)
#building MODEL #2 after dropping boreratio



X_train_new = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_new).fit() # fitting the model

print(lm.summary()) # model summary
#dropping constant to calculate VIF



X_train_new.drop('const', axis=1, inplace=True)
#checking VIF



checkVIF(X_train_new)
#dopping enginelocation because it has the highest p-value and also high VIF. it has very few values for rear as we saw earlier



X_train_new.drop(["enginelocation"], axis=1, inplace=True)
#building MODEL #3 after dropping enginelocation



X_train_new = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_new).fit() # fitting the model

print(lm.summary()) # model summary
#dropping constant to calculate VIF



X_train_new.drop('const', axis=1, inplace=True)
#checking VIF



checkVIF(X_train_new)
#dopping horsepower because it has the high VIF and exhibits multicollinearity. 

#it is highly correlated to engine size and can be dropped.



X_train_new.drop(["horsepower"], axis=1, inplace=True)
#building MODEL #4 after dropping horsepower



X_train_new = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_new).fit() # fitting the model

print(lm.summary()) # model summary
#dropping constant to calculate VIF



X_train_new.drop('const', axis=1, inplace=True)
#checking VIF



checkVIF(X_train_new)
#dopping carlength because it has the high VIF and exhibits multicollinearity. 

#it is highly correlated to engine size and can be dropped.



X_train_new.drop(["carlength"], axis=1, inplace=True)
#building MODEL #5 after dropping carlength



X_train_new = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_new).fit() # fitting the model

print(lm.summary()) # model summary
#dropping constant to calculate VIF



X_train_vif=X_train_new.drop('const', axis=1)
#checking VIF



checkVIF(X_train_vif)
#calculating price on train set using the model built



y_train_price = lm.predict(X_train_new)
# Plot the histogram of the error terms



fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
# Plotting y_train and y_train_price to understand the residuals.



plt.figure(figsize = (8,6))

plt.scatter(y_train,y_train_price)

plt.title('y_train vs y_train_price', fontsize=20)              # Plot heading 

plt.xlabel('y_train', fontsize=18)                          # X-label

plt.ylabel('y_train_price', fontsize=16)                          # Y-label
# Actual vs Predicted for TRAIN SET



plt.figure(figsize = (8,5))

c = [i for i in range(1,144,1)]

d = [i for i in range(1,144,1)]

plt.plot(c, y_train_price, color="blue", linewidth=1, linestyle="-")     #Plotting Actual

plt.plot(d, y_train, color="red",  linewidth=1, linestyle="-")  #Plotting predicted

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('Car Price', fontsize=16)  

plt.show()
# Error terms for TRAIN SET

plt.figure(figsize = (8,5))

c = [i for i in range(1,144,1)]

plt.scatter(c,y_train-y_train_price)



plt.title('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
# Applying the scaling on the test sets



num_vars = ['fueltype', 'aspiration', 'doornumber', 'enginelocation', 'enginesize','horsepower', 

            'peakrpm', 'highwaympg', 'carlength', 'carheight', 'boreratio', 'stroke', 'price']



df_test[num_vars] = scaler.transform(df_test[num_vars])
# Dividing into X_test and y_test



y_test = df_test.pop('price')

X_test = df_test
X_train_new.drop('const', axis=1, inplace=True)
# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label
# Actual vs Predicted

c = [i for i in range(1,63,1)]

d = [i for i in range(1,63,1)]

plt.plot(c, y_pred, color="blue", linewidth=1, linestyle="-")     #Plotting Actual

plt.plot(d, y_test, color="red",  linewidth=1, linestyle="-")  #Plotting predicted

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('Car Price', fontsize=16)  

plt.show()
# Error terms



fig = plt.figure()

c = [i for i in range(1,63,1)]

plt.scatter(c,y_test-y_pred)



fig.suptitle('Error Terms', fontsize=20)              # Plot heading 

plt.xlabel('Index', fontsize=18)                      # X-label

plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
#RMSE score for test set



import numpy as np

from sklearn import metrics

print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#RMSE score for train set



import numpy as np

from sklearn import metrics

print('RMSE :', np.sqrt(metrics.mean_squared_error(y_train, y_train_price)))
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
r2_score(y_train, y_train_price)