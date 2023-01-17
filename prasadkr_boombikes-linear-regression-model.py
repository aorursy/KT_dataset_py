# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
bikeSharing = pd.read_csv('/kaggle/input/boombikedata/day.csv')

bikeSharing.head()
bikeSharing.shape
bikeSharing.info()
# percentage of missing values in each column

round(100*(bikeSharing.isnull().sum()/len(bikeSharing)), 2).sort_values(ascending=False)
bikeSharing.describe()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

# setting the style for seaborn plots

%matplotlib inline
#Dropping instant and dteday since they dont have significance with data

bikeSharing.drop(['instant','dteday','casual','registered'],inplace=True,axis=1)

bikeSharing.head()
bikeSharing['season'] = bikeSharing['season'].map({1:'spring',2:'summer', 3:'fall', 4:'winter'})

bikeSharing['mnth'] = bikeSharing['mnth'].map({1:'Jan',2:'Feb', 3:'Mar', 4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sept',10:'Oct',11:'Nov',12:'Dec'})

bikeSharing['weekday'] = bikeSharing['weekday'].map({0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday'})

bikeSharing['weathersit'] = bikeSharing['weathersit'].map({1:'Clear-Partlycloudy',2:'Mist-Cloudy',3:'LightSnow-lightRain-Thunderstorm',4:'HeavyRain-IcePallets-Thunderstorm'})



bikeSharing.head()



bikeSharing.info()
def boxplot_cat_var(cat_var,target):

    plt.figure(figsize=(20, 12))

    for i in range(0,len(cat_var)):

        plt.subplot(2,3,i+1)

        sns.boxplot(x = cat_var[i], y = target, data = bikeSharing)

    plt.show()



cat_var =['season','yr','holiday','weekday','workingday','weathersit']

boxplot_cat_var(cat_var,'cnt')
sns.boxplot(x = 'mnth', y = 'cnt', data = bikeSharing)
# Defining the map function

def dummies(x,df):

    temp = pd.get_dummies(df[x], drop_first = True)

    df = pd.concat([df, temp], axis = 1)

    df.drop([x], axis = 1, inplace = True)

    return df

# Applying the function to the bikeSharing



bikeSharing = dummies('season',bikeSharing)

bikeSharing = dummies('mnth',bikeSharing)

bikeSharing = dummies('weekday',bikeSharing)

bikeSharing = dummies('weathersit',bikeSharing)

bikeSharing.head()
bikeSharing.shape
bikeSharing.describe()
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)



df_train, df_test = train_test_split(bikeSharing, train_size = 0.7, test_size = 0.3, random_state = 100)
df_train.shape
df_test.shape
# we can see patterns between variables 

sns.pairplot(df_train[[ 'temp','atemp', 'hum', 'windspeed','cnt']],diag_kind='kde')

plt.show()
#Correlation using heatmap

plt.figure(figsize = (30, 25))

sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Apply scaler() to all the columns except 'dummy' variables

num_vars = ['temp','atemp', 'hum', 'windspeed', 'cnt']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])



df_train.head()
df_train.describe()
y_train = df_train.pop('cnt')

X_train = df_train
#importing libs for RFE

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Running RFE with the output number of the variable equal to 15

lm = LinearRegression()

lm.fit(X_train,y_train)

rfe = RFE(lm, 15)

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
X_train_rfe = X_train[X_train.columns[rfe.support_]]

X_train_rfe.head()
def build_model(X,y):

    X = sm.add_constant(X) #Adding the constant

    lm = sm.OLS(y,X).fit() # fitting the model

    print(lm.summary()) # model summary

    return X

    

def checkVIF(X):

    vif = pd.DataFrame()

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return(vif)



X_train_new = build_model(X_train_rfe,y_train)
#Calculating the Variance Inflation Factor

checkVIF(X_train_new)
X_train_new=X_train_new.drop(["Jan"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#Calculating the Variance Inflation Factor

checkVIF(X_train_new)
X_train_new=X_train_new.drop(["holiday"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#Calculating the Variance Inflation Factor

checkVIF(X_train_new)
X_train_new=X_train_new.drop(["spring"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#Calculating the Variance Inflation Factor

checkVIF(X_train_new)
X_train_new=X_train_new.drop(["Jul"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#Calculating the Variance Inflation Factor

checkVIF(X_train_new)
#Correlation using heatmap

plt.figure(figsize = (30, 25))

sns.heatmap(X_train_new.corr(), annot = True, cmap="YlGnBu")

plt.show()
X_train_new=X_train_new.drop(["workingday"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
X_train_new=X_train_new.drop(["Saturday"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
#Calculating the Variance Inflation Factor

checkVIF(X_train_new)
lm = sm.OLS(y_train,X_train_new).fit()

y_train_cnt= lm.predict(X_train_new)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_cnt), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)   
#Scaling the test set

num_vars = ['temp','atemp', 'hum', 'windspeed', 'cnt']

df_test[num_vars] = scaler.fit_transform(df_test[num_vars])
#Dividing into X and y

y_test = df_test.pop('cnt')

X_test = df_test
# Now let's use our model to make predictions.

X_train_new = X_train_new.drop('const',axis=1)

# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
from sklearn.metrics import r2_score 

r2=r2_score(y_test, y_pred)

print(r2)
X_test_new.shape
# We already have the value of R^2 (calculated in above step)

# n is number of rows in X

n = X_test_new.shape[0]



# Number of features (predictors, p) is the shape along axis 1

p = X_test_new.shape[1]



# We find the Adjusted R-squared using the formula



adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

adjusted_r2
#EVALUATION OF THE MODEL

# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)   
print(lm.summary())