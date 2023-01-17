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
import pandas as pd

housing_data = pd.read_csv("../input/housing/Housing.csv")
housing_data.head()
housing_data['mainroad'] = housing_data['mainroad'].map({'yes':1,'no':0})

housing_data['guestroom'] = housing_data['guestroom'].map({'yes':1,'no':0})

housing_data['basement'] = housing_data['basement'].map({'yes':1,'no':0})

housing_data['hotwaterheating'] = housing_data['hotwaterheating'].map({'yes':1,'no':0})

housing_data['airconditioning'] = housing_data['airconditioning'].map({'yes':1,'no':0})

housing_data['prefarea'] = housing_data['prefarea'].map({'yes':1,'no':0})
status = pd.get_dummies(housing_data["furnishingstatus"])
status.head()
status = pd.get_dummies(housing_data["furnishingstatus"],drop_first=True)
status.head()
housing_data = pd.concat([housing_data,status],axis=1)
housing_data.head()
housing_data.drop(["furnishingstatus"],axis=1,inplace=True)
housing_data.head()
housing_data['areaperbedroom'] = housing_data['area']/housing_data['bedrooms']
housing_data['bbratio'] = housing_data['bathrooms']/housing_data['bedrooms']
housing_data.head()
# Definig a normalization function

def normalize(x):

    return ((x-np.min(x))/(max(x) - min(x)))



# applying normalize () to all columns

housing_data = housing_data.apply(normalize)
housing_data.head()
housing_data.columns
x = housing_data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',

       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',

       'parking', 'prefarea', 'semi-furnished', 'unfurnished',

       'areaperbedroom', 'bbratio']]
y = housing_data[['price']]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=100)
import statsmodels.api as sm

x_train = sm.add_constant(x_train)

lm_1 = sm.OLS(y_train,x_train).fit()
print(lm_1.summary())
# Checking VIF

def vif_cal(input_data, dependent_col):

    vif_df = pd.DataFrame(columns=['Var','Vif'])

    x_vars = input_data.drop([dependent_col],axis=1)

    xvar_names = x_vars.columns

    for i in range(0,xvar_names.shape[0]):

        y = x_vars[xvar_names[i]]

        x = x_vars[xvar_names.drop(xvar_names[i])]

        rsq = sm.OLS(y,x).fit().rsquared

        vif = round(1/(1-rsq),2)

        vif_df.loc[i] = [xvar_names[i],vif]

    return vif_df.sort_values(by='Vif',axis=0,ascending=False,inplace=False)
# Calculating vif value 

vif_cal(input_data=housing_data,dependent_col='price')
# Correlation matrix

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
# Let's see the correlation matrix

plt.figure(figsize=(16,10))

sns.heatmap(housing_data.corr(),annot=True)
# Dropping the variable and updating the model

x_train = x_train.drop('bbratio',1)
# craete a second fitted model

lm_2 = sm.OLS(y_train,x_train).fit()
# Let's see the summary of our second linear model

print(lm_2.summary())
# Dropping the variable and Updating the model

vif_cal(input_data=housing_data.drop(["bbratio"],axis=1),dependent_col="price")
# Dropping highly correlated variable and insignificant variables 

x_train = x_train.drop("bedrooms",1)
# Create a third fitted model

lm_3 = sm.OLS(y_train,x_train).fit()
# Let's see the summary of third linear model

print(lm_3.summary())
# calculating vif value

vif_cal(input_data=housing_data.drop(['bedrooms','bbratio'],axis=1),dependent_col='price')
# Dropping the variable and updating the model

x_train = x_train.drop('areaperbedroom',1)
# create a fourth fitted model

lm_4 = sm.OLS(y_train,x_train).fit()
print(lm_4.summary())
# Dropping the variable. and updating the model

x_train = x_train.drop('semi-furnished',1)
lm_5 = sm.OLS(y_train,x_train).fit()
print(lm_5.summary())
vif_cal(input_data=housing_data.drop(['bedrooms','bbratio','areaperbedroom','semi-furnished'],axis=1),dependent_col='price')
# Dropping the variable and updating the model

x_train = x_train.drop('basement',1)
# create a 6th fitted model

lm_6 = sm.OLS(y_train,x_train).fit()
print(lm_6.summary())
# Calculate VIF value 

vif_cal(input_data=housing_data.drop(['bedrooms','bbratio','areaperbedroom','semi-furnished','basement'],axis=1),dependent_col='price')
# Prediction using final model

x_test_m6 = sm.add_constant(x_test)
# Creating x_test _m6 dataframe by dropping variables from x_test_m6

x_test_m6 = x_test_m6.drop(['bedrooms','bbratio','areaperbedroom','semi-furnished','basement'],axis=1)
# making prediction

y_pred_m6 = lm_6.predict(x_test_m6)
# Model Evaluation

# Actual vs Predicted

c = [i for i in range(1,165,1)]

fig = plt.figure()

plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-") # Plotting actual

plt.plot(c,y_pred_m6,color="red",linewidth=2.5,linestyle="-") # Plotting predicted

fig.suptitle("Actual Vs Predicted",fontsize=20)

plt.xlabel('Index',fontsize=18)

plt.ylabel('Housing Price',fontsize=16)
# plotting y_test and y_pred

fig = plt.figure()

plt.scatter(y_test,y_pred_m6)

fig.suptitle('y_test vs y_pred',fontsize=20)

plt.xlabel('y_test',fontsize=18)

plt.ylabel('y_pred',fontsize=16)
from sklearn import metrics

print("RMSE: ",np.sqrt(metrics.mean_squared_error(y_test,y_pred_m6)))