# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Data Visualization libraries import

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Code for importing files and libraries

import os

output_path = os.path.abspath('/kaggle/output')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Read the file from the directory

df = pd.read_csv("/kaggle/input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv")
#Understanding the dataset in detail

df.info()
#Changing the names of the columns

df= df.rename(columns={'Cumulative number of case(s)': 'Cumulative_Cases', 'Number of deaths': 'Death_Count', 

                      'Number recovered': 'Recovered_Count'})
#Converting Date feature first to the datetime feature

df.Date = df.Date.apply(pd.to_datetime)
plt.figure(figsize = [10,5])

plt.title('Distribution of Confirmed Cases vs Death for SARS 2003')



sns.scatterplot(x = df['Cumulative_Cases'], y=df['Death_Count'])



plt.xlabel('Cumulative number of cases')

plt.ylabel("Deaths occured")
plt.figure(figsize = [10,5])

plt.title('Distribution of Confirmed Cases vs Death for SARS 2003')



sns.scatterplot(x = df['Cumulative_Cases'], y=df['Death_Count'], hue = df['Recovered_Count'])



plt.xlabel('Cumulative number of cases')

plt.ylabel("Deaths occured")
plt.figure(figsize = [10,5])

plt.title('Regression Line Distribution of Confirmed Cases vs Death for SARS 2003')



sns.regplot(x = df['Cumulative_Cases'], y=df['Death_Count'])



plt.xlabel('Cumulative number of cases')

plt.ylabel("Deaths occured")
# Performing feature engineering and extracting details from Date feature. 

# Since the entire dataset is of year 2003, so we are not considering year. 

# Only, we are extracting month and date values.

# It is important to convert the new feature into type int 



df['Day_of_the_year'] = df.Date.dt.strftime("%d").astype(int)

df['Week_of_the_year'] = df.Date.dt.strftime("%w").astype(int)

df['Month_of_the_year'] = df.Date.dt.strftime("%m").astype(int)

df.drop(['Date'], inplace = True, axis = 1)
obj_type_features = df.select_dtypes(include = "object").columns

print (obj_type_features)

print (df.Country) #Before Encoding
# We only have one feature - Country. So, now let's use Label Encoding.



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



#Now we will transform the data of feature Country

df.Country = le.fit_transform(df.Country)



#Now print and check Country feature values after encoding

print (df.Country)
from sklearn.model_selection import train_test_split



#Let us assume that we are predicting 

X = df.drop(['Death_Count'], axis = 1)

y = df['Death_Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#import the model class from sklearn 



from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
from sklearn import metrics



print ("Mean Absolute Error (MAE) - Test data : ", metrics.mean_absolute_error(y_test, y_pred))



print ("Mean Squared Error (MSE) - Test data : ", metrics.mean_squared_error(y_test, y_pred))



print ("Root Mean Squared Error (RMSE) - Test data : ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



print("Co-efficient of determination (R2 Score): ", metrics.r2_score(y_test, y_pred))
