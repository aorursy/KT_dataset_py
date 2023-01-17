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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Reading The Data File
df = pd.read_csv('/kaggle/input/insurance/insurance.csv') #reading through pandas liberary
pd.set_option('Display.max_rows',1339) #havin the entire DataFrame
df['added_ones'] = df['age'] + df['children'] + df['bmi'] + df['smoker']
#Fortunately not having a null value in any column/series
df[df.duplicated()] #havin a duplicate row/index
v = df.drop(df.index[581]) #removing a duplicate row/index
df.describe() #getting some info of numerical data
# About Smokers 
df['smoker'].value_counts() #getting the value of smokers/not smokers
df['smoker'].replace('yes' , 1 , inplace = True)
df['smoker'].replace('no' , 0 , inplace = True)
smoking_yes = df[df['smoker']== 1] #havin the data of only smokers
smoking_no = df[df['smoker'] == 0]#havin the data who are not smokers
#i have changed the value of column smokers..the one who smokes = 1
# the one who does not do that = 0
sns.set(style = 'darkgrid')
sns.countplot(x= 'smoker' , data = df)
# About Old Age People
#average age is 39 years which is really good
old_age = df[df['age'] >= 60] #getting the data whose age is more than 60 years
# if their age more than 60 years then they might affect
bins = [10 , 20 , 30 , 40 , 50 , 60 , 70]
plt.hist(df['age'] ,color = 'cyan', edgecolor = 'black',bins=bins)
plt.xlabel('Age Ranges')
plt.ylabel('Values')
plt.title('Age Distribution')
plt.show()
# About bmi
#the average of bmi is 30 which is not good
high_bmi = df[df['bmi'] > 24.9] #i have taken higher than 24.9 just because it should be between 18.5 to 24.9
plt.hist(df['bmi'] , color = 'grey')
plt.title('bmi Distribution')
plt.xlabel('bmi Ranges')
plt.ylabel('Values')
plt.show()
# About Regions
df['region'].value_counts() #each region has almost similar values
sns.countplot(x = ['southeast' , 'northwest' , 'southwest', 'northeast'])
# Every regions have almost same amount of people so that won't affect that much
# Machine Learning Section
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
lin_reg = LinearRegression()
labelencoder = LabelEncoder()
sns.heatmap(df.corr()) #Visualization By HeatMap
X = df.loc[: , 'added_ones'].values
Y = df.loc[: , 'charges'].values
X = X.reshape(1338,1) # We have to reshape X to have it in 2D array
lin_reg.fit(X,Y) #Fitting X & Y into LinearRegression model
Ypred = lin_reg.predict(X) #Prediction of charges
r2 = lin_reg.score(X,Y) #Getting the prediction by r-squared method..if it's 
                        # closer to 1 then our prediction is good 
print(r2)
plt.style.use('fivethirtyeight')
plt.scatter(X,Y ,color = 'blue'  )
plt.plot(X,Ypred , color = 'r' )
plt.title('Insurance Cost Prediction')
plt.tight_layout()
plt.show()
#By Observing , Prediction is insurance company's will be very less b'coz our r-squared score is 0.13 
# and that is very far from 1.

# So this is my first kaggle submission i hope you like it..Thank You