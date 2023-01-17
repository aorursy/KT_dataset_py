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
df=pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

df.head()
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder,StandardScaler



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df.info()

# only total_bedrooms have null values
mode=df['total_bedrooms'].mode()[0]

df['total_bedrooms'].fillna(mode,inplace=True)

# fill the bedrooms with most occurence of frequency of rooms bcoz mean value will be a decimal value and i dont wanted that 

any(df['total_bedrooms'].isnull())

#False shows there are no missing values left in total_bedrooms
df['ocean_proximity'].value_counts()
label=LabelEncoder()

df['ocean_proximity']=label.fit_transform(df['ocean_proximity'])

df.head(5)
df.describe()
figure, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))



df['housing_median_age'].plot(kind='hist',ax=ax1)



df['housing_median_age'].plot(kind='box', ax=ax2)



plt.tight_layout()



plt.show()
df.plot(kind='scatter',x='longitude',y='latitude')



plt.show()
figure, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))



df.plot(kind='scatter',x='median_house_value',y='total_rooms',ax=ax1)



df.plot(kind='scatter',x='median_house_value',y='total_bedrooms', ax=ax2)



plt.tight_layout()



plt.show()
corr_matrix=df.corr()



f, ax = plt.subplots(figsize=(11, 15))



heatmap = sns.heatmap(corr_matrix,

                      mask = np.triu(corr_matrix),

                      square = True,

                      linewidths = .5,

                      cmap ='coolwarm', 

                      cbar_kws = {'shrink': .4,'ticks' : [-1, -.5, 0, 0.5, 1]},

                      vmin = -1,

                      vmax = 1,

                      annot = True,

                      annot_kws = {"size": 12})



#add the column names as labels

ax.set_yticklabels(corr_matrix.columns, rotation = 0)

ax.set_xticklabels(corr_matrix.columns)



sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
input_values = df[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']]

output_values = df[['median_house_value']]



train_input,test_input,train_output,test_output=train_test_split(input_values,output_values,test_size=0.1,random_state=32)
model=LinearRegression()

model.fit(train_input,train_output)

model.score(test_input,test_output)
input_values = df[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']]

output_values = df[['median_house_value']]



sc=StandardScaler()

input_values=sc.fit_transform(input_values)

output_values=sc.fit_transform(output_values)



train_input,test_input,train_output,test_output=train_test_split(input_values,output_values,test_size=0.1,random_state=32)
model_svr=SVR(kernel='rbf',degree=2,C=10,verbose=3)

model_svr.fit(train_input,train_output)

model_svr.score(test_input,test_output)
input_values = df[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']]

output_values = df[['median_house_value']]



sc=StandardScaler()

input_values=sc.fit_transform(input_values)

output_values=sc.fit_transform(output_values)



train_input,test_input,train_output,test_output=train_test_split(input_values,output_values,test_size=0.1,random_state=32)
model_grad=GradientBoostingRegressor(max_depth= 8, max_features=6, min_samples_split=200, n_estimators=100,random_state=42)

model_grad.fit(train_input,train_output)

model_grad.score(test_input,test_output)