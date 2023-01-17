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
df=pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/calendar.csv')

df
df1=pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv')

df1
new_df=pd.merge(left=df,right=df1,how='left',left_on='wm_yr_wk',right_on='wm_yr_wk')

new_df
pd.isnull(new_df).sum()
new_df.info()
print("Uniques values of Event Name 1:",new_df['event_name_1'].unique())

print("Uniques values of Event Name 2:",new_df['event_name_2'].unique())

print("Uniques values of Event type 1:",new_df['event_type_1'].unique())

print("Uniques values of Event type 2:",new_df['event_type_2'].unique())
#new_df1=new_df.dropna(subset=['event_name_1','event_name_2','event_type_1','event_type_2'],how='all')

new_df1=new_df.dropna()

new_df1
new_df1.count()

new_df1.drop(['weekday'],axis=1)
new_df1_processed = pd.get_dummies(new_df1, prefix_sep="_",

                              columns=['event_type_1','event_name_1','event_name_2','event_type_2'],drop_first=True)

new_df1_processed 
new_df1_processed.drop(['weekday'],axis=1,inplace=True)

new_df1_processed = pd.get_dummies(new_df1_processed, prefix_sep="_",

                              columns=['store_id'],drop_first=True)

new_df1_processed
new_df1_processed['item_id'] = [id[ : -6] for id in new_df1_processed['item_id']] 

new_df1_processed
new_df1_processed = pd.get_dummies(new_df1_processed, prefix_sep="_",

                              columns=['item_id'],drop_first=True)

new_df1_processed
new_df1_processed['d']= [id[ 2: ] for id in new_df1_processed['d']] 

new_df1_processed
new_df1_processed.dtypes
corr=new_df1_processed.corr()

corr
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 420, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);

plt.subplots(figsize=(30,20))
new_df1_processed['d']=pd.to_numeric(new_df1_processed['d'])

print(new_df1_processed.dtypes)

X=new_df1_processed.drop(['sell_price','date'],axis=1)

y=new_df1_processed['sell_price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=42)

from sklearn.linear_model import LinearRegression

 

# instantiate the regressor class

regressor = LinearRegression()



# fit the build the model by fitting the regressor to the training data

regressor.fit(X_train, y_train)



# make a prediction set using the test set

prediction = regressor.predict(X_test)



# Evaluate the prediction accuracy of the model

from sklearn.metrics import mean_absolute_error, median_absolute_error

print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))

print("The Mean Absolute Error: %.2f price" % mean_absolute_error(y_test, prediction))

print("The Median Absolute Error: %.2f price" % median_absolute_error(y_test, prediction))
from sklearn.metrics import classification_report, confusion_matrix

print(regressor.score(X_train, y_train))

print(regressor.score(X_test, y_test))