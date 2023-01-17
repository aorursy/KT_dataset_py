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
import pandas as pd 

trd=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

trd.head()
import pandas as pd 

tsd=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

tsd.head()
w=trd.loc[trd.YrSold<2008][trd.SaleCondition=='Normal']

#print(w)

rw= len(w)/len(trd)

print("percentage of houses are for sale before 2008 are normal is : ",rw)
x=trd.loc[trd.LotArea<10000][trd.SaleCondition=='Normal']

#print(w)

rw= len(x)/len(trd)

print("percentage of houses are having area less than 10000 m^2 are normal is : ",rw)
value=trd.loc[trd.MSSubClass>60]

print(value)
"""author    s_agnik1511"""

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

trd = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train_y = trd.SalePrice

predictor_cols = ['LotArea']

train_X = trd[predictor_cols]

my_model = RandomForestRegressor()

my_model.fit(train_X, train_y)
trd.head()
"""author s_agnik1511"""

import pandas as pd

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('submission_sagnik.csv', index=False)
import pandas as pd

k=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

k.shape
import pandas as pd

import numpy as np

m=tsd.predict("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

Submission["taregt"]=m

Submission.to_csv("submission_sagnik123",index=False)

"""author s_agnik1511"""

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_X=test[predictor_cols]

predicted_prices=my_model.predict(test_X)

print(predicted_prices)
from sklearn.linear_model import LinearRegression

import pandas as pd

Model=LinearRegression()

x_test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

Model.fit(x_test)

Model.predict(x_test)


import pandas as pd # for reading the data frame

import numpy as np # for numerical calculation

import matplotlib.pyplot as plt # use for visualization

import seaborn as sns   # mostly used for statistical visualization 

%matplotlib inline      # used for inline ploting

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



print("Shape of train: ", train.shape) #  rows : 1459 columns : 81

print("Shape of test: ", test.shape)  # rows : 1459 columns : 80
train.head(20)
test.head(10)
df = pd.concat((train, test)) # here we concat the test and train data set

temp_df = df

print("Shape of df: ", df.shape)
temp_df.head() # by default its selected 5 rows and all the columns
temp_df.tail() # for vewig the last five rows
# To show the all columns

pd.set_option("display.max_columns",2000)  # used for viewing all the columns at onces

pd.set_option("display.max_rows",85)
df.head() 
df.shape
df.info()   # lets view the information about our data set like find the data types of our columns
df.describe() # used for finding the describtion about the data set like,  mean , standard deviation given below
df.select_dtypes(include=['int64', 'float64']).columns  # extracrt the columns whose dtype is intege and float
df.select_dtypes(include=['object']).columns  # find the columns whose dtype is object
# Set index as Id column

df = df.set_index("Id")
df.head()
# Show the null values using heatmap

plt.figure(figsize=(16,9))

sns.heatmap(df.isnull())       



# useing heat map we can see the missing values ... the white stripes indicates the missing values 

df.isnull().sum()   # from this we can see which columns has how many missig values  like LotFrontsge has 486 missing vales
# Get the percentages of null value

null_percent = df.isnull().sum()/df.shape[0]*100

null_percent





# from this we can say LotFrontage has 16 % and Alley has 93 % missing vslues
col_for_drop = null_percent[null_percent > 20].keys() # if the null value % 20 or > 20 so need to drop it
# drop columns

df = df.drop(col_for_drop, "columns")

df.shape
null_percent = df.isnull().sum()/df.shape[0]*100

null_percent # shows values which has less than 20 % missing values
# find the unique value count

for i in df.columns:

    print(i + "\t" + str(len(df[i].unique())))
# find unique values of each column

for i in df.columns:

    print("Unique value of:>>> {} ({})\n{}\n".format(i, len(df[i].unique()), df[i].unique()))
# Describe the target 

train["SalePrice"].describe()
# Plot the distplot of target

plt.figure(figsize=(10,8))

bar = sns.distplot(train["SalePrice"])
# correlation heatmap

plt.figure(figsize=(25,25))

ax = sns.heatmap(train.corr(), cmap = "coolwarm", annot=True, linewidth=2)



# here we use piearson corelation
# correlation heatmap of higly correlated features with SalePrice

hig_corr = train.corr()

hig_corr_features = hig_corr.index[abs(hig_corr["SalePrice"]) >= 0.5]

hig_corr_features

plt.figure(figsize=(10,8))

bar = sns.distplot(train["LotFrontage"])
plt.figure(figsize=(10,8))

bar = sns.distplot(train["MSSubClass"])