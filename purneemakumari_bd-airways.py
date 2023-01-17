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
#import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_rows',800)
pd.set_option('display.max_columns',500)

#for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
sns.set()

#import libraries from ML
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import random


#load the data
df = pd.read_csv("../input/airways-data/Airways data.csv")

#understanding the data
df.info()
df.describe()


#to check numerical and categorical variables
num_col = df.select_dtypes(include=np.number).columns
print("Numerical columns:\n",num_col)
cat_col = df.select_dtypes(exclude=np.number).columns
print("Categorical columns:\n",cat_col)


#import label encoder
from sklearn import preprocessing

#label_encoder object knows how to understand word labels
label_encoder = preprocessing.LabelEncoder()

#Encode labels in column 'Gender'
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Customer Type'] = label_encoder.fit_transform(df['Customer Type'])
df['Type of Travel'] = label_encoder.fit_transform(df['Type of Travel'])
df['Class'] = label_encoder.fit_transform(df['Class'])
df.head(10)
#encode y variable
sat_dummies = df.replace(to_replace = {'sat'})
#finding NA values
print(df.isna().sum())
print(df.shape)

#treating NA
print(df.dropna())

#performing EDA

#to check the distribution of y i.e satisfaction
sns.countplot(y = df.satisfaction,data=df)
plt.xlabel("Count of each target class")
plt.ylabel("Target classes")
plt.show()

#check the distribution of all the features
df.hist(figsize = (15,12),bins = 15)
plt.title("Feature distribution")
plt.show()
#heat map
plt.figure(figsize=(15,15))
p = sns.heatmap(df[num_col].corr(),annot = True,cmap='RdY1Gn',center =0)
sns.pairplot(df[num_col])