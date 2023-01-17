import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier ,GradientBoostingClassifier

from xgboost import XGBClassifier 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import Ridge,Lasso

from sklearn.metrics import roc_auc_score ,mean_squared_error,accuracy_score,classification_report,roc_curve,confusion_matrix

import warnings

warnings.filterwarnings('ignore')

from scipy.stats.mstats import winsorize

from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns',None)

import six

import sys

sys.modules['sklearn.externals.six'] = six
# accessing to the folder where the file is stored

path = '../input/banking-project-term-deposit/new_train.csv'



# Load the dataframe

dataframe = pd.read_csv(path)



print('Shape of the data is: ',dataframe.shape)



dataframe.head()





# IDENTIFYING NUMERICAL FEATURES



numeric_data = dataframe.select_dtypes(include=np.number) # select_dtypes selects data with numeric features

numeric_col = numeric_data.columns                                                                # we will store the numeric features in a variable



print("Numeric Features:")

print(numeric_data.head())

print("===="*20)


# IDENTIFYING CATEGORICAL FEATURES

categorical_data = dataframe.select_dtypes(exclude=np.number) # we will exclude data with numeric features

categorical_col = categorical_data.columns                                                                              # we will store the categorical features in a variable





print("Categorical Features:")

print(categorical_data.head())

print("===="*20)

# CHECK THE DATATYPES OF ALL COLUMNS:

    

print(dataframe.dtypes)

# To identify the number of missing values in every feature



# Finding the total missing values and arranging them in ascending order

total = dataframe.isnull().sum()



# Converting the missing values in percentage

percent = (dataframe.isnull().sum()/dataframe.isnull().count())

print(percent)

dataframe.head()


# dropping features having missing values more than 60%

dataframe = dataframe.drop((percent[percent > 0.6]).index,axis= 1)



# checking null values

print(dataframe.isnull().sum())
# imputing missing values with mean



for column in numeric_col:

    mean = dataframe[column].mean()

    dataframe[column].fillna(mean,inplace = True)

    

#   imputing with median

# for column in numeric_col:

#     mean = dataframe[column].median()

#     dataframe[column].fillna(mean,inpalce = True)

# we are finding the percentage of each class in the feature 'y'

class_values = (dataframe['y'].value_counts()/dataframe['y'].value_counts().sum())*100

print(class_values)
 
# Selecting the categorical columns

categorical_col = dataframe.select_dtypes(include=['object']).columns

plt.style.use('ggplot')

# Plotting a bar chart for each of the cateorical variable

for column in categorical_col:

    plt.figure(figsize=(20,4))

    plt.subplot(121)

    dataframe[column].value_counts().plot(kind='bar')

    plt.title(column)
# Impute mising values of categorical data with mode

for column in categorical_col:

    mode = dataframe[column].mode()[0]

    dataframe[column] = dataframe[column].replace('unknown',mode)

for column in numeric_col:

    plt.figure(figsize=(20,5))

    plt.subplot(121)

    sns.distplot(dataframe[column])

    plt.title(column)
for column in numeric_col:

    plt.figure(figsize=(20,5))

    plt.subplot(122)

    sns.boxplot(dataframe[column])

    plt.title(column)
dataframe.drop(['pdays','previous'],1,inplace=True)




for column in categorical_col:

    plt.figure(figsize=(20,4))

    plt.subplot(121)

    sns.countplot(x=dataframe[column],hue=dataframe['y'],data=dataframe)

    plt.title(column)    

    plt.xticks(rotation=90)
numeric_col = dataframe.select_dtypes(include=np.number).columns



for col in numeric_col:    

    dataframe[col] = winsorize(dataframe[col], limits=[0.05, 0.1],inclusive=(True, True))



# Now run the code snippet to check outliers again
# Initializing lable encoder

le = LabelEncoder()



# Initializing Label Encoder

le = LabelEncoder()



# Iterating through each of the categorical columns and label encoding them

for feature in categorical_col:

    try:

        dataframe[feature] = le.fit_transform(dataframe[feature])

    except:

        print('Error encoding '+feature)
dataframe.to_csv(r'preprocessed_data.csv',index=False)
from pandas_profiling import ProfileReport

prof = ProfileReport(dataframe)

prof