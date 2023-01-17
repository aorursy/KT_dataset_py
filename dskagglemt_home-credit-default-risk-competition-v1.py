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
# numpy and pandas for data manipulation

# already imported as a default by the Kaggel Kernels.



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns
# Read the training data

train_data = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

print('Training data shape: ', train_data.shape)

train_data.head()
# Check if there are any duplicate application.

train_data[train_data.duplicated(['SK_ID_CURR'])]

#train_data[train_data.duplicated(['SK_ID_CURR'])].count()
# Check if there are any duplicate rows.

train_data[train_data.duplicated()]
# Read the test data

test_data = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

print('Test data shape: ', test_data.shape)

test_data.head()
# Check if there are any duplicate rows.

test_data[test_data.duplicated()]
print(train_data['TARGET'].value_counts())
train_data['TARGET'].astype(int).plot.hist();
# Number of each type of column

train_data.dtypes.value_counts()
#Get list of categorical variables

s = (train_data.dtypes == 'object')

train_data_cat_var = list(s[s].index)

train_data_cat_var
#Get list of Numerical variables

train_data_num_var = list(train_data.select_dtypes(exclude=['object']).columns)

train_data_num_var
len(train_data_num_var)
train_data_num_var[1:10]
# Distribution of each feature

#pd.options.display.mpl_style = 'default'



import matplotlib

matplotlib.style.use('ggplot')



plt.figure(figsize=(20,5))

train_data.boxplot(column=train_data_num_var[1:10])



plt.show(block=True)

plt.figure(figsize=(20,5))

train_data.hist(column=train_data_num_var[1:5])

plt.show()
# Lets get the % of each null values.

total = train_data.isnull().sum().sort_values(ascending=False)

percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'], sort=False)

missing_data.head()
print("Total columns that have missing values :", str(len(missing_data[(missing_data['%']>0) ]) ))
print("Total columns that have missing values (More than 50%):", str(len(missing_data[(missing_data['%']>50) ]) ))

print("Total columns that have missing values (Less than 50%):", str(len(missing_data[(missing_data['%']<50) & (missing_data['%']>0)] ) ))
train_data[train_data_num_var[1:5]]
#Using Pearson Correlation



plt.figure(figsize=(20,10))



cor = train_data[train_data_num_var[1:20]].corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)



plt.show()
#Correlation with output variable

cor_target = abs(cor["TARGET"])



#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.05]

relevant_features
# Number of unique values in each categorical column

train_data.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
train_data.describe(include=[np.object])
cat_dict = dict()

for column in train_data:

    if train_data[column].dtype == 'object':

        if len(list(train_data[column].unique())) <= 2:

            cat_dict[column] = len(list(train_data[column].unique()))

#             print(cat_dict)

        else :

            cat_dict[column] = len(list(train_data[column].unique()))



print(cat_dict)

            
# write a function to get the distinct value in each categorical value

def get_Unique_Values(list_cat_var) :

    cat_dict = dict()

    for i in list_cat_var:

        cat_dict[i] = list(train_data[i].unique())

    return cat_dict
print(get_Unique_Values(['NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']))
print(get_Unique_Values(['CODE_GENDER', 'EMERGENCYSTATE_MODE', 'HOUSETYPE_MODE', 'NAME_EDUCATION_TYPE', 'FONDKAPREMONT_MODE']))
# Binary encoding

train_data['NAME_CONTRACT_TYPE'] = [0 if x == 'Cash loans' else 1 for x in train_data['NAME_CONTRACT_TYPE']]

train_data['FLAG_OWN_CAR'] = [0 if x == 'N' else 1 for x in train_data['FLAG_OWN_CAR']]

train_data['FLAG_OWN_REALTY'] = [0 if x == 'N' else 1 for x in train_data['FLAG_OWN_REALTY']]