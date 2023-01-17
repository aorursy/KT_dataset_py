import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Path of the file to read

file_path = '../input/titanic123/ChanDarren_RaiTaran_Lab2a.csv'



# Fill in the line below to read the file into a variable home_data

home_data = pd.read_csv(file_path)



# Print summary statistics in next liane
home_data.describe()
home_data.info()
home_data.head()
home_data.corr()
home_data["Fare"].value_counts()


#scatter plot

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'

plt.figure(figsize=(10,10))

sns.scatterplot(x=home_data['Age'], y=home_data['Survived'])



sns.regplot(x=home_data['Age'], y=home_data['Survived'])



# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'

sns.scatterplot(x=home_data['Age'], y=home_data['Survived'], hue=home_data['Pclass'])







import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



file_path = '../input/titanic123/ChanDarren_RaiTaran_Lab2a.csv'

home_data = pd.read_csv(file_path)

home_data.describe(include="all")



#home_data.head()

#home_data.info()



#home_data.corr()

print(home_data.columns)

print(pd.isnull(home_data).sum())

sns.barplot(x="Sex", y="Survived", data=home_data)

print("Percentage of females who survived:", home_data["Survived"][home_data["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", home_data["Survived"][home_data["Sex"] == 'male'].value_counts(normalize = True)[1]*100)

sns.barplot(x="Pclass", y="Survived", data=home_data)
sns.barplot(x="SibSp", y="Survived", data=home_data)
sns.barplot(x="Parch", y="Survived", data=home_data)
plt.show()
home_data["Age"] = home_data["Age"].fillna(-0.10)

bins = [-1, 0, 7, 9, 15, 29, 40, 65, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

home_data['AgeGroup'] = pd.cut(home_data["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=home_data)

plt.show()
#دا الوحيد اللي ما عرفتلو ولا فهمتلو ف مو راضي يزبط (اللي رسلتو جويريه)ا

#missing value

import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/titanic/train_data.csv', index_col='Id')

X_test_full = pd.read_csv('../input/titanic/test_data.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors

X = X_full.select_dtypes(exclude=['object'])

X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)



# Shape of training data (num_rows, num_columns)

print(X_train.shape)

# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])



# Fill in the line below: get names of columns with missing values

cols_with_missing = [col for col in X_train.columns

                     if X_train[col].isnull().any()] # Your code here

# Fill in the lines below: drop columns in training and validation data

reduced_X_train = X_train.drop(cols_with_missing, axis=1)

reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)



print("MAE (Drop columns with missing values):")

print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))



from sklearn.impute import SimpleImputer

# Fill in the lines below: imputation

my_imputer = SimpleImputer()

 # Your code here

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns

print("MAE (Imputation):")

print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))


