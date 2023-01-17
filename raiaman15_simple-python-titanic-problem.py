#Importing Libraries & checking versions

import warnings

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # used to plot and format it

import seaborn as sns

from sklearn.linear_model import LogisticRegression

np.__version__,pd.__version__,sns.__version__
# Enhancing Presentability of data in Jupyter Notebook

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',50)

pd.set_option('display.max_info_columns',50)
# Loading Train Dataset

df_data = pd.read_csv('../input/train.csv')

# Loading Test Dataset

test_data = pd.read_csv('../input/test.csv')

# Shape of Dataset

print(f'Rows: {df_data.shape[0]}')

print(f'Columns: {df_data.shape[1]}',end='\n\n')

# Overview of Columns

print(f'Columns in Dataset:\n{list(df_data.columns)}',end='\n\n')

print('Overview of each column')

df_data.info()
# Displaying top 5 rows of Dataset

df_data.head()
# Building set of Numerical, Categorical & Target columns

n_col = list(df_data.loc[:,df_data.columns != 'Survived'].select_dtypes([np.int64,np.float64]).columns)

c_col = list(df_data.select_dtypes([np.object]).columns)

t_col = ['Survived']

# Print list of numerical, categorical and target columns

print(f'Numerical Columns: {n_col}')

print(f'Categorical Columns: {c_col}')

print(f'Target Columns: {t_col}',end='\n\n')

# Details of Numerical Columns

print(f'Details of Numerical Columns:\n{df_data[n_col].describe()}',end='\n\n')

# Details of Categorical Columns

print(f'Details of Categorical Columns:\n{df_data[c_col].describe()}',end='\n\n')

# Details of Target Columns

print(f'Details of Target Columns:\n{df_data[t_col].describe()}',end='\n\n')
# Cleaning un-necessary columns

print(f'Percentage of null values in each column:\n{round(100*(df_data.isna().sum()/len(df_data.index)),2)}')

# Dropping columns which are insignificant for the analysis

df_data.drop(columns=['PassengerId'],inplace=True)

# Cabin have 77.10 % null values, hence dropping the column

df_data.drop(columns=['Cabin'],inplace=True)

# Name have all different values, which wouldt have any affect on target, hence dropping the column

df_data.drop(columns=['Name'],inplace=True)

# Removing the two columns from our columns set

n_col.remove('PassengerId')

c_col.remove('Cabin')

c_col.remove('Name')
# Cleaning un-necessary rows

# Dropping all rows which have more than 20 % missing values

# Dropping rows with most NaN value (Taking an arbitrary value >20 %)

threshold = int(.2*len(df_data.columns))

df_data.dropna(thresh=threshold, inplace=True)



# Dropping all rows where Age is null, since it is an important column

df_data.dropna(subset=['Age'], how='any', inplace=True)



# Dropping duplicate rows

df_data.drop_duplicates(inplace=True)
# Imputing Embarked Column as it have 0.22 % null values and most values of the column is 'S'

df_data.fillna(value='S', inplace=True)
# Shape of Dataset

print(f'Rows: {df_data.shape[0]}')

print(f'Columns: {df_data.shape[1]}')

print(f'Count of NaN values: {df_data.isna().sum()}')

print(f'Percentage of null values in each column:\n{round(100*(df_data.isna().sum()/len(df_data.index)),2)}')
# A function to plot univariate analysis plots for Numerical columns

def univariate_n(c):

    fig, ax = plt.subplots(1,2,figsize=(12,4))

    sns.distplot(df_data[c], ax=ax[0])

    sns.boxplot(x=df_data[c], ax=ax[1])

    fig.show()
# Plotting Histogram for Numerical columns

for c in n_col:

    univariate_n(c)
# Plotting bivariate analysis plots for Numerical columns

sns.pairplot(df_data, hue="Survived")

plt.show()
# Plotting Heatmap for Numerical columns

plt.figure(figsize=(12,10))

ax = sns.heatmap(df_data[n_col].corr(), vmin=0, vmax=1, cmap="YlGnBu", annot=True)

plt.show()
print(f'Percentage of null values in each column:\n{round(100*(test_data.isna().sum()/len(test_data.index)),2)}')
test_data['Fare'].mean()
test_data.fillna(value='35.6271884892086', inplace=True)
X = df_data[n_col]

y = df_data[t_col]

lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)

test_predict = lr.predict(test_data[n_col])

d = {'PassengerId': test_data['PassengerId'], 'Survived': test_predict}

dataframe_to_export = pd.DataFrame(data=d)

sns.countplot(dataframe_to_export['Survived'], palette = 'icefire')

# Exporting the Predicted values for evaluation at Kaggle

dataframe_to_export.to_csv(path_or_buf='submission.csv', index=False)