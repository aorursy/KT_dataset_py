import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('/kaggle/input/maisataitanicdata/titanic.csv')
#لطباعة رأس الجدول نقوم باستخدام الأمر head

data.head()
data.info()
############################

# عليك تعديل هذه الخانة قبل تسليم المشروع

# YOU HAVE TO EDIT THIS CELLL

############################



# كم عدد الصفوف ؟

nrows = data.shape[0]

# كم عدد الأعمدة ؟

ncols = data.shape[1]

# كم عدد الأعمدة الرقمية ؟ كم عدد الأعمدة الوصفية ؟

numeric = ['int16','float16', 'int32', 'float32', 'int64', 'float64']

n_num_cols = len(data.select_dtypes(include = numeric).columns)

n_str_cols = len(data.select_dtypes(include = ['object']).columns)
print('There are {0} rows and {1} columns in this dataset.'.format(nrows, ncols))

print('Of those columns, there are {0} numerical columns and {1} categorical columns.'.format(n_num_cols, n_str_cols))
############################

# عليك تعديل هذه الخانة قبل تسليم المشروع

# YOU HAVE TO EDIT THIS CELLL

############################



# Look at the outputs at `data.info()` again..

# Which columns have missing values ?

# replace the empty list below with column names that have missing values

# for example: ['Name', 'Fare', 'Sex']



cols_with_missing_vals = data.columns[data.isna().any()]
print('The following columns have missing values: {0}'.format(cols_with_missing_vals))
# delete 'PassengerId' column, using 'drop' command ; don't forget to include `inplace=True`

data.drop(columns='PassengerId', inplace=True)
############################

# عليك تعديل هذه الخانة قبل تسليم المشروع

# YOU HAVE TO EDIT THIS CELLL

############################



# drop the columns `Name` and `Ticket` as well the same way



# data.drop(columns='Ticket', inplace=True)

# data.drop(columns='Name', inplace=True)

cols_to_drop = ['Name','Ticket']

data.drop(cols_to_drop, axis=1, inplace= True)
final_column_set = data.columns.tolist()
# We can print all values of Age using this command:

age_vals = data['Age'].values
############################

# عليك تعديل هذه الخانة قبل تسليم المشروع

# YOU HAVE TO EDIT THIS CELLL

############################



# here, print the command that extracts the mean age using numpy: 

mean_age = data['Age'].mean()
assert np.allclose(mean_age, 29.69911764705882), 'You did not calculate the mean age correctly.'

print('OK!')
# Now, we will use the average age to fill in all missing values

data['Age'].fillna( mean_age, inplace=True)
data.head()
data['Survived'].value_counts()
############################

# عليك تعديل هذه الخانة قبل تسليم المشروع

# YOU HAVE TO EDIT THIS CELLL

############################



# survived_percentage = data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# survived_percentage = data.groupby('Sex').Survived.mean()

survived_percentage =  data['Survived'].value_counts()[1]/len(data)
print('About {0}% of people in this dataset have survived'.format(np.round(survived_percentage, 2)))
data['Pclass'].value_counts()
data['Sex'].value_counts()
data['Parch'].value_counts()
data['Embarked'].value_counts()
############################

# عليك تعديل هذه الخانة قبل تسليم المشروع

# YOU HAVE TO EDIT THIS CELLL

############################





# female_survived_percentage = data.loc[data['Sex'] == 'female', 'Survived'].mean()

female_survived_percentage = data[data['Sex'] == 'female']['Survived'].value_counts()[1]/data['Survived'].value_counts()[1]

print(female_survived_percentage)
print('nrows={}\nncols={}\nn_num_cols={}\nn_str_cols={}\nfinal_column_set={}\ncols_with_missing_vals={}\nmean_age={}\nsurvived_percentage={}\nfemale_survived_percentage={}'.format(

nrows,

ncols,

n_num_cols,

n_str_cols,

final_column_set,

cols_with_missing_vals,

mean_age,

survived_percentage,

female_survived_percentage

))
