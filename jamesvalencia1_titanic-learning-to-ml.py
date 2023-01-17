import numpy as np # Used for linear algebra
import pandas as pd # Used for data processing
import os

''' Prints a list of all files in directory '''
print(os.listdir("../input"))
# This code is the default coding that comes when you first start a notebook
# The 'os' module and 'listdir' function is not really necessary to keep with your code. Its just a FYI.

import warnings
warnings.filterwarnings('ignore')
# This code prevents any annoying warnings that might show up when running your code.
''' Load Data '''
''' 
Extra Lesson:
'try' and 'except' methods are like the training wheels to riding a bicycle. 
They are not necessary but are useful as a beginner starts coding. 
The code used in the 'try' method will first run and if there is an error to 
the code the 'except' method will output instead.
'''
try:
    df_train = pd.read_csv("../input/train.csv")
    df_test = pd.read_csv("../input/test.csv")
    print('Files are loaded')
except:
    print('Something went wrong.')
ship = df_train.append(df_test, ignore_index=True)
# The append function add the dataframe you put inside the parenthesis to the bottom of your selected dataframe.
# In this case, we are adding df_test to the bottom of df_train. We use ignore_index=True to prevent
# any confusion between the two dataframes. We are saying, ignore the index on df_test and just use
# the df_train index.
''' This will show the columns in the data '''
print('"ship" Dataset Columns: ', *ship.columns + ',')
# The star infront of the ship.columns code prints the list in a cleaner way. 
# Try running the code without the star and you will see what I mean.
''' This will check which column was removed in the test data '''
column_diff = list(column for column in df_train.columns if column not in df_test.columns) 
##  The style of for loop above is called List Comprehension. It runs much faster than nested for loops and is a must learn for beginners!
print('The only differences between test and train data: ', column_diff)
df_train['Survived'].value_counts(normalize=True) ## Normalize just makes the values into percentages for us
''' Checking the data types of each column '''
print('------- "ship" Data -------')
ship.info()
print('"ship" columns with null values:\n', ship.isnull().sum())
try:
    ship_orig = ship.copy()
    print('Copies have been made')
except:
    print('Something went wrong.')
''' Run this cell if you made a mistake to your data and wish to go back to your original '''
ship = ship_orig.copy()
print('Back up data has been used to restore your data.')
# drop unnecessary columns, these columns will not be used in this analysis and prediction
columns_to_drop = ['PassengerId', 'Ticket']
ship = ship.drop(columns_to_drop, axis=1)
print('The following columns have been dropped: ', columns_to_drop)
del columns_to_drop
ship.head()
ship.tail()
df_train.describe()
print('Missing Values in "ship" data: ', list(col for col in ship.columns if ship[col].isnull().any() == True))
ship['Embarked'].value_counts() ## Memorize the "value_counts" function as it is used quite often
ship['Embarked'].mode()
# Fill the two missing values with the most occurred value, which is "S".
ship["Embarked"] = ship["Embarked"].fillna("S")
print('Embarked column blanks have been filled with the value "S"')
# matplotlib is a module used for plotting data
import matplotlib.pyplot as plt
%matplotlib inline
x_data = ship['Fare']
plt.scatter(x_data.index, x_data)
plt.xlabel('Index Number')
plt.ylabel('Fare Price')
plt.show()
# We will fill the blank using some estimated guess
median_value = ship["Fare"].median()
print('The median value for the "Fare" column is: ', median_value)
ship["Fare"].fillna(median_value, inplace=True)
print('Fare column blanks have been filled with the median value of the column')
del median_value
print('Number of null values in Age column:', ship['Age'].isnull().sum())
print('The mean of the Age column: ', ship['Age'].mean())
## This will import the Imputer module from Sci-Kit Learn into our Kernel
from sklearn.impute import SimpleImputer
imputer_tool = SimpleImputer()

# Lets make a temporary variable with what will be imputed
ship_numeric = ship.select_dtypes(exclude=['object'])
# Note that Imputation works only with numerical type of data (i.e. Age, Weight, Temperature, etc.)
cols_with_missing = ['Age']

for col in cols_with_missing:
    ship_numeric.loc[:, col + '_was_missing'] = ship_numeric[col].isnull()

# Imputation
ship_numeric_imp = imputer_tool.fit_transform(ship_numeric.values)
ship_numeric = pd.DataFrame(ship_numeric_imp, index=ship_numeric.index, columns=ship_numeric.columns)
ship_numeric.head()
ship_numeric.loc[lambda df: df.Age_was_missing == 1, :][:5] ## This pulls up a sample of the records that had the age value missing.
ship['Age'] = ship_numeric['Age'].copy()
print('Are there any null values?', ship['Age'].isnull().any())
ship['Age'] = ship_numeric['Age'].copy()
ship['Age_was_missing'] = ship_numeric['Age_was_missing'].copy() # We will use this later
del ship_numeric, ship_numeric_imp # We don't need these variables anymore.
print('Age data has been copied back to main dataset')
print('Age_was_missing column was added to main dataset')
print('NaN "Cabin" values in "ship" dataset: %s out of %s' % (ship['Cabin'].isnull().sum(), len(ship)))
cabin_data = {}
cabin_data['With Cabin Name - Survived'] = ship['Cabin'].loc[ship['Cabin'].notnull() & (ship['Survived'] == 1)].count()
cabin_data['With Cabin Name - Deceased'] = ship['Cabin'].loc[ship['Cabin'].notnull() & (ship['Survived'] == 0)].count()
cabin_data['No Cabin Name - Survived'] = len(ship['Cabin'].loc[ship['Cabin'].isnull() & (ship['Survived'] == 1)])
cabin_data['No Cabin Name - Deceased'] = len(ship['Cabin'].loc[ship['Cabin'].isnull() & (ship['Survived'] == 0)])

plt.bar(list(cabin_data.keys()) ,(list(cabin_data.values())))
plt.xticks(rotation='vertical')
plt.show()
cabin_data
cabin_notnull = df_train['Cabin'].loc[df_train['Cabin'].notnull()].astype(str).str[0]
cabin_notnull = pd.DataFrame([cabin_notnull, df_train['Survived'].loc[df_train['Cabin'].notnull()]]).T
cabin_notnull['Cabin'].value_counts()
cabin_values = list(cabin_notnull['Cabin'].unique())
cabin_values
cabin_sums = [((cabin_notnull['Survived'].loc[cabin_notnull['Cabin'] == x].sum()) / (len(cabin_notnull.loc[cabin_notnull['Cabin'] == x]))) for x in cabin_values]
cabin_sums

fig, ax=plt.subplots()
ax.set(xlabel="Cabins", ylabel="Survival Rate")
ax.bar(cabin_values, cabin_sums)
plt.show()
print(*[('Survival rate for cabin %s:  %s \n' % (x, round(y, 2))) for x, y in zip(cabin_values, cabin_sums)])
#ship['Cabin_bool'] = ship["Cabin"].notnull().astype('int')
ship.drop("Cabin", axis=1, inplace=True)
print('Cabin column has been dropped.')
embark_dummies_titanic  = pd.get_dummies(ship['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

ship = ship.join(embark_dummies_titanic)

ship.drop(['Embarked'], axis=1,inplace=True)
print ('Embarked column has been dropped. C and Q columns have been added.')
# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(ship['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

ship.drop(['Pclass'],axis=1,inplace=True)

ship = ship.join(pclass_dummies_titanic)
print ('Pclass column has been changed.')
ship['Family'] =  ship["Parch"] + ship["SibSp"] # Adding together Parch and SibSp columns
ship['Family'].loc[ship['Family'] > 0] = 1
ship['Family'].loc[ship['Family'] == 0] = 0

'''Notice here: We have created a new column called "Family" and given it the values of only 1 or 0'''
'''When you create new columns of data, this is called Feature Engineering'''

# drop Parch & SibSp
ship = ship.drop(['SibSp','Parch'], axis=1)

print('SibSp and Parch columns have been dropped. Family column has been added.')
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 18 else sex
    
ship['Person'] = ship[['Age','Sex']].apply(get_person, axis=1) # The apply function will run the get_person function to the 'Age' and 'Sex' columns

# No need to use Sex column since we created Person column that contains the Sex of the passenger
ship.drop(['Sex'],axis=1,inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(ship['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

ship = ship.join(person_dummies_titanic)

# Now we can drop the Person column we created, since we converted it to a numerical datatype using the dummy method
ship.drop(['Person'],axis=1,inplace=True)
print('Age and Sex columns have been dropped. Child and Female columns have been added.')
ship['Title'] = ship['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
print('Title column has been created with values')
ship.head()
ship['Title'].value_counts()
titles = ship['Title'].unique()
titles
#titleGenderAge = pd.DataFrame(index = titles, columns = ['Gender', 'Min Age', 'Median Age', 'Max Age', 'Count'])
ship.groupby('Title', sort=False)['Age'].agg(['mean', 'min', 'median', 'max', 'count'])
ship['Title'].loc[ship['Title'] == 'the Countess'] = 'Mrs'
ship['Title'].loc[ship['Title'] == 'Ms'] = 'Mrs'
ship['Title'].loc[ship['Title'] == 'Lady'] = 'Mrs'
ship['Title'].loc[ship['Title'] == 'Dona'] = 'Mrs'
ship['Title'].loc[ship['Title'] == 'Mlle'] = 'Miss'
ship['Title'].loc[ship['Title'] == 'Mme'] = 'Miss'
print('Rare lady titles have been added to larger group titles')
print(ship['Title'].value_counts())
stat_min = 10 # while small is arbitrary, we'll use the common minimum in statistics: 
              # http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (ship['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

# apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: 
# https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
ship['Title'] = ship['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(ship['Title'].value_counts())
print("-"*10)
print('Rare title names have been combined into a "Misc" option')
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
label = LabelEncoder()
ship['Title_Code'] = label.fit_transform(ship['Title'])
print('Fit Transform function has been run.')
ship['Title_Code'].head()
ship.drop(['Name'],axis=1,inplace=True)
print('Name column has been dropped.')
title_data = {}
title_data['Mr - Survived'] = ship['Title_Code'].loc[(ship['Title_Code'] == 3) & (ship['Survived'] == 1)].count()
title_data['Mr - Deceased'] = ship['Title_Code'].loc[(ship['Title_Code'] == 3) & (ship['Survived'] == 0)].count()
title_data['Miss - Survived'] = ship['Title_Code'].loc[(ship['Title_Code'] == 2) & (ship['Survived'] == 1)].count()
title_data['Miss - Deceased'] = ship['Title_Code'].loc[(ship['Title_Code'] == 2) & (ship['Survived'] == 0)].count()
title_data['Mrs - Survived'] = ship['Title_Code'].loc[(ship['Title_Code'] == 4) & (ship['Survived'] == 1)].count()
title_data['Mrs - Deceased'] = ship['Title_Code'].loc[(ship['Title_Code'] == 4) & (ship['Survived'] == 0)].count()
title_data['Master - Survived'] = ship['Title_Code'].loc[(ship['Title_Code'] == 0) & (ship['Survived'] == 1)].count()
title_data['Master - Deceased'] = ship['Title_Code'].loc[(ship['Title_Code'] == 0) & (ship['Survived'] == 0)].count()
title_data['Misc - Survived'] = ship['Title_Code'].loc[(ship['Title_Code'] == 1) & (ship['Survived'] == 1)].count()
title_data['Misc - Deceased'] = ship['Title_Code'].loc[(ship['Title_Code'] == 1) & (ship['Survived'] == 0)].count()

plt.figure(figsize=(16,4))
plt.bar(list(title_data.keys()) ,(list(title_data.values())))
plt.show()
title_data

title_dummies_titanic  = pd.get_dummies(ship['Title'])
title_dummies_titanic.drop(['Misc', 'Mr'], axis=1, inplace=True)

ship.drop(['Title', 'Title_Code'],axis=1,inplace=True)

ship = ship.join(title_dummies_titanic)
print ('Title column has been dropped. "Master", "Mrs", and "Miss" has been added.')
ship.sample(5)
ship.sample(5)
print('Number of  people who survived with a Fare > 50: ',
      len(ship['Survived'].loc[(ship['Fare'] > 50) & (ship['Survived'] == 1)]))
print('Number of  people who died with a Fare > 50: ',
      len(ship['Survived'].loc[(ship['Fare'] > 50) & (ship['Survived'] == 0)]))

ship['High_Fare'] = [1 if x > 50 else 0 for x in ship['Fare']]
print('High Fare column created.')
ship.loc[ship['Age_was_missing'] == 1].sample(5)
ship.drop(['Age_was_missing'],axis=1,inplace=True)
print('Dropped "Age_was_missing" column from dataset.')
import seaborn as sns
#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(ship)
ship.drop(['Age'],axis=1,inplace=True)
# Machine Learning Tool called Sci-Kit Learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
df_tr = ship[ship.Survived.notnull()]
df_tr.head()
df_te = ship[ship.Survived.isnull()]
df_te.drop(['Survived'], axis=1, inplace=True)
df_te = df_te.reset_index(drop=True)
df_te.head()
x_train = df_tr.drop("Survived",axis=1)
y_train = df_tr["Survived"]
x_test  = df_te.copy()
print('Test and Train ML variables are ready.')
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
logreg.score(x_train, y_train)
random_forest = RandomForestClassifier(n_estimators=300)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test).astype(int)
random_forest.score(x_train, y_train)
df_coeff = pd.DataFrame(ship.columns.delete(0))
df_coeff.columns = ['Features']
df_coeff["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
df_coeff
df_test['Survived'] = y_pred.astype(int)
df_test.to_csv('SurvivedList.csv')
df_test
submission = df_test[["PassengerId", "Survived"]]
submission.head()
submission.to_csv('titanic.csv', index=False, header=['PassengerID', 'Survived'])
