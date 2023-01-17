import pandas

import numpy

import seaborn

import matplotlib.pyplot as plt



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



# For loading the data

train_data = pandas.read_csv('../input/train.csv', low_memory = False)

test_data = pandas.read_csv('../input/train.csv', low_memory = False)



# Set Pandas to show all columns in DataFrames

pandas.set_option('display.max_columns', None)



# Set Pandas to show all rows in DataFrames

pandas.set_option('display.max_rows', None)



# upper-case all DataFrame column names

train_data.columns = map(str.upper, train_data.columns)

test_data.columns = map(str.upper, test_data.columns)



# bug fix for display formats to avoid run time errors

pandas.set_option('display.float_format', lambda x:'%f'%x)



# train_data and test_data both are subset of the titanic data.

# So getting some basic information of the data from train_data as both of them are similar



# Getting heading of columns of data

print(train_data.head())



# Getting some details of data

print(train_data.describe())  
# Setting variable to numeric value as mostly all pandas operation/method work on numeric variable

train_data['SURVIVED'] = pandas.to_numeric(train_data['SURVIVED'], errors='coerce')

train_data['PCLASS'] = pandas.to_numeric(train_data['PCLASS'], errors='coerce')

train_data['AGE'] = pandas.to_numeric(train_data['AGE'], errors='coerce')

train_data['SIBSP'] = pandas.to_numeric(train_data['SIBSP'], errors='coerce')

train_data['PARCH'] = pandas.to_numeric(train_data['PARCH'], errors='coerce')



# Making new data set from given dataset with variables(column) which seem relevent in predicting the chance of survival

sub = train_data[['SURVIVED', 'PCLASS', 'AGE', 'SIBSP', 'PARCH', 'EMBARKED', 'SEX']]

print(sub.describe())
# Replacing unknown age by median of 'AGE' variable

sub['AGE'] = sub['AGE'].replace(numpy.nan, sub['AGE'].median())



# EMBARKED variable has some missing data. So filling it with category 'S' as this has most frequency

sub['EMBARKED'] = sub['EMBARKED'].fillna('S')



# Creating new variable for better visualization of age vs survival

def AGEGROUP(row):

    if (row['AGE'] > 60) :

        return 5        

    elif (row['AGE'] > 45) :

        return 4        

    elif (row['AGE'] > 19) :

        return 3   

    elif (row['AGE'] > 9) :

        return 2 

    elif (row['AGE'] >= 0) :

        return 1    

sub['AGEGROUP'] = sub.apply(lambda row: AGEGROUP (row),axis=1) 
# Now visualization of 'SURVIVED'

# Printing counts and percentage of diffrent survival

print(sub['SURVIVED'].value_counts(sort=False))

print(sub['SURVIVED'].value_counts(sort=False,normalize=True))



# Making variable  categorical 

sub['SURVIVED'] = sub['SURVIVED'].astype('category')



# Visualising counts of survival with bar graph

seaborn.countplot(x="SURVIVED", data=sub);

plt.xlabel('Survival Status')

plt.ylabel('Frequency')

plt.title('Count of Survival')



# Converting 'SURVIVED' to numeric value for further operations

sub['SURVIVED'] = pandas.to_numeric(sub['SURVIVED'], errors='coerce')
# Printing counts and percentage of category of Age Group

print(sub['AGEGROUP'].value_counts(sort=False))

print(sub['AGEGROUP'].value_counts(sort=False,normalize=True))



# Making variable AGEGROUP categorical and naming category

sub['AGEGROUP'] = sub['AGEGROUP'].astype('category')

sub['AGEGROUP'] = sub['AGEGROUP'].cat.rename_categories(["CHILDREN", "ADOLESCENENTS", "ADULTS", "MIDDLE AGE", "OLDS"])



# Visualising counts of Age Group with bar graph

seaborn.countplot(x="AGEGROUP", data=sub);

plt.xlabel('Age Group')

plt.ylabel('Frequency')

plt.title('Age group Distribution   ')



# Showing proportion of survival of different groups by plot

seaborn.factorplot(x="AGEGROUP", y="SURVIVED", data=sub, kind="bar", ci=None)

plt.xlabel('AGE GROUP')

plt.ylabel('Survive Percentage')

plt.title('Survive v/s Age Group')
# Now visualization of 'SIBSP'

# Printing counts and percentage of number of siblings and spouse

print(sub['SIBSP'].value_counts(sort=False))

print(sub['SIBSP'].value_counts(sort=False,normalize=True))



# Making variable categorical 

sub['SIBSP'] = sub['SIBSP'].astype('category')



# Visualising counts of siblings and spouse number with bar graph

seaborn.countplot(x="SIBSP", data=sub);

plt.xlabel('No. of Siblings and Spouse')

plt.ylabel('Frequency')

plt.title('Count of  number of Siblings and Spouse')



# Showing proportion of survival for different number of siblings and spouse

seaborn.factorplot(x="SIBSP", y="SURVIVED", data=sub, kind="bar", ci=None)

plt.xlabel('No. of Spouse and Siblings')

plt.ylabel('Survive Percentage')

plt.title('Survive v/s No. of Siblings and Spouse')
# Now visualization of 'PARCH'

# Printing counts and percentage of number of children and parent

print(sub['PARCH'].value_counts(sort=False))

print(sub['PARCH'].value_counts(sort=False,normalize=True))



# Making variable categorical 

sub['PARCH'] = sub['PARCH'].astype('category')



# Visualising counts of children and parent number with bar graph

seaborn.countplot(x="PARCH", data=sub);

plt.xlabel('No. of Children and Parent')

plt.ylabel('Frequency')

plt.title('Count of  number of Children and Parent')



# Showing proportion of survival for different number of parent and children

seaborn.factorplot(x="PARCH", y="SURVIVED", data=sub, kind="bar", ci=None)

plt.xlabel('No. of Children and Parent')

plt.ylabel('Survive Percentage')

plt.title('Survive v/s No. of Children and Parent')
# Now visualization of 'Gender'

# Printing counts and percentage of male and female

print(sub['SEX'].value_counts(sort=False))

print(sub['SEX'].value_counts(sort=False,normalize=True))



# Making variable  categorical 

sub['SEX'] = sub['SEX'].astype('category')



# Visualising counts of Gender with bar graph

seaborn.countplot(x="SEX", data=sub);

plt.xlabel('Gender')

plt.ylabel('Frequency')

plt.title('Count of Gender')



# Showing proportion of survival for different type of gender

seaborn.factorplot(x="SEX", y="SURVIVED", data=sub, kind="bar", ci=None)

plt.xlabel('Gender')

plt.ylabel('Survive Percentage')

plt.title('Survive v/s Sex')


# Now visualization of 'PCLASS'

# Printing counts and percentage of diffrent passanger class

print(sub['PCLASS'].value_counts(sort=False))

print(sub['PCLASS'].value_counts(sort=False,normalize=True))



# Making variable  categorical 

sub['PCLASS'] = sub['PCLASS'].astype('category')



# Visualising counts of diffrent passanger class with bar graph

seaborn.countplot(x="PCLASS", data=sub);

plt.xlabel('Passanger Class')

plt.ylabel('Frequency')

plt.title('Count of different passenger class')



# Showing proportion of survival for different types of passanger class

seaborn.factorplot(x="PCLASS", y="SURVIVED", data=sub, kind="bar", ci=None)

plt.xlabel('Passanger Class')

plt.ylabel('Survive Percentage')

plt.title('Survive v/s Passanger Class')
# Now visualization of 'EMBARKED'

# Printing counts and percentage of diffrent points of embarkation

print(sub['EMBARKED'].value_counts(sort=False))

print(sub['EMBARKED'].value_counts(sort=False,normalize=True))



# Making variable  categorical 

sub['EMBARKED'] = sub['EMBARKED'].astype('category')



# Visualising counts of diffrent points of embarkation with bar graph

seaborn.countplot(x="EMBARKED", data=sub);

plt.xlabel('Embarkation Point')

plt.ylabel('Frequency')

plt.title('Count of different embarkation point')



# Showing proportion of survival for different points of embarkation

seaborn.factorplot(x="EMBARKED", y="SURVIVED", data=sub, kind="bar", ci=None)

plt.xlabel('Embrakation Point')

plt.ylabel('Survive Percentage')

plt.title('Survive v/s Embarkation Points')
# Convert the male and female groups to integer form

sub['SEX'] = sub['SEX'].astype('category')

sub['SEX'] = sub['SEX'].cat.rename_categories([0,1])



# Convert the Embarked classes to integer form

sub['EMBARKED'] = sub['EMBARKED'].astype('category')

sub['EMBARKED'] = sub['EMBARKED'].cat.rename_categories([0,1,2])



# Replacing unknown age by median of 'AGE' variable of test data

test_data['AGE'] = test_data['AGE'].replace(numpy.nan, test_data['AGE'].median())



# In test dataset EMBARKED variable has some missing data. So filling it with category 'S' as this has most frequency

test_data['EMBARKED'] = test_data['EMBARKED'].fillna('S')



# Convert the male and female groups to integer form

test_data['SEX'] = test_data['SEX'].astype('category')

test_data['SEX'] = test_data['SEX'].cat.rename_categories([0,1])



# Convert the Embarked classes to integer form

test_data['EMBARKED'] = test_data['EMBARKED'].astype('category')

test_data['EMBARKED'] = test_data['EMBARKED'].cat.rename_categories([0,1,2])



# Create the target and features numpy arrays: target, features_one

target = sub['SURVIVED'].values

features = sub[['PCLASS', 'AGE', 'SIBSP', 'PARCH', 'EMBARKED', 'SEX']].values
# Import the `RandomForestClassifier`

from sklearn.ensemble import RandomForestClassifier



# Building and fitting my_forest

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(features,target)



# Print the score of the fitted random forest

print(my_forest.score(features, target))



# Compute predictions on our test set features then print the length of the prediction vector 

pred_forest = my_forest.predict(features)
# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId =numpy.array(test_data["PASSENGERID"]).astype(int)

pred_forest1 = pandas.DataFrame(pred_forest, PassengerId, columns = ["Survived"])



# Check that your data frame has 418 entries

print(pred_forest1.shape)



#Write your solution to a csv file with the name my_solution.csv

pred_forest1.to_csv("pred.csv", index_label = ["PassengerId"])