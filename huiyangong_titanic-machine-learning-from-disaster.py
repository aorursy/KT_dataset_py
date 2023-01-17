import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
print('Read')
train_data.head(10)
train_data.shape
train_data.info()
test_data.head(10)
test_data.shape
test_data.info()
def concat_df(train_data, test_data):
    return pd.concat([train_data, test_data],sort=True).reset_index(drop=True)

def divide_df(all_data):
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'],axis=1) 
all_data = concat_df(train_data, test_data)
all_data[885:895] # Check the second part of data consists of only NaN in "Survived" column.
print('Missing in the train data:') 
display(train_data.isnull().sum())

print('Missing in the test data:') 
display(test_data.isnull().sum())
age_missing = all_data['Age'].isnull().sum()
print("Missings for Age in the entire data: " + str(age_missing))
print("Missing percentage for Age in the entire data: " + str(round(age_missing*100/len(all_data),2))+ '%')
all_data['Age'].plot(kind='hist', figsize=(8, 5))

plt.title('Age of Passengers') # add a title to the histogram
plt.xlabel('Age') # add y-label
plt.ylabel('Number of passengers') # add x-label

plt.show()
fig = plt.figure(figsize=(8,8))

# top left
Pclass1 = all_data[all_data['Pclass']==1]
ax1 = fig.add_subplot(2, 2, 1)
ax1.hist(Pclass1['Age'])
ax1.set_xlabel("Age")
ax1.set_ylabel("Number of passengers in Class 1")

# top right
Pclass2 = all_data[all_data['Pclass']==2]
ax2 = fig.add_subplot(2, 2, 2)
ax2.hist(Pclass2['Age'])
ax2.set_xlabel("Age")
ax2.set_ylabel("Number of passengers in Class 2")

# bottom left
Pclass3 = all_data[all_data['Pclass']==3]
ax3 = fig.add_subplot(2, 2, 3)
ax3.hist(Pclass3['Age'])
ax3.set_xlabel("Age")
ax3.set_ylabel("Number of passengers in Class 3")


# show plots
fig.show()
# Median age for each category.
all_data.groupby(['Pclass','Sex'])['Age'].median()
# Median age for each category.
train_data.groupby(['Pclass','Sex'])['Age'].median()
# Replace missing age values with median in each category.
all_data['Age'] = all_data.groupby(['Pclass','Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
# Confrim that all missing age values is replaced.
age_missing = all_data['Age'].isnull().sum()
print("Missings for Age in the entire data: " + str(age_missing))
print("Missing percentage for Age in the entire data: " + str(round(age_missing*100/len(all_data),2))+ '%')
all_data.loc[all_data['Fare'].isnull()]
same_thomas = all_data.loc[(all_data['Pclass']==3) & (all_data['Embarked']== 'S') & (all_data['SibSp']==0)]
same_thomas
same_thomas['Fare'].plot(kind='hist', figsize=(8, 5), bins = 25)

plt.title('Fare of passengers with similar condition with Mr. Thomas') # add a title to the histogram
plt.xlabel('Fare') # add y-label
plt.ylabel('Frequency') # add x-label

plt.show()
# Use median instead of mean due to the pattern of the distribution.
median_thomas = same_thomas['Fare'].median()
median_thomas
all_data.loc[all_data['Fare'].isnull(),'Fare'] = median_thomas
# Confirm again.
all_data.loc[1043]
all_data.loc[all_data['Embarked'].isnull()]
same_amelie = all_data.loc[(all_data['Pclass']==1) & (all_data['Sex']== 'female') & (all_data['SibSp']==0)]
same_amelie['Embarked'].value_counts()
# Same thing applies on Mrs. Geroge. It seems that there is little correlation of 'Pclass', 'Sex' and 'SibSp' on 'Embarked'.
# Accroding to the following links, the embarkment is identified. So the missing values are replaced with 'S'.
# https://www.encyclopedia-titanica.org/titanic-survivor/amelia-icard.html
# https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html

all_data.loc[all_data['Embarked'].isnull(),'Embarked'] = 'S' 
# Confirm again.
all_data.loc[829]
# Check whether all missing values are replaced.
print('Missing in the entire data:') 
display(all_data.isnull().sum())
fig = plt.figure(figsize=(12,8))

# left
ax1 = fig.add_subplot(1, 2, 1)
ax1.boxplot(all_data['Age'])
ax1.set_xlabel("")
ax1.set_ylabel("Age")

# right
ax2 = fig.add_subplot(1, 2, 2)
ax2.boxplot(all_data['Fare'])
ax2.set_xlabel("")
ax2.set_ylabel("Fare")


# show plots
fig.show()
# cut: the bins are formed based on the values of the variable, regardless of how many cases fall into a category. 
# qcut: decompose a distribution so that there are the same number of cases in each category.
all_data['Fare'] = pd.qcut(all_data['Fare'],8)
all_data['Age'] = pd.cut(all_data['Age'].astype(int),8)
all_data['Fare'].value_counts()
all_data['Age'].value_counts()
age_survival = all_data[['Age','Survived']].groupby('Age')['Survived'].mean()
age_survival.plot(kind ='bar', figsize =(10,5), color = 'green')
plt.suptitle('Survival rate for each age categories')
fare_survival = all_data[['Fare','Survived']].groupby('Fare')['Survived'].mean()
fare_survival.plot(kind ='bar', figsize =(10,5), color = 'brown')
plt.suptitle('Survival rate for each fare categories')
