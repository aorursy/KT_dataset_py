# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



# Figures inline and set visualization style

%matplotlib inline

sns.set()



# Import data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Check the head of training data set

df_train.head()
print (df_train.info())
# Store target variable of training data in a safe place

survived_train = df_train.Survived



# Concatenate training and test sets to create a new Merged_data

Merged_data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])



# Check info Merged_data 

Merged_data.info()
#Below gives visual indicator of columns which are having Null values

# Generate a custom diverging colormap

cmap = sns.diverging_palette(90, 980, as_cmap=True)

sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False, cmap=cmap)
#Age distribution is shown in below graph

sns.set_style('whitegrid')

df_train['Age'].hist(bins=35)

plt.xlabel('Age')
#Fare distribution is shown in below graph

sns.set_style('whitegrid')

df_train['Fare'].hist(bins=40)

plt.xlabel('Fare')
sns.jointplot(x='Fare',y='Age',data=df_train)
#Below graph indicates that Female survived in grater proportion as compared to male.. 

sns.countplot(x="Survived",hue="Sex",data=df_train)
# Below number shows that 74% of woman survived where as only 18% of men survived

df_train.groupby(['Sex']).Survived.sum()/df_train.groupby(['Sex']).Survived.count()

df_train.groupby(['Sex','Pclass']).Survived.sum()/df_train.groupby(['Sex','Pclass']).Survived.count()

# Merged data counts by values which are grouped by Class and Sex

Merged_data.groupby(['Pclass','Sex']).count()
#LEt us impute values for Age values (Average of 1st class is 38 , average age 2nd class 29 and 3rd class is 24)

plt.figure(figsize=(10,5))

sns.boxplot(x='Pclass',y='Age',data=df_train)



# I will try to create function for imputation of age

# Fill the missing numerical variables

#Merged_data['Age'] = Merged_data.Age.fillna(Merged_data.Age.median())

#Merged_data['Fare'] = Merged_data.Fare.fillna(Merged_data.Fare.median())

# Function to fill the age of blank values

def Fill_Age(Cols):

    Age = Cols[0]

    Pclass = Cols[1]



    if pd.isnull(Age) :

        if Pclass==1:

            return 38

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age
Merged_data['Age']=Merged_data[['Age','Pclass']].apply(Fill_Age,axis=1)
# Cabin has too many blank values, Ticket and cabin are not useful so better to drop these columns

Merged_data.drop(['Ticket','Cabin'],axis=1,inplace=True)
Merged_data.head()
#This is method by which Age can be categorised in categories

Merged_data['CatAge'] = pd.qcut(Merged_data.Age, q=4, labels=False )

Merged_data['CatFare'] = pd.qcut(Merged_data.Age, q=5, labels=False )
#Let us also put numerical values for Sex and Embarked

Gender = pd.get_dummies(Merged_data['Sex'],drop_first=True,prefix='Gender')

Embarked = pd.get_dummies(Merged_data['Embarked'],drop_first=True,prefix='Embarked')
# LEt us concat the above dummy values with Merged DAta Frame

Modified_data =pd.concat([Merged_data,Embarked,Gender],axis=1)

Modified_data.head()
# Embarked , Age, Sex, Name and Fare

Modified_data.drop(['Embarked','Age','Sex','Name','Fare'],axis=1,inplace=True)

Modified_data.head()
cmap = sns.diverging_palette(90, 980, as_cmap=True)

sns.heatmap(Modified_data.isnull(),yticklabels=False,cbar=False, cmap=cmap)
Modified_data.drop(['PassengerId','SibSp'],axis=1,inplace=True)

Modified_data.head()
Modified_data.head()
data_train = Modified_data.iloc[:891]

data_test = Modified_data.iloc[891:]
X = data_train.values

test = data_test.values

y = survived_train.values
# Instantiate model and fit to data

clf = tree.DecisionTreeClassifier(max_depth=3)

clf.fit(X, y)
# Make predictions and store in 'Survived' column of 

Y_pred = clf.predict(test)

df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('Decision_Tree_Classification_Fare.csv', index=False)