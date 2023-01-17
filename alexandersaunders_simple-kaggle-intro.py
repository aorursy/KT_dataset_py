# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Imports the train csv file and displays the top 5 rows



train = pd.read_csv("../input/train.csv",index_col=0)

train.head()
# Creates a correlation matrix



train.corr()
# Replaces the text stored in the Sex field with numbers



train['Sex'] = train['Sex'].replace(to_replace='female', value=1)

train['Sex'] = train['Sex'].replace(to_replace='male', value=0)
# Displays a new correlation matrix with the Sex field now visible



train.corr()
# End of Part 1
# Begining of part 2

# Next we will introduce a simple Random Forest Classifier to generate and display a decision tree showing the branches for prediction whether a passenger survives or not.
# Some data cleaning

# Checking to see if there is any missing data

train_part2 = train.copy()



train_part2.isna().sum()
# Removes the Age & Cabin fields due to missing data

train_part2 = train_part2.drop(['Age','Cabin'],axis=1)



# Replaces the 2 na's in the Embarked column with the most common value

embarked_mode = train_part2['Embarked'].mode()

train_part2['Embarked']=train_part2['Embarked'].fillna(value=embarked_mode[0])
# This now shows that we don't have any missing data in the fields we are keeping

train_part2.isna().sum()
print("Find out the types of the remaining fields of data:")

print(train_part2.dtypes)



print("See how many unique values there are in each of the fields:")

print(train_part2.nunique())
# Remove the Name & Ticket fields as they contain text which isn't of much use to us at the moment

train_part2 = train_part2.drop(['Name','Ticket'],axis=1)



# Further remove fields SibSp, Parch & Fare to be left only with Pclass,Sex, & Embarked

# This is just to keep the model simple for now

train_part2 = train_part2.drop(['SibSp','Parch','Fare'],axis=1)

train_part2.head()
# Next we are going to OneHotEncode the Pclass & Embarked fields

# This works by creating a field with either a 1 or a 0 for each unique value within a field

# i.e. converting a field from being a list of values e.g. [a,b,c,b,a] would become

# a [1,0,0,0,1], b [0,1,0,1,0], c [0,0,1,0,0]
# Libary to carry out onehotencoding 

from sklearn.preprocessing import OneHotEncoder



onehot_encoder = OneHotEncoder(sparse=False)

train_encoded = onehot_encoder.fit_transform(train_part2)



print("The shape of the data before we carried out the OneHotEncoding was")

print(train_part2.shape)



print("The shape of the data after we carried out the OneHotEncoding was")

print(train_encoded.shape)
train_encoded = pd.DataFrame(train_encoded,columns=onehot_encoder.get_feature_names(train_part2.columns))

train_encoded.head()
# We don't need the Survived_0 fields so we can drop that,

# Now putting the data in to the right format to train the ML model



X = train_encoded.iloc[:,2:]

y = train_encoded.iloc[:,1]



print(X.head())

print(y.head())
# We are now ready to train our ML model

# Imports RandomForest Machine learning libary



from sklearn.ensemble import RandomForestClassifier
#Specify the parameters for the Random Forest model

clf = RandomForestClassifier(n_estimators=100, max_depth=3,random_state=0)
#Fit the model to our training data

clf.fit(X, y)
#Next we can see what the most important features are from the training data



df_eval = pd.DataFrame(X.columns,columns=['Field'])

df_eval['FeatureImportance'] = pd.Series(clf.feature_importances_)

df_eval.head(8)
# Select a sinlge decision tree to display by chaging the figure inside the [] as long as it is between 0 & 1 less than your n_estimators parameter specified above

estimator = clf.estimators_[99]
# Next we are going to create a .dot file which contains details around the decision tree selected above

from sklearn.tree import export_graphviz



# Export as dot file

# Note that while the decision tree depth can be defined in the parameters of the RandomForest Classifier, if you want to display a smaller depth then this can be changed here

export_graphviz(estimator, out_file='tree.dot', feature_names = X.columns, proportion=True, max_depth=3,rounded = True, precision = 2, filled = True)
# Converts the dot file to a .png file

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
# Displays the .png file

import matplotlib.pyplot as plt



plt.figure(figsize = (14, 18))

plt.imshow(plt.imread('tree.png'))

plt.axis('off')

plt.show()
# Next we need to manipulate the test data so that it is in the same format as the train data which was used to train the model

# Importing the test csv data, this is the same format as the train data, but without the Survived field

test = pd.read_csv("../input/test.csv",index_col=0)

test.head()
# Replace femail with 1's & male with 0's



test['Sex'] = test['Sex'].replace(to_replace='female', value=1)

test['Sex'] = test['Sex'].replace(to_replace='male', value=0)
# Check to see if there is any missing data from any other fields which we are wanting to use



test.isna().sum()
# Drop fields which we aren't interested in at the moment



test = test.drop(['Name','Age','SibSp','Parch','Ticket','Fare','Cabin'], axis=1)

test.head()
# Carry out OneHotEncoding on the test data which we have kept



onehot_encoder = OneHotEncoder(sparse=False)

test_encoded = onehot_encoder.fit_transform(test)

test_encoded = pd.DataFrame(test_encoded,columns=onehot_encoder.get_feature_names(test.columns))

test_encoded.head()
# Carry out a prediction using the RandomForest Classification model from above



y_pred = clf.predict(test_encoded)
# Put the prediction in to a DataFrame and manipulate it so that it is in a format which can be submitted



y_pred = pd.DataFrame(y_pred, columns=['Survived'])

y_pred['PassengerId'] = test.index

y_pred = y_pred[['PassengerId','Survived']] 

y_pred.head()
# Export the prediction as a csv file



y_pred.to_csv("y_pred.csv", index=False)



# This gives a public leaderboard score of 0.77990

# Which on the rolling leaderboard gave a position of 5129/11027 so a top half finish and we haven't done anything fancy and aren't even using half of the data!