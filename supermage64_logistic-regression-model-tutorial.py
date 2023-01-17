import pandas as pd

import numpy as np
# We will be using the Titanic Modelling Data from Kaggle (https://www.kaggle.com/hesh97/titanicdataset-traincsv)

#from google.colab import files

#uploaded = files.upload()
data1 = pd.read_csv('../input/titanic/Titanic Modelling data.csv')
avgfaresex = data1.groupby('Sex').mean()

avgfaresex = avgfaresex[['Fare']]

avgfaresex[['Avg Fare ,Sex']] = avgfaresex[['Fare']]

avgfaresex = avgfaresex.drop(columns=['Fare'])

avgfaresex.head()
# The below code would be used to merge the new created column to the original data sheet.

#

# data1 = data1.merge(avgfaresex, how = 'left', left_on = 'Sex',right_on = 'Sex')
# Things to deduce:

#

# If the gender of the passenger is female, the fares are likely to cost more than if a man were to buy the exact same ticket
minpclass = data1.groupby('Pclass').mean()

minpclass = minpclass[['Fare']]

minpclass['avg Fare ,Pclass'] = minpclass['Fare']

minpclass = minpclass.drop(columns=['Fare'])

minpclass.head()
# The below code would be used to merge the new created column to the original data sheet.

#

# data1 = data1.merge(minpclass, how = 'left', left_on = 'Pclass',right_on = 'Pclass')
# Things to deduce:

#

# The higher the class (1 being the highest), the more expensive the tickets are likely to be.
avgsage = data1.groupby('Survived').mean()

avgsage = avgsage[['Age']]

avgsage['avg Age, Survived'] = avgsage['Age']

avgsage = avgsage.drop(columns=['Age'])

avgsage.head()
# The below code would be used to merge the new created column to the original data sheet.

#

# data1 = data1.merge(avgsage, how ='left', left_on = 'Survived', right_on = 'Survived')
# Things to deduce:

#

# The older the age , the more likely the passenger is to not survive.
# Getting rid of useless columns (AccountYearId, Product Group (Year))

df = data1.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'])
# Replacing all empty sets with 0

df = df.fillna(0)

df['Age'] = df['Age'].fillna(0)

df.isna().sum()
# Turning boolean values into integers

#.astype(int)
# You then need to make an 'x' set and 'y' set to plug into the model.

# The x set is the perimiters the predictions will be based on.

# The y set is the predicted value (in this case it would be 'Survived').
df.columns
# x set

x = df[['Pclass', 'Sex', 'Age', 'Fare']]
# y set

y = df[['Survived']]
#Now the data sets must be split into training data and testing data.
from sklearn.model_selection import train_test_split
# The Great Split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#Logistic Regression Model (change this code to change the model)

from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression().fit(X_train, y_train)
# Now you have to train the model
# Run the predictions

X_test['prediction'] = logistic.predict(X_test)

X_test.head()
# Don't run this as a set up step

#X_test['prediction'].value_counts(dropna=0)
# Compare the coefficients with the columns they represent

logistic.coef_
x.columns

# Things to deduce:

#

# 1. The sex of the passenger will have the most effect on whether they survive or not

# 2. If the sex is a male (i.e the column has a 1), it  the chances of surviving decreases by a HUGE amount (the coefficient is a large negative)