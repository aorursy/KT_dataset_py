# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn import metrics

train_data = pd.read_csv("../input/train.csv",na_values= " ")

train_data.tail()
train_data.dtypes
missing_values_attributes = {} # Dictionary to contain attributes which contains missing values as key and the number of missing values as value



# Iterating over each attribute and checking whether that attribute contains missing values or not

for attribute in train_data:

    if train_data[attribute].isnull().any() == True:

        missing_values_attributes[attribute] = sum(train_data[attribute].isnull())

        

    
missing_values_attributes
# Imputing the 'Embarked' attribute with its mode value

train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])



# Imputing 'Age' with mean age

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())



# Dropping the 'Cabin' attribute

del train_data['Cabin']
categorical_attributes = ["Survived", "Pclass", "Embarked", "Sex" ] # Attributes to be converted to categorical



for attribute in categorical_attributes:

    train_data[attribute] = train_data[attribute].astype("category")



train_data.dtypes
# Checking the distribution of class

train_data['Survived'].value_counts()
# List containing all the relevant attributes to be included in the model

attr_list = [ item for item in train_data.columns if item not in  ["PassengerId", "Name", "Ticket", "Survived"]]



# List of categorical attributes

cat_vbl = [item for item in attr_list if train_data[item].dtype.name == "category"]



# List of numeric attributes

num_vbl = [item for item in attr_list if train_data[item].dtype.name != "category"]



# Creating data-frame from the attr_list

X = train_data.ix[:, attr_list]



# Creating label class/predicted class 

Y = train_data["Survived"]



# Standardizing the numerical variables

X[num_vbl] = X[num_vbl].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))



# Creating dummy variables for the categorical attributes

dumm = pd.get_dummies(X[cat_vbl])

new_X = pd.concat([X,dumm], axis=1)



# Dropping the actual categorical variables from the dataframe

new_X.drop(new_X[cat_vbl], axis=1, inplace=True)

categorical_attr = [item for item in attr_list if train_data[item].dtype == "category"]

num_att = [item for item in attr_list if train_data[item].dtype != "category"]
for item in attr_list:

    if train_data[item].dtype == train_data[""].dtype:

        print(item)
# Creating train test split (75-25 split)

X_train, X_test, Y_train, Y_test = train_test_split(new_X,Y,test_size=0.25,random_state=123)
model = LogisticRegression()

model = model.fit(X_train, Y_train)
model.score(X_train, Y_train)
predicted = model.predict(X_test)

print(predicted)
probs = model.predict_proba(X_test)

print(metrics.accuracy_score(Y_test, predicted))

print(metrics.roc_auc_score(Y_test, probs[:, 1]))