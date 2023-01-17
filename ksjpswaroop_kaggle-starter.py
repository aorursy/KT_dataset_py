# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import linear_model as lm

from sklearn.cross_validation import train_test_split

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# reading the train file and saving it as a pandas data frame #

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# dimensions of the input data (number of rows and columns) #

print("Train dataframe shape is : {}".format(train_df.shape))

print("Test dataframe shape is : {}".format(test_df.shape))
# name of the columns #

train_df.columns
test_df.columns
# taking a look at the first few rows #

train_df.head(10)
# getting the summary statistics of the numerical columns #

train_df.describe()
# getting the datatypes of the individual columns #

train_df.dtypes
# more information about the dataset #

train_df.info()
test_df.info()
# dropping the cabin variable #

train_df.drop(['Cabin'], axis=1, inplace=True)

test_df.drop(['Cabin'], axis=1, inplace=True)
# let us get some plots to see the data #

train_df.Survived.value_counts().plot(kind='bar', alpha=0.6)

plt.title("Distribution of Survival, (1 = Survived)")
# scatter plot between survived and age #

plt.scatter(range(train_df.shape[0]), np.sort(train_df.Age), alpha=0.2)

plt.title("Age Distribution")
train_df.Pclass.value_counts().plot(kind="bar", alpha=0.6)

plt.title("Class Distribution")
train_df.Embarked.value_counts().plot(kind='bar', alpha=0.6)

plt.title("Distribution of Embarked")
train_male = train_df.Survived[train_df.Sex == 'male'].value_counts().sort_index()

train_female = train_df.Survived[train_df.Sex == 'female'].value_counts().sort_index()



ind = np.arange(2)

width = 0.3

fig, ax = plt.subplots()

male = ax.bar(ind, np.array(train_male), width, color='r')

female = ax.bar(ind+width, np.array(train_female), width, color='b')

ax.set_ylabel('Count')

ax.set_title('DV count by Gender')

ax.set_xticks(ind + width)

ax.set_xticklabels(('DV=0', 'DV=1'))

ax.legend((male[0], female[0]), ('Male', 'Female'))

plt.show()
# getting the necessary columns for building the model #

train_X = train_df[["Pclass", "SibSp", "Parch", "Fare"]]

train_y = train_df["Survived"]

test_X = test_df[["Pclass", "SibSp", "Parch", "Fare"]]
# split the train data into two samples #

dev_X, val_X, dev_y, val_y = train_test_split(train_X, train_y, test_size=0.33, random_state=42)



# Build the machine learning model - in this case, logistic regression #

# Initialize the model #

clf = lm.LogisticRegression()



# Build the model on development sample #

clf.fit(dev_X, dev_y)



# Predict on the validation sample #

val_preds = clf.predict(val_X)

print(val_preds[:10])

# import the function that computes the accuracy score #

from sklearn.metrics import accuracy_score



accuracy_score(val_y, val_preds)
val_preds = clf.predict_proba(val_X)

val_preds[:10]