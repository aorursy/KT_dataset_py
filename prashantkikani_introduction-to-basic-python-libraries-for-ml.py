import pandas as pd
titanic_data = pd.read_csv("../input/titanic/train.csv")
type(titanic_data)
# first 5 rows of the table

titanic_data.head()
titanic_data.describe()
titanic_data.shape
titanic_data[['Name']].head()
titanic_data.columns
titanic_data['Sex'].value_counts()
titanic_data.sort_values(by="Fare")
titanic_data = titanic_data.fillna(value = 0)

titanic_data
titanic_data = titanic_data[titanic_data['Age'] > 0][titanic_data['Age'] < 100]

titanic_data
titanic_data = titanic_data[titanic_data['Fare'] > 0][titanic_data['Fare'] < 200]

titanic_data
titanic_data[titanic_data['Sex'] == "female"]
titanic_data['new_column'] = titanic_data['SibSp'] + titanic_data['Parch']

titanic_data.head(10)
import numpy as np
a = np.array([1, 2, 3, 67, 34])
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])
print(a)

a[0] = 77

print(a)
b = np.array([[1,2,3],[4,5,6]])

print(b.shape)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

print(a)
b = a[:2, 1:3]

print(b)
# To print 2nd raw of array a. Array index starts from 0.

print(a[1, :])
# To print 1st column of array a. Array index starts from 0.

print(a[:, 0])
# To print all numbers more than 2

print(a[a > 2])
# Datatypes

print(a.dtype)
x = np.array([1.0, 2.0, 3.0])

print(x.dtype)
x = np.array([1, 2], dtype=np.int64) # Force a particular datatype

print(x.dtype)
# Maths in Numpy arrays

x = np.array([[1,2,89],[3,4,54]], dtype=np.float64)

y = np.array([[5,6,45],[7,8,56]], dtype=np.float64)

print(x + y)
x - y
x * y
x / y
# Dot product

v = np.array([9,10,99])

x.dot(v)
# Sum of all the elements

x.sum()
# Transpose the array

x.T
# We can do operations to specific columns / rows also

v = np.array([0, 69, 1])

x[1, :] += v

x
x
v = np.array([1, 0, 1])

x + v
import sklearn
titanic_data[['Age']].head(10)
# import the preprocessing

from sklearn import preprocessing



# Make an object of StandardScaler

ss = preprocessing.MinMaxScaler(feature_range=(0, 1))



# Fit the StandardScaler

titanic_data['Age_scaled'] = ss.fit_transform(titanic_data['Age'].to_numpy().reshape(-1, 1))



# Show results

titanic_data[['Age_scaled']].head(10)
titanic_data['Age_scaled'].max(), titanic_data['Age_scaled'].min()
titanic_data[['Sex']].head()
# import the preprocessing

from sklearn import preprocessing



# Make an object of LabelEncoder

le = preprocessing.LabelEncoder()



# Fit the LabelEncoder

titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'])



# Show results

titanic_data[['Sex']].head()
from sklearn.model_selection import train_test_split
# rows & columns in data

titanic_data.shape
train_data, validation_data = train_test_split(titanic_data, test_size=0.2)

train_data.shape, validation_data.shape
891 * 0.8
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("In general air pressure and density decrease with altitude in the atmosphere.")

for token in doc:

    print(token.text, token.pos_)
doc = nlp("I was playing football. I kicked the ball.")

for token in doc:

    print(token.text, token.lemma_)
from spacy import displacy



doc = nlp("I was playing football & cricket last week.")



displacy.render(doc, style="dep",jupyter=True)
# import library

from textblob import TextBlob



sentence = TextBlob("Ronaldo & Messi are one of the best footballers in the world")

sentence.sentiment
sentence = TextBlob('Use 4 spaces per indentation level.')

print(sentence.words[2])

print(sentence.words[2].singularize())
print(sentence.words[-1])

print(sentence.words[-1].pluralize())
b = TextBlob("There are lot of speling miskate in this sentance")

b.correct()
import seaborn as sns



# We also import matplotlib library as a support

import matplotlib.pyplot as plt
sns.countplot(x="Sex", data=titanic_data)
age_dist = titanic_data['Age'].tolist()



sns.distplot(age_dist)

plt.show()
age_dist = titanic_data['Fare'].tolist()



sns.distplot(age_dist);
sns.pointplot(x="Sex", y="Pclass", data=titanic_data);
sns.pointplot(x="Pclass", y="Survived", data=titanic_data);
sns.lmplot(x="Pclass", y="Fare", data=titanic_data);
sns.stripplot(x="Sex", y="Fare", data=titanic_data);
sns.boxplot(x=titanic_data["Age"]);
sns.boxplot(x=titanic_data["Fare"]);
sns.pairplot(titanic_data[["Fare", "Age", "Pclass", "Survived"]], hue="Survived");