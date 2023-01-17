#importing modules

import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

import random



#pulling in data

df = pd.read_csv('../input/Uber_Ride_Reviews.csv')

df.drop(['sentiment'],axis = 1)

df.head()
#seperating by groups

groups = df.groupby('ride_rating').count()

Values = groups.ride_review

colors = ['r', 'g', 'b', 'c', 'm']

#making bar plot

plt.bar(([1,2,3,4,5]), Values, color= colors)

plt.title('Rating Distribution')

plt.xlabel('Rating')

plt.ylabel('Review Quantity')

plt.show()
#checking for nulls

null_count = df.isnull().sum()

null_count
#deleting all instances with ride_rating = 3

df = df[df.ride_rating != 3]



#seperating by groups

groups = df.groupby('ride_rating').count()

Values = groups.ride_review

colors = ['r', 'g', 'b', 'c']



#making bar plot

plt.bar(([1,2,4,5]), Values, color= colors)

plt.title('Rating Distribution')

plt.xlabel('Rating')

plt.ylabel('Review Quantity')

plt.show()
#creating new binary_class column

df['binary_class'] = np.where(df['ride_rating'] > 3, 1, 0)

df
#splitting into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['ride_review'], df['binary_class'], random_state = 0)



#setting random number between 1 and 1000

number = random.randint(1,1000)



#printing random training text and X_train shape

print ('Random Review:')

print(' ')

print(X_train[number])

print(' ')

print('X_train shape: ' + str(X_train.shape))
#importing countvectorizer

from sklearn.feature_extraction.text import CountVectorizer



#creating variable which assigns X_train to numbers

vect = CountVectorizer().fit(X_train)



#translates numbers back to text

vect.get_feature_names()[1:10]
#length of total words

len(vect.get_feature_names())
#creating matrix array for logistic regression

X_train_vectorized = vect.transform(X_train)

print (X_train_vectorized.toarray())
#creating log regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_vectorized, y_train)
#calculating AUC

from sklearn.metrics import roc_auc_score

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
#creating array variable of all the words

feature_names = np.array(vect.get_feature_names())



#creating array of all the regression coefficients per word

coef_index = model.coef_[0]



#creating df with both arrays in it

df = pd.DataFrame({'Word':feature_names, 'Coef': coef_index})



#sorting by coefficient

df.sort_values('Coef')
print(model.predict(vect.transform(['abandoned great'])))

print(model.predict(vect.transform(['great she the best'])))

print(model.predict(vect.transform(['charged slow horrible'])))

print(model.predict(vect.transform(['it was as average as a trip could be'])))

print(model.predict(vect.transform(['my family felt safe we got to our destination with ease'])))

print(model.predict(vect.transform(['i got to my destination quickly and affordably i had a smile on my face from start to finish'])))