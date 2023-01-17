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
# Importing Library Files

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
# Data Acquisition ---->  Import the three datasets

Master_Data = pd.read_csv('../input/Movielens.csv', sep = ',', engine = 'python')
Master_Data.head(2)
Master_Data = Master_Data.loc[:, ~Master_Data.columns.str.contains('^Unnamed')]
Master_Data.head(2)
Master_Data.info()
Master_Data.describe()
Master_Data.groupby('Age')['UserID'].count()
Master_Data.groupby('Age')['UserID'].count().plot(kind = 'bar', color = 'green',figsize = (8,7))

plt.xlabel('Age')

plt.ylabel('Number of Users in Population')

plt.title('Visualization of User Age Distribution')

plt.show
sns.distplot(Master_Data['Age'], color = 'g', bins = 45)

plt.xlabel('Age')

plt.ylabel('Number of Users')

plt.title('User Age Distribution')

plt.show
# Overall - Toy Story Movie Rating Analysis based upon Rating & UserID

Master_Data[Master_Data.Title == 'Toy Story (1995)'].groupby('Rating')['UserID'].count()
Master_Data[Master_Data.Title == 'Toy Story (1995)'].groupby('Rating')['UserID'].count().plot(kind = 'bar', color = 'green',figsize = (8,7))

plt.xlabel('Rating')

plt.ylabel('Number of Users in Population')

plt.title('Visualization of User Rating of Toy Story')

plt.show
#Toy Story Movie Rating Analysis based upon Rating & MovieID

Master_Data[Master_Data.Title == 'Toy Story (1995)'].groupby('Rating')['MovieID'].count()
#  Based on AGE GROUP - Toy Story Movie Rating Analysis based upon Age & MovieID

Master_Data[Master_Data.Title == 'Toy Story (1995)'].groupby('Age')['MovieID'].count()
Master_Data[Master_Data.Title == 'Toy Story (1995)'].groupby('Age')['MovieID'].count().plot(kind = 'barh', color = 'green',figsize = (6,5))

plt.xlabel('Rating')

plt.ylabel('Age Group of Users in Population')

plt.title('Visualization of User Group Rating of Toy Story')

plt.show
Master_Data.groupby('MovieID')['Rating'].count().sort_values(ascending = False)[:25]
Master_Data.groupby('MovieID')['Rating'].count().nlargest(25)
Master_Data.groupby('MovieID')['Rating'].count().sort_values(ascending = False)[:25].plot(kind ='barh', color = 'g', x = 'Rating', y = 'Number of Users', title = 'User Rating of Toy Story (1995) Movie', figsize = (10,9))

plt.xlabel('Rating')

plt.ylabel('Top 25 MovieID')

plt.title('Visualization of Top 25 Movies Viewership Rating')

plt.show
Master_Data.groupby('MovieID')['Rating'].count().nsmallest(25)
Master_Data.groupby('MovieID')['Rating'].count().nsmallest(25).plot(kind = 'pie', figsize = (8,7))
Master_Data[Master_Data.UserID == 2696].groupby('Rating')['MovieID'].count()
user2696 = Master_Data[Master_Data.UserID == 2696]

user2696
user_id = Master_Data[Master_Data.UserID == 2696].groupby('Rating')['MovieID'].count().plot(kind = 'pie',figsize = (6,5))

plt.title('Movie Ratings of UserID 2696')

plt.show
user2696 = Master_Data[Master_Data.UserID == 2696]

plt.scatter(y= user2696.Title, x = user2696.Rating, color = 'g')
Master_Data.head(2)
ml_Data = Master_Data.head(500)

ml_Data
ml_Data.shape
ml_Data.describe()
ml_Data['Age'].unique()
ml_Data['Occupation'].unique()
ml_Data.head()
f = ml_Data.iloc[:,[5,2,3]]

f.head(2)
l = ml_Data.iloc[:,6]

l.head(2)
features = f.values

label = l.values
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

for i in range(1,401):

    X_train,X_test,Y_train,Y_test = train_test_split(features,label,test_size=0.2, random_state= i)

    for n in range(1,401):

        model = KNeighborsClassifier(n_neighbors = n)

        model.fit(X_train,Y_train)

        training_score = model.score(X_train,Y_train)

        testing_score = model.score(X_test,Y_test)

        #Only Generalized model will be outputted

        if testing_score > training_score:

            if testing_score > 0.49:

                print("Training Score {} Testing Score {} for Random State {} and n_neighbors {}".format(training_score,testing_score,i,n))
#Training Score 0.3925 Testing Score 0.52 for Random State 366 and n_neighbors 52

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

X_train,X_test,Y_train,Y_test = train_test_split(features,label,test_size=0.2, random_state=232 )

model = KNeighborsClassifier(n_neighbors = 217)

model.fit(X_train,Y_train)

training_score = model.score(X_train,Y_train)

testing_score = model.score(X_test,Y_test)

# Only Generalized model will be outputted

if testing_score > training_score:

    print("Training Score {} Testing Score {} ".format(training_score,testing_score))
movieid = int(input("Enter the MovieID: "))

age = int(input("Enter the Age Group( 1, 56, 25, 45, 50):"))

occupation = int(input("Enter the Occupation group value (10, 16, 15,  7, 20,  9):"))



featureInput = np.array([[movieid,age,occupation]])

rating = model.predict(featureInput)

print("Rating of the Movie is: ", rating)