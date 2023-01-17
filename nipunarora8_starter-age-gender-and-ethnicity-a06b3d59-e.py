import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/age_gender.csv', delimiter=',')

df.dataframeName = 'age_gender.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head(5)
sns.countplot(x='age', data=df) #age distribution
sns.countplot(x='gender', data=df) #gender distribution
sns.countplot(x='ethnicity', data=df) #ethnicity distribution
df=df.sample(frac=1) # shuffle
df
X=df['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32")) #converting data to numpy array

X=np.array(X)/255.0 #normalization



X_t = []

for i in range(X.shape[0]):

    X_t.append(X[i].reshape(48,48,1)) #reshaping the data to (n,48,48,1)

X = np.array(X_t)



age=df['age'].values

gender=df['gender'].values
def plot(X,y):

    for i in range(5):

        plt.title(y[i],)

        plt.imshow(X[i].reshape(48,48))

        plt.show()
plot(X,age)
plot(X,gender)