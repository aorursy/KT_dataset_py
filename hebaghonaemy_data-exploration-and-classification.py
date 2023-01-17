import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



df = pd.read_csv('/kaggle/input/iris/Iris.csv')

df.head()
#Set id column to be index

df.set_index('Id' , inplace = True)

df.head()
print(df.info()) #information on data types,memory usage , null or non-null values
print(df.describe()) #descirptive statstics
df['Species'].value_counts()
sns.pairplot(df , hue = 'Species')
#determin bin size based on the values of the dataset

bins_sepal_len = np.arange(df['SepalLengthCm'].min()-0.5 , df['SepalLengthCm'].max()+0.5 , 0.5)

bins_sepal_wid = np.arange(df['SepalWidthCm'].min()-0.5 , df['SepalWidthCm'].max()+0.5 , 0.5)

bins_petal_len = np.arange(df['PetalLengthCm'].min()-0.5 , df['PetalLengthCm'].max()+0.5 , 0.5)

bins_petal_wid = np.arange(df['PetalWidthCm'].min()-0.5 , df['PetalWidthCm'].max()+0.5 , 0.5)



plt.figure(figsize = (15,15))



plt.subplot(221)

plt.hist(data = df, x = 'SepalLengthCm' , bins = bins_sepal_len)

plt.title('Sepal Length')



plt.subplot(222)

plt.hist(data = df, x = 'SepalWidthCm' , bins = bins_sepal_wid)

plt.title('Sepal Width')



plt.subplot(223)

plt.hist(data = df, x = 'PetalLengthCm' , bins = bins_petal_len)

plt.title('Petal Length')



plt.subplot(224)

plt.hist(data = df, x = 'PetalWidthCm' , bins = bins_petal_wid)

plt.title('Petal Width')
#Separ data according to there species

df1 = df[df.Species=='Iris-setosa']

df2 = df[df.Species=='Iris-versicolor']

df3 = df[df.Species=='Iris-virginica']




plt.figure(figsize = (15,15))



plt.hist(df1.SepalLengthCm,bins=30)

plt.hist(df2.SepalLengthCm,bins=30)

plt.hist(df3.SepalLengthCm,bins=30)

plt.legend(['Setosa','Versicolor','Virginica'])

plt.title('Sepal Length')



plt.subplot(221)

sns.distplot(df1['SepalLengthCm'] , kde = False , bins = 30)

sns.distplot(df2['SepalLengthCm'] , kde = False , bins = 30)

sns.distplot(df3['SepalLengthCm'] , kde = False , bins = 30)

plt.title('Sepal Length')



plt.subplot(222)

sns.distplot(df1['SepalWidthCm'] , kde = False, bins = 30)

sns.distplot(df2['SepalWidthCm'] , kde = False, bins = 30)

sns.distplot(df3['SepalWidthCm'] , kde = False, bins = 30)

plt.title('Sepal Width')



plt.subplot(223)

sns.distplot(df1['PetalLengthCm'] , kde = False ,bins = 30)

sns.distplot(df2['PetalLengthCm'] , kde = False ,bins = 30)

sns.distplot(df3['PetalLengthCm'] , kde = False ,bins = 30)

plt.title('Petal Length')



plt.subplot(224)

sns.distplot(df1['PetalWidthCm'] , kde = False , bins = 30)

sns.distplot(df2['PetalWidthCm'] , kde = False , bins = 30)

sns.distplot(df3['PetalWidthCm'] , kde = False , bins = 30)

plt.title('Petal Width')
sns.heatmap(df.corr() , annot=True)
#drop the results column

results = df['Species']

df = df.drop(columns = ['Species'])

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score





X_train , X_test , y_train , y_test = train_test_split(df, results, test_size = 0.25, random_state = 0)
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

score = accuracy_score(y_test , y_predict)



print(score)