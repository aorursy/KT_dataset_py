import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.naive_bayes import GaussianNB # importing Naive Bayes model

from sklearn.model_selection import train_test_split # split the data into train and test data

from sklearn.metrics import accuracy_score # calculate accuracy score of the model

from sklearn.metrics import confusion_matrix # generate confusion matrix



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/iris/Iris.csv')

df.head()
df.tail()
df.dtypes
df.describe()
df['Species'].unique()
df.set_index('Id',inplace=True)

df.head()
X = df.drop('Species',axis = 1)

y = df['Species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)
gauss = GaussianNB()

model = gauss.fit(X_train,y_train)
predict = model.predict(X_test)

accuracy = accuracy_score(predict,y_test)
print(accuracy)
confusion_matrix(predict,y_test)
import seaborn as sns
sns.countplot(df['Species'])
df.corr()
sns.pairplot(data=df,hue='Species')