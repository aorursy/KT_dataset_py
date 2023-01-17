import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read in the file to a pandas dataframe

df = pd.read_csv("../input/Iris.csv")
df.head()
df.info()
df.describe()
sns.countplot(data=df,x='Species')
sns.boxplot(data=df,x='Species',y='SepalLengthCm')
#Adjust the size of the plot

#plt.figure(figsize=(10,5))

sns.pairplot(data=df,hue='Species',size=3)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop(['Id','Species'],axis=1))
scaled_data = scaler.transform(df.drop(['Id','Species'],axis=1))
df.columns[1:-1]
scaled_df = pd.DataFrame(data=scaled_data,columns=df.columns[1:-1])
scaled_df.head()
df.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report,confusion_matrix
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit()
import tensorflow.contrib.learn.python.learn as learn
feature_columns
classifier = learn.DNNClassifier(hidden_units=[10,20,10],n_classes=3)