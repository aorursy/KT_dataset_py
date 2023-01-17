import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Data Visualization
df = pd.read_csv('../input/bank_note_data.csv')
df.head()
#Creating a countplot of classes
sns.countplot(x = 'Class' , data = df)
sns.pairplot(df , hue = 'Class')
#Normalizing the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('Class' , axis =1))
scaled_features = scaler.transform(df.drop('Class' , axis =1))
scaled_features = pd.DataFrame(scaled_features,columns=df.columns[:-1])

#Creating training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['Class'],
                                                    test_size=0.30)
#Creating and training the classifier
import tensorflow as tf
import tensorflow.contrib.learn as learn

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]
classifier = learn.DNNClassifier(hidden_units=[10, 20, 10],feature_columns = feature_columns, n_classes=2)
classifier.fit(X_train, y_train, steps=200, batch_size=20)
#Predictions
predictions = classifier.predict(X_test,as_iterable=False)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

