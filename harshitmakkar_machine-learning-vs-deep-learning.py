# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import warnings

warnings.simplefilter('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')
data.head()
data.info()
sns.pairplot(data=data,hue='Species')
data = data.drop('Id',axis=1)
data['Species'].unique()
#categorising species into numerical values



def turn_numeric(iris_x):

    if iris_x == 'Iris-setosa':

        return 0

    if iris_x == 'Iris-versicolor':

        return 1

    if iris_x == 'Iris-virginica':

        return 2

    else:

        print(iris_x)

        return
data['Species'] = data['Species'].apply(turn_numeric)
data.head()
data.isnull().values.any()
X = data.drop('Species',axis=1)

y = data['Species']
#scaling values so that euclidean distance can do its work

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)
#scaled features

scaled_features
df_feat = pd.DataFrame(data=scaled_features,columns=X.columns)
df_feat.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_feat, y, test_size=0.3,random_state=101)
from sklearn.neighbors import  KNeighborsClassifier
#elbow method to calculate k value



error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))



plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,marker='o')
KNN = KNeighborsClassifier(n_neighbors=7)

KNN.fit(X_train,y_train)

actual_pred = KNN.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,actual_pred))

print('\n')

print(confusion_matrix(y_test,actual_pred))
import tensorflow as tf
data.head()
X_deep = data.drop('Species',axis=1)

y_deep = data['Species']
X_deep_train, X_deep_test, y_deep_train, y_deep_test = train_test_split(X_deep, y_deep, test_size=0.3,random_state=101)
#creating feature columns

feat_cols = []



for col in X_deep.columns:

    feat_cols.append(tf.feature_column.numeric_column(col))
feat_cols
#creating an input function - 

input_func = tf.estimator.inputs.pandas_input_fn(x=X_deep_train,y=y_deep_train,batch_size=10,num_epochs=5,shuffle=True)
#defining a classifier

classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3,feature_columns=feat_cols)
#fitting the classifier on the input function

classifier.train(input_fn=input_func,steps=50)
#evaluation

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_deep_test,batch_size=len(X_deep_test),shuffle=False)
note_predictions = list(classifier.predict(input_fn=pred_fn))
#each dictionary corresponds to a prediction - class_ids is the prediction

note_predictions
final_preds  = []

for pred in note_predictions:

    final_preds.append(pred['class_ids'][0])
print(classification_report(y_deep_test,final_preds))

print('\n')

print(confusion_matrix(y_deep_test,final_preds))