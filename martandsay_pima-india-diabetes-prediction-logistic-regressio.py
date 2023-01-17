# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
df.head()
# Type of glass

df.Outcome.value_counts() # There are 500 patient with no diabetes(0) and 268 with diabetes.
# Split data into test and train

from sklearn.model_selection import train_test_split
col_names = ['Pregnancies', 'Glucose', 'BloodPressure',  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
df.head()
# Dependent Variable

X = df[col_names]
# Independent Variable

y = df.Outcome
# Splitting data into training and test set.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Feature Scaling

#from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()

#X_train = sc.fit_transform(X_train)

#X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred # predicted values
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

cnf_matrix = confusion_matrix(y_pred, y_test) # Confusion Matrix
cnf_matrix
class_names=[0,1] # name  of classes

fig, ax = plt.subplots(figsize=(10,8))

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
print("Accuracy:",accuracy_score(y_test, y_pred))

print("Precision:",precision_score(y_test, y_pred))
from keras.layers import Dense, Dropout

from keras.models import Sequential

# Splitting data into training and test set.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=45)
classifier = Sequential()
classifier.add(Dense(output_dim=4, init="uniform", activation="relu", input_dim=7)) #  first layer

classifier.add(Dropout(p=0.1))

classifier.add(Dense(output_dim=4, init="uniform", activation="relu")) # second layer

classifier.add(Dropout(p=0.1))



classifier.add(Dense(output_dim=1, init="uniform", activation="sigmoid")) # output layer
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
classifier.fit(X_train, y_train, batch_size=5, nb_epoch=100)