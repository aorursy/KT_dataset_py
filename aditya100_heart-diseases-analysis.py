import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/heart.csv')
df.head()
df.shape
df.dropna()
df.shape
df.describe()
plt.figure(figsize=(25, 10))
p = sns.heatmap(df.corr(), annot=True)
_ = plt.title('Correlation')
p = sns.countplot(x='sex', data=df)
p = sns.countplot(x='cp', data=df)
from sklearn.model_selection import train_test_split
Y = df['target']
X = df.drop(columns=['target'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #Splitting the dataset into training set and test set
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score # to find the accuracy for the trained model
# Setting Hyperparameters
clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='lbfgs', max_iter=1000)
_ = clf.fit(X_train, Y_train)
predictions = clf.predict(X_test) # Making Predictions
predictions
accuracy = accuracy_score(Y_test, predictions) # Finding Accuracy
accuracy
from sklearn.svm import SVC
# Setting Hyperparameters
clf_svm = SVC(gamma='auto', kernel='poly') 
_ = clf_svm.fit(X_train, Y_train)
predictions_svm = clf_svm.predict(X_test)
predictions_svm
accuracy_svm = accuracy_score(Y_test,predictions_svm)
accuracy_svm