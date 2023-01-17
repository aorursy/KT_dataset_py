import numpy as np 
import pandas as pd 
dataset = pd.read_csv("../input/Iris.csv")
dataset.head()
dataset.info()
dataset.describe()
X = dataset.iloc[:,1:5].values
y = dataset.iloc[:, 5].values
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy')
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
from matplotlib import pyplot as plt
import seaborn as sns
sns.pairplot(data=dataset.drop(['Id'],axis=1),hue='Species')
sns.heatmap(dataset.drop(['Id'],axis=1).corr())
sns.kdeplot(data=y_test,data2=y_pred,shade=True,bw='scott', cbar=True, cmap='Reds')

