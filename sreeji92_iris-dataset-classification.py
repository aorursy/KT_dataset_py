#importing basic libraires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
#Load Iris dataset
df = pd.read_csv("../input/Iris.csv")
df.head()
df.describe()
df.info()
df.isna().any()
#Removing the column "Id" as its not necessary for analysis or modeling
df.drop('Id',axis=1, inplace=True)
df.head()
df.plot.bar()
df.plot.hist()
#Encode catogrical target variable 'Series' into numerical value so that it can be 
#plugged into Scikit learn ML libraries

mapper = {'Iris-setosa' : 0,
          'Iris-virginica' : 1,
          'Iris-versicolor' :2
          }

df['Species'] = df['Species'].map(mapper)
df.head()
df.tail()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Prepare data for training
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Data Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape)
print(y_train.shape)
#Importing ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

#Training
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)

#Prediction
y_pred = classifier.predict(X_test)
print("Accuracy on Training Set: ", accuracy_score(y_train, classifier.predict(X_train)))
print("Classification Report on Training Set: \n", classification_report(y_train, classifier.predict(X_train)))
print("Confusion Matrix on Training Set: \n", confusion_matrix(y_train, classifier.predict(X_train)))
print("Average Accuracy on Training Set: \n", np.mean(cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy')))

print("Accuracy on Test Set", accuracy_score(y_test, classifier.predict(X_test)))
print("Classification Report on Test Set: ", classification_report(y_test, classifier.predict(X_test)))
print("Confusion Matrix on Test Set: ",confusion_matrix(y_test, classifier.predict(X_test)))
print("Average Accuracy on Test Set: \n", np.mean(cross_val_score(classifier, X_test, y_test, cv=10, scoring='accuracy')))


