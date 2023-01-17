# Immport the dependencies 

import matplotlib.pyplot as plt 
import seaborn as sb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
url='/kaggle/input/iris/Iris.csv'

iris=pd.read_csv(url)
iris.head()
iris.info()
iris.describe()
sb.pairplot(data=iris,hue='Species',palette='Dark2')
setosa=iris[iris.Species=='Iris-setosa']

sb.kdeplot(setosa['SepalLengthCm'],setosa['SepalWidthCm'],cmap="plasma", shade=True, shade_lowest=False)
from sklearn.model_selection import train_test_split

X = iris.drop('Species',axis=1)
print(X.shape)
y=iris['Species']
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier
#let's set the k(n_neighbors) value to 1 first and see the results 
knn=KNeighborsClassifier(n_neighbors=1)
# Train
knn.fit(X_train,y_train)
# Predict 
predictions= knn.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
print('\n')
print('accuracy score :',accuracy_score(y_test,predictions))
# let's parse a range of k values to see what a good value to choose 
# 
error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
# Let's plot  the results 
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
#k=3
knn=KNeighborsClassifier(n_neighbors=3)
# Train
knn.fit(X_train,y_train)
# Predict 
predictions= knn.predict(X_test)
#show the results 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
print('\n')
print('Accuracy score :',accuracy_score(y_test,predictions))
# Submit the results 
submission = pd.DataFrame({'Iris Id':X_test.Id , 'Species':predictions})
submission.to_csv('submission.csv', index=False)
print(" Submission  successfully saved!")

print(submission)
