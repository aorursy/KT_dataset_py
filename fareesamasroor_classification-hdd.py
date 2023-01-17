from sklearn.utils.multiclass import unique_labels
from sklearn.externals import joblib
#DecisionTree
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree

#Load dataset
data = pd.read_csv("../input/heartdd/heart.csv")
print(data.head())
print(data.columns)
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()

#test train split
x = data.iloc[:,:-1]
y = data.iloc[:,13]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# Normalize
X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values

# importing DecisionTreeClassifiertree
clf = tree.DecisionTreeClassifier()
#Training
lf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

print('Accuracy :',accuracy_score(y_test, y_pred))
#print("DecisionTreeClassifier score  ",format(clf.score(X_train, y_train)))

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
sns.heatmap(cm,annot=True)
plt.show()
joblib.dump(clf, 'Heart_model.pkl')

clf.predict([[61,0,4,130,330,0,2,169,0,0,1,0,3]])
#DecisionTree
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

#Load dataset
data = pd.read_csv("../input/heartdd/heart.csv")
print(data.head())
print(data.columns)
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()

#test train split
x = data.iloc[:,:-1]
y = data.iloc[:,13]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# Normalize
X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values

# importing DecisionTreeClassifiertree
knn = KNeighborsClassifier(n_neighbors=3)
#Training
lf = knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)

print('Accuracy :',accuracy_score(y_test, y_pred))
#print("DecisionTreeClassifier score  ",format(clf.score(X_train, y_train)))

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
sns.heatmap(cm,annot=True)
plt.show()
#naive_bayes
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

#Load dataset
data = pd.read_csv("../input/heartdd/heart.csv")
print(data.head())
print(data.columns)
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()

#test train split
x = data.iloc[:,:-1]
y = data.iloc[:,13]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# Normalize
X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values

# importing DecisionTreeClassifiertree
gnb = GaussianNB()
#Training
lf = gnb.fit(X_train, y_train)
y_pred=gnb.predict(X_test)

print('Accuracy :',accuracy_score(y_test, y_pred))
#print("DecisionTreeClassifier score  ",format(clf.score(X_train, y_train)))

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
sns.heatmap(cm,annot=True)
plt.show()
#DecisionTree
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
#Load dataset
data = pd.read_csv("../input/heartdd/heart.csv")
print(data.head())
print(data.columns)
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()

#test train split
x = data.iloc[:,:-1]
y = data.iloc[:,13]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# Normalize
X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values

# importing DecisionTreeClassifiertree
lg = LogisticRegression(random_state=0)
#Trainingk
lf = lg.fit(X_train, y_train)
y_pred=lg.predict(X_test)

print('Accuracy :',accuracy_score(y_test, y_pred))
#print("DecisionTreeClassifier score  ",format(clf.score(X_train, y_train)))

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
sns.heatmap(cm,annot=True)
plt.show()
#DecisionTree
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm

#Load dataset
data = pd.read_csv("../input/heartdd/heart.csv")
print(data.head())
print(data.columns)
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()

#test train split
x = data.iloc[:,:-1]
y = data.iloc[:,13]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# Normalize
X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values

# importing DecisionTreeClassifiertree
svm = svm.SVC()
#Training
svm= svm.fit(X_train, y_train)
y_pred=svm.predict(X_test)

print('Accuracy :',accuracy_score(y_test, y_pred))
#print("DecisionTreeClassifier score  ",format(clf.score(X_train, y_train)))

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
sns.heatmap(cm,annot=True)
plt.show()
