import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd
df_users = pd.read_csv("/kaggle/input/dataset/users.csv")

df_posts = pd.read_csv("/kaggle/input/dataset/posts.csv")

df_views = pd.read_csv("/kaggle/input/dataset/views.csv")
print(df_users.head())
print(df_posts.head())
print(df_views.head())
print(df_users.columns)
print(df_posts.columns)
print(df_views.columns)
print(df_users.shape)

print(df_posts.shape)

print(df_views.shape)
print(df_users.info())
df_users.rename(columns={"_id": "user_id"},inplace=True)
print(df_views.info())
df_views_new = pd.merge(df_views , df_users)
df_views_new.columns
df_views_new.head()
df_posts.rename(columns={"_id": "post_id"},inplace=True)
print(df_posts.columns)
main_df = pd.merge(df_views_new , df_posts)
main_df.head()
main_df.info()
main_df.drop('name',axis = 1,inplace =True)
main_df.columns
main_df.count()
main_df.isna()
main_df.isna().sum()
main_df.dropna(subset=['category'],how = 'any',inplace=True)
main_df.isna().sum()
main_df.count()
from sklearn.preprocessing import LabelEncoder

main_df = main_df.apply(LabelEncoder().fit_transform)
import numpy as np

import matplotlib.pyplot as plt

X = main_df.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values

y = main_df.iloc[:, [7]].values
print(main_df.head())
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
import numpy as np

y_train = np.ravel(y_train)
y_train.shape
#KNN

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

classifier.fit(X_train,y_train)

#predictiing the test result



y_pred = classifier.predict(X_test)



#confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('confusion matrix:\n',cm)







from sklearn import metrics

print('\naccuracy:',metrics.accuracy_score(y_test,y_pred))



print('\ny_pred:',y_pred)



#print('y_test:',y_test.tolist())

y = np.hstack(y_test)

print('\ny_test:',y)



from sklearn.metrics import classification_report

print ('\nclassification_report:\n',classification_report(y_pred, y_test))
#logistic regression

from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression(random_state=0)

classifier.fit(X_train, y_train)



# predictiing the test result



y_pred = classifier.predict(X_test)



# making the confusion matrix



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)

print('confusion matrix:\n',cm)







from sklearn import metrics

print('\naccuracy:',metrics.accuracy_score(y_test,y_pred))



print('\ny_pred:',y_pred)



#print('y_test:',y_test.tolist())

y = np.hstack(y_test)

print('\ny_test:',y)



from sklearn.metrics import classification_report

print ('\nclassification_report:\n',classification_report(y_pred, y_test))
#support vector machine

# Fitting SVM to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('confusion matrix:\n',cm)



from sklearn import metrics

print('\naccuracy:',metrics.accuracy_score(y_test,y_pred))



print('\ny_pred:',y_pred)



#print('y_test:',y_test.tolist())

y = np.hstack(y_test)

print('\ny_test:',y)



from sklearn.metrics import classification_report

print ('\nclassification_report:\n',classification_report(y_pred, y_test))
#decision tree

# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('confusion matrix:\n',cm)







from sklearn import metrics

print('\naccuracy:',metrics.accuracy_score(y_test,y_pred))



print('\ny_pred:',y_pred)



#print('y_test:',y_test.tolist())

y = np.hstack(y_test)

print('\ny_test:',y)



from sklearn.metrics import classification_report

print ('\nclassification_report:\n',classification_report(y_pred, y_test))
from sklearn import tree

tree.plot_tree(classifier)
#random forest

# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('confusion matrix:\n',cm)





from sklearn import metrics

print('\naccuracy:',metrics.accuracy_score(y_test,y_pred))



print('\ny_pred:',y_pred)



#print('y_test:',y_test.tolist())

y = np.hstack(y_test)

print('\ny_test:',y)



from sklearn.metrics import classification_report

print ('\nclassification_report:\n',classification_report(y_pred, y_test))
main_df.head()
main_df.drop('user_id',axis = 1,inplace =True)

main_df.columns
main_df.head()
import numpy as np

import matplotlib.pyplot as plt

X = main_df.iloc[:, [0, 1, 2, 3, 4, 5]].values

y = main_df.iloc[:, [6]].values
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
import numpy as np

y_train = np.ravel(y_train)
y_train.shape
#KNN

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

classifier.fit(X_train,y_train)

#predictiing the test result



y_pred = classifier.predict(X_test)



#confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('confusion matrix:\n',cm)







from sklearn import metrics

print('\naccuracy:',metrics.accuracy_score(y_test,y_pred))



print('\ny_pred:',y_pred)



#print('y_test:',y_test.tolist())

y = np.hstack(y_test)

print('\ny_test:',y)



from sklearn.metrics import classification_report

print ('\nclassification_report:\n',classification_report(y_pred, y_test))
#logistic regression

from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression(random_state=0)

classifier.fit(X_train, y_train)



# predictiing the test result



y_pred = classifier.predict(X_test)



# making the confusion matrix



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)

print('confusion matrix:\n',cm)







from sklearn import metrics

print('\naccuracy:',metrics.accuracy_score(y_test,y_pred))



print('\ny_pred:',y_pred)



#print('y_test:',y_test.tolist())

y = np.hstack(y_test)

print('\ny_test:',y)



from sklearn.metrics import classification_report

print ('\nclassification_report:\n',classification_report(y_pred, y_test))
#support vector machine

# Fitting SVM to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('confusion matrix:\n',cm)





from sklearn import metrics

print('\naccuracy:',metrics.accuracy_score(y_test,y_pred))



print('\ny_pred:',y_pred)



#print('y_test:',y_test.tolist())

y = np.hstack(y_test)

print('\ny_test:',y)



from sklearn.metrics import classification_report

print ('\nclassification_report:\n',classification_report(y_pred, y_test))
#decision tree

# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('confusion matrix:\n',cm)







from sklearn import metrics

print('\naccuracy:',metrics.accuracy_score(y_test,y_pred))



print('\ny_pred:',y_pred)



#print('y_test:',y_test.tolist())

y = np.hstack(y_test)

print('\ny_test:',y)



from sklearn.metrics import classification_report

print ('\nclassification_report:\n',classification_report(y_pred, y_test))
from sklearn import tree

tree.plot_tree(classifier)
#random forest

# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('confusion matrix:\n',cm)





from sklearn import metrics

print('\naccuracy:',metrics.accuracy_score(y_test,y_pred))



print('\ny_pred:',y_pred)



#print('y_test:',y_test.tolist())

y = np.hstack(y_test)

print('\ny_test:',y)



from sklearn.metrics import classification_report

print ('\nclassification_report:\n',classification_report(y_pred, y_test))