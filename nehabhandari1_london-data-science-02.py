import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics#Import scikit-learn metrics module for accuracy calculation

from sklearn.model_selection import train_test_split,cross_val_score

import seaborn as sns

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image 

from sklearn import tree

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

test = pd.read_csv("../input/data-science-london-scikit-learn/test.csv",header=None)

train = pd.read_csv("../input/data-science-london-scikit-learn/train.csv",header=None)

trainLabels = pd.read_csv("../input/data-science-london-scikit-learn/trainLabels.csv",header=None)
## finding size of all data

print("Train Shape:",train.shape)

print("Test Shape:",test.shape)

print("Labels Shape:",trainLabels.shape)
##getting first 5 rows of train

train.head()
trainLabels.columns = ['Target']

pd.crosstab(index=trainLabels['Target'].astype('category'),  # Make a crosstab

                              columns="count")   
train.iloc[:,0:10].describe()
Full_Data = pd.concat([train,trainLabels],axis=1)

Full_Data
Mean_Sum = Full_Data.groupby('Target').agg('mean')

Mean_Sum["Type"] = "Mean"



Sum_Sum = Full_Data.groupby('Target').agg('sum')

Sum_Sum["Type"] = "Sum"



Summ_By_Target = pd.concat([Mean_Sum,Sum_Sum])

Summ_By_Target
Full_Data[Full_Data['Target'] == 0].describe()
Full_Data[Full_Data['Target'] == 1].describe()
sns.lmplot(x="12", y="28", data=Full_Data.rename(columns=lambda x: str(x)), col='Target')

plt.show()
sns.lmplot(x="12", y="22", data=Full_Data.rename(columns=lambda x: str(x)))

plt.show()
##trying to combine predictor(x) and traget(y), as both are store in differnt varible and combing both will give entire 

##training data which will also include target variable.



X,y = train,np.ravel(trainLabels)



##spliting training data into train set and test set. train set has 70% of data while test set has 30% of data. 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Create Decision Tree classifer object



clf = DecisionTreeClassifier(max_depth=8)



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?

print("Test Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf = clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Import svm model

from sklearn import svm



#Create a svm Classifier

clf = svm.SVC(kernel='linear') # Linear Kernel



#Train the model using the training sets

clf.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
neighbor = np.arange(1,26)

k_fold = 10

train_acc = []

valid_acc = []

best_k = 0

trainLabels = np.ravel(trainLabels)

for i, k in enumerate(neighbor):

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train, y_train)

    train_acc.append(knn.score(X_train, y_train))

    valid_acc.append(np.mean(cross_val_score(knn, train, trainLabels, cv=k_fold)))

best_k = np.argmax(valid_acc)

print("Best k: ", best_k)
final_model = KNeighborsClassifier(n_neighbors = 2)

final_model.fit(train, trainLabels)

y_pred_knn=final_model.predict(X_test)

print("Training final: ", final_model.score(train, trainLabels))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_knn))
pred_test = final_model.predict(test)

pred_test[:5]

pred_test.shape
submission = pd.DataFrame(pred_test)

submission.columns = ['Solution']

submission['Id'] = np.arange(1,submission.shape[0]+1)

submission = submission[['Id', 'Solution']]

submission.head()
filename = 'London_Example.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)