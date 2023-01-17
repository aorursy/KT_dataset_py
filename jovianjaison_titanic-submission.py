# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn import preprocessing

import numpy as np

from sklearn.impute import SimpleImputer

train_data = pd.read_csv('../input/titanic/train.csv')



train_data = train_data.drop(columns=['PassengerId','Name','Ticket','Cabin'], axis=0)



imp = SimpleImputer(missing_values=np.nan,strategy="mean")

temp_df = np.array(train_data[['Pclass','Age','SibSp','Parch','Fare']])

train_data[['Pclass','Age','SibSp','Parch','Fare']] = pd.DataFrame(imp.fit_transform(temp_df))





imp = SimpleImputer(missing_values=np.nan,strategy="most_frequent")

temp_df = np.array(train_data[['Sex','Embarked']])

train_data[['Sex','Embarked']] = pd.DataFrame(imp.fit_transform(temp_df))



train_data = train_data.set_axis(['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'], axis=1, inplace=False)



train_data_Y = train_data['Survived']

train_data = train_data.drop(columns=['Survived'],axis=0)



input_labels = ['male','female']

encoder = preprocessing.LabelEncoder()

encoder.fit(input_labels)



train_data['Sex'] = encoder.transform(train_data['Sex'])



input_labels1 = ['S','C','Q']

encoder1 = preprocessing.LabelEncoder()

encoder1.fit(input_labels1)



train_data['Embarked'] = encoder1.transform(train_data['Embarked'])



print(train_data)



train_data = preprocessing.normalize(train_data,norm='l1')

print(train_data)



train_data = pd.DataFrame(imp.fit_transform(train_data))

train_data = train_data.set_axis(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'], axis=1, inplace=False)

train_data.insert(7,"Survived",train_data_Y,True)

print(train_data)

train_data = np.array(train_data)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection as ms

from sklearn import metrics



X,y = train_data[:,:-1],train_data[:,-1]

print(X.shape)

X = X.astype('float64') 

y = y.astype('int') 



X_train,X_test,y_train,y_test = ms.train_test_split(X,y,test_size=0.2,random_state=4)



'''k_range = range(1,26)

scores={}

scores_list=[]

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    y_pred = knn.predict(X_test)

    scores[k] = metrics.accuracy_score(y_test,y_pred)

    scores_list.append(scores[k])



import matplotlib.pyplot as plt



plt.plot(k_range,scores_list)

plt.xlabel('Value of K for knn')

plt.ylabel('Testing Accuracy')'''





knn = KNeighborsClassifier(n_neighbors=21)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

accuracy = 100.0*(y_test==y_pred).sum()/X_test.shape[0]

print("Validation accuracy is:",round(accuracy,2))

import matplotlib.pyplot as plt



def visualize_classifier(classifier, X, y, title=''):

    # Define the minimum and maximum values for X and Y

    # that will be used in the mesh grid

    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0

    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0



    # Define the step size to use in plotting the mesh grid 

    mesh_step_size = 0.01



    # Define the mesh grid of X and Y values

    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))



    

    # Run the classifier on the mesh grid

    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])



    # Reshape the output array

    output = output.reshape(x_vals.shape)



    # Create a plot

    plt.figure()



    # Specify the title

    plt.title(title)



    # Choose a color scheme for the plot 

    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)



    # Overlay the training points on the plot 

    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)



    # Specify the boundaries of the plot

    plt.xlim(x_vals.min(), x_vals.max())

    plt.ylim(y_vals.min(), y_vals.max())



    # Specify the ticks on the X and Y axes

    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))

    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))



    plt.show()



visualize_classifier(classifier, X, y)
test_data = pd.read_csv('../input/titanic/test.csv')



pid = test_data['PassengerId']

test_data = test_data.drop(columns=['PassengerId','Name','Ticket','Cabin'], axis=0)





imp = SimpleImputer(missing_values=np.nan,strategy="mean")

temp_df = np.array(test_data[['Pclass','Age','SibSp','Parch','Fare']])

test_data[['Pclass','Age','SibSp','Parch','Fare']] = pd.DataFrame(imp.fit_transform(temp_df))





imp = SimpleImputer(missing_values=np.nan,strategy="most_frequent")

temp_df = np.array(test_data[['Sex','Embarked']])

test_data[['Sex','Embarked']] = pd.DataFrame(imp.fit_transform(temp_df))



test_data = test_data.set_axis(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'], axis=1, inplace=False)







test_data['Sex'] = encoder.transform(test_data['Sex'])



test_data['Embarked'] = encoder1.transform(test_data['Embarked'])



print(test_data)



test_data = preprocessing.normalize(test_data,norm='l1')

print(test_data)
X = test_data[:,:]

print(X.shape)



X = X.astype('float64') 



y_pred = knn.predict(X)

print(y_pred)



y_pred = pd.DataFrame(y_pred)

y_pred = y_pred.set_axis(['Survived'],axis=1,inplace=False)

y_pred.insert(0,'PassengerId',pid,True)



print(y_pred)

y_pred.to_csv('gender_submission.csv',index=False)
