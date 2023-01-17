
"""
Created on Thu May 31 23:22:11 2018

@author: AJAY
"""
# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder   
# Importing the dataset
def getset(dataset,row):
    X= dataset.iloc[:, 0:-1].values
    Y = dataset.iloc[:, -1].values
    #replacing all the empty values in dataset with NaN
    dataset.replace(' ?', np.nan,inplace=True)
    labelencoder_X = LabelEncoder()
    abs= labelencoder_X.fit_transform(X[:, 1:2])
    #reshaping the np array
    X[:, 1:2]=np.reshape(abs, (row,1))
    #if value of row is 0(because row is 0 value is given to all NaN while label encoding) replace it with median
    X[:,1:2][X[:,1:2] == 0] = np.median(X[:, 1:2])
    
    labelencoder_X2 = LabelEncoder()
    abs2= labelencoder_X2.fit_transform(X[:, 6:7])
    X[:, 6:7]=np.reshape(abs2, (row,1))
    X[:,6:7][X[:,6:7] == 0] = np.median(X[:, 6:7])

    labelencoder_X2 = LabelEncoder()
    abs2= labelencoder_X2.fit_transform(X[:, 9:10])
    X[:, 9:10]=np.reshape(abs2, (row,1))
    X[:,9:10][X[:,9:10] == 0] = np.median(X[:, 9:10])

    labelencoder_X3 = LabelEncoder()
    abs3= labelencoder_X3.fit_transform(X[:, 13:14])
    X[:, 13:14]=np.reshape(abs3, (row,1))
    X[:,13:14][X[:,13:14] == 0] = np.median(X[:, 13:14])

    #for y matrix
    labelencoder_X4 = LabelEncoder()
    Y= labelencoder_X4.fit_transform(Y)

    labelencoder_X2 = LabelEncoder()
    abs2= labelencoder_X2.fit_transform(X[:, 3:4])
    X[:, 3:4]=np.reshape(abs2, (row,1))
    X[:,3:4][X[:,3:4] == 0] = np.median(X[:, 3:4])
    
    labelencoder_X2 = LabelEncoder()
    abs2= labelencoder_X2.fit_transform(X[:, 5:6])
    X[:, 5:6]=np.reshape(abs2, (row,1))
    X[:,5:6][X[:,5:6] == 0] = np.median(X[:, 5:6])

    labelencoder_X2 = LabelEncoder()
    abs2= labelencoder_X2.fit_transform(X[:, 8:9])
    X[:, 8:9]=np.reshape(abs2, (row,1))
    X[:,8:9][X[:,8:9] == 0] = np.median(X[:, 8:9])
    X=X[:, [0,1,3,5,6,8,9,10,11,13]]
    return X,Y
dataset = pd.read_csv('../input/adult.csv',names=None)
X,Y=getset(dataset,32561)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
results = confusion_matrix(Y_test, y_pred)
print ('Confusion Matrix :')
print(results)
print ('Accuracy Score :',accuracy_score(Y_test, y_pred))
print ('Report : ')
print (classification_report(Y_test, y_pred))

l2=[]
for i in range(3,20):
    classifier = KNeighborsClassifier(n_neighbors = i, metric = 'minkowski', p = 2)
    classifier.fit(X_train,Y_train)
    
    y_pred = classifier.predict(X_test)
    
    l2.append(accuracy_score(Y_test, y_pred))

print(l2)
l3=[]
for i in range(3,20):
    l3.append(i)

plt.plot(l3,l2)

#Using Gradienboosting
from sklearn.ensemble import GradientBoostingClassifier
gbc_classifier = GradientBoostingClassifier().fit(X_train,Y_train)
pred = gbc_classifier.predict(X_test)
gbc_accuracy = gbc_classifier.score(X_test, Y_test)
print('GBC: ', gbc_accuracy * 100, '%')


Y_op = (dataset.iloc[:, -1].values)

labelencoder_yop = LabelEncoder()
Y= labelencoder_yop.fit_transform(Y_op)
pl=[]
gain=dataset['capital.gain']
for i in gain.iteritems():
    if (str(Y[(i[0]-1)])=='0') and i[1]>0:
        pl.append(0)
    elif (str(Y[(i[0]-1)])=='1') and i[1]>0:
        pl.append(1)
    elif (str(Y[(i[0]-1)])=='0') and i[1]<=0:
        pl.append(2)
    else:
        pl.append(3)     
     
        





colors = ['lightgreen', 'darkseagreen', 'pink', 'lightcoral']


pielist=[]
pielist.append(pl.count(0))
pielist.append(pl.count(1))
pielist.append(pl.count(2))
pielist.append(pl.count(3))

plt.pie(pielist, shadow=True, colors=colors, explode = None,  autopct='%1.1f%%')
label=['Greater then 50k and gain +ve','Lesser then 50k and gain +ve','Greater then 50k and gain zero','Lesser then 50k and gain zero']
plt.legend(label, title='INCOME', bbox_to_anchor=(0.75, 0.75))