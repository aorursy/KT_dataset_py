from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
nRowsRead = 1000 # specify 'None' if want to read whole file
df1 = pd.read_csv('../input/Absenteeism_at_work.csv', delimiter=',', nrows = None)
df1.dataframeName = 'Absenteeism_at_work.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head()
X= df1.iloc[:,0:14]
y=df1['Absenteeism time in hours']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Unique labels in 'Abenteeism Time in Hours'\n", y.unique())
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
newy = np_utils.to_categorical(encoded_Y)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, newy, test_size=0.2)
seed = 7
# prepare models
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('K Nearest Neighbours', KNeighborsClassifier()))
models.append(('Decision Trees', DecisionTreeClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Support Vector Machine', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=X_train1.shape[1], units=6, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=10, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=len(y_train1[0]), kernel_initializer="uniform"))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train1, y_train1, batch_size = 100, epochs = 30,verbose=0)
score= classifier.evaluate(X_test1,y_test1)
y_pred=classifier.predict(X_test1)
def accuracy2(y_pred,y_test):
    l=len(y_pred)
    count=0
    for i in range(l):
        m=max(y_pred[i])
       # print(m)
        index= np.where(y_pred[i]==m)
        #print(index)
        one = np.where(y_test[i]==1)
        #print(one)
        if(one[0]==index[0]):
            count+=1
       # elif(one[0]==(index[0]-1) or one[0]==(index[0]+1)):
        #    count+=0.5
    return count/l
print("Neural Network:",accuracy2(y_pred,y_test1))
      
for name, model in models:
    model.fit(X_train,y_train)
    cv_results = model.score(X_test,y_test)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f " % (name, cv_results)
    print(msg)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test,y_test)
import graphviz 
yu=[str(i) for i in y.unique()]
dot_data = tree.export_graphviz(clf,feature_names=list(X_train),class_names=yu, filled=True, rounded=True)
graph = graphviz.Source(dot_data) 
graph
clf1 = LogisticRegression(random_state=0, solver='lbfgs',max_iter=10000000,multi_class='multinomial').fit(X_train, y_train)
y_pred=clf1.predict(X_test)
clf1.score(X_test, y_test)