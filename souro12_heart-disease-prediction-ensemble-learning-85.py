import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv")
df.head()
df.shape
df.isna().sum()
sns.countplot(df['age'])
sns.countplot(df['target'], label='count')
df.describe()
df.info()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
normalizedData = scaler.fit_transform(df)
X = normalizedData[:,0:13]
Y = normalizedData[:,13]
print(X)
kfold = model_selection.KFold(n_splits=10, random_state=7)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
from sklearn.ensemble import AdaBoostClassifier
seed = 10
num_trees = 60
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state = 21)
def models(x_train, y_train):
    
    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(x_train, y_train)
    
    #Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state = 0)
    tree.fit(x_train, y_train)
    
    #Support Vector Machines
    from sklearn.svm import SVC
    svm = SVC(kernel='linear')
    svm.fit(x_train,y_train)
    
    #Naive Bayes
    from sklearn.naive_bayes import GaussianNB 
    gnb = GaussianNB() 
    gnb.fit(x_train, y_train) 
    
    #Bagging Classifier
    from sklearn.neighbors import KNeighborsClassifier  
    knn = KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree', metric='minkowski', p=5)  
    knn.fit(x_train, y_train)  
    from sklearn.ensemble import BaggingClassifier
    bag = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
    bag.fit(x_train, y_train)
    
    print("Logistic Regression Training Accuracy:", log.score(x_train, y_train))
    print("Decision Tree Classifier Training Accuracy:", tree.score(x_train, y_train))
    print("SVM Training Accuracy:", svm.score(x_train, y_train))
    print("Naive Bayes Training Accuracy:", gnb.score(x_train, y_train))
    print("Bagging Classifier Training Accuracy:", knn.score(x_train, y_train))
    
    return log, tree, svm, gnb, bag
model = models(x_train, y_train)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print("Logistic Regression: ")
print()
print(classification_report(y_test, model[0].predict(x_test)))
print(accuracy_score(y_test, model[0].predict(x_test)))
print("Decision Tree Classifier: ")
print()
print(classification_report(y_test, model[1].predict(x_test)))
print(accuracy_score(y_test, model[1].predict(x_test)))
print("Support Vector Machines Classifier: ")
print()
print(classification_report(y_test, model[2].predict(x_test)))
print(accuracy_score(y_test, model[2].predict(x_test)))
print("Naive Bayes Classifier: ")
print()
print(classification_report(y_test, model[3].predict(x_test)))
print(accuracy_score(y_test, model[3].predict(x_test)))
print("Bagging Classifier: ")
print()
print(classification_report(y_test, model[4].predict(x_test)))
print(accuracy_score(y_test, model[4].predict(x_test)))
print("Test Accuracy: ")
print("Logistic Regression:", accuracy_score(y_test, model[0].predict(x_test))*100)
print("Decision Tree Classifier:", accuracy_score(y_test, model[1].predict(x_test))*100)
print("Support Vector Machines (SVM):", accuracy_score(y_test, model[2].predict(x_test))*100)
print("Naive Bayes:", accuracy_score(y_test, model[3].predict(x_test))*100)
print("Bagging Classifier: ", accuracy_score(y_test, model[4].predict(x_test))*100)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('decision', model2))
model3 = SVC()
estimators.append(('svm', model3))
model4 = GaussianNB()
estimators.append(('naive', model4))
model5 = BaggingClassifier()
estimators.append(('bag', model5))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, x_test, y_test)
print(results.mean()*100)
logistic = accuracy_score(y_test, model[0].predict(x_test))*100
decision = accuracy_score(y_test, model[1].predict(x_test))*100
svm = accuracy_score(y_test, model[2].predict(x_test))*100
naive = accuracy_score(y_test, model[3].predict(x_test))*100
bag = accuracy_score(y_test, model[4].predict(x_test))*100
voting = results.mean()*100
li = [logistic, decision, svm, naive, bag, voting]

plt.bar(['Logistic', 'Decision', 'SVM', 'Naive', 'Bag', 'Voting'], li)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Final Accuracy of the individual models')
plt.legend()
plt.show()
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix

model = Sequential()

model.add(Dense(8))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])

print('<< Compiling Model >>')

history_1 = model.fit(x_train,y_train ,batch_size = 32 ,epochs = 300)
y_pred_1 = model.predict(x_test)
y_pred_1 = (y_pred_1 > 0.5)
plt.plot(history_1.history['accuracy'], color='red')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy'], loc='upper left')
plt.show()
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_1)
print(cm)

print(accuracy_score(y_test, y_pred_1))
ann = 100*accuracy_score(y_test,y_pred_1)
print('percentage Accuracy : ',ann)