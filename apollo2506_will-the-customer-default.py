import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import itertools



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
data = pd.read_csv("../input/credit-risk/original.csv")

data.head()
data.info()
data.fillna(data.mean(),inplace=True)
data.drop(columns="clientid",inplace=True)

data["age"] = data["age"].astype("int")

data.head()
X = data[["income","age","loan"]]

y = data["default"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
neigh = KNeighborsClassifier(n_neighbors=4)



model = neigh.fit(X_train,y_train)



y_pred = model.predict(X_test)
print("Mean Squared Error{:.3f}".format(mean_squared_error(y_pred,y_test)))

print("Accuracy score:{:.3f}".format(accuracy_score(y_pred,y_test)*100))
results = pd.DataFrame({"Actual Values":y_test,

                        "Predicted Values":y_pred})

results.head()
results.to_csv("k-NN.csv",index=False)
cm = confusion_matrix(y_pred,y_test)



def plot_confusion_matrix(cm,classes,title='Confusion Matrix',cmap=plt.cm.Blues):

    

    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]

    plt.figure(figsize=(10,10))

    plt.imshow(cm,interpolation='nearest',cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    

    fmt = '.2f'

    thresh = cm.max()/2.

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):

        plt.text(j,i,format(cm[i,j],fmt),

                horizontalalignment="center",

                color="white" if cm[i,j] > thresh else "black")

        pass

    

    plt.ylabel('True Label')

    plt.xlabel('Predicted Label')

    pass



classes=['0','1']



plt.figure()

plot_confusion_matrix(cm,classes,title="KNN")

plt.show()
scaler = StandardScaler()



X = scaler.fit_transform(X)

X = pd.DataFrame(X)

X.columns = ["income","age","loan"]

X.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
neigh = KNeighborsClassifier(n_neighbors=3)



model = neigh.fit(X_train,y_train)



y_pred = model.predict(X_test)
print("Mean Squared Error: {:.3f}".format(mean_squared_error(y_pred,y_test)))

print("Accuracy score: {:.3f}".format(accuracy_score(y_pred,y_test)*100))
results_normalized = pd.DataFrame({"Actual Values":y_test,

                        "Predicted Values":y_pred})

results_normalized.head()
results_normalized.to_csv("k-NN_normalized.csv",index=False)
cm = confusion_matrix(y_pred,y_test)



def plot_confusion_matrix(cm,classes,title='Confusion Matrix',cmap=plt.cm.Blues):

    

    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]

    plt.figure(figsize=(10,10))

    plt.imshow(cm,interpolation='nearest',cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    

    fmt = '.2f'

    thresh = cm.max()/2.

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):

        plt.text(j,i,format(cm[i,j],fmt),

                horizontalalignment="center",

                color="white" if cm[i,j] > thresh else "black")

        pass

    

    plt.ylabel('True Label')

    plt.xlabel('Predicted Label')

    pass



classes=['0','1']



plt.figure()

plot_confusion_matrix(cm,classes,title="Normalized KNN");

plt.show();