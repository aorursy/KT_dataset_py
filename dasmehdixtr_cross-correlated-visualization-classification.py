import numpy as np

import glob

import pandas as pd
data = pd.read_csv('/kaggle/input/airsim-driver-behaviour-dataset/data.csv',index_col=0)

data
data.describe()
data.info()
labels = data['class'].unique()

labels
from scipy import signal

import matplotlib.pyplot as plt



for label in labels:

    data_label = data[data['class']==label]

    acc_label_x = data_label.iloc[:,1]

    plt.plot(np.arange(0,len(acc_label_x)), acc_label_x,label= label)

plt.legend()

plt.title('cross-correlation-features for AccelY')

plt.show()
from scipy import signal

import matplotlib.pyplot as plt



for label in labels:

    data_label = data[data['class']==label]

    acc_label_x = data_label.iloc[:,1]

    corr = signal.correlate(acc_label_x,np.ones(len(acc_label_x)),mode='same') / len(acc_label_x)

    clock= np.arange(64, len(acc_label_x), 128)

    plt.plot(clock, corr[clock],label= label)

plt.legend()

plt.title('cross-correlation-features for AccelY')

plt.show()

data_featured = pd.DataFrame()

data_prossed = pd.DataFrame()

labels = data['class'].unique()



for col in np.array([0,1,2,4,5,6,7,8,9]):    

    for label in labels:

        data_label = data[data['class']==label]

        acc_label_x = data_label.iloc[:,col]

        corr = signal.correlate(acc_label_x,np.ones(len(acc_label_x)),mode='same') / len(acc_label_x)

        clock= np.arange(64, len(acc_label_x), 128)

        plt.plot(clock, corr[clock],label= label)

        data_featured=pd.concat([data_featured,pd.DataFrame(corr)], ignore_index=True)

    data_prossed = pd.concat([data_prossed,data_featured],axis=1,ignore_index=True)

    data_featured = pd.DataFrame()

    plt.legend()

    plt.title('cross-correlation for feature {}'.format(col))

    plt.show()

data_prossed
data_prossed['class'] = data['class']

x = data_prossed.drop(["class"],axis=1)

y = data_prossed["class"].values

y
from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(X_train,y_train)





nb = GaussianNB()

nb.fit(X_train,y_train)

knn = KNeighborsClassifier(n_neighbors = 3) #n_neighbors = k

knn.fit(X_train,y_train)

svm = SVC(random_state = 1)

svm.fit(X_train,y_train)

print("SVM accuracy(train):",svm.score(X_train,y_train))

print('NB accuracy(train) :', nb.score(X_train,y_train))

print('SGD accuracy(train): ', sgd.score(X_train,y_train))

print('Knn accuracy(train): ',knn.score(X_train,y_train))

print("SVM accuracy(test):",svm.score(X_test,y_test))

print('NB accuracy(test) :', nb.score(X_test,y_test))

print('SGD accuracy(test): ', sgd.score(X_test,y_test))

print('Knn accuracy(test): ',knn.score(X_test,y_test))