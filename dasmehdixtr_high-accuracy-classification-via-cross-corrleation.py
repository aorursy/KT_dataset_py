import pandas as pd

import numpy as np
data = pd.read_csv('/kaggle/input/carla-driver-behaviour-dataset/full_data_carla.csv',index_col=0)

data.info()
from scipy import signal

import matplotlib.pyplot as plt

labels = data['class'].unique()

for label in labels:

    data_label = data[data['class']==label]

    acc_label_x = data_label.iloc[:,0]

    corr = signal.correlate(acc_label_x,np.ones(len(acc_label_x)),mode='same') / len(acc_label_x)

    clock= np.arange(64, len(acc_label_x), 128)

    plt.plot(clock, corr[clock],label= label)

plt.legend()

plt.title('cross-correlation-features for AccelX')

plt.show()
from scipy import signal

import matplotlib.pyplot as plt

data_featured = pd.DataFrame()

data_prossed = pd.DataFrame()

labels = data['class'].unique()



for col in np.array([0,1,2,4,5,6]):    

    for label in labels:

        data_label = data[data['class']==label]

        acc_label_x = data_label.iloc[:,col]

        corr = signal.correlate(acc_label_x,np.ones(len(acc_label_x)),mode='same') / len(acc_label_x)

        data_featured=pd.concat([data_featured,pd.DataFrame(corr)], ignore_index=True)

        #clock= np.arange(64, len(acc_label_x), 128)

        #plt.plot(clock, corr[clock],label= label)

    data_prossed = pd.concat([data_prossed,data_featured],axis=1,ignore_index=True)

    data_featured = pd.DataFrame()

#plt.legend()

#plt.title('cross-correlation-features for AccelX')

#plt.show()
data_prossed
data_prossed['class'] = data['class']

x = data_prossed.drop(["class"],axis=1)

y = data_prossed["class"].values
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

print("SVM accuracy is :",svm.score(X_test,y_test))

print('accuracy of bayes in test data is :', nb.score(X_test,y_test))

print('acc_of_sgd is: ', sgd.score(X_test,y_test))

print('acc_knn: ',knn.score(X_test,y_test))