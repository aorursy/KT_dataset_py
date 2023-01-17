import pandas as pd 

import os

import numpy as np



path = '/kaggle/input/two-person-interaction-kinect-dataset/'



full_data = pd.DataFrame()



for subdir, dirs, files in sorted(os.walk(path)):

    for file in sorted(files):

        if file.endswith('.txt'):

            #print('subdir:{},name:{}'.format(subdir[-6:-4],file))

            data = pd.read_csv(subdir+'/'+file,header=None)

            data['classs'] = subdir[-6:-4]

            full_data = pd.concat([full_data,data],ignore_index=True)



full_data.drop(full_data.columns[[0]],axis=1,inplace=True)

full_data.head()
full_data.describe()
full_data.dtypes
full_data['classs'].astype('category')
x = full_data.drop(["classs"],axis=1)

y = full_data.classs.values

print('size of x = ',x.shape)

print('size of y = ',y.shape)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,shuffle=True)
from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier



knn_clf = KNeighborsClassifier(n_neighbors=3)

print('......')

svm_clf = SVC(random_state=1)

nb_clf = GaussianNB()

nn_clf = MLPClassifier(solver='lbfgs',max_iter=20000)

sgd_clf = SGDClassifier()

rf_clf = RandomForestClassifier(n_estimators=100,random_state=1)



voting_clf = VotingClassifier(

        estimators=[('knn',knn_clf),('svm',svm_clf),('nb',nb_clf),

                    ('NN',nn_clf),('sgd',sgd_clf),('rf',rf_clf)], voting='hard')



from sklearn.metrics import accuracy_score

accuracies = {}

for clf in (knn_clf, svm_clf, nb_clf, nn_clf, sgd_clf, rf_clf, voting_clf):

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print(clf.__class__.__name__,accuracy_score(y_test, y_pred))

    accuracies[clf.__class__.__name__] = accuracy_score(y_test,y_pred)

from sklearn.model_selection import cross_val_score



accuracy_map = cross_val_score(estimator = rf_clf, X = x_train, y =y_train, cv = 8)

print("avg acc: ",np.mean(accuracy_map))

print("acg std: ",np.std(accuracy_map))
from matplotlib import pyplot as plt



plt.figure(figsize=(14, 8))

plt.suptitle('Accuracies of classifiers')

plt.subplot(121)

plt.bar(*zip(*sorted(accuracies.items())),color = 'g')

plt.xticks(fontsize=7)

plt.subplot(122)

plt.plot(*zip(*sorted(accuracies.items())),marker='P',linestyle='--',color='r')

plt.xticks(fontsize=7)

plt.grid()

plt.show()