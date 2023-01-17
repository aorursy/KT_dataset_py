#Models using only four features of Titanic data

#Excluding the most important feature of Gender



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



datatrain = pd.read_csv('../input/train.csv')

datatest = pd.read_csv('../input/test.csv')



#Converting NaN to 0



datatrain = datatrain.fillna(0)

datatest = datatest.fillna(0)



X = np.array(datatrain[['Pclass', 'Age', 'SibSp', 'Sex', 'Fare']])

Xt = np.array(datatest[['Pclass', 'Age', 'SibSp', 'Sex', 'Fare']])

Y = np.array(datatrain['Survived'])



for dataset in X:

    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)



for dataset in Xt:

    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    

    

#Using Gradient Boosting Classifier



from sklearn.ensemble import GradientBoostingClassifier



classifier1 = GradientBoostingClassifier()

classifier1.fit(X, Y)



Yo1 = classifier1.predict(Xt)

np.savetxt('OutputGBC.csv', Yo1, delimiter=',')



#Using SVM



from sklearn.svm import SVC

classifier2 = SVC()

classifier2.fit(X, Y)

Yo2 = classifier2.predict(Xt)

np.savetxt('OutputSVM.csv', Yo2, delimiter=',')
