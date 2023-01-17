#importing libraries

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
#import test and train set

train = pd.read_csv("../input/human-activity-data/train.csv")

test = pd.read_csv("../input/human-activity-data/test.csv")
train.head()
train.Activity.value_counts()
train.shape,test.shape
test.Activity.value_counts()
train.columns
train.describe()
#shuffling data

from sklearn.utils import shuffle

train= shuffle(train)

test = shuffle(test)
#separating features and labels

trainData= train.drop('Activity',axis=1).values

trainLabel= train.Activity.values



testData= test.drop('Activity', axis=1).values

testLabel= test.Activity.values
#encoding labels

from sklearn import preprocessing



encoder= preprocessing.LabelEncoder()



#encoding test labels

encoder.fit(testLabel)

testLabelE = encoder.transform(testLabel)



#encoding train labels

encoder.fit(trainLabel)

trainLabelE = encoder.transform(trainLabel)
#classification models:

#NN using MLP

#Logistic regression

#Random forest classifier

#KNN

#Decision Tree

#Grid search cv
# Applying supervised neural network using multi-layer-preceptron

import sklearn.neural_network as nn

mlpSGD  =  nn.MLPClassifier(hidden_layer_sizes=(90,)  \

                        , max_iter=1000 , alpha=1e-4  \

                        , solver='sgd' , verbose=10   \

                        , tol=1e-19 , random_state=1  \

                        , learning_rate_init=.001)
mlpADAM  =  nn.MLPClassifier(hidden_layer_sizes=(90,)  \

                        , max_iter=1000 , alpha=1e-4  \

                        , solver='adam' , verbose=10   \

                        , tol=1e-19 , random_state=1  \

                        , learning_rate_init=.001)
mlpLBFGS  =  nn.MLPClassifier(hidden_layer_sizes=(90,)  \

                        , max_iter=1000 , alpha=1e-4  \

                        , solver='lbfgs' , verbose=10   \

                        , tol=1e-19 , random_state=1  \

                        , learning_rate_init=.001)
nnModelSGD= mlpSGD.fit(trainData, trainLabelE)
nnModelSGD
nnModelADAM = mlpADAM.fit(trainData, trainLabelE)
#Logistic Regression

train_df= pd.read_csv('../input/human-activity-data/train.csv')

test_df= pd.read_csv('../input/human-activity-data/test.csv')
def get_all_data():

    train_values = train_df.values

    test_values = test_df.values

    np.random.shuffle(train_values)

    np.random.shuffle(test_values)

    X_train = train_values[:, :-1]

    X_test = test_values[:, :-1]

    y_train = train_values[:, -1]

    y_test = test_values[:, -1]

    return X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = get_all_data()
model = LogisticRegression()

model
model.fit(X_train, y_train)
model.score(X_test, y_test)
# logistic regression : 88.48%
#PCA

from sklearn.decomposition import PCA



X_train, X_test, y_train, y_test = get_all_data()

pca= PCA(n_components=200)

pca.fit(X_train)

X_train = pca.transform(X_train)

X_test = pca.transform(X_test)



model.fit(X_train, y_train)

model.score(X_test, y_test)

# Feature scaling between -1 to 1

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train, X_test, y_train, y_test = get_all_data()

scaler.fit(X_train)

X_train= scaler.transform(X_train)

X_test = scaler.transform(X_test)



model.fit(X_train, y_train)

model.score(X_test, y_test)

#Random forest classifier

from sklearn.ensemble import RandomForestClassifier



X_train, X_test, y_train, y_test = get_all_data()

scaler.fit(X_train)



X_train= scaler.transform(X_train)

X_test = scaler.transform(X_test)



model = RandomForestClassifier(n_estimators=500)



model.fit(X_train, y_train)

model.score(X_test, y_test)

#Random forest : 90.39
# KNN

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



#separating features and labels

trainData= train.drop('Activity',axis=1).values

trainLabel= train.Activity.values



testData= test.drop('Activity', axis=1).values

testLabel= test.Activity.values



#encoding labels

from sklearn import preprocessing



encoder= preprocessing.LabelEncoder()



#encoding test labels

encoder.fit(testLabel)

testLabelE = encoder.transform(testLabel)



#encoding train labels

encoder.fit(trainLabel)

trainLabelE = encoder.transform(trainLabel)
clf = KNeighborsClassifier(n_neighbors=24)



knnmodel= clf.fit(trainData, trainLabelE)

pred= clf.predict(testData)



acc= accuracy_score(testLabelE, pred)

print("KNN accuracy : %.5f" % (acc))
#KNN accuracy : 0.76476
#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



deTreeClf= DecisionTreeClassifier()

tree = deTreeClf.fit(trainData, trainLabelE)

testpred = tree.predict(testData)



acc1= accuracy_score(testLabelE,testpred)

print('Accuracy : %f'% acc1)
# Decision Tree Accuracy : 0.757758
# Grid Search CV

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC



parameters = {

    'kernel': ['linear', 'rbf', 'poly','sigmoid'],

    'C': [100, 50, 20, 1, 0.1]

}



selector = GridSearchCV(SVC(), parameters, scoring='accuracy') # we only care about accuracy here

selector.fit(trainData, trainLabel)



print('Best parameter set found:')

print(selector.best_params_)

print('Detailed grid scores:')

means = selector.cv_results_['mean_test_score']

stds = selector.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, selector.cv_results_['params']):

    print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))

    print()
clf=SVC(kernel='linear', C=100).fit(trainData, trainLabel)

y_pred= clf.predict(testData)

print('Accuracy score:',accuracy_score(testLabel,y_pred))

#Grid search : Accuracy score: 87.38
#Classification models

# logistic regression : 88.48%

#Random forest : 90.39

#KNN accuracy : 0.76476

# Decision Tree Accuracy : 0.757758

#Grid search : Accuracy score: 87.38