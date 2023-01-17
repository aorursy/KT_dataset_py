# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, warnings, gc

warnings.filterwarnings("ignore")



# Sklearn Classifier Algorithm

from sklearn.tree import ExtraTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import NearestCentroid, RadiusNeighborsClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import RidgeClassifier



# Sklearn (other)

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score



from tabulate import tabulate

url = '../input/all-datasets-for-practicing-ml/Class/Class_Abalone.csv'

data = pd.read_csv(url, header='infer')
# Total Records

print("Total Records: ", data.shape[0])
# Check for empty/null/missing records

print("Is Dataset Empty: ", data.empty)
# Records per Classes

data.Sex.value_counts()
# Instantiating Label Encoder

encoder = LabelEncoder()



# Columns List

columns = data.columns



# Encode the column 

data['Sex']= encoder.fit_transform(data['Sex']) 

    
# Inspect

data.head()
# Feature & Target Selection

target = ['Sex']   

features = columns [1:]



X = data[features]

y = data[target]





# Dataset Split

''' Training = 90% & Validation = 10%  '''

test_size = 0.1

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=True) 





# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_val = sc.transform(X_val)

# Instantiate Extra Tree Classifier

et = ExtraTreeClassifier(random_state=1)



# Bagging Classifier

bgc = BaggingClassifier(et, random_state=1, max_features=8, verbose=0)



# Train 

bgc.fit(X_train, y_train)



# Prediction

y_pred = bgc.predict(X_val)



# Accuracy

print("Extra Tree Classifier(Tree Module) Accuracy: ", '{:.2%}'.format(accuracy_score(y_val, y_pred)))



tab_data = []

tab_data.append(['Extra Tree(Tree)', '{:.2%}'.format(accuracy_score(y_val, y_pred))])
# Instantiate Classifier

etc = ExtraTreesClassifier(n_estimators=100, max_depth= 5,

                           verbose=0, random_state=1)



# Train

etc.fit(X_train, y_train)



# Prediction

y_pred = etc.predict(X_val)



# Accuracy

print("Extra Tree Classifier(Ensemble Module) Accuracy: ", '{:.2%}'.format(accuracy_score(y_val, y_pred)))

tab_data.append(['Extra Tree(Ensemble)', '{:.2%}'.format(accuracy_score(y_val, y_pred))])
# Instantiate Classifier

mlp = MLPClassifier(random_state=1, max_iter=300,solver='sgd',

                    batch_size=200, learning_rate='adaptive', learning_rate_init=0.001,

                    shuffle=True, verbose=0)



# Train

mlp.fit(X_train, y_train)



# Prediction

y_pred = mlp.predict(X_val)



# Accuracy

print("MLP Classifier Accuracy: ", '{:.2%}'.format(accuracy_score(y_val, y_pred)))

tab_data.append(['MLP', '{:.2%}'.format(accuracy_score(y_val, y_pred))])
# Instantiate Classifier

nc = NearestCentroid()



# Train

nc.fit(X_train, y_train)



# Prediction

y_pred = nc.predict(X_val)



# Accuracy

print("Nearest Centroid Classifier Accuracy: ", '{:.2%}'.format(accuracy_score(y_val, y_pred)))

tab_data.append(['Nearest Centroid', '{:.2%}'.format(accuracy_score(y_val, y_pred))])
# Instantiate Classifier

qda = QuadraticDiscriminantAnalysis()



# Train

qda.fit(X_train, y_train)



# Prediction

y_pred = qda.predict(X_val)



# Accuracy

print("Quadratic Discriminant Analysis Classifier Accuracy: ", '{:.2%}'.format(accuracy_score(y_val, y_pred)))

tab_data.append(['Quadratic Discriminant Analysis', '{:.2%}'.format(accuracy_score(y_val, y_pred))])
# Instantiate Classifier

rnc = RadiusNeighborsClassifier(radius=2.0, )



# Train

rnc.fit(X_train, y_train)



# Prediction

y_pred = rnc.predict(X_val)



# Accuracy

print("Radius Neighbours Classifier Accuracy: ", '{:.2%}'.format(accuracy_score(y_val, y_pred)))

tab_data.append(['Radius Neighbours', '{:.2%}'.format(accuracy_score(y_val, y_pred))])
# Instantiate Classifier

rc = RidgeClassifier(class_weight='balanced', random_state=1)



# Train

rc.fit(X_train, y_train)



# Prediction

y_pred = rc.predict(X_val)



# Accuracy

print("Ridge Classifier Accuracy: ", '{:.2%}'.format(accuracy_score(y_val, y_pred)))

tab_data.append(['Ridge Classifier', '{:.2%}'.format(accuracy_score(y_val, y_pred))])
print(tabulate(tab_data, headers=['Classifiers','Accuracy'], tablefmt='pretty'))