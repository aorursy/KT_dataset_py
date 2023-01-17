import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import warnings

warnings.filterwarnings('ignore')

try:

    t_file = pd.read_csv('../input/mushrooms.csv', encoding='ISO-8859-1')

    print('File load: Success')

except:

    print('File load: Failed')
#importing necessary modules

import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold, cross_val_score

from sklearn.cross_validation import train_test_split
mushroom = t_file.copy()

mushroom.head()
# checking how many labels are available for the 'class'

mushroom['class'].unique()
#Now using labelencoder, convert the categorical values to numeric

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in mushroom.columns:

    mushroom[col] = le.fit_transform(mushroom[col])

mushroom.head()
X = mushroom.iloc[:,1:23]

y = mushroom.iloc[:,0]
# Prepare default models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('QDA', QuadraticDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('DT', DecisionTreeClassifier()))

models.append(('RF', RandomForestClassifier(n_estimators=1000)))

models.append(('SVM', SVC()))

models.append(('NB', GaussianNB()))
# evaluating each model

results = []

name = []

for names, model in models:

    kfold = KFold(n_splits=10, random_state=7)

    mod_results = cross_val_score(model, X,y, cv=kfold, scoring='accuracy')

    results.append(mod_results)

    name.append(names)

    print("%s : %f" %(names, mod_results.mean()))

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Model Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(name)

plt.show()
result = []

accuracy = []

rmse = []

names = []

for name, model in models:

    xtrain,xtest,ytrain,ytest = train_test_split(X,y, test_size=0.3, random_state=3)

    model.fit(xtrain,ytrain)

    mod_pred = model.predict(xtest)

    accu = accuracy_score(ytest,mod_pred)

    error = np.sqrt(mean_squared_error(ytest,mod_pred))

    accuracy.append(accu)

    rmse.append(error)

    names.append(name)

    a = pd.DataFrame(accuracy)

    b = pd.DataFrame(rmse)

    print("%s : %f (%f)" %(name, accu,error))

#plot - accuracy

plt.figure(figsize=[7,4])

a.plot(kind='bar', alpha=0.7, color='g', rot=0)

plt.xticks([0,1,2,3,4,5,6,7], names)

plt.ylim(0.5,1)

plt.xlabel('Models')

plt.ylabel("Accuracy Score")

plt.legend("")

plt.show()



#plot - RMSE

plt.figure(figsize=[7,4])

b.plot(kind='bar', alpha=0.7, color='g', rot=0)

plt.xticks([0,1,2,3,4,5,6,7], names)

plt.ylim(0,0.7)

plt.xlabel('Models')

plt.ylabel("RMSE")

plt.legend("")

plt.show()