import pandas as pd

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
data = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv") 
data.head()
print(data.columns)
#visualize and explore the data

print(data.loc[20])



# Print the shape of the dataset

print(data.shape)
data['target_class'].value_counts().plot(kind='bar')
classes = data['target_class']

print(classes.value_counts())
df = pd.DataFrame(data)
df.notnull()
data.describe()
#Plotting the data

data.hist(figsize=(15,15))

plt.show()
import seaborn as sns
# Correlation matrix

corrmat = data.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corrmat,cmap='viridis',annot=True,linewidths=0.5,)
sns.pairplot(data, hue="target_class",palette="husl",diag_kind = "kde",kind = "scatter")

plt.suptitle("PairPlot of Data Without Std. Dev. Fields",fontsize=18)

plt.show() 

# Get all the columns from the dataFrame

columns = data.columns.tolist()



# Filter the columns to remove data we do not want

columns = [c for c in columns if c not in ["target_class"]]



# Store the variable we'll be predicting on

target = "target_class"



X = data[columns]

y = data[target]



# Print shapes

print(X.shape)

print(y.shape)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))



features_scaled = scaler.fit_transform(X)
print(X.loc[10])

print(y.loc[10])
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(features_scaled,y, test_size = 0.2)

X_test.shape
from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import cross_validate

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
seed= 1

scoring = 'accuracy'





models = []

models.append(('KNN', KNeighborsClassifier(n_neighbors = 3)))

models.append(('SVM', SVC(gamma = 'auto', kernel= 'rbf')))

models.append(('CART', DecisionTreeClassifier()))

models.append(('LR', LogisticRegression()))

models.append(('RFC', RandomForestClassifier(max_depth=5, n_estimators = 40)))



# evaluate each model in turn

results = []

names = []



for name, model in models:

    kfold = model_selection.KFold(n_splits=15, random_state = seed)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    #print(cv_results)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())

    print(msg)
from sklearn.ensemble import VotingClassifier



ensemble = VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1)



ensemble.fit(X_train, y_train)



predictions = ensemble.score(X_test, y_test)*100





print("The Voting Classifier Accuracy is: ", predictions)
# Compare Algorithms

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import CVScores



_, ax = plt.subplots()



cv = StratifiedKFold(10)



oz = CVScores(KNeighborsClassifier(n_neighbors = 3), ax=ax, cv=cv, scoring= 'accuracy')

oz.fit(X,y)

oz.poof()
for name, model in models:

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(name)

    print(accuracy_score(y_test, predictions)*100)

    print(classification_report(y_test, predictions))
from sklearn.metrics import cohen_kappa_score

cohen_score = cohen_kappa_score(y_test, predictions)

print("Kappa Score: ", cohen_score)



from sklearn.metrics import matthews_corrcoef



MCC = matthews_corrcoef(y_test, predictions)



print("MCC Score: ", MCC)
from sklearn import metrics

from sklearn.metrics import  confusion_matrix

predict = model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, predict)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
import numpy as np

score=[]

for i in range(1,15):

    clf = KNeighborsClassifier(n_neighbors = i)

    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    score.append(accuracy)

print("The best accuracy: ", max(score)*100)

print("The mean accuracy: ", np.mean(score)*100)



    

    