import pandas as pd
import numpy as np
import matplotlib.pyplot
from sklearn.metrics import confusion_matrix, accuracy_score
msh_df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
msh_df.head(5)
msh_df.columns
from pandas_profiling import ProfileReport
report = ProfileReport(msh_df, title = 'Mushroom Dataset')#, explorative = True)
report.to_widgets()
msh_df = msh_df.drop(['veil-type', 'gill-attachment'], axis = 1)
msh_df.head(2)
msh_df.columns
cat_col = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
       'stalk-surface-above-ring', 'stalk-surface-below-ring',
       'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color',
       'ring-number', 'ring-type', 'spore-print-color', 'population',
       'habitat']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for x in cat_col:
    msh_df[x] = le.fit_transform(msh_df[x])
msh_df.head()
x = msh_df.iloc[:, 1:]
x.head()
y = msh_df['class']
y.head()
from sklearn.feature_selection import SelectKBest, chi2
bst_features = SelectKBest(score_func = chi2, k = 20)
fit = bst_features.fit(x, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
fscore = pd.concat([dfcolumns, dfscores], axis = 1)
fscore.columns = ['Column', 'Score']
fscore
print(fscore.nlargest(20, 'Score'))
#higher the score highly the dependent variable is dependent on that variable
x_new = msh_df.iloc[:, 1:]
x_new.head()
y_new = msh_df['class']
y_new.head()
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x_new, y_new, test_size = .2, random_state = 1)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 1)
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
accuracy_score(ytest, ypred)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
accuracy_score(ytest, ypred)
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 1)
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
accuracy_score(ytest, ypred)
classifier = SVC(kernel = 'rbf', random_state = 1)
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
accuracy_score(ytest, ypred)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
accuracy_score(ytest, ypred)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
accuracy_score(ytest, ypred)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 1)
classifier.fit(xtrain, ytrain)
ypred = classifier.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
accuracy_score(ytest, ypred)
