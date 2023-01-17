import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
%matplotlib inline
wine=pd.read_csv("../input/winequality-red.csv")
wine.head()
wine.describe()
wine.info()
column_names=wine.columns.values

number_of_column=len(column_names)

rows=4

cols=3

fig, axarr=plt.subplots(rows,cols, figsize=(22,16))

counter=0

for i in range(rows):

    for j in range(cols):

        sns.barplot(x='quality', y=column_names[counter],data=wine, ax=axarr[i][j])

        counter+=1

        if counter==(number_of_column-1,):

            break
correlation=wine.corr()
plt.figure(figsize=(24,18))

heatmamp=sns.heatmap(correlation,annot=True,linewidths=0,vmin=-1,cmap='RdBu_r')
wine['quality'] = pd.cut(wine['quality'],bins=[0,4,6,10],labels = ['bad','netral','good'],include_lowest = False)
wine.head()
wine['quality'].value_counts()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

wine['quality']=labelencoder.fit_transform(wine['quality'])
wine['quality'].value_counts()
from sklearn.model_selection import train_test_split , cross_val_score
X = wine.drop('quality',axis=1)

y = wine['quality']
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state=5)
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()

X_train= standardscaler.fit_transform(X_train)

X_test = standardscaler.fit_transform(X_test)
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB
scoring = 'accuracy'

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))
from sklearn import model_selection

results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=42)

    cv_results = model_selection.cross_val_score(model, X_train,y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
fig = plt.figure()

plt.title('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)
from sklearn.metrics import accuracy_score

knn= KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

predicts =knn.predict(X_test)

print(accuracy_score(y_test, predicts))

print(confusion_matrix(y_test, predicts))

print(classification_report(y_test, predicts))
svc= SVC(kernel = 'rbf',random_state = 42,gamma= 0.9,C=1.2)

svc.fit(X_train,y_train)

predicts =svc.predict(X_test)

print(accuracy_score(y_test, predicts))

print(confusion_matrix(y_test, predicts))

print(classification_report(y_test, predicts))
forest = RandomForestClassifier(n_estimators=400,random_state = 42)

forest.fit(X_train,y_train)

predicts = forest.predict(X_test)

print(accuracy_score(y_test, predicts))

print(confusion_matrix(y_test, predicts))

print(classification_report(y_test, predicts))
tree = DecisionTreeClassifier()

tree.fit(X_train,y_train)

predicts = tree.predict(X_test)

print(accuracy_score(y_test, predicts))

print(confusion_matrix(y_test, predicts))

print(classification_report(y_test, predicts))
lr = LogisticRegression()

lr.fit(X_train,y_train)

predicts = lr.predict(X_test)

print(accuracy_score(y_test, predicts))

print(confusion_matrix(y_test, predicts))

print(classification_report(y_test, predicts))
lda =LinearDiscriminantAnalysis()

lda.fit(X_train,y_train)

predicts = lda.predict(X_test)

print(accuracy_score(y_test, predicts))

print(confusion_matrix(y_test, predicts))

print(classification_report(y_test, predicts))