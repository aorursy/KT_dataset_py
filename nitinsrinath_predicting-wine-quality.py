import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
wine=pd.read_csv('wine.csv')
wine=wine[['country', 'points', 'price', 'region_1', 'variety']]
wine=wine.dropna()

conditions = [
    (wine['points'] <=86),
    (wine['points'] >86) & (wine['points'] < 91),
    (wine['points'] >=91)]
choices = ['bleh', 'decent', 'great']
wine['howgood'] = np.select(conditions, choices, default='idk')
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in ['country', 'region_1', 'variety', 'howgood']:
    print(i, end=" ")
    wine[str(i)+'enc']=label_encoder.fit_transform(wine[i])
    le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(le_name_mapping)
wine.corr()
wine[['countryenc', 'howgoodenc', 'region_1enc', 'varietyenc']].hist(figsize=(20,15))
X=wine.drop(columns=['howgood', 'country', 'region_1', 'variety', 'howgoodenc', 'points'])
Y=wine['howgood']
chi_scores=chi2(X,Y)
p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
  {'C': [1, 10, 100, 1000], 'degree': [2,3,4], 'kernel':['poly']}
 ]

SVM = SVC(max_iter=100)
clf = GridSearchCV(SVM, param_grid)
clf.fit(X_train, Y_train)
clf.best_params_
SVM = SVC(C=1, gamma=0.001, kernel='rbf', max_iter=1000)

SVM.fit(X_train, Y_train)
y_pred=SVM.predict(X_test)
accuracy_score(Y_test, y_pred)
DTC= DecisionTreeClassifier()
DTC.fit(X_train, Y_train)
y_pred=DTC.predict(X_test)
accuracy_score(Y_test, y_pred)
RFtest=pd.DataFrame()
for i in range(1, 15):
    RFC = RandomForestClassifier(n_estimators=100, max_depth=i, random_state=0)
    RFC.fit(X_train, Y_train)
    y_train_pred=RFC.predict(X_train)
    y_test_pred=RFC.predict(X_test)
    RFtest.at[i, 'TrainAcc']=accuracy_score(Y_train, y_train_pred)
    RFtest.at[i, 'TestAcc']=accuracy_score(Y_test, y_test_pred)
plt.plot(RFtest.index, RFtest.TrainAcc, label='Train Acc')
plt.plot(RFtest.index, RFtest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.grid()
RFC = RandomForestClassifier(max_depth=15, random_state=0)

RFC.fit(X_train, Y_train)
y_pred=RFC.predict(X_test)
accuracy_score(Y_test, y_pred)
cvscoresRFC=cross_val_score(RFC, X, Y, cv=5)
cvscoresDTC=cross_val_score(DTC, X, Y, cv=5)
cvscoresSVM=cross_val_score(SVM, X, Y, cv=5)

fig, ax=plt.subplots(1,3, figsize=(20,5))
ax[0].boxplot(cvscoresRFC)
ax[1].boxplot(cvscoresDTC)
ax[2].boxplot(cvscoresSVM)

ax[0].set_ylabel('Test Accuracy')
ax[1].set_ylabel('Test Accuracy')
ax[2].set_ylabel('Test Accuracy')
ax[0].set_xlabel('Random Forest')
ax[1].set_xlabel('Decision Tree')
ax[2].set_xlabel('Support Vector')
feature_selection=pd.DataFrame()
for i in X.columns:
    use_x_train=X_train[i].values[:, np.newaxis]
    use_x_test=X_test[i].values[:, np.newaxis]
    RFC.fit(use_x_train, Y_train)
    y_pred=RFC.predict(use_x_test)
    feature_selection.at[i, 'testAcc']=accuracy_score(Y_test, y_pred)
feature_selection.sort_values('testAcc', ascending=False)
feature_selection=pd.DataFrame()
use_features1=[i for i in X.columns if i!='price']
for i in use_features1:
    use_x_train=pd.DataFrame(X_train[['price', i]])
    use_x_test=pd.DataFrame(X_test[['price', i]])
    RFC.fit(use_x_train, Y_train)
    y_pred=RFC.predict(use_x_test)
    feature_selection.at[i, 'testAcc']=accuracy_score(Y_test, y_pred)
feature_selection.sort_values('testAcc', ascending=False)  
feature_selection=pd.DataFrame()
use_features1=[i for i in X.columns if i not in ['price', 'region_1enc']]
for i in use_features1:
    use_x_train=pd.DataFrame(X_train[['price', 'region_1enc', i]])
    use_x_test=pd.DataFrame(X_test[['price', 'region_1enc', i]])
    RFC.fit(use_x_train, Y_train)
    y_pred=RFC.predict(use_x_test)
    feature_selection.at[i, 'testAcc']=accuracy_score(Y_test, y_pred)
feature_selection.sort_values('testAcc', ascending=False)  
cross_val_score(RFC, X, Y, cv=10)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
RFC.fit(X_train, Y_train)
y_train_pred=RFC.predict(X_train)
y_test_pred=RFC.predict(X_test)
print(accuracy_score(Y_train, y_train_pred))
print(accuracy_score(Y_test, y_test_pred))
confusion_matrix(Y_test, y_pred, labels=['bleh', 'decent','great'])
print(classification_report(Y_test, y_pred, labels=['bleh', 'decent','great']))
sample=X.sample(frac=1).head(10)
sample['prediction']=RFC.predict(sample)
sample
sns.scatterplot(x=wine.price, y=wine.points, hue=wine.country)
pd.DataFrame(wine.groupby('variety')['points'].mean()).sort_values('points', ascending=False).head()