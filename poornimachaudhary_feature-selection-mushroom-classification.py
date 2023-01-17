import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
df.head()
df.info()
df.describe()
df['class'].value_counts()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

for c in df.columns:

    df[c]=label.fit_transform(df[c])
df.head()
x = df.drop('class', axis=1)

y = df['class']
x = pd.get_dummies(x,columns=x.columns ,drop_first=True)
x.head()
corr = []

for i in range(x.shape[1]):

    c = np.corrcoef(x.iloc[:,i],y)

    corr.append(abs(c[0][1]))
corr_data = pd.DataFrame({'correlation': corr}, index=x.columns)
corr_data
plt.figure(figsize=(20,9))

sns.barplot(x=corr_data.index, y = corr_data['correlation'])

plt.xticks(rotation=90)
corr_data = corr_data.sort_values(by = 'correlation', ascending=False)
corr_imp = corr_data[corr_data['correlation'] >= 0.5]
corr_imp
corr_X = x[corr_imp.index]
corr_X
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(corr_X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(n_jobs=-1)

classifier.fit(X_train, y_train)
predictions1 = classifier.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(predictions1,y_test))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(predictions1,y_test))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
X_indices = np.arange(x.shape[-1])
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=20)

selector.fit(X_train, y_train)

scores = selector.scores_/1000



plt.figure(figsize=(50,10))

sns.barplot(data=pd.DataFrame({'Feature':x.columns, 'Scores': scores}),x='Feature',y='Scores',ci=None)

plt.xticks(rotation=90)
scores_data = pd.DataFrame({'Feature':x.columns, 'Scores': scores})
plt.figure(figsize=(12,8))

sns.distplot(scores_data['Scores'])
scores_data = scores_data.sort_values(by = 'Scores',ascending=False)
scores_x = scores_data.head(20)
scores_x = x[scores_x['Feature']]
scores_x
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(scores_x, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(n_jobs=-1)

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(predictions,y_test))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(predictions,y_test))
from sklearn.feature_selection import RFE
estimator = LogisticRegression(n_jobs=-1)
d = {}

for k in range(2, 25,2):  

    selector = RFE(estimator, n_features_to_select=k, step=2)

    selector = selector.fit(x, y)

    selector.support_

    selector.ranking_



    sel_fea  = [i for i,j in zip(x.columns,selector.ranking_) if j==1]



    x_new = x[sel_fea]



    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size=0.33, random_state=42)

    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(n_jobs=-1)

    classifier.fit(X_train, y_train)

    y_pred1 = classifier.predict(X_test)



    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_pred1,y_test)

    print("features: %s"%k, " Accuracy: %f"%acc)

    d[str(k)]=acc
selector = RFE(estimator, n_features_to_select=20, step=2)

selector = selector.fit(x, y)

selector.support_

selector.ranking_

sel_fea  = [i for i,j in zip(x.columns,selector.ranking_) if j==1]
x_new = x[sel_fea]
x_new
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(n_jobs=-1)

classifier.fit(X_train, y_train)
y_pred1 = classifier.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred1,y_test))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_pred1,y_test))
from sklearn.metrics import accuracy_score

print('correlation :')

print(accuracy_score(predictions1,y_test))

print('selectKBest :')

print(accuracy_score(predictions,y_test))

print('RFE :' )

print(accuracy_score(y_pred1,y_test))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x_new, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(n_jobs=-1)
classifier.fit(X_train, y_train)
y_pred1 = classifier.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_pred1,y_test))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_pred1,y_test))
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

classifier.fit(X_train,y_train)
y_pred2 = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,y_pred2))

print(confusion_matrix(y_test,y_pred2))
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train,y_train)
y_pred3 = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,y_pred3))

print(confusion_matrix(y_test,y_pred3))
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

classifier.fit(X_train,y_train)
y_pred4 = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,y_pred4))

print(confusion_matrix(y_test,y_pred4))
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10,criterion='entropy', random_state=0)

classifier.fit(X_train,y_train)
y_pred5 = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,y_pred5))

print(confusion_matrix(y_test,y_pred5))
df = pd.DataFrame({'y_test': y_test,'logistic_reg': y_pred1, 'KNN': y_pred2, 'Naive_Bayes': y_pred3

                  , 'Decision Tree': y_pred4, 'Random Forest': y_pred5})
df
from sklearn.metrics import accuracy_score
for i in df.columns[1:]:

    print(i+': ',accuracy_score(df['y_test'], df[i]))