import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
!pip install --user imblearn
df = pd.read_csv('../input/kyphosis.csv')
df.head()
sns.countplot('Kyphosis', data=df)
perc_present = np.around(len(df[df['Kyphosis'] == 'present'])/len(df) * 100, decimals=2)

perc_absent =  np.around(len(df[df['Kyphosis'] == 'absent'])/len(df) * 100, decimals=2)

print("Present class has {}% instances and absent class has {}% instances".format(perc_present, perc_absent))

print("Total dataset size:", len(df))
df = pd.get_dummies(df, drop_first=True)

df.head(5)
X = df.iloc[:,:-1]

y = df.iloc[:,-1]
from imblearn.over_sampling import RandomOverSampler

oversampler = RandomOverSampler(random_state=42)

X_res, y_res = oversampler.fit_resample(X,y)

print("Negative samples:", len(y_res[y_res==0]), "Positive samples:", len(y_res[y_res==1]), "after oversampling")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X_train_pca = pca.fit_transform(df.iloc[:,:-1].values)
df_new = df.copy()
df_new['col1'] = X_train_pca[:,0]

df_new['col2'] = X_train_pca[:,1]
df_new.head()
fig, ax = plt.subplots(figsize=(10,10))

sns.scatterplot('col1', 'col2', hue='Kyphosis_present', data=df_new, ax=ax)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=0)

classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Decision tree accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)
print("Accuracy of logistic regression:",accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
from sklearn.svm import SVC

classifier = SVC(kernel='rbf')

classifier.fit(X_train,y_train)

pred = classifier.predict(X_test)

print("SVM accuracy score:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, criterion='entropy')

classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

print("Accuracy of random forest: ",accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
params = {

    'criterion' : ['gini', 'entropy'],

    'splitter' : ['best', 'random']

}
from sklearn.model_selection import RandomizedSearchCV

classifier = DecisionTreeClassifier()

random_search = RandomizedSearchCV(classifier, params, verbose=1)

search = random_search.fit(X_res,y_res)

search.best_estimator_
from sklearn.model_selection import cross_val_score

classifier = search.best_estimator_

print("Accuracy achieved:", np.mean(cross_val_score(classifier,X_res,y_res, cv=10)))
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

print(classification_report(pred,y_test))