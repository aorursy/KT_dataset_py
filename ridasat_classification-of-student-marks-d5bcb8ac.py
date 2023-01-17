

import numpy as np

import pandas as pd 





df = pd.read_csv('../input/xAPI-Edu-Data.csv')

# Any results you write to the current directory are saved as output.

df.head()
print(df.shape)
df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

#breakdown by class

sns.countplot(x="Topic", data=df, palette="muted");

plt.show()
df['Failed'] = np.where(df['Class']=='L',1,0)

sns.factorplot('Topic','Failed',data=df,size=9)
pd.crosstab(df['Class'],df['Topic'])
sns.countplot(x='Class',data=df,palette='PuBu')

plt.show()
df.Class.value_counts()
sns.countplot(x='ParentschoolSatisfaction',data = df, hue='Class',palette='bright')

plt.show()
sns.factorplot('Relation','Failed',data=df)
sns.factorplot("gender","Failed",data=df)
Raised_hand = sns.boxplot(x="Class", y="raisedhands", data=df)

Raised_hand = sns.swarmplot(x="Class", y="raisedhands", data=df, color=".15")

plt.show()
Facetgrid = sns.FacetGrid(df,hue='Failed',size=6)

Facetgrid.map(sns.kdeplot,'raisedhands',shade=True)

Facetgrid.set(xlim=(0,df['raisedhands'].max()))

Facetgrid.add_legend()

ax = sns.boxplot(x="Class", y="Discussion", data=df)

ax = sns.swarmplot(x="Class", y="Discussion", data=df, color=".25")

plt.show()
Facetgrid = sns.FacetGrid(df,hue='Failed',size=7)

Facetgrid.map(sns.kdeplot,'Discussion',shade=True)

Facetgrid.set(xlim=(0,df['Discussion'].max()))

plt.show()
Vis_res = sns.boxplot(x="Class", y="VisITedResources", data=df)

Vis_res = sns.swarmplot(x="Class", y="VisITedResources", data=df, color=".25")

plt.show()
Facetgrid = sns.FacetGrid(df,hue='Failed',size=7)

Facetgrid.map(sns.kdeplot,'VisITedResources',shade=True)

Facetgrid.set(xlim=(0,df['VisITedResources'].max()))

plt.show()
Anounce_bp = sns.boxplot(x="Class", y="AnnouncementsView", data=df)

Anounce_bp = sns.swarmplot(x="Class", y="AnnouncementsView", data=df, color=".25")

plt.show() 
Facetgrid = sns.FacetGrid(df,hue='Failed',size=7)

Facetgrid.map(sns.kdeplot,'AnnouncementsView',shade=True)

Facetgrid.set(xlim=(0,df['AnnouncementsView'].max()))

plt.show()
df.groupby('Topic').median()

df['AbsBoolean'] = df['StudentAbsenceDays']

df['AbsBoolean'] = np.where(df['AbsBoolean'] == 'Under-7',0,1)

df['AbsBoolean'].groupby(df['Topic']).mean()
df[9:13].describe()
df['TotalQ'] = df['Class']

df['TotalQ'].loc[df.TotalQ == 'Low-Level'] = 0.0

df['TotalQ'].loc[df.TotalQ == 'Middle-Level'] = 1.0

df['TotalQ'].loc[df.TotalQ == 'High-Level'] = 2.0



continuous_subset = df.ix[:,9:13]



X = np.array(continuous_subset).astype('float64')

y = np.array(df['TotalQ'])

X.shape
from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import StandardScaler





X_train, X_test, y_train, y_test = train_test_split(

         X, y, test_size=0.3, random_state=0)





sc = StandardScaler()



sc.fit(X_train)



X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
from sklearn.linear_model import Perceptron



ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.svm import SVC



svm = SVC(kernel='linear', C=2.0, random_state=0)

svm.fit(X_train_std, y_train)



y_pred = svm.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())


print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


print(classification_report(y_test, y_pred))
svm = SVC(kernel='rbf', random_state=0, gamma=2, C=1.0)

svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
sns.countplot(x='StudentAbsenceDays',data = df, hue='Class',palette='bright')

plt.show()
sns.factorplot('StudentAbsenceDays','Failed',data=df)


continuous_subset['Absences'] = df['AbsBoolean']

X = np.array(continuous_subset).astype('float64')

y = np.array(df['TotalQ'])

X_train, X_test, y_train, y_test = train_test_split(

         X, y, test_size=0.3, random_state=0)

sc = StandardScaler()



sc.fit(X_train)



X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
svm.fit(X_train_std, y_train)



y_pred = svm.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())
print(classification_report(y_test, y_pred))
df.loc[(df['raisedhands']==2) & (df['VisITedResources']==9) & (df['AnnouncementsView']==7)]
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score

#clf = MLPClassifier(solver='lbfgs',alpha=1e-5,random_state=1)
sc = StandardScaler()

sc.fit(X)
clf = MLPClassifier(solver='lbfgs',alpha=.1,random_state=1)

clf.fit(X,y)

scores=cross_val_score(clf,X,y,cv=10)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
df = df.drop(df.index[[126]])

df.shape
df.loc[(df['raisedhands']==2) & (df['VisITedResources']==9) & (df['AnnouncementsView']==7)]
from sklearn.linear_model import Perceptron



ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))