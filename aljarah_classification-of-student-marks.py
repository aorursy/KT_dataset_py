# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



#import arff

#df = arff.load(open('../input/xAPI-Edu-Data.arff','rb'))

df = pd.read_csv('../input/xAPI-Edu-Data.csv')

# Any results you write to the current directory are saved as output.
df.head()
print(df.shape)
df.isnull().sum()
#breakdown by class

import matplotlib.pyplot as plt

sns.countplot(x="Topic", data=df, palette="muted");

plt.show()
sns.countplot(x='Class',data=df,palette='PuBu')

plt.show()
df.Class.value_counts()
sns.countplot(x='ParentschoolSatisfaction',data = df, hue='Class',palette='bright')

plt.show()
Raised_hand = sns.boxplot(x="Class", y="raisedhands", data=df)

Raised_hand = sns.swarmplot(x="Class", y="raisedhands", data=df, color=".25")

plt.show()
ax = sns.boxplot(x="Class", y="Discussion", data=df)

ax = sns.swarmplot(x="Class", y="Discussion", data=df, color=".25")

plt.show()
Vis_res = sns.boxplot(x="Class", y="VisITedResources", data=df)

Vis_res = sns.swarmplot(x="Class", y="VisITedResources", data=df, color=".25")

plt.show()
ax = sns.boxplot(x="Class", y="AnnouncementsView", data=df)

ax = sns.swarmplot(x="Class", y="AnnouncementsView", data=df, color=".25")

plt.show() 
df[9:13].describe()
df['Class'].loc[df.Class == 'L'] = 0.0

df['Class'].loc[df.Class == 'M'] = 1.0

df['Class'].loc[df.Class == 'H'] = 2.0



df['gender'].loc[df.gender == 'M'] = 0.0

df['gender'].loc[df.gender == 'F'] = 1.0



df['NationalITy'].loc[df.NationalITy == 'KW'] = 0

df['NationalITy'].loc[df.NationalITy == 'lebanon'] = 1

df['NationalITy'].loc[df.NationalITy == 'Egypt'] = 2

df['NationalITy'].loc[df.NationalITy == 'SaudiArabia'] = 3

df['NationalITy'].loc[df.NationalITy == 'USA'] = 4

df['NationalITy'].loc[df.NationalITy == 'Jordan'] = 5

df['NationalITy'].loc[df.NationalITy == 'venzuela'] = 6

df['NationalITy'].loc[df.NationalITy == 'Iran'] = 7

df['NationalITy'].loc[df.NationalITy == 'Tunis'] = 8

df['NationalITy'].loc[df.NationalITy == 'Morocco'] = 9

df['NationalITy'].loc[df.NationalITy == 'Syria'] = 10

df['NationalITy'].loc[df.NationalITy == 'Palestine'] = 11

df['NationalITy'].loc[df.NationalITy == 'Iraq'] = 12

df['NationalITy'].loc[df.NationalITy == 'Lybia'] = 13



df['PlaceofBirth'].loc[df.PlaceofBirth == 'KuwaIT'] = 0

df['PlaceofBirth'].loc[df.PlaceofBirth == 'lebanon'] = 1

df['PlaceofBirth'].loc[df.PlaceofBirth == 'Egypt'] = 2

df['PlaceofBirth'].loc[df.PlaceofBirth == 'SaudiArabia'] = 3

df['PlaceofBirth'].loc[df.PlaceofBirth == 'USA'] = 4

df['PlaceofBirth'].loc[df.PlaceofBirth == 'Jordan'] = 5

df['PlaceofBirth'].loc[df.PlaceofBirth == 'venzuela'] = 6

df['PlaceofBirth'].loc[df.PlaceofBirth == 'Iran'] = 7

df['PlaceofBirth'].loc[df.PlaceofBirth == 'Tunis'] = 8

df['PlaceofBirth'].loc[df.PlaceofBirth == 'Morocco'] = 9

df['PlaceofBirth'].loc[df.PlaceofBirth == 'Syria'] = 10

df['PlaceofBirth'].loc[df.PlaceofBirth == 'Iraq'] = 11

df['PlaceofBirth'].loc[df.PlaceofBirth == 'Palestine'] = 12

df['PlaceofBirth'].loc[df.PlaceofBirth == 'Lybia'] = 13



df['StageID'].loc[df.StageID == 'lowerlevel'] = 0

df['StageID'].loc[df.StageID == 'MiddleSchool'] = 1

df['StageID'].loc[df.StageID == 'HighSchool'] = 2



df['GradeID'].loc[df.GradeID == 'G-04'] = 0

df['GradeID'].loc[df.GradeID == 'G-07'] = 1

df['GradeID'].loc[df.GradeID == 'G-08'] = 2

df['GradeID'].loc[df.GradeID == 'G-06'] = 3

df['GradeID'].loc[df.GradeID == 'G-05'] = 4

df['GradeID'].loc[df.GradeID == 'G-09'] = 5

df['GradeID'].loc[df.GradeID == 'G-12'] = 6

df['GradeID'].loc[df.GradeID == 'G-11'] = 7

df['GradeID'].loc[df.GradeID == 'G-10'] = 8

df['GradeID'].loc[df.GradeID == 'G-02'] = 9



df['SectionID'].loc[df.SectionID == 'A'] = 0

df['SectionID'].loc[df.SectionID == 'B'] = 1

df['SectionID'].loc[df.SectionID == 'C'] = 2



df['Topic'].loc[df.Topic == 'IT'] = 0

df['Topic'].loc[df.Topic == 'Math'] = 1

df['Topic'].loc[df.Topic == 'Arabic'] = 2

df['Topic'].loc[df.Topic == 'Science'] = 3

df['Topic'].loc[df.Topic == 'English'] = 4

df['Topic'].loc[df.Topic == 'Quran'] = 5

df['Topic'].loc[df.Topic == 'Spanish'] = 6

df['Topic'].loc[df.Topic == 'French'] = 7

df['Topic'].loc[df.Topic == 'History'] = 8

df['Topic'].loc[df.Topic == 'Biology'] = 9

df['Topic'].loc[df.Topic == 'Chemistry'] = 10

df['Topic'].loc[df.Topic == 'Geology'] = 11



df['Semester'].loc[df.Semester == 'F'] = 0

df['Semester'].loc[df.Semester == 'S'] = 1



df['Relation'].loc[df.Relation == 'Father'] = 0

df['Relation'].loc[df.Relation == 'Mum'] = 1



df['ParentAnsweringSurvey'].loc[df.ParentAnsweringSurvey == 'Yes'] = 0

df['ParentAnsweringSurvey'].loc[df.ParentAnsweringSurvey == 'No'] = 1



df['ParentschoolSatisfaction'].loc[df.ParentschoolSatisfaction == 'Good'] = 0

df['ParentschoolSatisfaction'].loc[df.ParentschoolSatisfaction == 'Bad'] = 1



df['StudentAbsenceDays'].loc[df.StudentAbsenceDays == 'Under-7'] = 0

df['StudentAbsenceDays'].loc[df.StudentAbsenceDays == 'Above-7'] = 1





'''print(df['StageID'].unique())

print(df['GradeID'].unique())

print(df['SectionID'].unique())

print(df['Topic'].unique())

print(df['Semester'].unique())

print(df['Relation'].unique())

print(df['ParentAnsweringSurvey'].unique())

print(df['ParentschoolSatisfaction'].unique())

print(df['StudentAbsenceDays'].unique())'''









continuous_subset = df.ix[:,0:16]



X = np.array(continuous_subset).astype('float64')

y = np.array(df['Class']).astype('float64')

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
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score



svm = SVC(kernel='linear', C=2.0, random_state=0)

svm.fit(X_train_std, y_train)



y_pred = svm.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))






'''cross validation'''

clf = SVC(kernel='linear', C=1)

scores = cross_val_score(clf, X, y, cv=10)

scores 

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





from sklearn.linear_model import Perceptron



ppn = Perceptron(n_iter=100, eta0=0.1, random_state=0)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print('Misclassified percentage samples: %d' % (y_test != y_pred).sum())





'''cross validation'''

scores = cross_val_score(ppn, X, y, cv=10)

scores 

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
svm = SVC(kernel='rbf', random_state=0, gamma=2, C=1.0)

svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
sns.countplot(x='StudentAbsenceDays',data = df, hue='Class',palette='bright')

plt.show()
df['AbsBoolean'] = df['StudentAbsenceDays']

df['AbsBoolean'] = np.where(df['AbsBoolean'] == 'Under-7',0,1)

continuous_subset['Absences'] = df['AbsBoolean']

X = np.array(continuous_subset).astype('float64')

y = np.array(df['Class']).astype('float64')

X_train, X_test, y_train, y_test = train_test_split(

         X, y, test_size=0.3, random_state=0)

sc = StandardScaler()



sc.fit(X_train)



X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
svm.fit(X_train_std, y_train)



y_pred = svm.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
svm = SVC(kernel='rbf', random_state=0, gamma=2, C=1.0)

svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())
import seaborn

import pandas

from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import cross_validation

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier



num_instances=480

num_folds=10

seed = 7

scoring = 'accuracy'



models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(kernel='rbf', random_state=0, gamma=2, C=1.0)))

models.append(('RF', RandomForestClassifier(n_estimators=200, n_jobs=-1, criterion='gini')))

models.append(('MLP',  MLPClassifier(hidden_layer_sizes=(32,), random_state=1, max_iter=100, warm_start=True)))

models.append(('AdaBoost', AdaBoostClassifier()))





# evaluate each model in turn

results = []

names = []

for name, model in models:

	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)

	cv_results = cross_validation.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)
from sklearn.utils.testing import all_estimators



estimators = all_estimators()



for name, class_ in estimators:

    if hasattr(class_, 'predict_proba'):

        print(name)