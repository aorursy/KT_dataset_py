# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt 
import seaborn as sns 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/student-performance-in-class/iitstudentperformance.csv')
df
df.gender.unique(),df.NationalITy.unique(),df.NationalITy.unique(),df.StageID.unique(),df.GradeID.unique(),df.Semester.unique()
df.Class.unique(),df.Relation.unique()
df[['gender_']]=pd.get_dummies(df['gender'],drop_first=True)
df[['semester_']]=pd.get_dummies(df['Semester'],drop_first=True)
df[['ParentAnsweringSurvey_']]=pd.get_dummies(df['ParentAnsweringSurvey'],drop_first=True)
df[['ParentschoolSatisfaction_']]=pd.get_dummies(df['ParentschoolSatisfaction'],drop_first=True)
df[['Relation_']]=pd.get_dummies(df['Relation'],drop_first=True)
df[['StudentAbsenceDays_']]=pd.get_dummies(df['StudentAbsenceDays'],drop_first=True)
df1=df.drop(columns =['gender', 'Semester','ParentAnsweringSurvey','ParentschoolSatisfaction','Relation','StudentAbsenceDays']) 
mapping = {'H' : 3, 'M' : 2, 'L' : 1}
df1['Class'] = df1['Class'].map(mapping)
mapping = {'HighSchool' : 3, 'MiddleSchool' : 2, 'lowerlevel' : 1}
df1['StageID'] = df1['StageID'].map(mapping)
mapping = {'C' : 3, 'B' : 2, 'A' : 1}
df1['SectionID'] = df1['SectionID'].map(mapping)
mapping = {'G-04':4, 'G-07':7, 'G-08':8, 'G-06':6, 'G-05':5, 'G-09':9, 'G-12':12, 'G-11':11,
       'G-10':10, 'G-02':2}
df1['GradeID'] = df1['GradeID'].map(mapping)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder().fit(df1['Topic'])
df1['Topic'] = encoder.transform(df['Topic'])

df.StudentAbsenceDays.unique()
encoder1 = LabelEncoder().fit(df1['PlaceofBirth'])
df1['PlaceofBirth'] = encoder1.transform(df1['PlaceofBirth'])
encoder1 = LabelEncoder().fit(df1['NationalITy'])
df1['NationalITy'] = encoder1.transform(df1['NationalITy'])
df1
df2=df1
df2
feature_scale=[feature for feature in df2.columns if feature not in ['semester_']]
feature_scale

'''
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df2[feature_scale])
'''
df2['semester_'].value_counts()
df2['gender_'].value_counts()
df2.isnull().sum()
df2['raisedhands'].hist(bins=20)
print(df2.boxplot(column='VisITedResources'))
sns.countplot(df2.NationalITy)
feature_scale=[feature for feature in df2.columns if feature not in ['semester_']]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df2[feature_scale])
data = pd.concat([df2[['semester_']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(df2[feature_scale]), columns=feature_scale)], axis=1)
data
Y=data[['semester_']]
X=data.drop(columns=['semester_'],axis=1)
X

'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2,k=10)
fit = bestfeatures.fit(X,Y)
'''

'''
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores

print(featureScores.nlargest(10,'Score'))
f=featureScores.nlargest(10,'Score')
f_=f['Specs']
X_=X[f_]
Y_=Y
'''

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
feat_importances.nlargest(7)
X_=data[['AnnouncementsView',
'raisedhands',
'VisITedResources',
'Topic',       
'Discussion',
'GradeID',
'NationalITy']]
Y_=Y
X_
arr = np.array(Y_)
Y_=arr.reshape(480,)
Y_
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
knnclassifier = KNeighborsClassifier(n_neighbors=4)
print(cross_val_score(knnclassifier, X_, Y_, cv=5, scoring ='accuracy'))
print(cross_val_score(knnclassifier, X_, Y_, cv=5, scoring ='accuracy').mean())
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, X_, Y_, cv=10, scoring = 'accuracy'))
print(cross_val_score(logreg, X_, Y_, cv=10, scoring = 'accuracy').mean())
#decision tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
dec = DecisionTreeClassifier(criterion='entropy',max_depth=20)
print (cross_val_score(dec, X_, Y_, cv=10, scoring = 'accuracy'))
print (cross_val_score(dec, X_, Y_, cv=10, scoring = 'accuracy').mean())
from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
print (cross_val_score(model, X_, Y_, cv=10, scoring = 'accuracy'))
print (cross_val_score(model, X_, Y_, cv=10, scoring = 'accuracy').mean())
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200
)
print(cross_val_score(model, X_, Y_, cv=10, scoring = 'accuracy'))
print(cross_val_score(model, X_, Y_, cv=10, scoring = 'accuracy').mean())
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.20, random_state=0)
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score,f1_score,log_loss
#logistic regression
from sklearn.linear_model import LogisticRegression
model= LogisticRegression(C=1,solver='liblinear')
model.fit(x_train,y_train)
predictions2=model.predict(x_test)
predictions2
accuracy_score(y_test,predictions2),f1_score(y_test,predictions2)
#decision tree
model = DecisionTreeClassifier(criterion='entropy',max_depth=100)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
predictions
from sklearn.metrics import accuracy_score,f1_score,log_loss
accuracy_score(y_test,predictions),f1_score(y_test,predictions)
#k-nearest-neighbors
k=9
model=KNeighborsClassifier(n_neighbors=k)
model.fit(x_train,y_train)
#jaccard_index
predictions3 = model.predict(x_test)
print(accuracy_score(y_test, predictions3))
f1_score(y_test,predictions3)
from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(x_train,y_train)
# Actual class predictions
pred = model.predict(x_test)
print(accuracy_score(y_test, pred))
f1_score(y_test,pred)
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200
)
classifier.fit(x_train,y_train)
pred = classifier.predict(x_test)
print(accuracy_score(y_test, pred))
f1_score(y_test,pred)
