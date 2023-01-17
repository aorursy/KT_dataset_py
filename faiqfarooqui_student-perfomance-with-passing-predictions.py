import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
passingmarks = 40
df = pd.read_csv("../input/StudentsPerformance.csv")

df.head()
df.shape
df.describe()
df.isnull().sum()
from matplotlib import rcParams

# figure size in inches
rcParams['figure.figsize'] = 20,20.27
sns.countplot(x="math score", data = df)
plt.show()
sns.countplot(x="reading score", data = df)
plt.show()
sns.countplot(x="writing score", data = df)
plt.show()
df['Math_PassStatus'] = np.where(df['math score']<passingmarks, 'F', 'P')
df.Math_PassStatus.value_counts()
df['Reading_PassStatus'] = np.where(df['reading score']<passingmarks, 'F', 'P')
df.Reading_PassStatus.value_counts()
df['Writing_PassStatus'] = np.where(df['writing score']<passingmarks, 'F', 'P')
df.Writing_PassStatus.value_counts()
sns.countplot(x='parental level of education', data = df, hue='Math_PassStatus', palette='bright')
plt.show()
sns.countplot(x='parental level of education', data = df, hue='Reading_PassStatus', palette='bright')
plt.show()
sns.countplot(x='parental level of education', data = df, hue='Writing_PassStatus', palette='bright')
plt.show()
df['OverAll_PassStatus'] = df.apply(lambda x : 'F' if x['Math_PassStatus'] == 'F' or 
                                    x['Reading_PassStatus'] == 'F' or x['Writing_PassStatus'] == 'F' else 'P', axis =1)

df.OverAll_PassStatus.value_counts()
sns.countplot(x='parental level of education', data = df, hue='OverAll_PassStatus', palette='bright')
plt.show()
df['Total_Marks'] = df['math score']+df['reading score']+df['writing score']
df['Percentage'] = df['Total_Marks']/3
sns.countplot(x="Percentage", data = df, palette="muted")
plt.show()
def GetGrade(Percentage, OverAll_PassStatus):
    if ( OverAll_PassStatus == 'F'):
        return 'F'    
    if ( Percentage >= 80 ):
        return 'A'
    if ( Percentage >= 70):
        return 'B'
    if ( Percentage >= 60):
        return 'C'
    if ( Percentage >= 50):
        return 'D'
    if ( Percentage >= 40):
        return 'E'
    else: 
        return 'F'

df['Grade'] = df.apply(lambda x : GetGrade(x['Percentage'], x['OverAll_PassStatus']), axis=1)

df.Grade.value_counts()
sns.countplot(x="Grade", data = df, order=['A','B','C','D','E','F'],  palette="muted")
plt.show()
sns.countplot(x='parental level of education', data = df, hue='Grade', palette='bright')
plt.show()
df.head()
sns.countplot(x='test preparation course', data = df, hue='Grade', palette='bright')
plt.show()
df['test preparation course'].value_counts()
sns.countplot(x='race/ethnicity', data = df, hue='Grade', palette='bright')
plt.show()
sns.countplot(x='gender', data = df, hue='Grade', palette='bright')
plt.show()
df.head()
d = {'female': 1, 'male': 0}
d1 = {'some college': 1, 'associate\'s degree': 2, 'bachelor\'s degree': 3,'master\'s degree':4,'high school':5,'some high school':6}
d2 =  {'standard': 1, 'free/reduced': 0}
d3 = {'none': 1, 'completed': 0}
d4 = {'group A': 1, 'group B': 2, 'group C': 3,'group D':4,'group E':5}

df['gender'] = df['gender'].map(d)
df['race/ethnicity'] = df['race/ethnicity'].map(d4)
df['parental level of education'] = df['parental level of education'].map(d1)
df['lunch'] = df['lunch'].map(d2)
df['test preparation course'] = df['test preparation course'].map(d3)
df.head()
d = {'P': 1, 'F': 0}
df['Math_PassStatus'] = df['Math_PassStatus'].map(d)
d = {'P': 1, 'F': 0}
df['Reading_PassStatus'] = df['Reading_PassStatus'].map(d)
d = {'P': 1, 'F': 0}
df['Writing_PassStatus'] = df['Writing_PassStatus'].map(d)
df.head()
d = {'P': 1, 'F': 0}
df['OverAll_PassStatus'] = df['OverAll_PassStatus'].map(d)
d = {'A': 1, 'B': 2,'C': 3, 'D': 4,'E': 5, 'F': 6}
df['Grade'] = df['Grade'].map(d)
df.head()
features = list(df.columns[:5])
features
from sklearn import tree
y = df["OverAll_PassStatus"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf,X, y, cv=10)
print(scores.mean())
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
scores = cross_val_score(clf,X, y, cv=10)
print(scores.mean())
from sklearn import svm, datasets

C = 1.0
svc = svm.SVC(kernel='linear', C=C)
scores = cross_val_score(svc,X, y, cv=10)
print(scores.mean())
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=10)
scores = cross_val_score(knn,X, y, cv=10)
print(scores.mean())
for i in range(1,50):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    cv_scores = cross_val_score(knn,X, y, cv=10)
    print('score '+str(i)+': '+ str(cv_scores.mean())+'\n')
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
all_features_minmax = scaler.fit_transform(X)

clf = MultinomialNB()
cv_scores = cross_val_score(clf, X, y, cv=10)

cv_scores.mean()
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
cv_scores = cross_val_score(clf, X, y, cv=10)

cv_scores.mean()
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(random_state=1)
cv_scores = cross_val_score(clf, X, y, cv=10)

cv_scores.mean()
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor(random_state=1)
cv_scores = cross_val_score(clf, X, y, cv=10)

cv_scores.mean()














