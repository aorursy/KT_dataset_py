import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 
df = pd.read_csv('../input/heart-disease-uci/heart.csv')

df.shape
df
df.info()
df.describe()
plt.figure(figsize=(5,5))



def plot(df,variable):

  plt.hist(df[variable])
df.describe().columns
df_num = df[['age','trestbps','chol','thalach','oldpeak']]

df_num
df_cat = df[['sex', 'cp', 'fbs', 'restecg',

       'exang',  'slope', 'ca', 'thal', 'target']]

df_cat
df_cat.columns
df.sex.value_counts()
sns.countplot(df_cat['sex'])
sns.countplot(df_cat['cp'])
sns.countplot(df_cat['fbs'])
sns.countplot(df_cat['restecg'])
sns.countplot(df_cat['exang'])
sns.countplot(df_cat['slope'])
sns.countplot(df_cat['ca'])
sns.countplot(df_cat['thal'])
# correlations 



df_Corr = df.corr()

top_corr= df_Corr.index



plt.figure(figsize=(10,7))

g=sns.heatmap(df[top_corr].corr(),cmap="BuGn_r")
sns.countplot(df['target'])
## balancing sex data

from sklearn.utils import resample

df_majority_males = df[df.sex==1]

df_minority_females = df[df.sex==0]



## upgrading the sex_class

df_minority_upsampled = resample(df_minority_females,replace=True,n_samples=207, random_state=123) 
df_upsampled = pd.concat([df_majority_males, df_minority_upsampled])

df_upsampled
df_upsampled.sex.value_counts()   ### blanced sex column
df_new = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],drop_first=True)

df_new
## preprocessing



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

colms = ['age','trestbps','chol','thalach','oldpeak']



df_new[colms] = scaler.fit_transform(df_new[colms]) 
df_new.head()
X= df_new.drop(['target'],axis=1)

y=df_new['target']



print(y)

X
from sklearn.model_selection import train_test_split



Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=123)

#### Model Selection (KNN)



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



knn_scores = []

for k in range(1,21):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    score=cross_val_score(knn_classifier,Xtrain,ytrain,cv=10)

    knn_scores.append(score.mean())
plt.figure(figsize=(30,10))

plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')

for i in range(1,21):

    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))

plt.xticks([i for i in range(1, 21)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')



knn_clf = KNeighborsClassifier(n_neighbors = 3)

score = cross_val_score(knn_clf,Xtrain,ytrain,cv=10)

score.mean()
knn= KNeighborsClassifier(n_neighbors = 3)  ### GridSearchCV
from sklearn.model_selection import GridSearchCV



parameters = {'weights':['uniform','distance'],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}



knn_clf_new = GridSearchCV(knn, parameters)
knn_clf_new
score = cross_val_score(knn_clf_new,Xtrain,ytrain,cv=10)

score
score.mean()

from sklearn.ensemble import RandomForestClassifier



clf_random = RandomForestClassifier()

score = cross_val_score(clf_random,Xtrain,ytrain,cv=10)

score.mean()

from sklearn.tree import DecisionTreeClassifier



clf_tree = DecisionTreeClassifier()

score = cross_val_score(clf_tree, Xtrain,ytrain, cv=10)

score.mean()
parameters = {'max_features':('auto', 'sqrt', 'log2'),'splitter':('best', 'random'),'criterion':('gini', 'entropy')}



tree_clf_new = GridSearchCV(DecisionTreeClassifier(), parameters,cv=3)
tree_clf_new.estimator

tree_clf_new.score
score
score = cross_val_score(tree_clf_new,Xtrain,ytrain,cv=5)

score.mean()
## best model

from sklearn.model_selection import cross_val_predict

knn = KNeighborsClassifier(n_neighbors = 3)

knn_model = knn.fit(Xtrain,ytrain)
prediction = knn_model.predict(Xtest)
from sklearn.metrics import accuracy_score

accuracy_score(ytest, prediction)