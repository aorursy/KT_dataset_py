import pandas as pd

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")
df.head()
df.dtypes
df.isnull().sum()
df.fillna(value=df.mode().loc[0], inplace=True)

#df.dropna(inplace=True)

df.isnull().sum()
X = df[["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","type","feature10","feature11"]].copy()

y = df["rating"]
X = pd.get_dummies(data=X, columns=["type"])

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



#from sklearn.preprocessing import MinMaxScaler



#scaler1 = MinMaxScaler()

#scaler1.fit(X_train)



#X_train = scaler1.transform(X_train)

#X_test = scaler1.transform(X_test)



#from sklearn.preprocessing import RobustScaler

#scaler2 = RobustScaler()

#scaler2.fit(X_train)



#X_train = scaler2.transform(X_train)

#X_test = scaler2.transform(X_test)
clf = DecisionTreeClassifier(max_depth=12)



clf = clf.fit(X_train,y_train)



y_pred1 = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(metric='manhattan', n_neighbors=19, weights='distance')

classifier.fit(X_train, y_train)

y_pred2 = classifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))
from sklearn.ensemble import RandomForestClassifier

rfmodel = RandomForestClassifier(criterion='entropy', min_samples_split=2, n_estimators=100)

rfmodel.fit(X_train, y_train)

y_predrf = rfmodel.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_predrf))
from sklearn.svm import SVC

svclassifier = SVC(kernel="rbf", gamma=0.9)

svclassifier.fit(X_train, y_train)
y_pred3 = svclassifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred3))
kf = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")

kf.fillna(value=df.mean(), inplace=True)

kf.isnull().sum()

myid1 = kf["id"].copy()

X = kf[["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","type","feature10","feature11"]].copy()

X = pd.get_dummies(data=X, columns=["type"])



scaler = StandardScaler()

scaler.fit(X)



X = scaler.transform(X)



Y_pred =  rfmodel.predict(X)

boo1 = pd.DataFrame(Y_pred)

ans = pd.concat([myid1,boo1],axis=1)

ans.columns=["id","rating"]

ans = ans.set_index("id")

ans.to_csv("ans_Randomforest.csv")
kf = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")

kf.fillna(value=df.mean(), inplace=True)

kf.isnull().sum()

myid1 = kf["id"].copy()

X = kf[["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","type","feature10","feature11"]].copy()

X = pd.get_dummies(data=X, columns=["type"])



scaler = StandardScaler()

scaler.fit(X)



X = scaler.transform(X)



Y_pred =  clf.predict(X)

boo1 = pd.DataFrame(Y_pred)

ans = pd.concat([myid1,boo1],axis=1)

ans.columns=["id","rating"]

ans = ans.set_index("id")

ans.to_csv("ans_decisionTree.csv")
kf = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")

kf.fillna(value=df.mean(), inplace=True)

kf.isnull().sum()

myid1 = kf["id"].copy()

X = kf[["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","type","feature10","feature11"]].copy()

X = pd.get_dummies(data=X, columns=["type"])



scaler = StandardScaler()

scaler.fit(X)



X = scaler.transform(X)



Y_pred =  svclassifier.predict(X)

boo1 = pd.DataFrame(Y_pred)

ans = pd.concat([myid1,boo1],axis=1)

ans.columns=["id","rating"]

ans = ans.set_index("id")

ans.to_csv("ans_svc.csv")
kf = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")

kf.fillna(value=df.mean(), inplace=True)

kf.isnull().sum()

myid1 = kf["id"].copy()

X = kf[["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","type","feature10","feature11"]].copy()

X = pd.get_dummies(data=X, columns=["type"])



scaler = StandardScaler()

scaler.fit(X)



X = scaler.transform(X)



Y_pred =  classifier.predict(X)

boo1 = pd.DataFrame(Y_pred)

ans = pd.concat([myid1,boo1],axis=1)

ans.columns=["id","rating"]

ans = ans.set_index("id")

ans.to_csv("ans_kne.csv")
sns.distplot(df['feature1'],kde = False)
coo = df.corr()

top = coo.index

plt.figure(figsize=(20,20))

g = sns.heatmap(df[top].corr(),annot=True,cmap="RdYlGn")
from sklearn.model_selection import GridSearchCV

#parameters = [{'C': [1,10,100,1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9]}]

#grid_search = GridSearchCV(estimator = svclassifier,

#                           param_grid = parameters,

#                           scoring = 'accuracy',

#                           cv = 10,

#                           n_jobs = -1)

#grid_search = grid_search.fit(X_train, y_train)
#grid_params = {

#    'n_neighbors' : [3,5,11,19],

#    'weights' : ['uniform','distance'],

#    'metric' : ['euclidean','manhattan']

#}





#grid_search = GridSearchCV(estimator = classifier,

#                           param_grid = grid_params,

#                           verbose=1,

#                           cv = 3,

#                           n_jobs = -1)

#grid_search = grid_search.fit(X_train, y_train)
grid_param = {

    'n_estimators': [30,50,100],

    'criterion' : ['entropy', 'gini'],

    'min_samples_split' : [2,10,20,50]

}





grid_search = GridSearchCV(estimator = rfmodel,

                           param_grid = grid_param,

                           verbose=1,

                           cv = 3,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
accuracy = grid_search.best_score_
accuracy
grid_search.best_params_