import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv(r'train.csv')
test = pd.read_csv(r'test.csv')
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())
train.head()
test.head()
train.select_dtypes(include=['object'])
train=pd.get_dummies(train,columns=['col2','col11','col37','col44','col56']).drop(columns=['col44_No','col11_No'])
test=pd.get_dummies(test,columns=['col2','col11','col37','col44','col56']).drop(columns=['col44_No','col11_No'])
train.head()
f, ax = plt.subplots(figsize=(20, 20))
corr = train.drop(columns=['ID','Class']).corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
to_drop
f, ax = plt.subplots(figsize=(20, 20))
corr = train.drop(columns=['ID','Class']+to_drop).corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
X_train = train.drop(columns=['ID','Class'])
y_train = train.drop(columns=['ID'])['Class']
X_test = test.drop(columns=['ID'])
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.fit_transform(X_test))
from sklearn.model_selection import train_test_split
X_t1, X_t2, y_t1, y_t2 = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
np.random.seed(42)

# Metrics and cross-validation
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
train_accuracy = []
test_accuracy = []

for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_t1,y_t1)
    train_accuracy.append(knn.score(X_t1,y_t1))
    test_accuracy.append(knn.score(X_t2,y_t2))

plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,20),train_accuracy,color='blue', marker='o', markerfacecolor='red', markersize=5)
test_score,=plt.plot(range(1,20),test_accuracy,color='red', marker='o', markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Accuracy", "Test Accuracy"])
plt.title('Accuracy vs K neighbors')
plt.xlabel('K neighbors')
plt.ylabel('Accuracy')
tree = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
lr = LogisticRegression()
nb = GaussianNB()
svm = SVC()
knn = KNeighborsClassifier(n_neighbors=10)
scorer_f1 = make_scorer(f1_score, average = 'micro')
print('knn',end='\n\n')
res = cross_validate(knn, X_train, y_train, cv=10, scoring=(scorer_f1), return_train_score=True)
print("Train Accuracy for 10 folds= ",np.mean(res['train_score']))
print("Test Accuracy for 10 folds = ",np.mean(res['test_score']))

print('dt',end='\n\n')
res = cross_validate(tree, X_train, y_train, cv=10, scoring=(scorer_f1), return_train_score=True)
print("Train Accuracy for 10 folds= ",np.mean(res['train_score']))
print("Test Accuracy for 10 folds = ",np.mean(res['test_score']))


print('rf',end='\n\n')
res = cross_validate(rf, X_train, y_train, cv=10, scoring=(scorer_f1), return_train_score=True)
print("Train Accuracy for 10 folds= ",np.mean(res['train_score']))
print("Test Accuracy for 10 folds = ",np.mean(res['test_score']))

print('lr',end='\n\n')
res = cross_validate(lr, X_train, y_train, cv=10, scoring=(scorer_f1), return_train_score=True)
print("Train Accuracy for 10 folds= ",np.mean(res['train_score']))
print("Test Accuracy for 10 folds = ",np.mean(res['test_score']))

print('nb',end='\n\n')
res = cross_validate(nb, X_train, y_train, cv=10, scoring=(scorer_f1), return_train_score=True)
print("Train Accuracy for 10 folds= ",np.mean(res['train_score']))
print("Test Accuracy for 10 folds = ",np.mean(res['test_score']))

print('svm',end='\n\n')
res = cross_validate(svm, X_train, y_train, cv=10, scoring=(scorer_f1), return_train_score=True)
print("Train Accuracy for 10 folds= ",np.mean(res['train_score']))
print("Test Accuracy for 10 folds = ",np.mean(res['test_score']))
score_train_RF = []
score_test_RF = []

for i in range(5,50,1):
    rf = RandomForestClassifier(n_estimators = 100, max_depth=i)
    rf.fit(X_t1, y_t1)
    score_train_RF.append(rf.score(X_t1,y_t1))
    score_test_RF.append(rf.score(X_t2,y_t2))

plt.figure(figsize=(10,6))
train_score,=plt.plot(range(5,50,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(5,50,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')
from sklearn.model_selection import GridSearchCV

score_train_RF = []
score_test_RF = []

for i in range(10,500,10):
    rf = RandomForestClassifier(n_estimators = i, max_depth=40)
    rf.fit(X_t1, y_t1)
    sc_train = rf.score(X_t1,y_t1)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_t2,y_t2)
    score_test_RF.append(sc_test)
    
plt.figure(figsize=(10,6))
train_score,=plt.plot(range(10,500,10),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(10,500,10),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')
score_train_RF = []
score_test_RF = []

for i in range(100,200,10):
    rf = RandomForestClassifier(n_estimators = i, max_depth=40)
    rf.fit(X_t1, y_t1)
    sc_train = rf.score(X_t1,y_t1)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_t2,y_t2)
    score_test_RF.append(sc_test)
    
plt.figure(figsize=(10,6))
train_score,=plt.plot(range(100,200,10),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(100,200,10),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')
rf_temp = RandomForestClassifier(n_estimators = 120)        #Initialize the classifier object

parameters = {'max_depth':[6,9,10,40, 50],'min_samples_split':[2, 3, 4, 5,6,7,8,9]}    #Dictionary of parameters

scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators=120, max_depth = 40, min_samples_split = 8)
rf_best.fit(X_t1, y_t1)
rf_best.score(X_t2,y_t2)
rf_best.fit(X_train,y_train)
y_test=rf_best.predict(X_test)
y_test
test['Class']=y_test
test.head()
test[['ID','Class']].to_csv('sub2.csv',index=False)
features = X_train.columns.values
imp = rf_best.feature_importances_
indices = np.argsort(imp)[::-1]

#plot
plt.figure(figsize=(20,20))
plt.bar(range(len(indices)), imp[indices], color = 'b', align='center')
plt.xticks(range(len(indices)), features[indices], rotation='vertical')
plt.xlim([-1,len(indices)])
X1 = X_train.drop(columns=[64,68,62,8,69,61,65,67,63,66,60,59,23])
X2 = X_test.drop(columns=[64,68,62,8,69,61,65,67,63,66,60,59,23])
rf_best.fit(X1,y_train)
y_test=rf_best.predict(X2)
test['Class']=y_test
test[['ID','Class']].to_csv('sub3.csv',index=False)
y_test
