import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
test = pd.read_csv('../input/titanic/test.csv')
train = pd.read_csv('../input/titanic/train.csv')

#gender_submission.head()
#test.head()
#train.head()
train.head()

print(gender_submission.shape)
print(test.shape)
print(train.shape)
train.info(), test.info()
train.isnull().sum()

test.isnull().sum()

def pie_chart(df, column ,explode , labels,title,no):
    
    plt.pie(df[column].value_counts(),
            explode=explode,    #explode=[0.04,0]
            startangle=90, 
            autopct='%1.1f%%',
            labels=labels, #labels=['Males','Females']
            colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99'],
            pctdistance=.6,
            textprops={'fontsize': 20})
    plt.title(title)
    plt.figure(no)

pie_chart(train, "Pclass" ,[0.05,0.05,0.05], ['3','1',"2"],"Pclass for train data",0)
pie_chart(test ,"Pclass" ,[0.05,0.05,0.05], ['3','1',"2"],"Pclass for test data",1)

survived = train[train["Survived"] == 1]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[20, 5])

sns.countplot(train['Sex'], ax =axes[0]).set_title("number of males and females on the ship", fontsize=18)

sns.countplot(survived['Sex'], ax =axes[1]).set_title("number of males and females who survived", fontsize=18)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[20, 7])

sns.countplot(train["Embarked"], ax =axes[0]).set_title("Train data Embarked distribution", fontsize=18)

sns.countplot(test["Embarked"], ax =axes[1]).set_title("Test data Embarked distribution", fontsize=18)

train["Embarked"].fillna("S", inplace = True)
train[train["Age"] <2]
train_notnull_age = train[pd.notnull(train["Age"])]
test_notnull_age = test[pd.notnull(test["Age"])]
train_notnull_age
train_notnull_age["Age"] = train_notnull_age["Age"].astype(int)
test_notnull_age["Age"] = test_notnull_age["Age"].astype(int)


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=[30, 9])

sns.countplot(train_notnull_age['Age'],ax =axes[0]).set_title("Age distribution for train data", fontsize=20)
sns.countplot(test_notnull_age['Age'],ax =axes[1]).set_title("Age distribution for test data", fontsize=20)
train["Age"].fillna(train["Age"].median(), inplace = True)
test["Age"].fillna(test["Age"].median(), inplace = True)
train["Age"] = train["Age"].astype(int)
test["Age"] = test["Age"].astype(int)

test["Fare"].fillna(test["Fare"].median(), inplace = True)
train["Fare"].describe()
train[train["Fare"] < 4]
fare = train[train["Fare"] > 4]
fare["Fare"]
train.loc[ train.Fare == 0, "Fare" ] = fare["Fare"].median()

train[train["Fare"] < 4]
train.isnull().sum(), test.isnull().sum()

train["Cabin"].describe()
train.head()
train_without_cabin= train[train["Cabin"].isnull()]
train_with_cabin= train[train["Cabin"].notnull()]

test_without_cabin= test[test["Cabin"].isnull()]
test_with_cabin= test[test["Cabin"].notnull()]
train_without_cabin["Pclass"].value_counts(), train_with_cabin["Pclass"].value_counts()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[24, 5])

sns.countplot(train_without_cabin['Pclass'],ax =axes[0]).set_title("Pclass without cabin for train data", fontsize=18)
sns.countplot(test_without_cabin['Pclass'],ax =axes[1]).set_title("Pclass without cabin for test data", fontsize=18)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[24, 5])

sns.countplot(train_with_cabin['Pclass'],ax =axes[0]).set_title("Pclass with cabin for train data", fontsize=18)
sns.countplot(test_with_cabin['Pclass'],ax =axes[1]).set_title("Pclass with cabin for test data", fontsize=18)

train_with_cabin[train_with_cabin["Pclass"] == 3]
train_with_cabin["Cabin"].unique()
train["Cabin"].loc[ train["Cabin"].str.contains("C", na=False) ] = "C"
train["Cabin"].loc[ train["Cabin"].str.contains("E", na=False) ] = "E"
train["Cabin"].loc[ train["Cabin"].str.contains("A", na=False) ] = "A"
train["Cabin"].loc[ train["Cabin"].str.contains("B", na=False) ] = "B"
train["Cabin"].loc[ train["Cabin"].str.contains("G", na=False) ] = "G"
train["Cabin"].loc[ train["Cabin"].str.contains("F", na=False) ] = "F"
train["Cabin"].loc[ train["Cabin"].str.contains("D", na=False) ] = "D"
train["Cabin"].loc[ train["Cabin"].str.contains("B", na=False) ] = "B"

test["Cabin"].loc[ test["Cabin"].str.contains("C", na=False) ] = "C"
test["Cabin"].loc[ test["Cabin"].str.contains("E", na=False) ] = "E"
test["Cabin"].loc[ test["Cabin"].str.contains("A", na=False) ] = "A"
test["Cabin"].loc[ test["Cabin"].str.contains("B", na=False) ] = "B"
test["Cabin"].loc[ test["Cabin"].str.contains("G", na=False) ] = "G"
test["Cabin"].loc[ test["Cabin"].str.contains("F", na=False) ] = "F"
test["Cabin"].loc[ test["Cabin"].str.contains("D", na=False) ] = "D"
test["Cabin"].loc[ test["Cabin"].str.contains("B", na=False) ] = "B"

train.head(20)
train["Cabin"].unique()
train_with_cabin= train[train["Cabin"].notnull()]
train_with_cabin= train_with_cabin[train_with_cabin["Pclass"] == 3]

test_with_cabin= train[train["Cabin"].notnull()]
test_with_cabin= test_with_cabin[test_with_cabin["Pclass"] == 3]
test_with_cabin["Cabin"].unique(), train_with_cabin["Cabin"].unique()

train["Cabin"].fillna("G", inplace = True)
test["Cabin"].fillna("G", inplace = True)

train["Cabin"].value_counts(),test["Cabin"].value_counts()
train["Cabin"].unique(), test["Cabin"].unique()
train.isnull().sum(), test.isnull().sum()
sns.pairplot(train, palette='deep')
ax = sns.barplot(data=train.drop(["PassengerId","Age","Fare"],axis=1), capsize=.2)
plt.tick_params(axis='x', rotation=30)
ax = sns.barplot(data=train[["Age","Fare"]], capsize=.2)
plt.tick_params(axis='x', rotation=30)
colormap = plt.cm.RdBu
plt.figure(figsize=(14,9))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
train.columns , test.columns
train["family"] = train["SibSp"] + train["Parch"]+1
train.drop(["Parch","SibSp"],axis=1,inplace=True)


test["family"] = test["SibSp"] + test["Parch"]+1
test.drop(["Parch","SibSp"],axis=1,inplace=True)

train['Title'] = train['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
train['Is_Married'] = 0
train['Is_Married'].loc[train['Title'] == 'Mrs'] = 1

test['Title'] = test['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
test['Is_Married'] = 0
test['Is_Married'].loc[test['Title'] == 'Mrs'] = 1
train['Title'].unique()
plt.rcParams['figure.figsize'] = (11, 6)

sns.countplot(train["Title"]).set_title("Is_Married feature distribution", fontsize=18)
train.head()
colormap = plt.cm.RdBu
plt.figure(figsize=(14,9))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
#train = pd.get_dummies(train, columns=["Embarked","Sex"])
#test = pd.get_dummies(test, columns=["Embarked","Sex"])
#train.head()
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
train["Ticket"]=le.fit_transform(train["Ticket"])
train["Embarked"]=le.fit_transform(train["Embarked"])
train["Name"]=le.fit_transform(train["Name"])
train["Sex"]=le.fit_transform(train["Sex"])
train["Cabin"]=le.fit_transform(train["Cabin"])
train["Title"]=le.fit_transform(train["Title"])


test["Ticket"]=le.fit_transform(test["Ticket"])
test["Embarked"]=le.fit_transform(test["Embarked"])
test["Name"]=le.fit_transform(test["Name"])
test["Sex"]=le.fit_transform(test["Sex"])
test["Cabin"]=le.fit_transform(test["Cabin"])
test["Title"]=le.fit_transform(test["Title"])

#list(Name)
#le.inverse_transform([2])
#test['Sex'] = test['Sex'].apply(lambda x: 0 if x == "male" else 1)

#train['Sex'] = train['Sex'].apply(lambda x: 0 if x == "male" else 1)
X_train = train.drop(["Survived"], axis=1)
Y_train = train["Survived"]
X_test  = test.copy()
X_train.shape, Y_train.shape, X_test.shape
#Y_test = gender_submission["Survived"]
X_train.head()
X_test.head()
#from sklearn.feature_selection import f_regression
#f_regression(X_train, Y_train, center=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
#y_train = scaler.fit_transform(y_train)
#y_test = scaler.fit_transform(y_test)
#scaler.transform(X_train)
#from sklearn.preprocessing import MaxAbsScaler
#MAscaler = MaxAbsScaler()
#X_train = MAscaler.fit_transform(X_train)
#X_test = MAscaler.fit_transform(X_test)
#from sklearn.preprocessing import MinMaxScaler
#MMscaler = MinMaxScaler(feature_range=(0,1))
#X_train = MMscaler.fit_transform(X_train)
#X_test = MMscaler.fit_transform(X_test)
#from sklearn.preprocessing import Normalizer
#normal = Normalizer(norm="l2")
#X_train = normal.fit_transform(X_train)
#X_test = normal.fit_transform(X_test)
scoring="accuracy"
k_fold = KFold(n_splits= 10 ,shuffle = True,random_state=0)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(solver='liblinear',max_iter=2000)
LR.fit(X_train,Y_train)


LR_acc = round(LR.score(X_train, Y_train) * 100, 2)
print(LR_acc)

LR_pred = LR.predict(X_test)

LR_CV = cross_val_score(LR, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(LR_CV)
print(LR_CV.mean())

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([('classifier' , LogisticRegression())])

param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']}]

LR2 = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)


best_LR = LR2.fit(X_train, Y_train)

LR_acc = round(best_LR.score(X_train, Y_train) * 100, 2)
LR_acc

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,Y_train)

gnb_acc = round(gnb.score(X_train, Y_train) * 100, 2)
print(gnb_acc)

cv = cross_val_score(gnb,X_train,Y_train,cv=k_fold)
print(cv)
print(cv.mean())

svc = SVC()
svc.fit(X_train, Y_train)
SVC_pred = svc.predict(X_test)

acc_SVC = round(svc.score(X_train, Y_train) * 100, 2)
print(acc_SVC)


SVC_CV = cross_val_score(svc, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring).mean()
print(SVC_CV)

param_grid = [
    {'kernel' : ['linear', 'rbf', 'poly']}]
    #'gamma' : [0.1, 1, 10, 100]
    #'C' : [0.1, 1, 10, 100, 1000],
    #'degree' : [0, 1, 2, 3, 4, 5, 6]}]

svc2 = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)


best_svc2 = svc2.fit(X_train, Y_train)

svc2_acc = round(best_svc2.score(X_train, Y_train) * 100, 2)
svc2_acc
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
KNN_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print(acc_knn)

KNN_CV = cross_val_score(knn, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring).mean()
print(KNN_CV)

param_grid = {'n_neighbors' : [3,5,7,9],
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree'],
              'p' : [1,2]}

Knn2 = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)


best_Knn2 = Knn2.fit(X_train, Y_train)

Knn2_acc = round(best_Knn2.score(X_train, Y_train) * 100, 2)
Knn2_acc

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
rand_forrest_pred = random_forest.predict(X_test)


acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)

random_forest_CV = cross_val_score(random_forest, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring).mean()
print(random_forest_CV)

param_grid =  {'n_estimators': [400,450,500,550],
               'criterion':['gini','entropy'],
                                  'bootstrap': [True],
                                  'max_depth': [15, 20, 25],
                                  'max_features': ['auto','sqrt', 10],
                                  'min_samples_leaf': [2,3],
                                  'min_samples_split': [2,3]}

random_forest2 = GridSearchCV(random_forest, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)


best_random_forest2 = random_forest2.fit(X_train, Y_train)

random_forest2_acc = round(best_random_forest2.score(X_train, Y_train) * 100, 2)
random_forest2_acc
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state = 1)

param_grid = {
    'n_estimators': [450,500,550],
    'colsample_bytree': [0.75,0.8,0.85],
    'max_depth': [None],
    'reg_alpha': [1],
    'reg_lambda': [2, 5, 10],
    'subsample': [0.55, 0.6, .65],
    'learning_rate':[0.5],
    'gamma':[.5,1,2],
    'min_child_weight':[0.01],
    'sampling_method': ['uniform']
}

clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X_train,Y_train)
xgb_acc = round(best_clf_xgb.score(X_train, Y_train) * 100, 2)
xgb_acc
from keras.callbacks import ModelCheckpoint, EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

#Layers
#Input Layer
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
#Hidden Layers
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))


#Ouput Layer
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Binary cross_entropy is for classification loss function.
model.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fit the model
model.fit(X_train, Y_train, batch_size = 32, epochs = 6000, verbose=2, callbacks=[es])
model.summary()

y_pred = model.predict(X_test)
y_fin = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])
_, train_acc = model.evaluate(X_train, Y_train, verbose=0)
print('Train: %.3f'%(train_acc))
#from sklearn.model_selection import cross_val_predict
#from sklearn.metrics import confusion_matrix
#predictions = cross_val_predict(L, X_train, Y_train, cv=3)
#confusion_matrix(Y_train, predictions)
final_report = pd.DataFrame({'classifier': ["Log_reg_score","SVC_score","KNN_score","random_forest_score","XGBoost_score","NN"]
                            ,'train_acc(Tuned)':  [LR_acc ,svc2_acc ,Knn2_acc ,random_forest2_acc ,xgb_acc ,train_acc*100]
                   })

final_report