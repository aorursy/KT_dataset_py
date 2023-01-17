import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import warnings
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC
import os 
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")
combine = [train_df,test_df]
train_df.columns.values
# pclass relationship ---- we need to keep variable pclass 
train_df[['Pclass', 'Survived']].groupby(['Pclass'],as_index = False).mean().sort_values(by='Survived', ascending=False)
# parch relationship 
train_df[['Parch', 'Survived']].groupby(['Parch'],as_index = False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp', 'Survived']].groupby(['SibSp'],as_index = False).mean().sort_values(by='Survived', ascending=False)
train_df.columns.values
train_df[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean().sort_values(by="Sex",ascending = False)
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
sns.distplot(train_df["Age"],rug = True)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
train_df.describe(include=["O"])
train_df.isnull().sum()
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
## Droop Ticket and Cabin
print ("before dropping, the dimensions of train and test sets are ", combine[0].shape, combine[1].shape)

train_df.drop(["Ticket","Cabin"],axis= 1, inplace=True)
test_df.drop(["Ticket","Cabin"],axis= 1, inplace=True)
combine = [train_df,test_df]
print ("After dropping,the dimensions of train and test sets are ", combine[0].shape, combine[1].shape)
## Extract Title from Name 
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine: 
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

total_titles = train_df["Title"].append(test_df["Title"])
major_titles = total_titles.value_counts() >= 40 
rare_titles = list(major_titles[major_titles != True].index)

for dataset in combine:
    dataset["Title"]  = dataset["Title"].replace(rare_titles,"Rare")
    dataset.drop("Name",axis = 1, inplace=True)

print ("Counts of Titles after transformation")
print (train_df["Title"].value_counts())
print ("_"*40)
print ("Cross tables between title and sex")
print (pd.crosstab(train_df["Title"],train_df["Sex"]))

print ("_"*40)
print ("Cross tables between title and sex in test dataset")
print (pd.crosstab(test_df["Title"],test_df["Sex"]))



# Age null values --- 177 null values 
print ("Before transformation, the null values in feature 'Age'")
for dataset in combine:
    print (dataset["Age"].isnull().sum())


grid = sns.FacetGrid(train_df, col='Sex', row='Pclass',size=2.2, aspect=1.6,palette="Set1")
grid.map(plt.hist, 'Age',alpha=.5, bins=20)
grid.add_legend()
def hand_over(value):
    key = float(str(value).split(".")[-1][0])
    if key <=4:
        result = int(str(value).split(".")[0])
    else:
        result = int(str(value).split(".")[0])+1
    return (result)
for dataset in combine:
    Median_age = pd.DataFrame(0.0,index = range(1,4),columns=["female","male"])
    for i in range(1,4): 
        for j in range(2): # = 0--female 
            tuple_name = (i,list(Median_age.columns.values)[j])
            median_age = dataset.groupby(["Pclass","Sex"]).get_group(tuple_name).mean()["Age"]
            Median_age[list(Median_age.columns.values)[j]][i] = median_age
    Median_age = Median_age.applymap(lambda x : hand_over(x))
    for i in range(1,4):
        for j in range(2):
            dataset.loc[(dataset["Age"].isnull()) & (dataset["Sex"]==list(Median_age.columns.values)[j])\
                        & (dataset["Pclass"]==i),"Age"] = Median_age[list(Median_age.columns.values)[j]][i]
print ("After transformation, the null values in feature 'Age'")
for dataset in combine:
    print (dataset["Age"].isnull().sum())
# Null value in Other features
for dataset in combine:
    print (dataset.isnull().sum())
## 2 null values in embarked ---training 
## 1 null value in fare ---test
# for training 
train_df["Embarked"] = train_df["Embarked"].fillna("S")
print (combine[0]["Embarked"].isnull().sum())
# for test 
sns.distplot(test_df["Fare"])
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())
print (combine[1]["Fare"].isnull().sum())
# Drop Name and Passenger ID
print ("Before Dropping, the dimensions are ", combine[0].shape, combine[1].shape)
train_df.drop(["PassengerId"],axis = 1,inplace=True)
test_df.drop(["PassengerId"],axis =1,inplace=True)
print ("after Dropping, the dimensions are ", combine[0].shape, combine[1].shape)

# generate dummy variables 
train_df = pd.get_dummies(train_df, columns=["Embarked","Title","Sex"])
test_df = pd.get_dummies(test_df, columns=["Embarked","Title","Sex"])
combine = [train_df,test_df]
# generate new variable: is alone? 
for dataset in combine:
    dataset["IsAlone"] = (dataset["SibSp"]+dataset["Parch"]==0)*1
train_df["AgeBand"] = pd.cut(train_df["Age"],5)
def apply_age_band(x):
    if x <= sorted(list(set(train_df["AgeBand"])))[0].right:
        return (sorted(list(set(train_df["AgeBand"])))[0])
    elif x > sorted(list(set(train_df["AgeBand"])))[-1].left:
        return (sorted(list(set(train_df["AgeBand"])))[-1])
    else:
        return ([ele for ele in sorted(list(set(train_df["AgeBand"]))) if x in ele][0])
combine[0][["AgeBand","Survived"]].groupby("AgeBand",as_index = False).mean().sort_values("AgeBand",ascending = True)
test_df["AgeBand"] = test_df["Age"].map(lambda x: apply_age_band(x))
age_dict = {d:i+1 for i,d in enumerate(sorted(list(set(train_df["AgeBand"]))))}
for dataset in combine:
    dataset["AgeBand_class"] = dataset["AgeBand"].map(lambda x: age_dict[x] )
train_df.head(2)
test_df.head(2)
train_lr = train_df.copy()
test_lr = test_df.copy()
train_lr.drop("AgeBand",axis = 1,inplace=True)
test_lr.drop("AgeBand",axis = 1,inplace=True)
test_df_orginal = pd.read_csv("/content/drive/My Drive/Kaggle/Titanic/test.csv")
test_lr.index = test_df_orginal["PassengerId"]
del test_df_orginal
X_train,Y_train = train_lr[list(train_lr.columns.values)[1:]], train_lr["Survived"]
X_test = test_lr.copy()
X_train.shape, Y_train.shape, X_test.shape

logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print ("Accuracy score is ",acc_log)
Results = pd.DataFrame(columns=["PassengerId","Survived"])
Results["PassengerId"] = X_test.index
Results["Survived"] = Y_pred
# import os 
# Results.to_csv(os.path.join("/content/drive/My Drive/Kaggle/Titanic/Model 2","Logstic.csv"),index = False)

warnings.filterwarnings('ignore')
sfs = SFS(LogisticRegression(),
           k_features=(1,17),
           forward=True,
           floating=False,
           scoring = 'accuracy',
           cv = 3)
sfs.fit(X_train,Y_train)

logreg2 = LogisticRegression()
logreg2.fit(X_train[list(sfs.k_feature_names_)],Y_train)
print(list(sfs.k_feature_names_))
print (round(logreg2.score(X_train[list(sfs.k_feature_names_)],Y_train)*100,2))
Y_pred = logreg2.predict(X_test[list(sfs.k_feature_names_)])
Results = pd.DataFrame(columns=["PassengerId","Survived"])
Results["PassengerId"] = X_test.index
Results["Survived"] = Y_pred
import os 
Results.to_csv(os.path.join("/content/drive/My Drive/Kaggle/Titanic/Model 2","Logstic_forwardselect.csv"),index = False)
warnings.filterwarnings('ignore')
sfs_3= SFS(LogisticRegression(),
           k_features=(1,17),
           forward=True,
           floating=False,
           scoring = 'accuracy',
           cv = 0)
sfs_3.fit(X_train,Y_train)
logreg3 = LogisticRegression()
logreg3.fit(X_train[list(sfs_3.k_feature_names_)],Y_train)
print (list(sfs_3.k_feature_names_))
print (round(logreg3.score(X_train[list(sfs_3.k_feature_names_)],Y_train)*100,2))
Y_pred = logreg3.predict(X_test[list(sfs_3.k_feature_names_)])
Results = pd.DataFrame(columns=["PassengerId","Survived"])
Results["PassengerId"] = X_test.index
Results["Survived"] = Y_pred
# import os 
Results.to_csv(os.path.join("/content/drive/My Drive/Kaggle/Titanic/Model 2","Logstic_backwardselect.csv"),index = False)
warnings.filterwarnings('ignore')
sfs_4= SFS(LogisticRegression(),
           k_features=(4,16),
           forward=True,
           floating=True,
           scoring = 'accuracy',
           cv = 3)
sfs_4.fit(X_train,Y_train)
logreg4 = LogisticRegression()
logreg4.fit(X_train[list(sfs_4.k_feature_names_)],Y_train)
X_train, Y_train = train_svc[list(train_svc.columns.values)[1:]], train_svc["Survived"]
X_test = test_svc.copy()
svc = SVC(kernel="linear")
selector = RFE(svc, step=1)
selector = selector.fit(X_train, Y_train)
Y_pred = selector.predict(X_test)
Results = pd.DataFrame(columns=["PassengerId","Survived"])
Results["PassengerId"] = X_test.index
Results["Survived"] = Y_pred
# Results.to_csv(os.path.join("/content/drive/My Drive/Kaggle/Titanic/Model 2","SVC_rfeselect.csv"),index = False)
train_svc = train_df.copy()
test_svc = test_df.copy()
train_svc.drop("AgeBand",axis = 1,inplace=True)
test_svc.drop("AgeBand",axis = 1,inplace=True)
## correlation among features --- No significant correlated features exist 
correlated_features = set()
correlated_matrix = train_svc.drop("Survived",axis=1).corr()
for i in range(correalted_matrix.shape[0]):
    for j in range(i+1,correalted_matrix.shape[0]):
        if abs(correalted_matrix.iloc[i,j]) > 0.8:
            colname = correalted_matrix.columns[i]
            correlated_features.add(colname)
print (correalted_features)
# Define train and test set 
X_train, Y_train = train_svc[list(train_svc.columns.values)[1:]], train_svc["Survived"]
X_test = test_svc.copy()
svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X_train, Y_train)
print('Optimal number of features: {}'.format(rfecv.n_features_))
Y_pred = rfecv.predict(X_test)
Results = pd.DataFrame(columns=["PassengerId","Survived"])
xx  = pd.read_csv("/content/drive/My Drive/Kaggle/Titanic/test.csv")
Results["PassengerId"] = xx.PassengerId
Results["Survived"] = Y_pred
Results.to_csv(os.path.join("/content/drive/My Drive/Kaggle/Titanic/Model 2","SVC_refcv.csv"),index = False)
train_knn = train_df.copy()
test_knn = test_df.copy()
print (list(train_knn.columns.values)[1:])
train_knn.drop("AgeBand",axis = 1,inplace=True)
test_knn.drop("AgeBand",axis = 1,inplace=True)
test_df_orginal = pd.read_csv("/content/drive/My Drive/Kaggle/Titanic/test.csv")
test_knn.index = test_df_orginal["PassengerId"]
del test_df_orginal
X_train, Y_train = train_knn[list(train_knn.columns.values)[1:]], train_knn["Survived"]
X_test = test_knn.copy()
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
# ACC_KNN = []
# for num in range(1,10):
#     knn = KNeighborsClassifier(n_neighbors = num)
#     knn.fit(X_train, Y_train)
#     Y_pred = knn.predict(X_test)
#     acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
#     ACC_KNN.append(acc_knn)
# plt.plot(range(1,10),ACC_KNN)
Y_pred = knn.predict(X_test)
Results = pd.DataFrame(columns=["PassengerId","Survived"])
Results["PassengerId"] = X_test.index
Results["Survived"] = Y_pred
Results.to_csv(os.path.join("/content/drive/My Drive/Kaggle/Titanic/Model 2","KNN.csv"),index = False)
train_rf = train_df.copy()
test_rf = test_df.copy()
print (list(train_rf.columns.values)[1:])
train_rf.drop("AgeBand",axis = 1,inplace=True)
test_rf.drop("AgeBand",axis = 1,inplace=True)
test_df_orginal = pd.read_csv("/content/drive/My Drive/Kaggle/Titanic/test.csv")
test_rf.index = test_df_orginal["PassengerId"]
del test_df_orginal
X_train, Y_train = train_rf[list(train_rf.columns.values)[1:]], train_rf["Survived"]
X_test = test_rf.copy()
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print (random_forest.score(X_train, Y_train))
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
Y_pred = random_forest.predict(X_test)
Results = pd.DataFrame(columns=["PassengerId","Survived"])
Results["PassengerId"] = X_test.index
Results["Survived"] = Y_pred
Results.to_csv(os.path.join("/content/drive/My Drive/Kaggle/Titanic/Model 2","randomforest.csv"),index = False)