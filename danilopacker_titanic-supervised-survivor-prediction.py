#Generate Map with RMS Titanic previous route and crash location 

import folium

loc = [41.726931,-49.948253]

# create a plain world map

world_map = folium.Map(location=loc, zoom_start=3,tiles = "Stamen Terrain")



folium.Marker([41.726931, -49.948253],

              popup='Crash location RMS Titanic',

              icon=folium.Icon(color='green')

             ).add_to(world_map)



# display map

world_map
import pandas as pd

import matplotlib.pylab as plt

import numpy as np

from sklearn import preprocessing

from sklearn import utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import jaccard_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn import svm

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Import Dataset

db = "../input/titanic/train.csv"

test = pd.read_csv("../input/titanic/test.csv")

df = pd.read_csv(db)

df.head()
df.describe(include="all")
print(pd.isnull(df).sum())
print(pd.isnull(test).sum())
# create new column with the travelers alone

def alone(df):

    if (df['Parch'] == 0) and (df['SibSp'] == 0):

        return 1

    else:

        return 0

    

df['Alone'] = df.apply(alone, axis=1)

test['Alone'] = test.apply(alone, axis=1)

sns.barplot(x="Alone", y="Survived", data=df)



#print percentage of people that was traveling alone

print("Percentage alone travelers who survived:", df["Survived"][df["Alone"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage not alone travelers who survived:", df["Survived"][df["Alone"] == 0].value_counts(normalize = True)[1]*100)
sns.countplot(x='Survived', hue='Sex', data = df)
sns.barplot(x="Pclass", y="Survived", data=df)
df["Age"] = df["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)





sns.barplot(x="AgeGroup", y="Survived", data=df)

plt.show
# create new column with the Family Size

df['FamilySize'] = df['Parch'] + df['SibSp']

test['FamilySize'] = test['Parch'] + test['SibSp']

sns.barplot(x="FamilySize", y="Survived", data=df)

plt.show
combine = [df, test]



for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

pd.crosstab(df['Title'], df['Sex'])
#replace various titles with more common names

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    

df.head()
#create an Age Group

df["Age"] = df["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



# Fill the missing values

mr_age = df[df["Title"] == 1]["AgeGroup"].mode() #Young Adult

miss_age = df[df["Title"] == 2]["AgeGroup"].mode() #Student

mrs_age = df[df["Title"] == 3]["AgeGroup"].mode() #Adult

master_age = df[df["Title"] == 4]["AgeGroup"].mode() #Baby

royal_age = df[df["Title"] == 5]["AgeGroup"].mode() #Adult

rare_age = df[df["Title"] == 6]["AgeGroup"].mode() #Adult



age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}



for x in range(len(df["AgeGroup"])):

    if df["AgeGroup"][x] == "Unknown":

        df["AgeGroup"][x] = age_title_mapping[df["Title"][x]]

        

for x in range(len(test["AgeGroup"])):

    if test["AgeGroup"][x] == "Unknown":

        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
#Transform each Age Group into numerical

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

df['AgeGroup'] = df['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

# Drop the age that not be usefull for now

df = df.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)
sex_mapping = {"female": 1, "male": 0 }

df['Sex'] = df['Sex'].map(sex_mapping).astype(int)

test['Sex'] = test['Sex'].map(sex_mapping).astype(int)
df.head()
df['Embarked'].value_counts().idxmax()
df["Embarked"].replace(np.nan, "S", inplace=True)

test["Embarked"].replace(np.nan, "S", inplace=True)
embarked_mapping = {"S": 1, "C": 2, "Q": 3}

df['Embarked'] = df['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)

df.head()
test["Fare"].replace(np.nan, test["Fare"].mean(), inplace=True)
plt.figure(figsize=(16,8))

sns.heatmap(df.corr(),annot=True,linewidths=1,linecolor='w')

plt.xlabel('Columns')

plt.ylabel('Columns')

plt.title('Heatmap')

plt.savefig('Heatmap.png')
df = df.drop(["SibSp"], axis = 1)

df = df.drop(["Parch"], axis = 1)

df = df.drop(['Cabin'], axis = 1)

df = df.drop(["Ticket"], axis = 1)

df = df.drop(["Name"], axis = 1)
test = test.drop(["SibSp"], axis = 1)

test = test.drop(["Parch"], axis = 1)

test = test.drop(['Cabin'], axis = 1)

test = test.drop(["Ticket"], axis = 1)

test = test.drop(["Name"], axis = 1)
print(pd.isnull(df).sum())

print(pd.isnull(test).sum())
plt.figure(figsize=(16,8))

sns.heatmap(df.corr(),annot=True,linewidths=1,linecolor='w')

plt.xlabel('Columns')

plt.ylabel('Columns')

plt.title('Heatmap')

plt.savefig('Heatmap.png')
y = df['Survived']

x = df.drop(['Survived','PassengerId'],axis=1)
x = preprocessing.StandardScaler().fit(x).transform(x)
LRlib = LogisticRegression(C=0.01, solver='liblinear').fit(x,y)

LRnew = LogisticRegression(C=0.01, solver='newton-cg').fit(x,y)

LRsag = LogisticRegression(C=0.01, solver='sag').fit(x,y)

LRsaga = LogisticRegression(C=0.01, solver='saga').fit(x,y)

LRlbf = LogisticRegression(C=0.01, solver='lbfgs').fit(x,y)
yLIB = cross_val_predict(LRlib, x, y, cv=5)

yNEW = cross_val_predict(LRnew, x, y, cv=5)

ySAG = cross_val_predict(LRsag, x, y, cv=5)

ySAGA = cross_val_predict(LRsaga, x, y, cv=5)

yLBFGS = cross_val_predict(LRlbf, x, y, cv=5)
lr_cross_val = []

lr_cross_val.append(cross_val_score(LRlib, x, y, cv=5).mean())

lr_cross_val.append(cross_val_score(LRnew, x, y, cv=5).mean())

lr_cross_val.append(cross_val_score(LRsag, x, y, cv=5).mean())

lr_cross_val.append(cross_val_score(LRsaga, x, y, cv=5).mean())

lr_cross_val.append(cross_val_score(LRlbf, x, y, cv=5).mean())
lr_f1_score = []

lr_f1_score.append(f1_score(y, yLIB, average='weighted'))

lr_f1_score.append(f1_score(y, yNEW, average='weighted'))

lr_f1_score.append(f1_score(y, ySAG, average='weighted'))

lr_f1_score.append(f1_score(y, ySAGA, average='weighted'))

lr_f1_score.append(f1_score(y, yLBFGS, average='weighted'))
lr_jaccard = []

lr_jaccard.append(jaccard_score(y, yLIB,average='weighted'))

lr_jaccard.append(jaccard_score(y, yNEW,average='weighted'))

lr_jaccard.append(jaccard_score(y, ySAG,average='weighted'))

lr_jaccard.append(jaccard_score(y, ySAGA,average='weighted'))

lr_jaccard.append(jaccard_score(y, yLBFGS,average='weighted'))
dataLR = {'Solver': ['Liblinear','Newton-cg','Sag','Saga','LBFGS'],'Cross Validation': lr_cross_val, 'F1_Score': lr_f1_score, 'Jaccard': lr_jaccard}

SVM_Results = pd.DataFrame(dataLR) 

SVM_Results.set_index('Solver', inplace = True)

SVM_Results
best_fit_cv = []

best_fit_f1 = []

best_fit_jaccard = []



best_LR = LogisticRegression(C=0.01, solver='lbfgs').fit(x,y)

y_best_LR = cross_val_predict(best_LR, x, y, cv=5)

best_fit_cv.append(cross_val_score(best_LR, x, y, cv=5).mean())

best_fit_f1.append(f1_score(y, y_best_LR, average='weighted'))

best_fit_jaccard.append(jaccard_score(y, y_best_LR,average='weighted'))
bestTree = DecisionTreeClassifier(criterion="entropy",max_depth = 4).fit(x,y)

yhat_cross_predictDT = bestTree.predict(x)
#Check f1 score, jaccard, and cross validation score



yhatScoreDT = cross_val_score(bestTree, x, y, cv = 5)

Tree_jaccard = jaccard_score(y, yhat_cross_predictDT,average='weighted')

Tree_f1_score = f1_score(y, yhat_cross_predictDT, average='weighted')



Tree_Scores = []

Tree_Scores.append(yhatScoreDT.mean())

Tree_Scores.append(Tree_f1_score)

Tree_Scores.append(Tree_jaccard)
dataTree = {'Validation': ['Cross Validation','Jaccard','F1_Score'],'Results': Tree_Scores}

Tree_Results = pd.DataFrame(dataTree) 

Tree_Results.set_index('Validation', inplace = True)

Tree_Results
best_fit_cv.append(cross_val_score(bestTree, x, y, cv = 5).mean())

best_fit_f1.append(f1_score(y, yhat_cross_predictDT, average='weighted'))

best_fit_jaccard.append(jaccard_score(y, yhat_cross_predictDT,average='weighted'))
neighBestCross = KNeighborsClassifier(n_neighbors = 5).fit(x,y)

KNN_cross_predict = neighBestCross.predict(x)
Knn_f1_score = f1_score(y, KNN_cross_predict, average='weighted')

Knn_jaccard_score = jaccard_score(y, KNN_cross_predict, average='weighted')

yhat_cross_score = cross_val_score(neighBestCross, x, y, cv= 5)



KNN_Scores = []

KNN_Scores.append(yhat_cross_score.mean())

KNN_Scores.append(Knn_jaccard_score)

KNN_Scores.append(Knn_f1_score)
dataKNN = {'Validation': ['Cross Validation','Jaccard','F1_Score'],'Results': KNN_Scores}

KNN_Results = pd.DataFrame(dataKNN) 

KNN_Results.set_index('Validation', inplace = True)

KNN_Results
best_fit_cv.append(cross_val_score(neighBestCross, x, y, cv= 5).mean())

best_fit_f1.append(f1_score(y, KNN_cross_predict, average='weighted'))

best_fit_jaccard.append(jaccard_score(y, KNN_cross_predict,average='weighted'))
rbf = svm.SVC(kernel='rbf',gamma='scale').fit(x,y)

lin = svm.SVC(kernel='linear',gamma='scale').fit(x,y)

poly = svm.SVC(kernel='poly',gamma='scale').fit(x,y)

sig = svm.SVC(kernel='sigmoid',gamma='scale').fit(x,y)
svm_rbf_cross_predict = rbf.predict(x)

svm_lin_cross_predict = rbf.predict(x)

svm_poly_cross_predict = rbf.predict(x)

svm_sig_cross_predict = rbf.predict(x)
svm_cross_val = []

svm_cross_val.append(cross_val_score(rbf, x, y, cv = 5).mean())

svm_cross_val.append(cross_val_score(lin, x, y, cv = 5).mean())

svm_cross_val.append(cross_val_score(poly, x, y, cv = 5).mean())

svm_cross_val.append(cross_val_score(sig, x, y, cv = 5).mean())
svm_f1_score = []

svm_f1_score.append(f1_score(y, svm_rbf_cross_predict, average='weighted'))

svm_f1_score.append(f1_score(y, svm_lin_cross_predict, average='weighted'))

svm_f1_score.append(f1_score(y, svm_poly_cross_predict, average='weighted'))

svm_f1_score.append(f1_score(y, svm_sig_cross_predict, average='weighted'))
svm_jaccard = []

svm_jaccard.append(jaccard_score(y, svm_rbf_cross_predict,average='weighted'))

svm_jaccard.append(jaccard_score(y, svm_lin_cross_predict,average='weighted'))

svm_jaccard.append(jaccard_score(y, svm_poly_cross_predict,average='weighted'))

svm_jaccard.append(jaccard_score(y, svm_sig_cross_predict,average='weighted'))
data = {'Kernel': ['RBF','Linear','Polynomial','Sigmoid'],'Cross Validation': svm_cross_val, 'F1_Score': svm_f1_score, 'Jaccard': svm_jaccard}

SVM_Results = pd.DataFrame(data) 

SVM_Results.set_index('Kernel', inplace = True)

SVM_Results
best_SVM = svm.SVC(kernel='rbf',gamma='scale').fit(x,y)

y_best_SVM = rbf.predict(x)

best_fit_cv.append(cross_val_score(rbf, x, y, cv = 5).mean())

best_fit_f1.append(f1_score(y, svm_rbf_cross_predict, average='weighted'))

best_fit_jaccard.append(jaccard_score(y, svm_rbf_cross_predict,average='weighted'))

BestFit = {'Algorithm': ['Logistic Regression','Decision Tree','K-Nearest Neighbors','SVM (Support Vector Machines)'],'Cross Validation': best_fit_cv, 'F1_Score': best_fit_f1, 'Jaccard': best_fit_jaccard}

Results = pd.DataFrame(BestFit) 

Results.set_index('Algorithm', inplace = True)

Results
x_test = test.drop(['PassengerId'],axis=1)

x_test = preprocessing.StandardScaler().fit(x_test).transform(x_test)
DT_test = bestTree.predict(x_test)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = DT_test



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)