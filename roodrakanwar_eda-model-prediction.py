#For data processing

import pandas as pd

import numpy as np



#For visualizations

import matplotlib.pyplot as plt

import seaborn as sns



#For ignoring warnings

import warnings

warnings.filterwarnings("ignore")
df1 = pd.read_csv("../input/titanic/train.csv")

tf1 = pd.read_csv("../input/titanic/test.csv")

result = pd.read_csv("../input/titanic/gender_submission.csv")
df1.head()
tf1.head()
df = df1.copy()

df1.describe()
tf = tf1.copy()

tf1.describe()
df.shape
tf.shape
df.dtypes
tf.dtypes
df.isnull().sum()
tf.isnull().sum()
sns.set(rc={'figure.figsize':(15,5)})

sns.heatmap(df.isnull(),yticklabels=False)
sns.heatmap(tf.isnull(),yticklabels=False)
df['Survived'].value_counts()
final = pd.concat([df,tf],axis = 0)

final.drop(['Survived'],axis = 1,inplace = True)
final.head()
final.shape
index_NaN_age = list(final["Age"][final["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = final["Age"].median()

    age_pred = final["Age"][((final['SibSp'] == final.iloc[i]["SibSp"]) & (final['Parch'] == final.iloc[i]["Parch"]) & (final['Pclass'] == final.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        final['Age'].iloc[i] = age_pred

    else :

        final['Age'].iloc[i] = age_med
final['Age'].isnull().sum()
final['Age'].fillna(final['Age'].median(),inplace = True)

final['Fare'].isnull().sum()
final["Fare"] = final["Fare"].fillna(final["Fare"].median())

final['Fare'].dtypes
final["Fare"] = final["Fare"].map(lambda n: np.log(n) if n > 0 else 0)

new = final['Name'].str.split('.', n=1, expand = True)

final['First'] = new[0]

final['Last'] = new[1]

new1 = final['First'].str.split(',', n=1, expand = True)

final['Last Name'] = new1[0]

final['Title'] = new1[1]

new2 = final['Title'].str.split('', n=1, expand = True)
final['Title'].value_counts()
final.drop(['First','Last','Name','Last Name'],axis = 1,inplace = True)
final.replace(to_replace = [ ' Don', ' Rev', ' Dr', ' Mme',

        ' Major', ' Sir', ' Col', ' Capt',' Jonkheer'], value = ' Honorary(M)', inplace = True)

final.replace(to_replace = [ ' Ms', ' Lady', ' Mlle',' the Countess', ' Dona'], value = ' Honorary(F)', inplace = True)
df3 = final.copy()

df3 =  df3[:891]

df3 = pd.concat([df3,df1['Survived']],axis = 1)

df3.head()
final['Title'].value_counts()
final = pd.get_dummies(final, columns = ["Title"])
final.head()
final["Family"] = final["SibSp"] + final["Parch"] + 1
final['Single'] = final['Family'].map(lambda s: 1 if s == 1 else 0)

final['SmallF'] = final['Family'].map(lambda s: 1 if  s == 2  else 0)

final['MedF'] = final['Family'].map(lambda s: 1 if 3 <= s <= 4 else 0)

final['LargeF'] = final['Family'].map(lambda s: 1 if s >= 5 else 0)
final['Embarked'].fillna("S",inplace = True)
final = pd.get_dummies(final, columns = ["Embarked"], prefix="Embarked_from_")
final.Cabin.isnull().sum()
final.Cabin.value_counts()
final['Cabin_final'] = df['Cabin'].str[0]
final['Cabin_final'].fillna('Unknown',inplace = True)
final['Cabin_final'].value_counts()
final.drop(['Cabin'],axis = 1,inplace = True)
final = pd.get_dummies(final, columns = ["Cabin_final"],prefix="Cabin_")
final.head()
final.Ticket.unique()
final.Ticket.value_counts()
final['Ticket'] = final['Ticket'].astype(str)

final['Ticket_length'] = final.Ticket.apply(len)

final['Ticket_length'].astype(int)

final['Ticket_length'].unique()
final['Ticket_length'] = np.where(((final.Ticket_length == 3) | (final.Ticket_length == 4) | (final.Ticket_length == 5)),4,final.Ticket_length)



final['Ticket_length'] = np.where(((final.Ticket_length == 6)),5,final.Ticket_length)



final['Ticket_length'] = np.where(((final.Ticket_length == 7) | (final.Ticket_length == 8) | (final.Ticket_length == 9) | (final.Ticket_length == 10) | (final.Ticket_length == 13)

                                 | (final.Ticket_length == 17)| (final.Ticket_length == 16)| (final.Ticket_length == 13)| (final.Ticket_length == 12) | (final.Ticket_length == 15)

                                 | (final.Ticket_length == 11)| (final.Ticket_length == 18)),12,final.Ticket_length)

final['Ticket_length'].value_counts()
final['Ticket_length'] = final['Ticket_length'].astype(str)



final['Ticket_length'] = np.where(((final.Ticket_length == '4')),'Below 6',final.Ticket_length)

final['Ticket_length'] = np.where(((final.Ticket_length == '5')),'At 6',final.Ticket_length)

final['Ticket_length'] = np.where(((final.Ticket_length == '12')),'Above 6',final.Ticket_length)

conversion = pd.get_dummies(final.Ticket_length, prefix = 'Ticket Length')

final = pd.concat([final , conversion], axis = 1)

 

final.drop(['Ticket','Ticket_length'],axis = 1, inplace = True)
final.head()
final = pd.get_dummies(final, columns = ["Sex"],prefix="Gender_")
final.head()

final.drop(['PassengerId'],axis = 1,inplace = True)

final.drop(['SibSp','Parch','Family'],axis = 1,inplace = True)
final.dtypes
final.isnull().sum()
sns.countplot(x = 'Survived', data = df1)
sns.countplot(x = 'Pclass', data = df1)
sns.countplot(x = 'Title', data = df3)
sns.countplot(x = 'Sex', data = df1)
sns.set(rc={'figure.figsize':(40,5)})

sns.countplot(x = 'Age', data = df1)
x = df1['Age']

sns.distplot(x, hist=True, rug=True)
x = df1['Fare']

sns.distplot(x, hist=True, rug=True)
x = final['Fare']

sns.distplot(x, hist=True, rug=True)
sns.set(rc={'figure.figsize':(15,5)})

sns.countplot(x = 'SibSp', data = df1)
sns.countplot(x = 'Parch', data = df1)
sns.countplot(x = 'Embarked', data = df1)
sns.catplot(x ='Survived', y ='Age', data = df1)
sns.catplot(x ='Survived', y ='SibSp', data = df1)
sns.catplot(x ='Survived', y ='Parch', data = df1)
sns.catplot(x = 'Sex',y='Survived',hue = 'Pclass', kind = 'bar', data = df1, col = 'Pclass', color = 'purple')
sns.catplot(x = 'Title',y='Survived',hue = 'Sex', kind = 'bar', data = df3, col = 'Sex', palette = 'GnBu_d',aspect =2)
sns.catplot(x = 'SibSp',y='Survived',hue = 'Pclass',kind = 'violin', data = df3, palette = 'BuGn_r', col = 'Pclass')
sns.catplot(x = 'Parch',y='Survived',hue = 'Pclass',kind = 'violin', data = df3, palette = 'cubehelix', col = 'Pclass')
sns.catplot(x = 'Embarked',y='Survived',kind = 'point', data = df3, hue = 'Pclass', col = 'Pclass')
sns.jointplot(x=df1['Age'], y=df1['SibSp'], kind = 'kde')
correlation = final.copy()

sur = pd.concat([df['Survived'],result['Survived']],axis = 0)

correlation = pd.concat([correlation,sur],axis = 1)
plt.figure(figsize=(30,30))

sns.heatmap(correlation.corr(), annot=True, linewidth=0.5, cmap='coolwarm')
#The models trained

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.naive_bayes import BernoulliNB



#For Scaling and Hyperparameter Tuning

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn import metrics



#Voting Classifier

from sklearn.ensemble import VotingClassifier 
x_train = final[:891]

feature_scaler = MinMaxScaler()

x_train = feature_scaler.fit_transform(x_train)
y_train = final[891:]

feature_scaler = MinMaxScaler()

y_train = feature_scaler.fit_transform(y_train)
x_test = df1['Survived']
y_test = result['Survived']
accuracy = []

estimator = []
LR = LogisticRegression()

estimator.append(('LR',LogisticRegression()))

cv = cross_val_score(LR,x_train,x_test,cv=10)

accuracy1 = cv.mean()

accuracy.append(accuracy1)

print(cv)

print(cv.mean())
LR.fit(x_train,x_test)

model1pred = LR.predict(y_train)

submission1 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission1['PassengerId'] = result['PassengerId']

submission1['Survived'] = model1pred

submission1.to_csv('LogisticRegression(No HT).csv',index = False)
LR.score(y_train,y_test)
SVC = LinearSVC()

#estimator.append(('LSVC',LinearSVC()))

cv = cross_val_score(SVC,x_train,x_test,cv=10)

accuracy2 = cv.mean()

accuracy.append(accuracy2)

print(cv)

print(cv.mean())

SVC.fit(x_train,x_test)

SVC.score(y_train,y_test)
SVC.fit(x_train,x_test)

model2pred = SVC.predict(y_train)

submission2 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission2['PassengerId'] = result['PassengerId']

submission2['Survived'] = model2pred

submission2.to_csv('LinearSVC(No HT).csv',index = False)
poly = svm.SVC(kernel = 'poly', gamma = 'scale')

#estimator.append(('PSVC',svm.SVC(kernel = 'poly', gamma = 'scale')))

cv = cross_val_score(poly,x_train,x_test,cv=10)

accuracy3 = cv.mean()

accuracy.append(accuracy3)

print(cv)

print(cv.mean())
poly.fit(x_train,x_test)

poly.score(y_train,y_test)
model3pred = poly.predict(y_train)

submission3 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission3['PassengerId'] = result['PassengerId']

submission3['Survived'] = model3pred

submission3.to_csv('PolynomialSVC(No HT).csv',index = False)
DT = DecisionTreeClassifier(random_state = 5)

estimator.append(('DT',DecisionTreeClassifier(random_state = 5)))

cv = cross_val_score(DT,x_train,x_test,cv=10)

accuracy4 = cv.mean()

accuracy.append(accuracy4)

print(cv)

print(cv.mean())
DT.fit(x_train,x_test)

DT.score(y_train,y_test)
model4pred = DT.predict(y_train)

submission4 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission4['PassengerId'] = result['PassengerId']

submission4['Survived'] = model4pred

submission4.to_csv('Decision Tree(No HT).csv',index = False)
GNB = GaussianNB()

estimator.append(('GNB',GaussianNB()))

cv = cross_val_score(GNB,x_train,x_test,cv=10)

accuracy5 = cv.mean()

accuracy.append(accuracy5)

print(cv)

print(cv.mean())
GNB.fit(x_train,x_test)

GNB.score(y_train,y_test)
model5pred = GNB.predict(y_train)

submission5 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission5['PassengerId'] = result['PassengerId']

submission5['Survived'] = model5pred

submission5.to_csv('Gaussian NB(No HT).csv',index = False)
MNB = MultinomialNB()

estimator.append(('MNB',MultinomialNB()))

cv = cross_val_score(MNB,x_train,x_test,cv=10)

accuracy6 = cv.mean()

accuracy.append(accuracy6)

print(cv)

print(cv.mean())
MNB.fit(x_train,x_test)

MNB.score(y_train,y_test)
MNB.fit(x_train,x_test)

model6pred = MNB.predict(y_train)

submission6 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission6['PassengerId'] = result['PassengerId']

submission6['Survived'] = model6pred

submission6.to_csv('MultinomialNB(No HT).csv',index = False)
RF = RandomForestClassifier(random_state = 5)

estimator.append(('RF',RandomForestClassifier(random_state = 5)))

cv = cross_val_score(RF,x_train,x_test,cv=10)

accuracy7 = cv.mean()

accuracy.append(accuracy7)

print(cv)

print(cv.mean())
RF.fit(x_train,x_test)

RF.score(y_train,y_test)
RF.fit(x_train,x_test)

model7pred = RF.predict(y_train)

submission7 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission7['PassengerId'] = result['PassengerId']

submission7['Survived'] = model7pred

submission7.to_csv('RandomForest(No HT).csv',index = False)
GBC = GradientBoostingClassifier(random_state = 5)

estimator.append(('GBC',GradientBoostingClassifier(random_state = 5)))

cv = cross_val_score(GBC,x_train,x_test,cv=10)

accuracy8 = cv.mean()

accuracy.append(accuracy8)

print(cv)

print(cv.mean())
GBC.fit(x_train,x_test)

GBC.score(y_train,y_test)
GBC.fit(x_train,x_test)

model8pred = GBC.predict(y_train)

submission8 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission8['PassengerId'] = result['PassengerId']

submission8['Survived'] = model8pred

submission8.to_csv('GradientBoosting(No HT).csv',index = False)
XGB = XGBClassifier(random_state = 5)

estimator.append(('XGB', XGBClassifier(random_state = 5)))

cv = cross_val_score(XGB,x_train,x_test,cv=10)

accuracy9 = cv.mean()

accuracy.append(accuracy9)

print(cv)

print(cv.mean())
XGB.fit(x_train,x_test)

XGB.score(y_train,y_test)
XGB.fit(x_train,x_test)

model9pred = XGB.predict(y_train)

submission9 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission9['PassengerId'] = result['PassengerId']

submission9['Survived'] = model9pred

submission9.to_csv('XGBoosting(No HT).csv',index = False)
Krange = range(1,20)

scores = {}

scores_list = []

for k in Krange:

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(x_train,x_test)

    y_pred = knn.predict(y_train)

    scores[k] = metrics.accuracy_score(result['Survived'],y_pred)

    scores_list.append(metrics.accuracy_score(result['Survived'],y_pred))

    

plt.plot(Krange,scores_list)

plt.xlabel("Value of K")

plt.ylabel("Accuracy")
KNN = KNeighborsClassifier(n_neighbors = 11)

estimator.append(('KNN',KNeighborsClassifier(n_neighbors = 11)))

cv = cross_val_score(KNN,x_train,x_test,cv=10)

accuracy10 = cv.mean()

accuracy.append(accuracy10)

print(cv)

print(cv.mean())
KNN.fit(x_train,x_test)

KNN.score(y_train,y_test)
KNN.fit(x_train,x_test)

model10pred = KNN.predict(y_train)

submission10 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission10['PassengerId'] = result['PassengerId']

submission10['Survived'] = model10pred

submission10.to_csv('KNN(No HT).csv',index = False)
Models = ['Logistic Regression','Linear SVM','Polynomial SVM','Decision Tree','Gaussian NB','Multinomial NB','Random Forest Classifier','Gradient Boost Classifier','XG Boosting','K-Nearest Neighbors']

total = list(zip(Models,accuracy))

output1 = pd.DataFrame(total, columns = ['Models','Accuracy'])

o = output1.groupby(['Models'])['Accuracy'].mean().reset_index().sort_values(by='Accuracy',ascending=False)

o.head(10).style.background_gradient(cmap='Reds')

vot_soft = VotingClassifier(estimators = estimator, voting ='soft') 

vot_soft.fit(x_train, x_test) 

y_pred = vot_soft.predict(y_train)

vot_soft.score(y_train,y_test)
modelpred1 = vot_soft.predict(y_train)

sub1 = pd.DataFrame(columns = ['PassengerId','Survived'])

sub1['PassengerId'] = result['PassengerId']

sub1['Survived'] = modelpred1

sub1.to_csv('SoftVoting(NO HT).csv',index = False)
vot_hard = VotingClassifier(estimators = estimator, voting ='hard') 

vot_hard.fit(x_train, x_test) 

y_pred = vot_hard.predict(y_train)

vot_hard.score(y_train,y_test)
modelpred2 = vot_hard.predict(y_train)

sub2 = pd.DataFrame(columns = ['PassengerId','Survived'])

sub2['PassengerId'] = result['PassengerId']

sub2['Survived'] = modelpred2

sub2.to_csv('HardVoting(NO HT).csv',index = False)
Accuracy = []

Estimator = []
"""

C = [0.01,0.1, 1, 10,50, 100]

penalty = ['l2']

solver = ['newton-cg','lbfgs','liblinear']

class_weight = ['dict','balanced','None']

max_iter = [900,1000,1100,1200]



Log = LogisticRegression()



parameters = {'C': [0.01,0.1, 1, 10,50, 100],'penalty' : ['l2'],'solver' : ['newton-cg','lbfgs','liblinear'],'class_weight':['dict','balanced','None'],'max_iter':[900,1000,1100,1200]}



log_regressor = GridSearchCV(Log, parameters, scoring='accuracy',cv =10)

log_regressor.fit(x_train, x_test)

log_regressor.best_params_

"""
#log_regressor.best_score_
lr = LogisticRegression(C = 100,penalty = 'l2', solver = 'newton-cg',class_weight = 'dict', max_iter = 900)

Estimator.append(('lr',LogisticRegression(C = 1,penalty = 'l2', solver = 'newton-cg',class_weight = 'dict', max_iter = 900)))

cv = cross_val_score(lr,x_train,x_test,cv=10)

Accuracy1 = cv.mean()

Accuracy.append(Accuracy1)

print(cv)

print(cv.mean())
lr.fit(x_train,x_test)

lr.score(y_train,y_test)
model11pred = lr.predict(y_train)

submission11 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission11['PassengerId'] = result['PassengerId']

submission11['Survived'] = model11pred

submission11.to_csv('LogisticRegression(HT).csv',index = False)
'''

penalty = ['l1','l2']

loss = ['hinge','squared_hinge']

class_weight = ['dict','balanced','None']

C = [.1,1,10,50,100,150]



SVM = LinearSVC()



parameters = {'penalty':['l1','l2'],'loss':['hinge','squared_hinge'],'class_weight':['dict','balanced','None'] ,'C': [.1,1,10,50,100,150]}



SVM_classifier = GridSearchCV(SVM, parameters, scoring='accuracy' ,cv =10)

SVM_classifier.fit(x_train, x_test)

SVM_classifier.best_params_

'''
#SVM_classifier.best_score_
svc = LinearSVC(C = 0.1,penalty = 'l2', loss = 'hinge',class_weight = 'balanced')

cv = cross_val_score(svc,x_train,x_test,cv=10)

Accuracy2 = cv.mean()

Accuracy.append(Accuracy2)

print(cv)

print(cv.mean())
svc.fit(x_train,x_test)

svc.score(y_train,y_test)
model12pred = svc.predict(y_train)

submission12 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission12['PassengerId'] = result['PassengerId']

submission12['Survived'] = model12pred

submission12.to_csv('SVCLinear(HT).csv',index = False)
'''

kernel = ['poly']

degree = [1,2,3]

class_weight = ['balanced','dict']

C = [.1,1,10,]

gamma = ['scale','auto']



s = svm.SVC()



parameters = {'kernel':['poly'],'class_weight':['balanced','dict'] ,'C': [.1,1,10],'degree':[1,2,3],'gamma':['scale','auto']}



svcc = GridSearchCV(s, parameters, scoring='accuracy' ,cv =10)

svcc.fit(x_train, x_test)

svcc.best_params_

'''
#svcc.best_score_
SVM_all = svm.SVC(C = 1,degree = 2, kernel = 'poly',class_weight = 'balanced',gamma = 'scale')

cv = cross_val_score(svc,x_train,x_test,cv=10)

Accuracy3 = cv.mean()

Accuracy.append(Accuracy3)

print(cv)

print(cv.mean())
SVM_all.fit(x_train,x_test)

SVM_all.score(y_train,y_test)
model13pred = SVM_all.predict(y_train)

submission13 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission13['PassengerId'] = result['PassengerId']

submission13['Survived'] = model13pred

submission13.to_csv('PolynomialSVM(HT).csv',index = False)
'''

criterion = ['gini','entropy']

splitter = ['best','random']

max_depth = [5,10,15,20,25]

min_samples_split = [2,3,4,5]

class_weight = ['dict','balanced','None']

random_state = [5,6]





Tree = DecisionTreeClassifier()



parameters = {'criterion': ['gini','entropy'],'splitter': ['best','random'], 'max_depth':[5,10,15,20,25],'min_samples_split':[2,3,4,5],'class_weight':['dict','balanced','None'],'random_state':[5,6]}



tree_classifier = GridSearchCV(Tree, parameters, scoring='accuracy' ,cv = 10)

tree_classifier.fit(x_train, x_test)

tree_classifier.best_params_

'''
#tree_classifier.best_score_

dt = DecisionTreeClassifier(class_weight = 'balanced',criterion = 'entropy',max_depth = 5,min_samples_split = 2,splitter = 'best',random_state = 6)

Estimator.append(('dt',DecisionTreeClassifier(class_weight = 'balanced',criterion = 'entropy',max_depth = 5,min_samples_split = 2,splitter = 'best',random_state = 6)))

cv = cross_val_score(dt,x_train,x_test,cv=10)

Accuracy4 = cv.mean()

Accuracy.append(Accuracy4)

print(cv)

print(cv.mean())
dt.fit(x_train,x_test)

dt.score(y_train,y_test)
model14pred = SVM_all.predict(y_train)

submission14 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission14['PassengerId'] = result['PassengerId']

submission14['Survived'] = model14pred

submission14.to_csv('DecisionTrees(HT).csv',index = False)
'''

alpha = [0.01,0.1, 1, 10, 100]

fit_prior = [True,False]



mnb = MultinomialNB()



parameters = {'alpha': [0.01,0.1, 1, 10, 100],'fit_prior' : [True,False]}



mn = GridSearchCV(mnb, parameters, scoring='accuracy',cv =10)

mn.fit(x_train, x_test)

mn.best_params_

'''
#mn.best_score_
mnb = MultinomialNB(alpha = 1,fit_prior = True)

Estimator.append(('mnb',MultinomialNB(alpha = 1,fit_prior = True)))

cv = cross_val_score(mnb,x_train,x_test,cv=10)

Accuracy5 = cv.mean()

Accuracy.append(Accuracy5)

print(cv)

print(cv.mean())
mnb.fit(x_train,x_test)

mnb.score(y_train,y_test)
model15pred = mnb.predict(y_train)

submission15 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission15['PassengerId'] = result['PassengerId']

submission15['Survived'] = model15pred

submission15.to_csv('MultinomialNB(HT).csv',index = False)
'''

n_estimators = [250,500,750,1000]

criterion = ['gini','entropy']

max_depth = [5,10,15,20,25]

min_samples_split = [2,3,4,5]

bootstrap = [True,False]

oob_score = [True,False]

class_weight = ['balanced','balanced_subsample','dict']

max_features = ['auto','sqrt','log2']



RF = RandomForestClassifier()



parameters = {'n_estimators': [250,500,750,1000],'criterion': ['gini','entropy'],'max_depth':[5,10,15,20,25],'min_samples_split':[2,3,4,5],'bootstrap':[True,False]

              ,'oob_score':[True,False],'class_weight':['balanced','balanced_subsample','dict'],'max_features':['auto','sqrt','log2']}



RFClassifier = RandomizedSearchCV(RF, parameters, scoring='accuracy' ,cv =50)

RFClassifier.fit(x_train, x_test)

RFClassifier.best_params_

'''
#RFClassifier.best_score_
"""

n_estimators = [650,700,750,800,850]

criterion = ['gini']

max_depth = [4,5]

min_samples_split = [5,6]

bootstrap = [False,True]

oob_score = [False,True]

class_weight = ['balanced_subsample']

max_features = ['log2']



rF = RandomForestClassifier()



parameters = {'n_estimators': [650,700,750,800,850],'criterion': ['gini'],'max_depth':[5,6],'min_samples_split':[4,5],'bootstrap':[False,True]

              ,'oob_score':[False,True],'class_weight':['balanced_subsample'],'max_features':['log2']}



RClassifier = GridSearchCV(rF, parameters, scoring='accuracy' ,cv =5)

RClassifier.fit(x_train,x_test)

RClassifier.best_params_

"""
#RClassifier.best_score_
rf = RandomForestClassifier(oob_score = True,n_estimators =650 ,min_samples_split = 4,max_features = 'log2',max_depth =6,criterion = 'gini',class_weight = 'balanced_subsample',bootstrap = True)

Estimator.append(('rf',RandomForestClassifier(oob_score = True,n_estimators =650 ,min_samples_split = 4,max_features = 'log2',max_depth =6,criterion = 'gini',class_weight = 'balanced_subsample',bootstrap = True)))

cv = cross_val_score(rf,x_train,x_test,cv=10)

Accuracy6 = cv.mean()

Accuracy.append(Accuracy6)

print(cv)

print(cv.mean())
rf.fit(x_train,x_test)

rf.score(y_train,y_test)
model16pred = rf.predict(y_train)

submission16 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission16['PassengerId'] = result['PassengerId']

submission16['Survived'] = model16pred

submission16.to_csv('RandomForest(HT).csv',index = False)
'''

n_estimators = [250,500,750,1000]

learning_rate = [.01,.1,1,5]

subsample = [.01,.1,1,5]

min_samples_split = [2,3,4,5]

max_depth = [5,10,15,20,25]

loss = ['deviance','exponential']

max_features = ['auto','sqrt','log2']



GB = GradientBoostingClassifier()



parameters = {'n_estimators': [250,500,750,1000],'loss': ['deviance','exponential'],'max_features':['auto','sqrt','log2'],'learning_rate':[.01,.1,1,5],'subsample':[.01,.1,1,5],

             'min_samples_split':[2,3,4,5],'max_depth':[5,10,15,20,25]}



GBClassifier = RandomizedSearchCV(GB, parameters, scoring='accuracy' ,cv =50)

GBClassifier.fit(x_train, x_test)

GBClassifier.best_params_

'''
#GBClassifier.best_score_
'''

n_estimators = [150,200,250,300,350]

learning_rate = [.01,.1]

subsample = [.05,.1]

min_samples_split = [3,4,5]

max_depth = [9,10,11]

loss = ['exponential']

max_features = ['auto']



GB = GradientBoostingClassifier()



parameters = {'n_estimators': [150,200,250,300,350],'loss': ['exponential'],'max_features':['auto'],'learning_rate':[.01,.1],'subsample':[.05,.1],

             'min_samples_split':[3,4,5],'max_depth':[9,10,11]}



GBClassifier = GridSearchCV(GB, parameters, scoring='accuracy' ,cv =5)

GBClassifier.fit(x_train, x_test)

GBClassifier.best_params_

'''
#GBClassifier.best_score_
gbc = GradientBoostingClassifier(loss = 'exponential',n_estimators =200 ,min_samples_split = 4,max_features = 'auto',max_depth =9,learning_rate = .01,subsample = .1)

Estimator.append(('gbc',GradientBoostingClassifier(loss = 'exponential',n_estimators =200 ,min_samples_split = 4,max_features = 'auto',max_depth =9,learning_rate = .01,subsample = .1)))

cv = cross_val_score(gbc,x_train,x_test,cv=10)

Accuracy7 = cv.mean()

Accuracy.append(Accuracy7)

print(cv)





print(cv.mean())
gbc.fit(x_train,x_test)

gbc.score(y_train,y_test)
model17pred = gbc.predict(y_train)

submission17 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission17['PassengerId'] = result['PassengerId']

submission17['Survived'] = model17pred

submission17.to_csv('GradientBoosting(HT).csv',index = False)
'''

min_child_weight = [1,5,10]

gamma = [.5,1,1.5,2,2.5]

subsample = [.6,.8,1]

colsample_bytree = [.6,.8,1]

eta = [.01,.05,.1,.5,.2]

max_depth = [3,4,5,6,7,8,9,10]



XB = XGBClassifier()



parameters = {'min_child_weight': [1,5,10],'gamma': [.5,1,1.5,2,2.5],'subsample':[.6,.8,1],'colsample_bytree':[.6,.8,1],'subsample':[.6,.8,1],

             'eta':[.01,.05,.1,.5,.2],'max_depth':[3,4,5,6,7,8,9,10]}



XBClassifier = RandomizedSearchCV(XB, parameters, scoring='accuracy' ,cv =50)

XBClassifier.fit(x_train, x_test)

XBClassifier.best_params_

'''
#XBClassifier.best_score_
'''

min_child_weight = [4,5,6]

gamma = [1,1.5,2.0,2.5,3]

subsample = [.6,.8,1,1.2]

colsample_bytree = [.6,.8,1,1.2]

eta = [.5,.01]



max_depth = [5,6,7,8]



XB = XGBClassifier()



parameters = {'min_child_weight': [4,5,6],'gamma': [1,1.5,2.0,2.5,3],'subsample':[.6,.8,1,1.2],'colsample_bytree':[.6,.8,1,1.2],

             'eta':[.5,.01],'max_depth':[5,6,7,8]}



XBClassifier = GridSearchCV(XB, parameters, scoring='accuracy' ,cv =5)

XBClassifier.fit(x_train, x_test)

XBClassifier.best_params_

'''
#XBClassifier.best_score_
xgb = XGBClassifier(colsample_bytree = .6,eta = 0.5,gamma = 1,max_depth = 5,min_child_weight = 6,subsample = 1)

Estimator.append(('xgb',XGBClassifier(colsample_bytree = .6,eta = 0.5,gamma = 1,max_depth = 5,min_child_weight = 6,subsample = 1)))

cv = cross_val_score(xgb,x_train,x_test,cv=10)

Accuracy8 = cv.mean()

Accuracy.append(Accuracy8)

print(cv)

print(cv.mean())
xgb.fit(x_train,x_test)

gbc.score(y_train,y_test)
model18pred = xgb.predict(y_train)

submission18 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission18['PassengerId'] = result['PassengerId']

submission18['Survived'] = model18pred

submission18.to_csv('XGBoosting(HT).csv',index = False)
x_train1 = final[:891]

feature_scaler = StandardScaler()

x_train1 = feature_scaler.fit_transform(x_train1)

y_train1 = final[891:]

feature_scaler = StandardScaler()

y_train1 = feature_scaler.fit_transform(y_train1)
Krange1 = range(1,20)

scores1 = {}

scores_list1 = []

for k in Krange1:

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(x_train1,x_test)

    y_pred = knn.predict(y_train1)

    scores1[k] = metrics.accuracy_score(result['Survived'],y_pred)

    scores_list1.append(metrics.accuracy_score(result['Survived'],y_pred))

    

plt.plot(Krange,scores_list)

plt.xlabel("Value of K")

plt.ylabel("Accuracy")
knn = KNeighborsClassifier(n_neighbors = 11)

Estimator.append(('knn',KNeighborsClassifier(n_neighbors = 13)))

cv = cross_val_score(knn,x_train1,x_test,cv=10)

Accuracy9 = cv.mean()

Accuracy.append(Accuracy9)

print(cv)

print(cv.mean())
knn.fit(x_train1,x_test)

knn.score(y_train1,y_test)
model19pred = knn.predict(y_train)

submission19 = pd.DataFrame(columns = ['PassengerId','Survived'])

submission19['PassengerId'] = result['PassengerId']

submission19['Survived'] = model19pred

submission19.to_csv('KNN(StdScaler).csv',index = False)
models = ['Logistic Regression','SVM Linear Classifier','SVM Polynomial Classifier','Decision Tree','Multinomial NB','Random Forest Classifier','Gradient Boost Classifier','XG Boosting','K-Nearest Neighbors(StdScaler)']

total = list(zip(models,Accuracy))

output2 = pd.DataFrame(total, columns = ['Models after Hyperparameter Tuning','Accuracy after HT'])
r = output2.groupby(['Models after Hyperparameter Tuning'])['Accuracy after HT'].mean().reset_index().sort_values(by='Accuracy after HT',ascending=False)

r.head(10).style.background_gradient(cmap='Reds')

vot_soft1 = VotingClassifier(estimators = Estimator, voting ='soft') 

vot_soft1.fit(x_train, x_test) 

y_pred = vot_soft1.predict(y_train)

vot_soft1.score(y_train,y_test)



modelpred3 = vot_soft1.predict(y_train)

sub3 = pd.DataFrame(columns = ['PassengerId','Survived'])

sub3['PassengerId'] = result['PassengerId']

sub3['Survived'] = modelpred3

sub3.to_csv('SoftVoting(HT).csv',index = False)
vot_soft1.fit(x_train, x_test) 

vot_soft1.score(y_train,y_test)
vot_hard1 = VotingClassifier(estimators = Estimator, voting ='hard') 

vot_hard1.fit(x_train, x_test) 

y_pred = vot_hard1.predict(y_train)

vot_hard1.score(y_train,y_test)



modelpred4 = vot_hard1.predict(y_train)

sub4 = pd.DataFrame(columns = ['PassengerId','Survived'])

sub4['PassengerId'] = result['PassengerId']

sub4['Survived'] = modelpred4

sub4.to_csv('HardVoting(HT).csv',index = False)
vot_hard1.fit(x_train, x_test) 

vot_hard1.score(y_train,y_test)
output = pd.concat([output1,output2],axis = 1)

output.sort_values(by=['Accuracy after HT'], inplace=True, ascending=False)

output.head(10)