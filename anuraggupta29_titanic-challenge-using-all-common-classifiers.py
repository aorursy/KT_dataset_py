#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as r
#function to generate random colors
def randomColor(n):
    color = []
    colorArr = ['00','11','22','33','44','55','66','77','88','99','AA','BB','CC','DD','EE','FF']
    for _ in range(n):
        color.append('#' + colorArr[r.randint(0,15)] + colorArr[r.randint(0,15)] + colorArr[r.randint(0,15)])
    return color
#get datasets
traindf = pd.read_csv('/kaggle/input/titanic/train.csv')
display(traindf.head())
print(traindf.shape)
testdf = pd.read_csv('/kaggle/input/titanic/test.csv')
display(testdf.head())
print(testdf.shape)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize = (18,9))

#Gender vs Survuval Ratio
gender_survived = traindf[['Sex', 'Survived']].groupby('Sex').mean()
gender_survived.plot(kind='barh', 
                     color = [randomColor(len(gender_survived))], 
                     ax = axes[0,0],
                     title = 'Gender vs Survival Ratio',
                     xlim = (0,1),
                     grid = True,)

#Pclass vs Survival Ratio
pclass_survived = traindf[['Pclass', 'Survived']].groupby('Pclass').mean()
pclass_survived.plot(kind='bar', 
                     color = [randomColor(len(pclass_survived))], 
                     ax=axes[0,1],
                     title = 'Pclass vs Survival Ratio',
                     ylim = (0,1),
                     grid = True,)

#Embarked vs Survival Ratio
embarked_survived = traindf[['Embarked', 'Survived']].groupby('Embarked').mean()
embarked_survived.plot(kind='bar', 
                       color = [randomColor(len(embarked_survived))], 
                       ax=axes[0,2],
                       title = 'Embarked vs Survival Ratio',
                       ylim = (0,1),
                       grid = True,)

#Parch vs Survival Ratio
parch_survived = traindf[['Parch', 'Survived']].groupby('Parch').mean()
parch_survived.plot(kind='bar', 
                    color = [randomColor(len(parch_survived))], 
                    ax=axes[1,0], 
                    title = 'Parch vs Survival Ratio',
                    ylim = (0,1),
                    grid = True,)

#SibSp vs Survival Ratio
sibsp_survived = traindf[['SibSp', 'Survived']].groupby('SibSp').mean()
sibsp_survived.plot(kind='bar', 
                    color = [randomColor(len(sibsp_survived))], 
                    ax=axes[1,1],
                    title = 'Parch vs Survival Ratio',
                    ylim = (0,1),
                    grid = True,)

#Age Group vs Survival Ratio
agegrp = {'child':(0,13), 'teen':(13,20), 'young_adult':(20,35), 'middle_adult':(35,45), 'old_adult':(45,60), 'senior_citizen':(60,100)}
age_survival = traindf[['Age','Survived']].dropna().reset_index(drop=True)
age_survival['age_grp'] = None

for i in range(len(age_survival)):
    for grp in agegrp:
        temp = agegrp[grp]
        if age_survival.loc[i,'Age'] in range(temp[0],temp[1]):
            age_survival.loc[i,'age_grp'] = grp
            break
            
age_survival = age_survival.drop(columns=['Age']).groupby('age_grp').mean()

age_survival.plot(kind= 'barh', 
                  color = [randomColor(6)], 
                  legend=False, ax=axes[1,2],
                  title = 'Age group vs Survival Ratio',
                  xlim = (0,1),
                  grid = True,)

fig.tight_layout()
fig.show()
traindf2 = traindf[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
missingdf = traindf2.transpose()
missingdf['missing values'] = missingdf.apply(lambda x: len(traindf)-x.count(), axis=1)
missingdf = missingdf[['missing values']]
missingdf
traindf2.drop(columns=['Age'], inplace=True)
traindf2['Embarked'].fillna(traindf2['Embarked'].mode()[0], inplace=True)
display(traindf2.head())
y = traindf2["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]
X = pd.concat([pd.get_dummies(traindf2[features[0]]),pd.get_dummies(traindf2[features[1:]])], axis = 1, sort = False)
display(X.head())
print(X.shape)
#All values have been converted to categorical variables
#Normalizing data not required
#X = preprocessing.StandardScaler.fit(X).transform(X)
#importing ML packages
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 5)
kval = [i for i in range(1,11)]
knnArr = []

for k in kval:
    knnmodel = KNeighborsClassifier(n_neighbors = k)
    knnmodel.fit(xtrain,ytrain)
    knnArr.append(round(accuracy_score(ytest, knnmodel.predict(xtest))*100, 4))
    
knndf = pd.DataFrame({'kval':kval, 'accuracy':knnArr}).set_index('kval')
knndf.plot(kind='line', ylim=(75,85))
plt.title('KNN Analysis')
plt.xlabel('K value')
plt.ylabel('Accuracy')
print('Max KNN Accuracy :',max(knnArr))
degreeval = [i for i in range(1,7)]
svmPolyArr = []
for d in degreeval:
    svmmodel = svm.SVC(kernel='poly', degree=d)
    svmmodel.fit(xtrain, ytrain)
    svmPolyArr.append(round(accuracy_score(ytest, svmmodel.predict(xtest))*100, 4))
    
svmpolydf = pd.DataFrame({'degreeval':degreeval, 'accuracy':svmPolyArr}).set_index('degreeval')
svmpolydf.plot(kind='line', ylim=(75,85))
plt.title('SVM Analysis')
plt.xlabel('Poly-Degree value')
plt.ylabel('Accuracy')
print('Max SVM Poly Accuracy :', max(svmPolyArr))
gammaval = [0.0001, 0.001, 0.01, 0.1, 1, 10]
svmrbfArr = []
for g in gammaval:
    svmmodel2 = svm.SVC(kernel='rbf', gamma = g)
    svmmodel2.fit(xtrain, ytrain)
    svmrbfArr.append(round(accuracy_score(ytest, svmmodel2.predict(xtest))*100, 4))

svmrbfdf = pd.DataFrame({'gammaval':gammaval, 'accuracy':svmrbfArr}).set_index('gammaval')
svmrbfdf.plot(kind='line', ylim=(60,90), logx = True)
plt.title('SVM Analysis')
plt.xlabel('rbf gamma value')
plt.ylabel('Accuracy')
print('Max SVM rbf Accuracy :', max(svmrbfArr))
depthval = [i for i in range(1, 11)]
dtArr = []

for d in depthval:
    dtmodel = DecisionTreeClassifier(criterion="entropy", max_depth = d)
    dtmodel.fit(xtrain, ytrain)
    dtArr.append(round(accuracy_score(ytest, dtmodel.predict(xtest))*100, 4))
    
dtdf = pd.DataFrame({'depthval':depthval, 'accuracy':dtArr}).set_index('depthval')
dtdf.plot(kind='line', ylim=(77,85))
plt.title('Decision Tree Analysis')
plt.xlabel('Max Depth value')
plt.ylabel('Accuracy')
print('Max Decison Tree Accuracy :', max(dtArr))
cval = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
lrArr = []

for c in cval:
    lrmodel = LogisticRegression(solver="liblinear", C = c)
    lrmodel.fit(xtrain, ytrain)
    lrArr.append(round(accuracy_score(ytest, lrmodel.predict(xtest))*100, 4))
    
lrdf = pd.DataFrame({'cval':cval, 'accuracy':lrArr}).set_index('cval')
lrdf.plot(kind='line', ylim=(60,90), logx = True)
plt.title('Logistic Regression Analysis')
plt.xlabel('C value')
plt.ylabel('Accuracy')
print('Max Logistic Regression Accuracy :', max(lrArr))
treesval = [5,10,50,100,500]
rfArr = []

for t in treesval:
    rfmodel = RandomForestClassifier(n_estimators=t, max_depth=3) #increasing max depth was not producing any better results.
    rfmodel.fit(xtrain, ytrain)
    rfArr.append(round(accuracy_score(ytest, rfmodel.predict(xtest))*100, 4))
    
rfdf = pd.DataFrame({'depthval':treesval, 'accuracy':rfArr}).set_index('depthval')
rfdf.plot(kind='line', ylim=(75,85), logx = True)
plt.title('Random Forest Analysis')
plt.xlabel('Trees count')
plt.ylabel('Accuracy')
print('Max Random Forest Accuracy :', max(rfArr))
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, CategoricalNB, BernoulliNB
nbclass = ['GaussianNB', 'MultinomialNB', 'ComplementNB', 'CategoricalNB', 'BernoulliNB']
nbArr = []
gnb = BernoulliNB()
mnb = MultinomialNB()
compnb = ComplementNB()
catnb = CategoricalNB()
bnb = BernoulliNB()

nbArr.append(round(accuracy_score(ytest, gnb.fit(xtrain, ytrain).predict(xtest))*100,4))
nbArr.append(round(accuracy_score(ytest, mnb.fit(xtrain, ytrain).predict(xtest))*100,4))
nbArr.append(round(accuracy_score(ytest, compnb.fit(xtrain, ytrain).predict(xtest))*100,4))
nbArr.append(round(accuracy_score(ytest, catnb.fit(xtrain, ytrain).predict(xtest))*100,4))
nbArr.append(round(accuracy_score(ytest, bnb.fit(xtrain, ytrain).predict(xtest))*100,4))

nbdf = pd.DataFrame({'classifier': nbclass, 'accuracy':nbArr}).set_index('classifier')
nbdf.plot(kind='line', ylim=(75,85))
plt.title('Naive Bayes Analysis')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
print('Max Naive Bayes Accuracy :', max(nbArr))
#train odel using entire dataset
dtmodelfinal = DecisionTreeClassifier(criterion="entropy", max_depth = 10)
dtmodelfinal.fit(X, y)
print('Training accuracy over entire dataset:')
print(round(accuracy_score(y, dtmodelfinal.predict(X))*100, 4))
len(testdf[features].dropna())
Xfinaltest = pd.concat([pd.get_dummies(testdf[features[0]]),pd.get_dummies(testdf[features[1:]])], axis = 1, sort = False)
display(Xfinaltest.head())
#make prediction on test dataset
Ypred = dtmodelfinal.predict(Xfinaltest)
resultdf = pd.DataFrame({'PassengerId': testdf['PassengerId'], 'Survived': Ypred})
display(resultdf.head())
resultdf.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")