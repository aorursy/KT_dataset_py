# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
sub.head()
print(len(train))

train.head()
train.isnull().sum()
test.isnull().sum()
arrive = []

dead = []



for i in range(len(train['Sex'])) :

    if train['Survived'][i] == 0 :

        dead.append(train['Sex'][i])

    elif train['Survived'][i] == 1 :

        arrive.append(train['Sex'][i])

        

print('Survived Person : {} \nDead Person : {}'.format(len(arrive),len(dead)))
k = []

for i in train['Pclass'] :

    if i not in k :

        k.append(i)

print(np.sort(k))
ar_1_cnt,ar_2_cnt,ar_3_cnt = 0,0,0

de_1_cnt,de_2_cnt,de_3_cnt = 0,0,0



for i in range(len(train['Pclass'])) :

    if train['Pclass'][i] == 1 :

        if train['Survived'][i] == 1 :

            ar_1_cnt += 1

        else :

            de_1_cnt += 1

    elif train['Pclass'][i] == 2 :

        if train['Survived'][i] == 1 :

            ar_2_cnt += 1

        else :

            de_2_cnt += 1

    else :

        if train['Survived'][i] == 1 :

            ar_3_cnt += 1

        else :

            de_3_cnt += 1

        

print('Survived-Class1 : {}\t Survied-Class2 : {}\tSurvied-Class3 : {}'.format(ar_1_cnt,ar_2_cnt,ar_3_cnt))

print('Dead-Class1 : {}\t Dead-Class2 : {}\tDead-Class3 :{}'.format(de_1_cnt,de_2_cnt,de_3_cnt))
plt.figure()

plt.title('Class-Survied and Dead')

plt.bar([1,2,3],[ar_1_cnt,ar_2_cnt,ar_3_cnt],

        tick_label = ['class1','class2','class3'],

        label = 'Survived',

        color = 'seagreen')

plt.bar([1,2,3],[de_1_cnt,de_2_cnt,de_3_cnt],

        tick_label = ['class1','class2','class3'],

        bottom = [ar_1_cnt,ar_2_cnt,ar_3_cnt],

        label = 'Dead',

        color = 'coral')

plt.legend()
ar_male_cnt = 0

ar_female_cnt = 0

de_male_cnt = 0

de_female_cnt = 0



for i in arrive :

    if i == 'male' :

        ar_male_cnt += 1

    else :

        ar_female_cnt += 1

        

for i in dead :

    if i == 'male' :

        de_male_cnt += 1

    else :

        de_female_cnt += 1

        

print('Survived-male : {}\t Survied-female : {}'.format(ar_male_cnt,ar_female_cnt))

print('Dead-male : {}\t\t Dead-female : {}'.format(de_male_cnt,de_female_cnt))
fig, ax = plt.subplots(1,2, figsize=(10,6))

ax[0].pie([ar_male_cnt , ar_female_cnt] , labels= ['male','female'] , colors = ['seagreen','coral'],

       autopct = '%.1f%%')

ax[0].title.set_text('Sex - Survived')



ax[1].pie([de_male_cnt , de_female_cnt] , labels = ['male','female'] , colors = ['seagreen','coral'],

       autopct = '%.1f%%')

ax[1].title.set_text('Sex - Dead')

plt.show()
print('Age-nan :',train['Age'].isnull().sum())

print('Age-total :',len(train['Age']))

print('누락 비율 : ','%.2f'%(train['Age'].isnull().sum()/len(train['Age']) * 100),'%')
print('Age mean : ',train['Age'].mean())

print('Age median : ',train['Age'].median())

print('Age max : ',train['Age'].max())

print('Age min : ',train['Age'].min())
plt.figure()

plt.hist(train['Age'])

plt.title('Age-Histogram')

plt.show()
ar_Age = []

de_Age = []

for i in range(len(train['Age'])) :

    if train['Survived'][i] == 1 :

        if train['Age'][i] == train['Age'][i] :  #nan값은 자기자신과 비교할 수 없음

            ar_Age.append(train['Age'][i])

    else :

        if train['Age'][i] ==train['Age'][i] :

            de_Age.append(train['Age'][i])



arr1,bins1= np.histogram(ar_Age)

arr2,bins2= np.histogram(de_Age)

plt.figure()

plt.plot(bins1[:-1],arr1 , label = 'Survived',color = 'seagreen')

plt.plot(bins2[:-1],arr2 , label = 'Dead',color = 'coral')

plt.title('Age-Survived and Dead(delete NaN data)')

plt.legend()

plt.show()
ar_Age = []

de_Age = []

for i in range(len(train['Age'])) :

    if train['Survived'][i] == 1 :

        if train['Age'][i] == train['Age'][i] :  #nan값은 자기자신과 비교할 수 없음

            ar_Age.append(train['Age'][i])

        else :

            ar_Age.append(train['Age'].mean())

    else :

        if train['Age'][i] ==train['Age'][i] :

            de_Age.append(train['Age'][i])

        else :

            de_Age.append(train['Age'].mean())



arr1,bins1= np.histogram(ar_Age)

arr2,bins2= np.histogram(de_Age)

plt.figure()

plt.plot(bins1[:-1],arr1 , label = 'Survived',color = 'seagreen')

plt.plot(bins2[:-1],arr2 , label = 'Dead',color = 'coral')

plt.title('Age-Survived and Dead(NaN data replaced Age-mean)')

plt.legend()

plt.show()
ar_nan_cnt = 0

de_nan_cnt = 0



for i in range(len(train['Age'])) :

    if train['Survived'][i] == 1 :

        if train['Age'][i] != train['Age'][i]  :

            ar_nan_cnt += 1

    else :

        if train['Age'][i] != train['Age'][i]  :

            de_nan_cnt += 1

            

plt.figure(figsize=(5,5))

plt.title('Age-Nan data Survived and Dead')

plt.pie([ar_nan_cnt,de_nan_cnt],labels = ['Survived','Dead'],autopct='%.1f%%',colors=['seagreen','coral'])

plt.show()
train.head()
ar_sib,ar_parch = [], []

de_sib,de_parch = [], []

for i in range(len(train['SibSp'])) :

    if train['Survived'][i] == 1 :

        ar_sib.append(train['SibSp'][i])

        ar_parch.append(train['Parch'][i])

    else :

        de_sib.append(train['SibSp'][i])

        de_parch.append(train['Parch'][i])
fig, ax = plt.subplots(1,2, figsize=(10,6))



ar_sib_hist = plt.hist(ar_sib)

de_sib_hist= plt.hist(de_sib)

ar_parch_hist = plt.hist(ar_parch)

de_parch_hist = plt.hist(de_parch)



plt.cla()

ax[0].plot(ar_sib_hist[1][:-1],ar_sib_hist[0],label = 'Survived',marker='o' , color = 'seagreen')

ax[0].plot(de_sib_hist[1][:-1],de_sib_hist[0],label = 'Dead',marker='o', color = 'coral')

ax[0].title.set_text('With Sister or Brother-Survived and Dead')

ax[0].legend()

#plt.figure()

ax[1].plot(ar_parch_hist[1][:-1],ar_parch_hist[0] ,label = 'Survived',marker='o' , color = 'seagreen')

ax[1].plot(de_parch_hist[1][:-1],de_parch_hist[0] ,label = 'Dead',marker='o' , color = 'coral')

ax[1].title.set_text('With Parents or Children Survived and Dead')

ax[1].legend()
ar_fare , de_fare = [],[]

for i in range(len(train['Fare'])) :

    if train['Survived'][i] == 1 :

        ar_fare.append(train['Fare'][i])

    else :

        de_fare.append(train['Fare'][i])



ar_fare_hist = plt.hist(ar_fare)

de_fare_hist = plt.hist(de_fare)

plt.cla()

plt.plot(ar_fare_hist[1][:-1],ar_fare_hist[0],label = 'Survived',marker='o' , color = 'seagreen')

plt.plot(de_fare_hist[1][:-1],de_fare_hist[0],label = 'Dead',marker='o', color = 'coral')

plt.ylim(-50,450)

plt.legend()
e = []

for i in train['Embarked'] :

    if i not in e :

        e.append(i)

print(np.sort(e))
ar_C_cnt,ar_Q_cnt,ar_S_cnt, ar_nan_cnt = 0,0,0,0

de_C_cnt,de_Q_cnt,de_S_cnt, de_nan_cnt = 0,0,0,0



for i in range(len(train['Embarked'])) :

    

        if train['Embarked'][i] == 'C' :

            if train['Survived'][i] == 1 :

                ar_C_cnt += 1

            else :

                de_C_cnt += 1

                

        elif train['Embarked'][i] == 'Q' :

            if train['Survived'][i] == 1 :

                ar_Q_cnt += 1

            else :

                de_Q_cnt += 1

                

        elif train['Embarked'][i] == 'S' :

            if train['Survived'][i] == 1 :

                ar_S_cnt += 1

            else :

                de_S_cnt += 1

        else :

            if train['Embarked'][i] == 1 :

                ar_nan_cnt += 1

            else :

                de_nan_cnt += 1

        

print('Survived-C : {}\t Survied-Q : {}\tSurvied-S : {}\tSurvied-nan : {}'.format(ar_C_cnt,ar_Q_cnt,ar_S_cnt,ar_nan_cnt))

print('Dead-C : {}\t Dead-Q : {}\tDead-S :{}\tDead-nan: {}'.format(de_C_cnt,de_Q_cnt,de_S_cnt,de_nan_cnt))
plt.figure()

plt.title('Embarked-Survied and Dead')

plt.bar([1,2,3,4],[ar_C_cnt,ar_Q_cnt,ar_S_cnt,ar_nan_cnt],

        tick_label = ['C','Q','S','nan'],

        label = 'Survived',

        color = 'seagreen')

plt.bar([1,2,3,4],[de_C_cnt,de_Q_cnt,de_S_cnt,de_nan_cnt],

        tick_label = ['C','Q','S','nan'],

        bottom = [ar_C_cnt,ar_Q_cnt,ar_S_cnt,ar_nan_cnt],

        label = 'Dead',

        color = 'coral')

plt.legend()
x = []

y = []



cate = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']



for i in cate :

    tmpx = []

    for j in train[i] :

        tmpx.append(j)

    x.append(tmpx)

x = np.array(x).T



for i in train['Survived'] :

    y.append(i)

    

y = np.array(y)



print(' X')

print(x[:5])



print(' Y')

print(y[:5])
for i in range(len(x)) :

    if x[i][1] == 'male' :

        x[i][1] = 0

    elif x[i][1] == 'female' :

        x[i][1] = 1



for i in range(len(x)) :

    if x[i][-1] == 'C' :

        x[i][-1] = 0

    elif x[i][-1] == 'Q' :

        x[i][-1] = 1

    elif x[i][-1] == 'S' :

        x[i][-1] = 2

    else :

        x[i][-1] = 2  #누락값을 S로 대체 (S가 가장많이 죽었고 nan값 두개 모두 Dead이기때문)



for i in range(len(x)) :

    if x[i][2] == 'nan' :

        x[i][2] = round(train['Age'].mean(),2)  #나이의 nan값을 평균값으로 대체



x = x.astype(np.float64)

print(x[:5])
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(x,y)

knn = clf.score(x,y)*100

clf = RandomForestClassifier()

clf.fit(x,y)

rfc = clf.score(x,y)*100

clf = ExtraTreesClassifier()

clf.fit(x,y)

etc = clf.score(x,y)*100

clf = DecisionTreeClassifier()

clf.fit(x,y)

etc = clf.score(x,y)*100

clf = LinearDiscriminantAnalysis()

clf.fit(x,y)

lda = clf.score(x,y)*100

clf = AdaBoostClassifier(learning_rate=0.1)

clf.fit(x,y)

abc = clf.score(x,y)*100

clf = GradientBoostingClassifier()

clf.fit(x,y)

gbc = clf.score(x,y)*100

clf = LogisticRegression()

clf.fit(x,y)

lr = clf.score(x,y)*100

clf = MLPClassifier()

clf.fit(x,y)

mlp = clf.score(x,y)*100

clf = SVC()

clf.fit(x,y)

svc = clf.score(x,y)*100



print('KNeighborsClassifier : {0:0.2f}%'.format(knn))

print('RandomForestClassifier : {0:0.2f}%'.format(rfc))

print('ExtraTreesClassifier : {0:0.2f}%'.format(etc))

print('DecisionTreeClassifier : {0:0.2f}%'.format(etc))

print('LinearDiscriminantAnalysis : {0:0.2f}%'.format(lda))

print('AdaBoostClassifier : {0:0.2f}%'.format(abc))

print('GradientBoostingClassifier : {0:0.2f}%'.format(gbc))

print('LogisticRegression : {0:0.2f}%'.format(lr))

print('MLPClassifier : {0:0.2f}%'.format(mlp))

print('SVC : {0:0.2f}%'.format(svc))
kfold = StratifiedKFold(n_splits=10)
import seaborn as sns

random_state = 2

classifiers = []

classifiers.append(KNeighborsClassifier(n_neighbors=3))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(LinearDiscriminantAnalysis())

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(SVC(random_state=random_state))



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, x, y = y, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["KNeighboors","RandomForest","ExtraTrees",

                                                                                      "DecisionTree","LinearDiscriminantAnalysis","AdaBoost",

                                                                                      "GradientBoosting","LogisticRegression","MultipleLayerPerceptron","SVC"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
cv_res.sort_values(by='CrossValMeans', ascending=False).reset_index(drop=True)
test.head()
test.isnull().sum()
testx = []

for i in cate :

    tmpx = []

    for j in test[i] :

        tmpx.append(j)

    testx.append(tmpx)

testx = np.array(testx).T

print(testx[:5])

print(testx.shape)
for i in range(len(testx)) :

    if testx[i][1] == 'male' :

        testx[i][1] = 0

    elif testx[i][1] == 'female' :

        testx[i][1] = 1



for i in range(len(testx)) :

    if testx[i][-1] == 'C' :

        testx[i][-1] = 0

    elif testx[i][-1] == 'Q' :

        testx[i][-1] = 1

    elif testx[i][-1] == 'S' :

        testx[i][-1] = 2

    else :

        testx[i][-1] = 2  #누락값을 S로 대체 (S가 가장많이 죽었고 nan값 두개 모두 Dead이기때문)



for i in range(len(testx)) :

    if testx[i][2] == 'nan' :

        testx[i][2] = round(train['Age'].mean(),2)  #나이의 nan값을 평균값으로 대체

        

for i in range(len(testx)) :

    if testx[i][-2] == 'nan' :

        testx[i][-2] = 0  #fare의 nan값을 평균값으로 대체(낮은 가격의 fare가 가장 많기 때문)



testx = testx.astype(np.float64)

print(testx[:5])

print(testx.shape)
clf = GradientBoostingClassifier()

clf.fit(x,y)

Y_pred = clf.predict(testx)



submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission
submission.to_csv('submission.csv', index=False)