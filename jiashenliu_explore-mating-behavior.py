import numpy as np  

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cross_validation import train_test_split

from ggplot import *
df = pd.read_csv('../input/baboon_mating.csv')

df1 = df

del df1['female_id']

del df1['male_id']

del df1['cycle_id']

df1.head()
df2 = df1

del df2['conceptive']

train_1,test_1 = train_test_split(df2,test_size=0.2,random_state=99)
print(train_1.describe(include = 'all'))
Cor_matrxi = train_1.iloc[:,1:].corr(method='pearson', min_periods=1)

print(Cor_matrxi)
fig, ax = plt.subplots()

heatmap = ax.pcolor(Cor_matrxi, cmap=plt.cm.Blues, alpha=0.8)

fig = plt.gcf()

fig.set_size_inches(6, 6)

ax.set_frame_on(False)

ax.set_yticks(np.arange(15) + 0.5, minor=False)

ax.set_xticks(np.arange(15) + 0.5, minor=False)

ax.set_xticklabels(train_1.columns[1:17], minor=False)

ax.set_yticklabels(train_1.columns[1:17], minor=False)

plt.xticks(rotation=90)
variables = ['consort','female_hybridscore','male_hybridscore','female_gendiv','male_gendiv','female_age','males_present','females_present','gen_distance_transform','rank_interact','female_age_transform','assort_index','gen_distance']
con = train_1[variables]
for i in range(1,13):

    g= ggplot(con,aes(x= 'consort',y=variables[i]))+geom_boxplot()+ggtitle('Box Plot of Consorting Result and '+variables[i])+theme_bw()

    print(g)
con['label'] = con['consort'].apply(lambda x:str(x))
g=ggplot(con,aes(x='female_hybridscore',y='male_hybridscore',color='label')) +geom_point() +theme_bw()+facet_grid('label')+ggtitle('Hybrid Score VS Consorting Behavior')

print(g)
g=ggplot(con,aes(x='female_gendiv',y='male_gendiv',color='label')) +geom_point() +theme_bw()+facet_grid('label')+ggtitle('Gen Div VS Consorting Behavior')

print(g)
g=ggplot(con,aes(x='females_present',y='males_present',color='label')) +geom_point() +theme_bw()+facet_grid('label')+ggtitle('Present Data VS Consorting Behavior')

print(g)
del con['label']
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve
Classifiers = [

    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=200),

    AdaBoostClassifier(),

    GaussianNB(),

    GradientBoostingClassifier(n_estimators=200)]
All_features = train_1.iloc[:,1:]

Test_features = test_1.iloc[:,1:]

Label = train_1.iloc[:,0]

Model = []

Accuracy = []

for clf in Classifiers:

    fit=clf.fit(All_features,Label)

    pred=fit.predict(Test_features)

    Model.append(clf.__class__.__name__)

    Accuracy.append(accuracy_score(test_1['consort'],pred))

    prob = fit.predict_proba(Test_features)[:,1]

    print('Accuracy of '+clf.__class__.__name__ +' is '+str(accuracy_score(test_1['consort'],pred)))

    fpr, tpr, _ = roc_curve(test_1['consort'],prob)

    tmp = pd.DataFrame(dict(fpr=fpr, tpr=tpr))

    g = ggplot(tmp, aes(x='fpr', y='tpr')) +geom_line() +geom_abline(linetype='dashed')+ ggtitle('Roc Curve of '+clf.__class__.__name__)

    print(g)
Classifiers_2 = [

    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),

    SVC(kernel="rbf", C=0.025, probability=True),

    RandomForestClassifier(n_estimators=200),

    GradientBoostingClassifier(n_estimators=200)]
All_features_2 = con.iloc[:,1:]

Test_features_2 = test_1[variables[1:]]

Label = con.iloc[:,0]

Model_2 = []

Accuracy_2 = []

for clf in Classifiers_2:

    fit=clf.fit(All_features_2,Label)

    pred=fit.predict(Test_features_2)

    Model_2.append(clf.__class__.__name__)

    Accuracy_2.append(accuracy_score(test_1['consort'],pred))

    prob = fit.predict_proba(Test_features_2)[:,1]

    print('Accuracy of '+clf.__class__.__name__ +' is '+str(accuracy_score(test_1['consort'],pred)))

    fpr, tpr, _ = roc_curve(test_1['consort'],prob)

    tmp = pd.DataFrame(dict(fpr=fpr, tpr=tpr))

    g = ggplot(tmp, aes(x='fpr', y='tpr')) +geom_line() +geom_abline(linetype='dashed')+ ggtitle('Roc Curve of '+clf.__class__.__name__)

    print(g)
Model = GradientBoostingClassifier(n_estimators=200)

Fit = Model.fit(All_features,Label)

importances = Model.feature_importances_

indices = np.argsort(importances)[::-1]

plt.figure()

plt.title("Feature importances")

plt.bar(range(All_features.shape[1]), importances[indices],

       color="r",  align="center")

plt.xticks(range(All_features.shape[1]),indices)

plt.xlim([-1, All_features.shape[1]])

plt.show()
print(All_features.columns[13],All_features.columns[4],All_features.columns[10],All_features.columns[1],All_features.columns[11],All_features.columns[12])