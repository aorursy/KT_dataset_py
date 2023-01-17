import sys

import pandas as pd

import numpy as np

import scipy as sp

import IPython

from IPython import display

import sklearn

import random

import time

import warnings 

from subprocess import check_output #Pythonからコマンドを実行するやつ

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection, model_selection, metrics

from xgboost import XGBClassifier

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.plotting import scatter_matrix

warnings.filterwarnings('ignore')
%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
data_train = pd.read_csv('../input/titanic-kaggle/train.csv')

data_test = pd.read_csv('../input/titanic-kaggle/test.csv')
data_c = data_train.copy(deep = True)
data_train #data_trainはSurvivedが記載されているトレーニング用データ

# data_c はdata_trainをdeep copyした複製データ
data_test #テスト用のSurvivedを予測するデータ
data_cleaner = [data_c,data_test] #トレーニング用データdata_cとテストのdata_testのDataFrameを配列化
print(data_c.isnull().sum()) # AgeとCabinとEmbarkedにNULLが存在する。

print(data_test.isnull().sum()) #こちらはAgeとFare、CabinにNULLが存在する
data_c.describe(include= 'all')
for dataset in data_cleaner:

    dataset['Age'].fillna(dataset['Age'].median(),inplace = True)

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

# data_cとdata_test両方のNULLを定量をMedian,定性は最頻（Mode）で置き換え
data_c.drop(['PassengerId','Cabin','Ticket'], axis=1 ,inplace=True)

#PassengerIdは乗船番号なので削除。

#CabinもNULLが多いのでいったん削除。

#チケットも削除。
data_c.sample(5)
print(data_c.isnull().sum()) # data_c

print(data_test.isnull().sum())
for dataset in data_cleaner:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    #家族人数を表すカラムを追加

    dataset['IsAlone'] = 1

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

    #1人の人はフラグ1を立てる。

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    dataset['FareBin'] = pd.qcut(dataset['Fare'],4)

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
data_test.sample(5)
data_c.sample(5)
data_c['Title'].value_counts()
title_names = (data_c['Title'].value_counts() < 10)

data_c['Title'] = data_c['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
data_c['Title'].value_counts()

#出現が10個以下のものはMiscにする
data_c
data_test
label = LabelEncoder()

for dataset in data_cleaner:

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
print(data_c.sample(5))

print(data_test.sample(5))
#目的変数yを設定

Target = ['Survived']
#説明変数xを設定

data_x = ['Sex','Pclass','Embarked','Title','SibSp','Parch','Age','Fare','FamilySize','IsAlone']

#IsAloneをアルゴリズム計算用に入れてみる？



#アルゴリズム計算用

data_x_calc = ['Sex_Code','Pclass','Embarked_Code','Title_Code','SibSp','Parch','Age','Fare','IsAlone']



data_xy = Target + data_x

print(data_xy)
#離散型パラメータ

data_x_bin = ['Sex_Code','Pclass','Embarked_Code','Title_Code','FamilySize','AgeBin_Code','FareBin_Code']

data_xy_bin = Target + data_x_bin

print(data_xy_bin)
#object型の3カラムをget_dummiesでダミー変数へ変換

data_c_dummy = pd.get_dummies(data_c[data_x])



data_c_x_dummy = data_c_dummy.columns.tolist()

data_c_xy_dummy = Target + data_c_x_dummy

print(data_c_xy_dummy)
print(data_c.isnull().sum())

print("-"*10)

print(data_c.info())

print("-"*10)



print(data_test.isnull().sum())

print("-"*10)

print(data_test.info())
data_train.describe(include='all')
#トレーニング用データとして3種

#calcとしてアルゴリズム用としたもの、ビニング処理で離散型？にしたもの、ダミー型にしたもの。

train_c_x,test_c_x,train_c_y,test_c_y = model_selection.train_test_split(data_c[data_x_calc],data_c[Target],random_state = 0,test_size = 0.25)

train_c_x_bin,test_c_x_bin,train_c_y_bin,test_c_y_bin = model_selection.train_test_split(data_c[data_x_bin],data_c[Target],random_state = 0,test_size = 0.25)

train_c_x_dummy,test_c_x_dummy,train_c_y_dummy,test_c_y_dummy = model_selection.train_test_split(data_c_dummy[data_c_x_dummy],data_c[Target],random_state = 0,test_size = 0.25)



print(data_c.shape)

print(train_c_x.shape)

print(test_c_x.shape)

#各次元ごとの要素数 #IsAloneを独自でいれたので次元数少し多くなってる
for x in data_x:

    if data_c[x].dtype != 'float64' :

        print('Survival Correlation by:', x)

        print(data_c[[x,Target[0]]].groupby(x, as_index=False).mean())

print(pd.crosstab(data_c['Title'],data_c[Target[0]]))
data_c.info()
plt.figure(figsize=[16,12])



plt.subplot(2,3,1)

plt.boxplot(x=data_c['Fare'], showmeans = True , meanline = True)

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(2,3,2)

plt.boxplot(data_c['Age'], showmeans = True, meanline = True) #showmeans=点線が表示される

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(2,3,3)

plt.boxplot(data_c['FamilySize'], showmeans = True, meanline = True)

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')





plt.subplot(2,3,4)

plt.hist(x = [data_c[data_c['Survived'] == 1]['Fare'], data_c[data_c['Survived'] == 0]['Fare']],

        stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')



plt.subplot(2,3,5)

plt.hist(x = [data_c[data_c['Survived'] == 1]['Age'], data_c[data_c['Survived'] == 0]['Age']],

        stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')



plt.subplot(2,3,6)

plt.hist(x = [data_c[data_c['Survived'] == 1]['FamilySize'], data_c[data_c['Survived'] == 0]['FamilySize']],

        stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')



#どちらもX軸に重ねて表示しているので、[]で2つそのもののDataFrameをかこっている

#赤がDead、緑がSurvived
fig,saxis = plt.subplots(2,3,figsize=(16,12))



sns.barplot(x = 'Embarked', y = 'Survived', data = data_c, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', data = data_c , ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived',order = [1,0] ,data = data_c, ax = saxis[0,2])



sns.pointplot(x = 'FareBin', y = 'Survived', data = data_c, ax = saxis[1,0])

sns.pointplot(x = 'AgeBin', y = 'Survived', data = data_c, ax = saxis[1,1])

sns.pointplot(x = 'FamilySize', y = 'Survived', data = data_c, ax = saxis[1,2])
fig,(axis1,axis2,axis3,axis4) = plt.subplots(1,4,figsize=(20,12))



sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data_c, ax = axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data_c, split = True, ax = axis2)

axis2.set_title('Pclass vs Age Survival Comparison')



sns.boxplot(x = 'Pclass', y = 'FamilySize', hue = 'Survived', data = data_c, ax = axis3)

axis3.set_title('Pclass vs Family Size Survival Comparison')



sns.violinplot(x = 'Pclass', y = 'IsAlone', hue = 'Survived', data = data_c, split = True,ax =axis4)

axis4.set_title('Pclass vs IsAlone Survival Comparison')
fig,qaxis = plt.subplots(1,3,figsize=(14,12))



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data = data_c, ax =qaxis[0])

qaxis[0].set_title('Sex vs Embarked Survival Comparison')



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data = data_c, ax = qaxis[1])

qaxis[1].set_title('Sex vs Pclass Survival Comparison')



sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data = data_c, ax = qaxis[2])

qaxis[2].set_title('Sex vs IsAlone Survival Comparison')
fig,(maxis1,maxis2) = plt.subplots(1,2,figsize=(14,12))



sns.pointplot(x='FamilySize',y='Survived',hue = 'Sex', data = data_c,

             palette={'male':'blue','female':'pink'},

             markers=['*','o'],linestyles=['-','--'],ax = maxis1)

sns.pointplot(x='Pclass',y='Survived',hue = 'Sex', data = data_c,

             palette={'male':'blue','female':'pink'},

             markers=['*','o'],linestyles=['-','--'], ax = maxis2)
e = sns.FacetGrid(data_c,col = 'Embarked')

e.map(sns.pointplot,'Pclass','Survived','Sex', ci=95.0,palette = 'muted')

e.add_legend()
a = sns.FacetGrid(data_c,hue='Survived',aspect=4)

a.map(sns.kdeplot,'Age',shade=True)

a.set(xlim=(0,data_c['Age'].max()))

a.add_legend()
h = sns.FacetGrid(data_c,row= 'Sex',col = 'Pclass', hue = 'Survived')

h.map(plt.hist,'Age',alpha = 0.75)

h.add_legend()
pp = sns.pairplot(data_c,hue = 'Survived',palette = 'deep', height=2,diag_kind='hist')

pp.set(xticklabels=[])

# pp = sns.pairplot(data_c,hue = 'Survived',palette = 'deep', height=2,diag_kind='kde',diag_kws=dict(shade=True),plot_kws=dict(s=10))

#これがエラーになるのは、ペアプロットで使用されるkdeplotのbwが指定されていないため。

#palrplotで自動的kdeplotになるときにkdeplotの詳細どうやっていじるんだ

#↑もしかしたらdicts.optionalのここかもしれんな
def correlation_heatmap(df):

    fig,ax = plt.subplots(figsize=(14,12))

    colormap = sns.diverging_palette(240,10,as_cmap =True)

    

    fig = sns.heatmap(

        df.corr(), #相関係数

        cmap = colormap,

        square=True,

        cbar_kws={'shrink':.9},

        ax=ax,

        annot=True,

        linewidths=0.1,vmax=1.0,linecolor='white',

        annot_kws={'fontsize':12}

    )

    

    plt.title('Pearson Correlation of Features', y=1.05,size=15)



correlation_heatmap(data_c)
#まずはそれぞれデフォルト設定で様々なものを試す。

MLA = [

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    gaussian_process.GaussianProcessClassifier(),

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    neighbors.KNeighborsClassifier(),

    svm.SVC(probability=True), #Trueはクラスごとの確率も計算

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    XGBClassifier()

]



cv_split = model_selection.ShuffleSplit(n_splits =10,test_size = .25,train_size = .75)



MLA_columns = ['MLA Name','MLA Parameters','MLA Train Accuracy Mean','MLA Test Accuracy Mean','MLA Test Accuracy 3*STD','MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)

print(MLA_compare)



MLA_predict = data_c[Target]



row_index = 0



#それぞれのモデルを一気に回していく

for alg in MLA:

    

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA Name'] = MLA_name

    MLA_compare.loc[row_index,'MLA Parameters'] = str(alg.get_params())

    cv_results = model_selection.cross_validate(alg,data_c[data_x_bin],data_c[Target], cv = cv_split)

    MLA_compare.loc[row_index,'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index,'MLA Train Accuracy Mean'] = cv_results['score_time'].mean()

    MLA_compare.loc[row_index,'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()

    MLA_compare.loc[row_index,'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3

    alg.fit(data_c[data_x_bin],data_c[Target])

    MLA_predict[MLA_name] = alg.predict(data_c[data_x_bin])

    row_index += 1



MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'],ascending = False,inplace = True)

MLA_compare    
sns.barplot(x='MLA Test Accuracy Mean',y = 'MLA Name', data = MLA_compare, color = 'm')



plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')

#テストでの精度ランキング
#各rowにランダム関数で1か0を付与する

for index, row in data_c.iterrows():

    if random.random() > .5:

        data_c.at[index,'Random_Predict'] = 1

    else:

        data_c.at[index,'Random_Predict'] = 0



data_c['Random_Score'] = 0

data_c.loc[(data_c['Survived'] == data_c['Random_Predict']),'Random_Score'] = 1

#locでTrue or Falseで判断



print('Random Model Accuracy: {:.2f}%'.format(data_c['Random_Score'].mean()*100))
pivot_female = data_c[data_c.Sex=='female'].groupby(['Sex','Pclass','Embarked','FareBin'])['Survived'].mean()

print('Survival Decision Tree w/Female Node: \n',pivot_female)



pivot_male = data_c[data_c.Sex=='male'].groupby(['Sex','Title'])['Survived'].mean()

print('\n\nSurvival Decision Tree w/Male Node: \n',pivot_male)
def mytree(df):

    Model = pd.DataFrame(data = {'Predict':[]})

    male_title = ['Master']

    

    for index,row in df.iterrows():

        Model.loc[index,'Predict'] = 0

        

        if (df.loc[index,'Sex'] == 'female'):

            Model.loc[index,'Predict'] = 1

        

        if ((df.loc[index,'Sex'] == 'female') &

           (df.loc[index,'Pclass'] == 3) &

           (df.loc[index,'Embarked'] == 'S') &

           (df.loc[index,'Fare'] >8) 

           ):

            Model.loc[index,'Predict'] = 0

        

        if ((df.loc[index,'Sex'] == 'male') &

           (df.loc[index,'Title'] == 'Master')

           ):

            Model.loc[index,'Predict'] = 1

    return Model



Tree_Predict = mytree(data_c)

print(Tree_Predict)



print('Decision Tree Model Accuracy/Precision Score: {:.2f}%\n'.format(metrics.accuracy_score(data_c['Survived'],Tree_Predict)*100))

print(metrics.classification_report(data_c['Survived'], Tree_Predict))
import itertools

def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:np.newaxis]

        print('Normalized confusion matrix')

    else:

        print('Confusion matrix,without normalization')

    

    print(cm)

    

    plt.imshow(cm,interpolation='nearest',cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2

    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



cnf_matrix = metrics.confusion_matrix(data_c['Survived'], Tree_Predict)

np.set_printoptions(precision=2)



class_names = ['Dead','Survived']



plt.figure()

plot_confusion_matrix(cnf_matrix, classes = class_names,title='Confusion matrix,without normalization')



plt.figure()

plot_confusion_matrix(cnf_matrix, classes= class_names,normalize=True,title='Normalized confusion matrix')
dtree = tree.DecisionTreeClassifier(random_state = 0)

base_results = model_selection.cross_validate(dtree, data_c[data_x_bin], data_c[Target], cv  = cv_split)

dtree.fit(data_c[data_x_bin], data_c[Target])

print('-'*30)

print('BEFORE DT Parameters: ', dtree.get_params())

print("BEFORE DT Training w/bin score mean: {:.2f}". format(base_results['score_time'].mean()*100)) 

print("BEFORE DT Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))

print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))

print("BEFORE DT Test w/bin set score min: {:.2f}". format(base_results['test_score'].min()*100))

print('-'*10)
param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini

              'splitter': ['best'], #splitting methodology; two supported strategies - default is best

              'max_depth': [4], #max depth tree can grow; default is none

              'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2

              'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1

              #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all

              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation

             }



tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)

tune_model.fit(data_c[data_x_bin], data_c[Target])



print('AFTER DT Parameters: ', tune_model.best_params_)

print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_score_time'][tune_model.best_index_]*100)) 

print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)
print('BEFORE DT RFE Training Shape Old: ', data_c[data_x_bin].shape) 

print('BEFORE DT RFE Training Columns Old: ', data_c[data_x_bin].columns.values)



print("BEFORE DT RFE Training w/bin score mean: {:.2f}". format(base_results['score_time'].mean()*100)) 

print("BEFORE DT RFE Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))

print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))

print('-'*10)
dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy', cv = cv_split)

dtree_rfe.fit(data_c[data_x_bin], data_c[Target])



#transform x&y to reduced features and fit new model

#alternative: can use pipeline to reduce fit and transform steps: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

X_rfe = data_c[data_x_bin].columns.values[dtree_rfe.get_support()]

rfe_results = model_selection.cross_validate(dtree, data_c[X_rfe], data_c[Target], cv  = cv_split)



#print(dtree_rfe.grid_scores_)

print('AFTER DT RFE Training Shape New: ', data_c[X_rfe].shape) 

print('AFTER DT RFE Training Columns New: ', X_rfe)



print("AFTER DT RFE Training w/bin score mean: {:.2f}". format(rfe_results['score_time'].mean()*100)) 

print("AFTER DT RFE Test w/bin score mean: {:.2f}". format(rfe_results['test_score'].mean()*100))

print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}". format(rfe_results['test_score'].std()*100*3))

print('-'*10)
rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)

rfe_tune_model.fit(data_c[X_rfe], data_c[Target])





print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)

print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_score_time'][tune_model.best_index_]*100)) 

print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}". format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))

print('-'*10)
import graphviz 

dot_data = tree.export_graphviz(dtree, out_file=None, 

                                feature_names = data_x_bin, class_names = True,

                                filled = True, rounded = True)

graph = graphviz.Source(dot_data) 

graph
correlation_heatmap(MLA_predict)
MLA_predict
MLA_predict.columns.to_list()
vote_est = [

    ('ada',ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=None))),

    ('bc',ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_depth=None))),

    ('etc',ensemble.ExtraTreesClassifier()),

    ('gbc',ensemble.GradientBoostingClassifier()),

    ('rfc',ensemble.RandomForestClassifier()),

    ('gpc',gaussian_process.GaussianProcessClassifier()),

    ('lr',linear_model.LogisticRegressionCV()),

    ('bnb',naive_bayes.BernoulliNB()),

    ('gnb',naive_bayes.GaussianNB()),

    ('knn',neighbors.KNeighborsClassifier()),

    ('svc',svm.SVC(probability=True)),

    ('xgb',XGBClassifier())

]
# #ソフト投票で良かったので、ここで悪いやつを抜いていく

# vote_est = [

#     ('ada',ensemble.AdaBoostClassifier(learning_rate=0.75,n_estimators=1000,random_state=0,base_estimator=tree.DecisionTreeClassifier(max_depth=None))),

#     ('bc',ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_depth=None))),

#     ('gbc',ensemble.GradientBoostingClassifier(learning_rate=0.05,loss='exponential',max_depth=2,n_estimators=200,random_state=0)),

#     ('lr',linear_model.LogisticRegressionCV(fit_intercept=True,multi_class='ovr',random_state=0,solver='sag')),

#     ('bnb',naive_bayes.BernoulliNB(alpha=0.1)),

#     ('svc',svm.SVC(C=1,decision_function_shape='ovo',gamma=0.1,kernel='poly',random_state=0,probability=True)),

#     ('xgb',XGBClassifier(max_depth=6,n_estimators=300,seed=0,learning_rate=0.01))

# ]
#ハード投票または多数決ルール

vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

vote_hard_cv = model_selection.cross_validate(vote_hard, data_c[data_x_bin], data_c[Target], cv  = cv_split)

vote_hard.fit(data_c[data_x_bin], data_c[Target])

print(vote_hard_cv)

print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['score_time'].mean()*100)) 

print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))

print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))

print('-'*10)
#Soft Voteまたは加重確率

vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

vote_soft_cv = model_selection.cross_validate(vote_soft, data_c[data_x_bin], data_c[Target], cv  = cv_split)

vote_soft.fit(data_c[data_x_bin], data_c[Target])



print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['score_time'].mean()*100)) 

print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))

print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))

print('-'*10)
#パラメータサーチのための入力。

grid_n_estimator = [10, 50, 100, 300,1000]

grid_ratio = [.1, .25, .5, .75, 1.0]

grid_learn = [.01, .03, .05, .1, .25, .5, 1, .75]

grid_max_depth = [2, 4, 6, 8, 10, None]

grid_min_samples = [5, 10, .03, .05, .10]

grid_criterion = ['gini', 'entropy']

grid_bool = [True, False]

grid_seed = [0]
#各モデル設定項目

grid_param = [

            [{

            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

            'n_estimators': grid_n_estimator, #default=50

            'learning_rate': grid_learn, #default=1

            'random_state': grid_seed

            }],

    

            [{

            #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier

            'n_estimators': grid_n_estimator, #default=10

            'max_samples': grid_ratio, #default=1.0

            'random_state': grid_seed

             }],



    

            [{

            #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier

            'n_estimators': grid_n_estimator, #default=10

            'criterion': grid_criterion, #default=”gini”

            'max_depth': grid_max_depth, #default=None

            'random_state': grid_seed

             }],





            [{

            #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

            'loss': ['deviance', 'exponential'], #default=’deviance’

            'learning_rate': [.05, .1], #default=0.1 

            'n_estimators': [200,300,400], #default=100 

            'max_depth': grid_max_depth, #default=3   

            'random_state': grid_seed

             }],



    

            [{

            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

            'n_estimators': grid_n_estimator, #default=10

            'criterion': grid_criterion, #default=”gini”

            'max_depth': grid_max_depth, #default=None

            'oob_score': [True], #default=False

            'random_state': grid_seed

             }],

    

            [{    

            #GaussianProcessClassifier

            'max_iter_predict': grid_n_estimator, #default: 100

            'random_state': grid_seed

            }],

        

    

            [{

            #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV

            'fit_intercept': grid_bool, #default: True

            'multi_class':['outo','ovr','multinomial'],

            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs

            'random_state': grid_seed

             }],

            

    

            [{

            #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

            'alpha': grid_ratio, #default: 1.0

             }],

    

    

            #GaussianNB - 

            [{}],

    

            [{

            #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

            'n_neighbors': [1,2,3,4,5,6,7], #default: 5

            'weights': ['uniform', 'distance'], #default = ‘uniform’

            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

            }],

            

    

            [{

            #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

            #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r

            #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], #この種類増やすと時間相当掛かります。

            'C': [1,2,3,4,5], #default=1.0

            'gamma': grid_ratio, #edfault: auto

            'decision_function_shape': ['ovo', 'ovr'], #default:ovr

            'probability': [True],

            'random_state': grid_seed

             }],



    

            [{

            #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html

            'learning_rate': grid_learn, #default: .3

            'max_depth': [1,2,4,6,8,10,0], #default 2

            'n_estimators': grid_n_estimator, 

            'seed': grid_seed  

             }]   

        ]
#実行に相当時間がかかります。(SVCのカーネルぬいたのでそうでもないです)

start_total = time.perf_counter()

for clf, param in zip (vote_est,grid_param):

    start = time.perf_counter()

    best_search = model_selection.GridSearchCV(estimator = clf[1],param_grid = param, cv = cv_split, scoring = 'roc_auc')

    best_search.fit(data_c[data_x_bin],data_c[Target])

    run = time.perf_counter() - start

    

    best_param = best_search.best_params_

    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))

    clf[1].set_params(**best_param) 



run_total = time.perf_counter() - start_total

print('Total optimization time was {:.2f} minutes.'.format(run_total/60))



print('-'*10)
grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

grid_hard_cv = model_selection.cross_validate(grid_hard, data_c[data_x_bin], data_c[Target], cv  = cv_split)

grid_hard.fit(data_c[data_x_bin], data_c[Target])



print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['score_time'].mean()*100)) 

print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))

print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))

print('-'*10)
grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

grid_soft_cv = model_selection.cross_validate(grid_soft, data_c[data_x_bin], data_c[Target], cv  = cv_split)

grid_soft.fit(data_c[data_x_bin], data_c[Target])



print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['score_time'].mean()*100)) 

print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))

print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))

print('-'*10)
#モデリング用データ準備

print(data_test.info())

print("-"*10)
#ハイパーパラメータで一番良かったやつだけパラメータに入れてみます。(とりあえず時間かからないし)

#少し下のセルでめちゃくちゃ時間かかったやつをまた繰り返してみます。



# 手作りの意思決定ツリー 0.77990

#data_test['Survived'] = mytree(data_test).astype(int)







#完全なデータセットモデリング提出スコアを含む決定ツリー 0.77990

# submit_dt = tree.DecisionTreeClassifier()

# submit_dt = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)

# submit_dt.fit(data_c[data_x_bin], data_c[Target])

# print('Best Parameters: ', submit_dt.best_params_) 

# data_test['Survived'] = submit_dt.predict(data_test[data_x_bin])



#SVC 0.76555

# submit_svc = svm.SVC(probability=True)

# submit_svc = model_selection.GridSearchCV(svm.SVC(probability=True), param_grid= {'C': [1], 'decision_function_shape': ['ovo'], 'gamma': [0.1], 'kernel': ['poly'], 'probability': [True], 'random_state': [0]}, scoring = 'roc_auc', cv = cv_split)

# submit_svc.fit(data_c[data_x_bin],data_c[Target])

# print('Best Parameters: ', submit_svc.best_params_)

# data_test['Survived'] = submit_bc.predict(data_test[data_x_bin])





# 完全なデータセットモデリング提出スコア付きのバギング 0.76555

# submit_bc = ensemble.BaggingClassifier()

# submit_bc = model_selection.GridSearchCV(ensemble.BaggingClassifier(), param_grid= {'n_estimators':[1000], 'max_samples': [0.1], 'oob_score': grid_bool, 'random_state': [0]}, scoring = 'roc_auc', cv = cv_split)

# submit_bc.fit(data_c[data_x_bin], data_c[Target])

# print('Best Parameters: ', submit_bc.best_params_) 

# data_test['Survived'] = submit_bc.predict(data_test[data_x_bin])







# 完全なデータセットモデリング提出スコア付きの追加ツリー 0.74641

# submit_etc = ensemble.ExtraTreesClassifier()

# submit_etc = model_selection.GridSearchCV(ensemble.ExtraTreesClassifier(), param_grid={'n_estimators': [300], 'criterion': ['entropy'], 'max_depth': [8], 'random_state': [0]}, scoring = 'roc_auc', cv = cv_split)

# submit_etc.fit(data_c[data_x_bin], data_c[Target])

# print('Best Parameters: ', submit_etc.best_params_) 

# data_test['Survived'] = submit_etc.predict(data_test[data_x_bin])







# 完全なデータセットモデリング提出スコア付きのランダムなフォセット　0.74641

# submit_rfc = ensemble.RandomForestClassifier()

# submit_rfc = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), param_grid={'n_estimators': [50], 'criterion': ['entropy'], 'max_depth': [6], 'random_state': [0]}, scoring = 'roc_auc', cv = cv_split)

# submit_rfc.fit(data_c[data_x_bin], data_c[Target])

# print('Best Parameters: ', submit_rfc.best_params_) 

# data_test['Survived'] = submit_rfc.predict(data_test[data_x_bin])









# adaブーストw /フルデータセットモデリング提出スコア 0.75598

# submit_abc = ensemble.AdaBoostClassifier()

# submit_abc = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(), param_grid={'n_estimators': [1000], 'learning_rate': [0.75], 'algorithm': ['SAMME', 'SAMME.R'], 'random_state': [0]}, scoring = 'roc_auc', cv = cv_split)

# submit_abc.fit(data_c[data_x_bin], data_c[Target])

# print('Best Parameters: ', submit_abc.best_params_) 

# data_test['Survived'] = submit_abc.predict(data_test[data_x_bin])







# 完全なデータセットモデリング提出スコアによる勾配ブースティング 0.77511

# submit_gbc = ensemble.GradientBoostingClassifier()

# submit_gbc = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid={'learning_rate': [0.05], 'n_estimators': [200], 'max_depth': [2], 'random_state':[0]}, scoring = 'roc_auc', cv = cv_split)

# submit_gbc.fit(data_c[data_x_bin], data_c[Target])

# print('Best Parameters: ', submit_gbc.best_params_) 

# data_test['Survived'] = submit_gbc.predict(data_test[data_x_bin])





# 完全なデータセットモデリング提出スコアによる極端なブースティング 0.76555

# submit_xgb = XGBClassifier()

# submit_xgb = model_selection.GridSearchCV(XGBClassifier(), param_grid= {'learning_rate': [0.01], 'max_depth': [6], 'n_estimators': [300], 'seed': [0]}, scoring = 'roc_auc', cv = cv_split)

# submit_xgb.fit(data_c[data_x_bin], data_c[Target])

# print('Best Parameters: ', submit_xgb.best_params_) 

# data_test['Survived'] = submit_xgb.predict(data_test[data_x_bin])





# 完全なデータセットモデリング提出スコア付きのハード投票分類子 0.77033

#data_test['Survived'] = vote_hard.predict(data_test[data_x_bin])

#data_test['Survived'] = grid_hard.predict(data_test[data_x_bin]) 





# 完全なデータセットモデリング提出スコアを含むソフト投票分類器 0.78468

data_test['Survived'] = vote_soft.predict(data_test[data_x_bin])

#data_test['Survived'] = grid_soft.predict(data_test[data_x_bin]) 

submit = data_test[['PassengerId','Survived']]

submit.to_csv("submit.csv", index=False)



print('Validation Data Distribution: \n', data_test['Survived'].value_counts(normalize = True))

submit.sample(10)