import sys  #该库提供与解析器交互的函数

import numpy as np

import pandas as pd

import scipy as sp #提供科学计算的函数

from IPython import display #漂亮的表格



# misc libraries

import random 

import time



# ignore warnings

import warnings

warnings.filterwarnings("ignore")
# 基础模型算法

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

import xgboost



# 特征工程工具

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



# 可视化工具

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.plotting import scatter_matrix



# 魔法函数，显示matplotlib生成函数的图像及设置格式

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
# 读取数据

data_train = pd.read_csv("../input/titanic/train.csv")

data_test = pd.read_csv("../input/titanic/test.csv")

data_train_origin = pd.read_csv("../input/titanic/train.csv")

# 构建data_cleaner，方便清洗时一起操作

data_cleaner = [data_train, data_test]
# 初步观察

print(data_train.info())

data_train.head(10)
# Completing:查看缺失值

print("训练集中各特征中缺失值情况：\n",data_train.isnull().sum().sort_values(ascending=False))

print("-"*10)

print("测试集中各特征中缺失值情况：\n",data_test.isnull().sum().sort_values(ascending=False))

print("-"*10)
# Completing：填充缺失数据，删除不需要数据

drop_columns = ['PassengerId', 'Ticket']

for dataset in data_cleaner:

    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

    dataset['Cabin'].fillna("Seat", inplace=True)

    dataset.drop(drop_columns, axis=1, inplace=True)

print(data_train.isnull().sum())

print("-"*10)

print(data_test.isnull().sum())
# Creating：特征工程

for dataset in data_cleaner:

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0  # 是否独身特征构建

    dataset['Title'] = dataset['Name'].str.split(",", expand=True)[1].str.split(".", expand=True)[0]  # Title特征构建

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)  # 将Fare均等分分组，每组4人

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5) # 将Age分成5组等间距分组
# 清除数量少于10个的title

stat_min = 10

for dataset in data_cleaner:

    title_names = dataset['Title'].value_counts() < stat_min

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x]==True else x)

    print(dataset['Title'].value_counts())

    print("="*10)
data_train.head()
# Convert：转换数据

label = LabelEncoder()

for dataset in data_cleaner:

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Cabin_Code'] = label.fit_transform(dataset['Cabin'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

# 目标值

Target = ['Survived']

# 原始特征变量

x_origin = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']

x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare']

xy_origin = Target + x_origin

# 离散化特征变量

x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

xy_bin = Target + x_bin

# dummy型特征变量

data_train_dummy = pd.get_dummies(data_train[x_origin])

x_dummy = data_train_dummy.columns.tolist()

xy_dummy = Target + x_dummy
data_train.head()
# 初步探索与分析

# 每个特征对应生存率影响

for x in x_origin:

        print('生存率相关变量:', x)

        print(data_train[[x, Target[0]]].groupby(x, as_index=False).mean())

        print('-'*10, '\n')
plt.figure(figsize=[16,12])



plt.subplot(231)

plt.boxplot(x=data_train['Fare'], showmeans = True, meanline = True)

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(232)

plt.boxplot(data_train['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(233)

plt.boxplot(data_train['FamilySize'], showmeans = True, meanline = True)

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')



plt.subplot(234)

plt.hist(x = [data_train[data_train['Survived']==1]['Fare'], data_train[data_train['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [data_train[data_train['Survived']==1]['Age'], data_train[data_train['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x = [data_train[data_train['Survived']==1]['FamilySize'], data_train[data_train['Survived']==0]['FamilySize']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()
# 热力图

corrmat = data_train.corr()

plt.subplots(figsize =(18, 16))

colormap = sns.diverging_palette(220, 10, as_cmap = True)

sns.heatmap(

        data_train.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

plt.title("Correlation map")
# 监督学习分类问题，选择相应的机器学习算法选择及初始化

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    #XGBClassifier

    xgboost.XGBClassifier()    

    ]
#分割数据集，进行交叉验证

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = 0.3, train_size = 0.6, random_state = 0 )

cv_results = model_selection.cross_validate(MLA[0], data_train[x_bin], data_train[Target], cv  = cv_split)

print(cv_results)
# 建立表格以对比各算法

MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)

MLA_predict = {}

# 计算并保存相关数据

row_index = 0

for alg in MLA:

    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    cv_results = model_selection.cross_validate(alg, data_train[x_bin], data_train[Target], cv  = cv_split)   

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   

    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    



    # 保存预测模型及结果

    alg.fit(data_train[x_bin], data_train[Target])

    MLA_predict[MLA_name] = alg.predict(data_train[x_bin])

    

    row_index+=1



    

# 打印表格

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare
sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=MLA_compare)