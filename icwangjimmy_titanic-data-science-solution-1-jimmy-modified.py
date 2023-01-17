# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning algorithms using scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
train_df.info()
train_df.head()
test_df.head()
print(train_df.columns.values)
# 按照Pclass分类的，Age和Fare的散点图，颜色按照是否survive区分
g = sns.FacetGrid(train_df, hue="Survived", col="Pclass", margin_titles=True, palette={1:"seagreen", 0:"gray"}, height=5)
g=g.map(plt.scatter, "Fare", "Age", edgecolor="w").add_legend();
#boxplot一般用来寻找outlier的数据点
#按照Pclass分的Age分布图
plt.figure(figsize=(12,6))
ax= sns.boxplot(x="Pclass", y="Age", data=train_df)
ax= sns.stripplot(x="Pclass", y="Age", data=train_df, jitter=True, edgecolor="gray")
plt.show()
#按照Pclass分的Fare分布图
plt.figure(figsize=(12,6))
ax= sns.boxplot(x="Pclass", y="Fare", data=train_df)
ax= sns.stripplot(x="Pclass", y="Fare", data=train_df, jitter=True, edgecolor="gray")
plt.show()
# histograms 直方图一般用来单变量的分布情况
train_df.hist(figsize=(18,18))
plt.figure()
#Age变量可以再深入研究一下
train_df["Age"].hist();
#按照是否survive来看看age变量的分布
f,ax=plt.subplots(1,2,figsize=(16,8))
train_df[train_df['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)

train_df[train_df['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()
# scatter plot matrix 散点图矩阵
pd.plotting.scatter_matrix(train_df,figsize=(15,15))
plt.figure()
# violinplots 小提琴图和boxplot功能类似，不过可以展示对分布的概率密度估计
# 这里是按Sex分开的Age分布
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.figure(figsize=(12,8))
sns.violinplot(data=train_df,x="Sex", y="Age")
#这里更进一步将是否survive分开表示
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=train_df,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=train_df,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()
# Using seaborn pairplot to see the bivariate relation between each pair of features
# seaborn也可以做配对关系散点图，显示更好一些
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
sns.pairplot(train_df, hue="Sex")
# sns.pairplot(train_df, hue="Sex", diag_kind="kde")
# seaborn's kdeplot, plots univariate or bivariate density estimates.
#Size can be changed by tweeking the value of height used
# 这里显示的是按是否survive分开的Fare分布估计
sns.FacetGrid(train_df, hue="Survived", height=7).map(sns.kdeplot, "Fare").add_legend()
plt.show()
#讲单变量分布和双变量关系图同时显示
sns.jointplot(x='Fare',y='Age',data=train_df)
#附加了线性回归作为参考线
sns.jointplot(x='Fare',y='Age' ,data=train_df, kind='reg')
#蜂群图是stripplot的一种改进
plt.figure(figsize=(12,6))
sns.swarmplot(x='Pclass',y='Age',data=train_df)
#热点图用颜色显示栅格数据
plt.figure(figsize=(10,10)) 
sns.heatmap(train_df.corr(),annot=True,cmap='cubehelix_r') 
plt.show()
#柱状图也是很常用的
plt.figure(figsize=(6,6)) 
train_df['Pclass'].value_counts().plot(kind="bar");
#带有Fare点估计的柱状图
plt.figure(figsize=(6,6)) 
sns.barplot(x="Pclass", y="Fare", data=train_df)
#带有Age点估计的柱状图
plt.figure(figsize=(6,6)) 
sns.barplot(x="Pclass", y="Age", data=train_df)
#如果对零点并不关心，只关心相对大小，是对柱状图的一种替代
plt.figure(figsize=(6,6)) 
sns.pointplot('Pclass', 'Survived',hue='Sex', data=train_df)
plt.show()
#分布图是对直方图和kde的一种组合
f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(train_df[train_df['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(train_df[train_df['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(train_df[train_df['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()
#饼图也是常用图的一种
f,ax=plt.subplots(1,2,figsize=(18,8)) #one row two columns
train_df['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')

sns.countplot('Survived',data=train_df,ax=ax[1])
ax[1].set_title('Survived')
plt.show()
# countplot和value_counts功能类似
f,ax=plt.subplots(1,2,figsize=(18,8))
train_df[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=train_df,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()
# preview the data
train_df.head(10)
train_df.tail(10)
train_df.info()
print('_'*40)
test_df.info()
print(train_df.isnull().sum())
print('_'*40)
print(test_df.isnull().sum())
# train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
train_df.describe(percentiles=[.61, .62])
# Review Parch distribution using `percentiles=[.75, .8]`
#train_df.describe(percentiles=[.75, .8])
# SibSp distribution `[.68, .69]`
# train_df.describe(percentiles=[.68, .69])
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`
# train_df.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99])
train_df.describe(include=['O'])
corr = train_df.corr()
_ , ax = plt.subplots( figsize =( 12 , 10 ) )
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
_ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).count()
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Parch_Survived_df = train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# sns.scatterplot("Parch", "Survived", data=Parch_Survived_df)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).count()
g = sns.FacetGrid(train_df, col='Survived', height=5)
g.map(plt.hist, 'Age', bins=20)
facet = sns.FacetGrid( train_df, aspect=2 , col = "Survived", height=5 )
facet.map( sns.kdeplot , 'Age' , shade= True )
facet.add_legend()

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=3, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
import warnings
warnings.filterwarnings('ignore')
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=3, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.figure(figsize=(6,6)) 
sns.countplot('Embarked', hue='Survived', data=train_df)
plt.show()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=3, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
guess_ages = np.zeros((2,3))
guess_ages
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
#             guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            guess_ages[i,j] = age_guess
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
# train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
# test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]

train_df.head()
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()
# test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].dropna().mode()[0], inplace=True)
test_df.head()
pd.cut(train_df['Fare'], 4).value_counts()
pd.qcut(train_df['Fare'], 4).value_counts()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)
test_df.head(10)
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_1 = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
logreg.coef_[0]
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_2 = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
# Linear SVC
#Linear Support Vector Classification.
#Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility 
#in the choice of penalties and loss functions and should scale better to large numbers of samples.
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_5 = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_3 = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
# Perceptron
# Perceptron is a classification algorithm which shares the same underlying implementation with SGDClassifier. 
# In fact, Perceptron() is equivalent to SGDClassifier(loss=”perceptron”, eta0=1, learning_rate=”constant”, penalty=None).
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred_4 = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
# Stochastic Gradient Descent
# Linear classifiers (SVM, logistic regression, a.o.) with SGD training.
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred_6 = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
# Decision Tree
decision_tree = DecisionTreeClassifier()
# decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
# Y_pred_7 = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(decision_tree, out_file=None, 
                                feature_names = X_train.columns.tolist(), class_names = True,
                                filled = True, rounded = True)
graph = graphviz.Source(dot_data) 
graph
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, max_depth=7)
random_forest.fit(X_train, Y_train)
Y_pred_8 = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
random_forest.get_params
#Artificial neural network
# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=400, alpha=1e-4,
                    solver='sgd', verbose=False, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, Y_train)
Y_pred_9 = mlp.predict(X_test)
mlp.score(X_train, Y_train)
acc_ann = round(mlp.score(X_train, Y_train) * 100, 2)
acc_ann
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Perceptron', 'ANN',
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_perceptron, acc_ann,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
%%time
# Choose random forest to do model tuning. 
rfc = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [100, 200, 400], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [3, 5, 10, 20], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, Y_train)

# Set the clf to the best combination of parameters
rfc = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
rfc.fit(X_train, Y_train)
grid_obj.best_estimator_
Y_pred_10 = rfc.predict(X_test)
rfc.score(X_train, Y_train)
acc_rfc = round(rfc.score(X_train, Y_train) * 100, 2)
acc_rfc
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_10
    })
submission.to_csv('submission.csv', index=False)

