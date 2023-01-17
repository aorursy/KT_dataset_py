# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#导入所需库

import re 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
#加载数据集

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

# 数据可视化

sns.set_style('whitegrid')

train_data.head()
# 展示数据训练集信息

train_data.info()

print("-" * 50)

test_data.info()
train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
#replace missing value with U0

train_data['Cabin'] = train_data.Cabin.fillna('U0') # train_data.Cabin[train_data.Cabin.isnull()]='U0'
from sklearn.ensemble import RandomForestRegressor



#choose training data to predict age

age_df = train_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]

age_df_notnull = age_df.loc[(train_data['Age'].notnull())]

age_df_isnull = age_df.loc[(train_data['Age'].isnull())]

X = age_df_notnull.values[:,1:]

Y = age_df_notnull.values[:,0]

# use RandomForestRegression to train data

RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)

RFR.fit(X,Y)

predictAges = RFR.predict(age_df_isnull.values[:,1:])

train_data.loc[train_data['Age'].isnull(), ['Age']]= predictAges



train_data.info()

#性别与生存之间的关系

train_data.groupby(['Sex','Survived'])['Survived'].count()
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

#船舱等级与生存之间的关系

train_data.groupby(['Pclass','Survived'])['Pclass'].count()

train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()

train_data.groupby(['Sex', 'Pclass', 'Survived'])['Survived'].count()

fig, ax = plt.subplots(1, 2, figsize = (18, 8))

sns.violinplot("Pclass", "Age", hue="Survived", data=train_data, split=True, ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0, 110, 10))



sns.violinplot("Sex", "Age", hue="Survived", data=train_data, split=True, ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0, 110, 10))



plt.show()

plt.figure(figsize=(12,5))

plt.subplot(121)

train_data['Age'].hist(bins=70)

plt.xlabel('Age')

plt.ylabel('Num')



plt.subplot(122)

train_data.boxplot(column='Age', showfliers=False)

plt.show()

facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train_data['Age'].max()))

facet.add_legend()

# average survived passengers by age

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

train_data["Age_int"] = train_data["Age"].astype(int)

average_age = train_data[["Age_int", "Survived"]].groupby(['Age_int'],as_index=False).mean()

sns.barplot(x='Age_int', y='Survived', data=average_age)

train_data['Age'].describe()

bins = [0, 12, 18, 65, 100]

train_data['Age_group'] = pd.cut(train_data['Age'], bins)

by_age = train_data.groupby('Age_group')['Survived'].mean()

by_age

by_age.plot(kind = 'bar')

train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_data['Title'], train_data['Sex'])

train_data[['Title','Survived']].groupby(['Title']).mean().plot.bar()

fig, axis1 = plt.subplots(1,1,figsize=(18,4))

train_data['Name_length'] = train_data['Name'].apply(len)

name_length = train_data[['Name_length','Survived']].groupby(['Name_length'],as_index=False).mean()

sns.barplot(x='Name_length', y='Survived', data=name_length)

# 将数据分为有兄弟姐妹的和没有兄弟姐妹的两组：

sibsp_df = train_data[train_data['SibSp'] != 0]

no_sibsp_df = train_data[train_data['SibSp'] == 0]

plt.figure(figsize=(10,5))

plt.subplot(121)

sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')

plt.xlabel('sibsp')



plt.subplot(122)

no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')

plt.xlabel('no_sibsp')



plt.show()

parch_df = train_data[train_data['Parch'] != 0]

no_parch_df = train_data[train_data['Parch'] == 0]



plt.figure(figsize=(10,5))

plt.subplot(121)

parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')

plt.xlabel('parch')



plt.subplot(122)

no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')

plt.xlabel('no_parch')



plt.show()

fig,ax=plt.subplots(1,2,figsize=(18,8))

train_data[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Parch and Survived')

train_data[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])

ax[1].set_title('SibSp and Survived')

train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp'] + 1

train_data[['Family_Size','Survived']].groupby(['Family_Size']).mean().plot.bar()

plt.figure(figsize=(10,5))

train_data['Fare'].hist(bins = 70)



train_data.boxplot(column='Fare', by='Pclass', showfliers=False)

plt.show()

train_data['Fare'].describe()

fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]

fare_survived = train_data['Fare'][train_data['Survived'] == 1]



average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])

average_fare.plot(yerr=std_fare, kind='bar', legend=False)



plt.show()

# Replace missing values with "U0"

train_data.loc[train_data.Cabin.isnull(), 'Cabin'] = 'U0'

train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)

train_data[['Has_Cabin','Survived']].groupby(['Has_Cabin']).mean().plot.bar()

# create feature for the alphabetical part of the cabin number

train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

# convert the distinct cabin letters with incremental integer values

train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]

train_data[['CabinLetter','Survived']].groupby(['CabinLetter']).mean().plot.bar()

sns.countplot('Embarked', hue='Survived', data=train_data)

plt.title('Embarked and Survived')

sns.factorplot('Embarked', 'Survived', data=train_data, size=3, aspect=2)

plt.title('Embarked and Survived rate')

plt.show()

embark_dummies  = pd.get_dummies(train_data['Embarked'])

train_data = train_data.join(embark_dummies)

train_data.drop(['Embarked'], axis=1,inplace=True)



embark_dummies = train_data[['S', 'C', 'Q']]

embark_dummies.head()

# Replace missing values with "U0"

train_data['Cabin'][train_data.Cabin.isnull()] = 'U0'

# create feature for the alphabetical part of the cabin number

train_data['CabinLetter'] = train_data['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())

# convert the distinct cabin letters with incremental integer values

train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]





train_data['CabinLetter'].head()

from sklearn import preprocessing



assert np.size(train_data['Age']) == 891

# StandardScaler will subtract the mean from each value then scale to the unit variance

scaler = preprocessing.StandardScaler()

train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1))

train_data['Age_scaled'].head()

# Divide all fares into quartiles

train_data['Fare_bin'] = pd.qcut(train_data['Fare'], 5)

train_data['Fare_bin'].head()

# qcut() creates a new variable that identifies the quartile range, but we can't use the string

# so either factorize or create dummies from the result



# factorize

train_data['Fare_bin_id'] = pd.factorize(train_data['Fare_bin'])[0]



# dummies

fare_bin_dummies_df = pd.get_dummies(train_data['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))

train_data = pd.concat([train_data, fare_bin_dummies_df], axis=1)

train_df_org = pd.read_csv('../input/train.csv')

test_df_org = pd.read_csv('../input/test.csv')

test_df_org['Survived'] = 0

combined_train_test = train_df_org.append(test_df_org)

PassengerId = test_df_org['PassengerId']

combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)

# 为了后面的特征分析，这里我们将 Embarked 特征进行facrorizing

combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]



# 使用 pd.get_dummies 获取one-hot 编码

emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'], prefix=combined_train_test[['Embarked']].columns[0])

combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)

# 为了后面的特征分析，这里我们也将 Sex 特征进行facrorizing

combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]



sex_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])

combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)

# what is each person's title? 

combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

title_Dict = {}

title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))

title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))

title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))

title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))

title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))

title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))



combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)

# 为了后面的特征分析，这里我们也将 Title 特征进行facrorizing

combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]



title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])

combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)



combined_train_test['Name_length'] = combined_train_test['Name'].apply(len)



combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))

combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform('count')

combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']

combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)

combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)

combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]



fare_bin_dummies_df = pd.get_dummies(combined_train_test['Fare_bin_id']).rename(columns=lambda x: 'Fare_' + str(x))

combined_train_test = pd.concat([combined_train_test, fare_bin_dummies_df], axis=1)

combined_train_test.drop(['Fare_bin'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder



# 建立PClass Fare Category

def pclass_fare_category(df, pclass1_mean_fare, pclass2_mean_fare, pclass3_mean_fare):

    if df['Pclass'] == 1:

        if df['Fare'] <= pclass1_mean_fare:

            return 'Pclass1_Low'

        else:

            return 'Pclass1_High'

    elif df['Pclass'] == 2:

        if df['Fare'] <= pclass2_mean_fare:

            return 'Pclass2_Low'

        else:

            return 'Pclass2_High'

    elif df['Pclass'] == 3:

        if df['Fare'] <= pclass3_mean_fare:

            return 'Pclass3_Low'

        else:

            return 'Pclass3_High'

        

Pclass1_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([1]).values[0]

Pclass2_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([2]).values[0]

Pclass3_mean_fare = combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get([3]).values[0]



# 建立Pclass_Fare Category

combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(pclass_fare_category, args=(

    Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)

pclass_level = LabelEncoder()



# 给每一项添加标签

pclass_level.fit(np.array(

    ['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))



# 转换成数值

combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(combined_train_test['Pclass_Fare_Category'])



# dummy 转换

pclass_dummies_df = pd.get_dummies(combined_train_test['Pclass_Fare_Category']).rename(columns=lambda x: 'Pclass_' + str(x))

combined_train_test = pd.concat([combined_train_test, pclass_dummies_df], axis=1)

combined_train_test['Pclass'] = pd.factorize(combined_train_test['Pclass'])[0]

def family_size_category(family_size):

    if family_size <= 1:

        return 'Single'

    elif family_size <= 4:

        return 'Small_Family'

    else:

        return 'Large_Family'



combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1

combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)



le_family = LabelEncoder()

le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))

combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])



family_size_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'],

                                        prefix=combined_train_test[['Family_Size_Category']].columns[0])

combined_train_test = pd.concat([combined_train_test, family_size_dummies_df], axis=1)

missing_age_df = pd.DataFrame(combined_train_test[

    ['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass']])



missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]

missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]

missing_age_test.head()

from sklearn import ensemble

from sklearn import model_selection

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor



def fill_missing_age(missing_age_train, missing_age_test):

    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)

    missing_age_Y_train = missing_age_train['Age']

    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)



    # model 1  gbm

    gbm_reg = GradientBoostingRegressor(random_state=42)

    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}

    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')

    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)

    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))

    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))

    print('GB Train Error for "Age" Feature Regressor:' + str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))

    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)

    print(missing_age_test['Age_GB'][:4])

    

    # model 2 rf

    rf_reg = RandomForestRegressor()

    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}

    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')

    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)

    print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))

    print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))

    print('RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))

    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)

    print(missing_age_test['Age_RF'][:4])



    # two models merge

    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)

    # missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)



    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])

    print(missing_age_test['Age'][:4])



    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)



    return missing_age_test







combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)

combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]

combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)



# 如果要提取数字信息，则也可以这样做，现在我们对数字票单纯地分为一类。

# combined_train_test['Ticket_Number'] = combined_train_test['Ticket'].apply(lambda x: pd.to_numeric(x, errors='coerce'))

# combined_train_test['Ticket_Number'].fillna(0, inplace=True)



# 将 Ticket_Letter factorize

combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]

combined_train_test.loc[combined_train_test.Cabin.isnull(), 'Cabin'] = 'U0'

combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)

Correlation = pd.DataFrame(combined_train_test[

    ['Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass', 

     'Pclass_Fare_Category', 'Age', 'Ticket_Letter', 'Cabin']])

colormap = plt.cm.viridis

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(Correlation.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

g = sns.pairplot(combined_train_test[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked',

       u'Family_Size', u'Title', u'Ticket_Letter']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])

scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age','Fare', 'Name_length']])

combined_train_test[['Age','Fare', 'Name_length']] = scale_age_fare.transform(combined_train_test[['Age','Fare', 'Name_length']])

combined_data_backup = combined_train_test

combined_train_test.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id', 'Pclass_Fare_Category', 

                          'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'],axis=1,inplace=True)

train_data = combined_train_test[:891]

test_data = combined_train_test[891:]



titanic_train_data_X = train_data.drop(['Survived'],axis=1)

titanic_train_data_Y = train_data['Survived']

titanic_test_data_X = test_data.drop(['Survived'],axis=1)

titanic_train_data_X.shape

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier



def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):



    # random forest

    rf_est = RandomForestClassifier(random_state=0)

    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}

    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)

    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)

    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))

    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))

    print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))

    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),

                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']

    print('Sample 10 Features from RF Classifier')

    print(str(features_top_n_rf[:10]))



    # AdaBoost

    ada_est =AdaBoostClassifier(random_state=0)

    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}

    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)

    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)

    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))

    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))

    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))

    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),

                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']

    print('Sample 10 Feature from Ada Classifier:')

    print(str(features_top_n_ada[:10]))



    # ExtraTree

    et_est = ExtraTreesClassifier(random_state=0)

    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}

    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)

    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)

    print('Top N Features Best ET Params:' + str(et_grid.best_params_))

    print('Top N Features Best ET Score:' + str(et_grid.best_score_))

    print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))

    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),

                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']

    print('Sample 10 Features from ET Classifier:')

    print(str(features_top_n_et[:10]))

    

    # GradientBoosting

    gb_est =GradientBoostingClassifier(random_state=0)

    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}

    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)

    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)

    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))

    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))

    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))

    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),

                                           'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']

    print('Sample 10 Feature from GB Classifier:')

    print(str(features_top_n_gb[:10]))

    

    # DecisionTree

    dt_est = DecisionTreeClassifier(random_state=0)

    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}

    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)

    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)

    print('Top N Features Best DT Params:' + str(dt_grid.best_params_))

    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))

    print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))

    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),

                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']

    print('Sample 10 Features from DT Classifier:')

    print(str(features_top_n_dt[:10]))

    

    # merge the three models

    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt], 

                               ignore_index=True).drop_duplicates()

    

    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et, 

                                   feature_imp_sorted_gb, feature_imp_sorted_dt],ignore_index=True)

    

    return features_top_n , features_importance

feature_to_pick = 30

feature_top_n, feature_importance = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)

titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])

titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])

rf_feature_imp = feature_importance[:10]

Ada_feature_imp = feature_importance[32:32+10].reset_index(drop=True)



# make importances relative to max importance

rf_feature_importance = 100.0 * (rf_feature_imp['importance'] / rf_feature_imp['importance'].max())

Ada_feature_importance = 100.0 * (Ada_feature_imp['importance'] / Ada_feature_imp['importance'].max())



# Get the indexes of all features over the importance threshold

rf_important_idx = np.where(rf_feature_importance)[0]

Ada_important_idx = np.where(Ada_feature_importance)[0]



# Adapted from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html

pos = np.arange(rf_important_idx.shape[0]) + .5



plt.figure(1, figsize = (18, 8))



plt.subplot(121)

plt.barh(pos, rf_feature_importance[rf_important_idx][::-1])

plt.yticks(pos, rf_feature_imp['feature'][::-1])

plt.xlabel('Relative Importance')

plt.title('RandomForest Feature Importance')



plt.subplot(122)

plt.barh(pos, Ada_feature_importance[Ada_important_idx][::-1])

plt.yticks(pos, Ada_feature_imp['feature'][::-1])

plt.xlabel('Relative Importance')

plt.title('AdaBoost Feature Importance')



plt.show()

from sklearn.model_selection import KFold



# Some useful parameters which will come in handy later on

ntrain = titanic_train_data_X.shape[0]

ntest = titanic_test_data_X.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 7 # set folds for out-of-fold prediction

kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)



def get_out_fold(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.fit(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC



rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6, 

                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)



ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)



et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)



gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)



dt = DecisionTreeClassifier(max_depth=8)



knn = KNeighborsClassifier(n_neighbors = 2)



svm = SVC(kernel='linear', C=0.025)

# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models

x_train = titanic_train_data_X.values # Creates an array of the train data

x_test = titanic_test_data_X.values # Creats an array of the test data

y_train = titanic_train_data_Y.values

# Create our OOF train and test predictions. These base results will be used as new features

rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test) # Random Forest

ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test) # AdaBoost 

et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test) # Extra Trees

gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test) # Gradient Boost

dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test) # Decision Tree

knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test) # KNeighbors

svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test) # Support Vector



print("Training is complete")

x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)

x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)

from xgboost import XGBClassifier



gbm = XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, 

                        colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)

predictions = gbm.predict(x_test)

StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})

StackingSubmission.to_csv('StackingSubmission.csv',index=False,sep=',')

from sklearn.model_selection import learning_curve



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), verbose=0):

    """

    Generate a simple plot of the test and traning learning curve.



    Parameters

    ----------

    estimator : object type that implements the "fit" and "predict" methods

        An object of that type which is cloned for each validation.



    title : string

        Title for the chart.



    X : array-like, shape (n_samples, n_features)

        Training vector, where n_samples is the number of samples and

        n_features is the number of features.



    y : array-like, shape (n_samples) or (n_samples, n_features), optional

        Target relative to X for classification or regression;

        None for unsupervised learning.



    ylim : tuple, shape (ymin, ymax), optional

        Defines minimum and maximum yvalues plotted.



    cv : integer, cross-validation generator, optional

        If an integer is passed, it is the number of folds (defaults to 3).

        Specific cross-validation objects can be passed, see

        sklearn.cross_validation module for the list of possible objects



    n_jobs : integer, optional

        Number of jobs to run in parallel (default 1).

    """

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt

X = x_train

Y = y_train



# RandomForest

rf_parameters = {'n_jobs': -1, 'n_estimators': 500, 'warm_start': True, 'max_depth': 6, 'min_samples_leaf': 2, 

                 'max_features' : 'sqrt','verbose': 0}



# AdaBoost

ada_parameters = {'n_estimators':500, 'learning_rate':0.1}



# ExtraTrees

et_parameters = {'n_jobs': -1, 'n_estimators':500, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0}



# GradientBoosting

gb_parameters = {'n_estimators': 500, 'max_depth': 5, 'min_samples_leaf': 2, 'verbose': 0}



# DecisionTree

dt_parameters = {'max_depth':8}



# KNeighbors

knn_parameters = {'n_neighbors':2}



# SVM

svm_parameters = {'kernel':'linear', 'C':0.025}



# XGB

gbm_parameters = {'n_estimators': 2000, 'max_depth': 4, 'min_child_weight': 2, 'gamma':0.9, 'subsample':0.8, 

                  'colsample_bytree':0.8, 'objective': 'binary:logistic', 'nthread':-1, 'scale_pos_weight':1}

title = "Learning Curves"

plot_learning_curve(RandomForestClassifier(**rf_parameters), title, X, Y, cv=None,  n_jobs=4, train_sizes=[50, 100, 150, 200, 250, 350, 400, 450, 500])

plt.show()
