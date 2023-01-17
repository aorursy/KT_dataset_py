# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('/kaggle/input/titanic/train.csv')

test_df=pd.read_csv('/kaggle/input/titanic/test.csv')
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



%matplotlib inline

import matplotlib.pylab as pylab

plt.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
data_cleaner=[train_df,test_df]

train_df.info()
train_df.head()
print('null values in train data ','\n',train_df.isnull().sum())

print('null values in test data:','\n',test_df.isnull().sum())

train_df.describe(include='all')
train_df['Embarked'].mode()
for data in data_cleaner:

    data['Age'].fillna(data['Age'].median(),inplace=True)

    

    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)

    

    data['Fare'].fillna(data['Fare'].mean(),inplace=True)



drop_column=['PassengerId','Cabin','Ticket']

train_df.drop(drop_column,axis=1,inplace=True)



print(train_df.isnull().sum())



print('\n',test_df.isnull().sum())

    
#Feature Engineering



for data in data_cleaner:

    data['FamilySize']=data['SibSp']+data['Parch']+1

    data['IsAlone']=1

    data['IsAlone'].loc[data['FamilySize']>1]=0

    data['Title']=data['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]

    data['FareBin'] = pd.qcut(data['Fare'], 4)

    data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)

    

stat_min = 10

title_names = (train_df['Title'].value_counts() < stat_min)

train_df['Title'] = train_df['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)



print(train_df['Title'].value_counts())

print("-"*10)

label = LabelEncoder()

for dataset in data_cleaner:    

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
Target = ['Survived']

train_df_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts

train_df_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation

train_df_xy =  Target + train_df_x

print('Original X Y: ', train_df_xy, '\n')
train_df_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

train_df_xy_bin = Target + train_df_x_bin

print('Bin X Y: ', train_df_xy_bin, '\n')
train_df_dummy = pd.get_dummies(train_df[train_df_x])

train_df_x_dummy = train_df_dummy.columns.tolist()

train_df_xy_dummy = Target + train_df_x_dummy

print('Dummy X Y: ', train_df_xy_dummy, '\n')



train_df_dummy.head()
print('Train columns with null values: \n', train_df.isnull().sum())

print("-"*10)

print (train_df.info())

print("-"*10)
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(train_df[train_df_x_calc], train_df[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(train_df[train_df_x_bin], train_df[Target] , random_state = 0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(train_df_dummy[train_df_x_dummy], train_df[Target], random_state = 0)





print("Data1 Shape: {}".format(train_df.shape))

print("Train1 Shape: {}".format(train1_x.shape))

print("Test1 Shape: {}".format(test1_x.shape))
train1_x_bin.head()
for x in train_df_x:

    if train_df[x].dtype != 'float64' :

        print('Survival Correlation by:', x)

        print(train_df[[x, Target[0]]].groupby(x, as_index=False).mean())

        print('-'*10, '\n')

        



#using crosstabs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html

print(pd.crosstab(train_df['Title'],train_df[Target[0]]))
plt.figure(figsize=[16,12])



plt.subplot(231)

plt.boxplot(x=train_df['Fare'], showmeans = True, meanline = True)

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(232)

plt.boxplot(train_df['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(233)

plt.boxplot(train_df['FamilySize'], showmeans = True, meanline = True)

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')



plt.subplot(234)

plt.hist(x = [train_df[train_df['Survived']==1]['Fare'], train_df[train_df['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [train_df[train_df['Survived']==1]['Age'], train_df[train_df['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x = [train_df[train_df['Survived']==1]['FamilySize'], train_df[train_df['Survived']==0]['FamilySize']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()
fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'Embarked', y = 'Survived', data=train_df, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=train_df, ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=train_df, ax = saxis[0,2])



sns.pointplot(x = 'FareBin', y = 'Survived',  data=train_df, ax = saxis[1,0])

sns.pointplot(x = 'AgeBin', y = 'Survived',  data=train_df, ax = saxis[1,1])

sns.pointplot(x = 'FamilySize', y = 'Survived', data=train_df, ax = saxis[1,2])
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))



sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = train_df, ax = axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = train_df, split = True, ax = axis2)

axis2.set_title('Pclass vs Age Survival Comparison')



sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = train_df, ax = axis3)

axis3.set_title('Pclass vs Family Size Survival Comparison')
fig, qaxis = plt.subplots(1,3,figsize=(14,12))



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=train_df, ax = qaxis[0])

axis1.set_title('Sex vs Embarked Survival Comparison')



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=train_df, ax  = qaxis[1])

axis1.set_title('Sex vs Pclass Survival Comparison')



sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=train_df, ax  = qaxis[2])

axis1.set_title('Sex vs IsAlone Survival Comparison')
#more side-by-side comparisons

fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(14,12))



#how does family size factor with sex & survival compare

sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=train_df,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)



#how does class factor with sex & survival compare

sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train_df,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)
pp = sns.pairplot(train_df, hue = 'Survived', palette = 'deep', height=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )

pp.set(xticklabels=[])

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(train_df)
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



    

        

    ]
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )



MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)



#create table to compare MLA predictions

MLA_predict = train_df[Target]



row_index = 0

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

    cv_results = model_selection.cross_validate(alg, train_df[train_df_x_bin], train_df[Target], cv  = cv_split,return_train_score=True)



    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   

    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    

    #save MLA predictions - see section 6 for usage

    alg.fit(train_df[train_df_x_bin], train_df[Target])

    MLA_predict[MLA_name] = alg.predict(train_df[train_df_x_bin])

    

    row_index+=1

    

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare

#MLA_predict



MLA_compare
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')



#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
test_df.head()
model=svm.SVC(probability=True)

model.fit(train_df[train_df_x_bin],train_df[Target])

predictions=model.predict(test_df[train_df_x_bin])



output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})

output.head()
output.to_csv('my_submission_1.csv', index=False)

print("Your submission was successfully saved!")