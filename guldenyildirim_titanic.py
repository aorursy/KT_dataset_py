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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split ,GridSearchCV ,cross_validate ,cross_val_score ,ShuffleSplit

from sklearn.preprocessing import StandardScaler , OneHotEncoder

from  sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier , ExtraTreeClassifier

from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.linear_model import LogisticRegression , RidgeClassifierCV , PassiveAggressiveClassifier , SGDClassifier

from sklearn.svm import  SVC

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.metrics import  classification_report , confusion_matrix ,roc_curve , roc_auc_score



import seaborn as sns

# pip install setuptools numpy scipy scikit-learn -U

# pip install lightgbm

import lightgbm as lgb

from lightgbm import LGBMClassifier

# pip install xgboost

# pip install setuptools

# cd python-package; python setup.py install

from xgboost import XGBClassifier



import warnings

warnings.filterwarnings('ignore')


train_df =pd.read_csv("/kaggle/input/titanic/train.csv")

test_df =pd.read_csv("/kaggle/input/titanic/test.csv")





#remember python assignment or equal passes by reference vs values, so we use the copy function

rev1_train=train_df.copy(deep=True)

rev1_test=test_df.drop('PassengerId', axis=1)


data_cleaner = [rev1_train, rev1_test]





for dataset in data_cleaner:

    dataset['surname'] = dataset['Name'].str.split(',', expand=True)[0]

    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]





# unique titleları ayıklamak için ; herhangi bir titledaki kişi sayısı 10 dan  küçük ise Misc olarak değiştirir.

title_names = (rev1_train['Title'].value_counts() < 10)

rev1_train['Title'] = rev1_train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

title_names = (rev1_train['Title'].value_counts() < 10)

rev1_test['Title'] = rev1_test['Title'].apply(lambda x: 'Misc' if x not in title_names else ('Misc' if title_names.loc[x] == True else x))



for dataset in data_cleaner :

    dataset.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

    dataset['Pclass'] = rev1_train.Pclass.astype('str')  

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1                                                       # initialize to yes/1 is alone

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0                          # now update to no/0 if family size is greater than 1

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)



#     dataset['Age'].fillna(dataset['Age'].median(), inplace=True)                  bunları daha sonra preprosesor içinde yaptırdım

#     dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

#     dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)



# for x in rev1_train.columns:

#     if x not in ['Survived','Age','Fare','surname'] :

#         print('Survival Correlation by:', x)

#         print(rev1_train.groupby(x).Survived.mean())

#         print('-'*10, '\n')





# graph distribution of quantitative data

# plt.figure(figsize=[16,12])

#

# plt.subplot(231)

# plt.boxplot(x=rev1_train['Fare'], showmeans = True, meanline = True)

# plt.title('Fare Boxplot')

# plt.ylabel('Fare ($)')



# plt.subplot(232)

# plt.boxplot(rev1_train['Age'], showmeans = True, meanline = True)

# plt.title('Age Boxplot')

# plt.ylabel('Age (Years)')



# plt.subplot(233)

# plt.boxplot(rev1_train['FamilySize'], showmeans = True, meanline = True)

# plt.title('Family Size Boxplot')

# plt.ylabel('Family Size (#)')

#

# plt.subplot(234)

# plt.hist(x = [rev1_train[rev1_train['Survived']==1]['Fare'], rev1_train[rev1_train['Survived']==0]['Fare']],

#          stacked=True, color = ['g','r'],label = ['Survived','Dead'])

# plt.title('Fare Histogram by Survival')

# plt.xlabel('Fare ($)')

# plt.ylabel('# of Passengers')

# plt.legend()

#

# plt.subplot(235)

# plt.hist(x = [rev1_train[rev1_train['Survived']==1]['Age'], rev1_train[rev1_train['Survived']==0]['Age']],

#          stacked=True, color = ['g','r'],label = ['Survived','Dead'])

# plt.title('Age Histogram by Survival')

# plt.xlabel('Age (Years)')

# plt.ylabel('# of Passengers')

# plt.legend()

#

# plt.subplot(236)

# plt.hist(x = [rev1_train[rev1_train['Survived']==1]['FamilySize'], rev1_train[rev1_train['Survived']==0]['FamilySize']],

#          stacked=True, color = ['g','r'],label = ['Survived','Dead'])

# plt.title('Family Size Histogram by Survival')

# plt.xlabel('Family Size (#)')

# plt.ylabel('# of Passengers')

# plt.legend()

#

# #we will use seaborn graphics for multi-variable comparison

# fig, saxis = plt.subplots(2, 3,figsize=(16,12))

#

# sns.barplot(x = 'Embarked', y = 'Survived', data=rev1_train, ax = saxis[0,0])

# sns.barplot(x = 'Pclass', y = 'Survived', data=rev1_train, ax = saxis[0,1])

# sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=rev1_train, ax = saxis[0,2])

#

# sns.pointplot(x = 'FareBin', y = 'Survived',  data=rev1_train, ax = saxis[1,0])

# sns.pointplot(x = 'AgeBin', y = 'Survived',  data=rev1_train, ax = saxis[1,1])

# sns.pointplot(x = 'FamilySize', y = 'Survived', data=rev1_train, ax = saxis[1,2])

#

# plt.figure()

# sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = rev1_train )

# plt.title('Pclass vs Fare Survival Comparison')

#

# #graph distribution of qualitative data

#

# fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

#

# sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=rev1_train, ax=axis1)

# axis1.set_title('Sex vs Embarked Survival Comparison')

#

# sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=rev1_train, ax  = axis2)

# axis2.set_title('Sex vs Pclass Survival Comparison')

#

# sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=rev1_train, ax  = axis3)

# axis3.set_title('Sex vs IsAlone Survival Comparison')

#

# #histogram comparison of sex, class, and age by survival- satırlarda cinsiyet, sütunlarda pclass olacak şekilde hayattan kalanların ve ölenlerin yaşları yaşlarının histogramını çizdirir.

# h = sns.FacetGrid(rev1_train, row = 'Sex', col = 'Pclass', hue = 'Survived')

# h.map(plt.hist, 'Age', alpha = .75)

# h.add_legend() # eksenlere açıklama ekler

#

#

# # correlation heatmap of dataset

# def correlation_heatmap(df):

#     _, ax = plt.subplots(figsize=(14, 12))

#     colormap = sns.diverging_palette(220, 10, as_cmap=True)     # renk paleti , bir renk listesi de olabilir.

#

#     _ = sns.heatmap(

#         rev1_train.corr(),

#         cmap=colormap,                 #matplotlib colormap name or object, or list of colors, optional

#         square=True,

#         cbar_kws={'shrink': .9},       #dict of key, value mappings, optional

#         ax=ax,                         #Axes in which to draw the plot, otherwise use the currently-active Axes.

#         annot=True,                    #If True, write the data value in each cell. If an array-like with the same shape as data, then use this to annotate the heatmap instead of the raw data.

#         linewidths=0.1, vmax=1.0, linecolor='white',

#         annot_kws={'fontsize': 12}     #dict of key, value mappings, optional

#     )

#

#     plt.title('Pearson Correlation of Features', y=1.05, size=15)

#

# correlation_heatmap(rev1_train)             #### sex , pclass, title categorical olmadığı için sadece kategorif ve sayısal değerler için kolelasyon hesabı yapıyor.#########



rev1_train.drop(['PassengerId'], axis=1, inplace=True)



y=rev1_train[['Survived']]

X=rev1_train.drop('Survived', axis=1)



X_train , X_test , y_train , y_test= train_test_split(X ,y , test_size = 0.2 , random_state= 40)




numeric_transformer = Pipeline(steps=[

    ('imputation', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())])



categorical_transformer = Pipeline(steps=[

    ('imputation', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()      #ya da direkt sürun isimleri verilebilir.

categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()



numeric_feature_index = [i for i, col in enumerate(X.columns) if col in numeric_features]

categorical_feature_index = [i for i, col in enumerate(X.columns) if col in categorical_features]



from sklearn.compose import ColumnTransformer



preprocessor = ColumnTransformer(

    transformers=[

                    ('num', numeric_transformer, numeric_feature_index),

                    ('cat', categorical_transformer, categorical_feature_index)

                ], remainder='passthrough')





steps = [('preprocessor', preprocessor), ('classifier' , KNeighborsClassifier()) ]

seed=41

param_grid=[

            # {

            # 'classifier': [KNeighborsClassifier()] ,

            # 'classifier__n_neighbors': [4,6,8,10] ,

            # 'classifier__weights': ['uniform','distance'],

            # },

            # {

            # 'classifier': [DecisionTreeClassifier(random_state=seed)],

            #  'classifier__splitter': ['best', 'random']

            # },

            {

            'classifier' : [RandomForestClassifier(random_state=seed)],

            'classifier__n_estimators' : [10,50,100,200,400],

            'classifier__max_depth': [ None, 7, 9, 11],

            'classifier__min_samples_split': [2 ,3 ]

            }

            # {'classifier':[LGBMClassifier(random_state=seed)] },

            # {'classifier' : [SVC(random_state=seed)]},

            # {'classifier' : [LogisticRegression(random_state=seed)],

            #  'classifier__C': np.logspace(-5, 8, 15),

            #  'classifier__solver' : ['liblinear','saga']

            #  }

            # {

            # 'classifier' : [GradientBoostingClassifier(random_state=seed)],

            #  'classifier__n_estimators':[100 ,120,150,200],

            #  'classifier__warm_start' : ['True' ,'False']

            #  }

            # {'classifier': [XGBClassifier()]}

            ]







pipeline=Pipeline(steps)

knncv=GridSearchCV(pipeline , param_grid, cv=5 , scoring='neg_mean_squared_error', verbose=True )

knncv.fit(X_train, y_train)


y_pred_prob=knncv.predict_proba(X_test)[:,1]

y_pred=knncv.predict(X_test)



print("neg_mean_squared_error: {}".format(knncv.score(X_test, y_test)))

print("Roc_Auc_score: {}".format(roc_auc_score(y_test, y_pred_prob)))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test,y_pred))

print("Tuned Model Parameters: {}".format(knncv.best_params_))



fpr, tpr , thresholds  = roc_curve(y_test,y_pred_prob)



# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()



results=knncv.predict(rev1_test)

test_df.shape

results.shape



result_file_df=pd.DataFrame({'PassengerId' : test_df['PassengerId'] , 'Survived' : results })

gender_submission=result_file_df.to_csv("C://Users/hp admin/Desktop/pyhton-case files/results.csv" , index=False)