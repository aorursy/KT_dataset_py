import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



sns.set(context='notebook', style='white', palette='colorblind')
from sklearn.preprocessing import StandardScaler, LabelEncoder



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier



from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, cross_val_score
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

df = pd.concat([train, test], axis=0, ignore_index=True)



print(f'Train:{train.shape}\nTest:{test.shape}\nDf:{df.shape}')
df.sample(3)
df.info()
#columns with missing values

df.isna().sum()[df.isna().sum()>0]
df.describe()
df.describe(exclude='number')
df.sample(3)
# replacing values in male for 0 and female for 1 in 'Sex' column

df.Sex=df.Sex.map({'male':0, 'female':1}).astype('int')
# correlation between features

plt.figure(figsize=(10,4))

sns.heatmap(df.drop('PassengerId', axis=1).corr(), annot=True, center=0)
sns.FacetGrid(train, col='Survived').map(sns.distplot, "Age", hist=False, kde=True, rug=False, kde_kws={'shade':True})
sns.catplot(x="Pclass", y="Survived", data=train, kind="bar", height=3, aspect=2)
sns.catplot(x="Sex", y="Survived", hue="Pclass", data=train, kind="bar", height=3, aspect=2)
f,ax = plt.subplots(1, 2, figsize=(15,4))



sns.violinplot("Pclass","Age", hue="Sex", data=train, split=True, ax=ax[0])

ax[0].set_title('Pclass vs Age')

ax[0].set_yticks(range(0,110,10))



sns.violinplot("Sex", "Age", hue="Survived", data=train, split=True, ax=ax[1])

ax[1].set_title('Sex vs Age')

ax[1].set_yticks(range(0,110,10))

plt.show()
sns.catplot(x="Parch", y="Survived", data=train, kind="bar", height=4, aspect=2)
plt.figure(figsize=(3,3))

sns.barplot(x="Sex", y="Survived", data=train)

sns.despine()
sns.catplot(x="Embarked", y="Survived", data=train, kind="bar", height=3, aspect=2)
# creating a new feature using a linear function and dropping the old features to avoid redundancy and overfitting

df['Family_Size']=df.SibSp + df.Parch

df.groupby('Family_Size')['Survived'].mean()
df['Title'] = df['Name']



for name_string in df['Name']:

    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)



mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}



df.replace({'Title': mapping}, inplace=True)

    

gr = sns.countplot(x="Title", data=df)

gr.set_xticklabels(gr.get_xticklabels(), rotation=0)

plt.show(gr)



sns.catplot(x="Title",y="Survived",data=df.iloc[:len(train)],kind="bar")

plt.show(sns)    
df.Title.value_counts()
plt.figure(figsize=(5,4))

sns.countplot(x='Title', data=df)



plt.ylabel('')

plt.xlabel('')

plt.title('Count Plot - Titles')
sns.catplot(x="Title", y="Survived", data=df, kind="bar")
df['Family_Size'] = df['Parch'] + df['SibSp']
df['Last_Name'] = df['Name'].apply(lambda x: str.split(x, ",")[0])



df['Fare'].fillna(df['Fare'].mean(), inplace=True)



DEFAULT_SURVIVAL_VALUE = 0.5



df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE



for grp, grp_df in df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    

    if (len(grp_df) != 1):

        # A Family group is found.

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin==0.0):

                df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0

                

for _, grp_df in df.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin==0.0):

                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0
sns.catplot(x="Family_Size", y="Survived", data = df.iloc[:len(train)], kind="bar")

plt.title('Survival Prediction per Family Size')

plt.ylabel('')
facet = sns.FacetGrid(train, hue='Sex', aspect=3)

facet.map(sns.kdeplot,'Age', shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()
df['Family_Survival'].value_counts()
df['Fare'].fillna(df['Fare'].median(), inplace = True)



df['FareBin'] = pd.qcut(df['Fare'], 5)



label = LabelEncoder()

df['FareBin_Code'] = label.fit_transform(df['FareBin'])



df.drop(['Fare'], 1, inplace=True)
df['FareBin_Code'].value_counts()
# filling missing values in 'age' column

titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']



for title in titles:

    age_to_impute = df.groupby('Title')['Age'].median()[titles.index(title)]

    df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = age_to_impute
df['AgeBin'] = pd.qcut(df['Age'], 4)



label = LabelEncoder()

df['AgeBin_Code'] = label.fit_transform(df['AgeBin'])
df['AgeBin_Code'].value_counts()
sns.FacetGrid(data=df, hue = "Title", height=4, aspect=2).map(sns.kdeplot, "Age", shade=True)

plt.legend()
df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',

               'Embarked', 'Last_Name', 'FareBin', 'AgeBin', 'Survived', 'Title', 'Age'], axis = 1, inplace = True)
df.sample(2)
X_train = df[:len(train)]

X_test = df[len(train):]



y_train = train['Survived']
scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
kfold = StratifiedKFold(n_splits=8)
RFC = RandomForestClassifier()



rf_param_grid = {"max_depth": [None],

              "max_features": [3,"sqrt", "log2"],

              "min_samples_split": [n for n in range(1, 9)],

              "min_samples_leaf": [5, 7],

              "bootstrap": [False, True],

              "n_estimators" :[200, 500],

              "criterion": ["gini", "entropy"]}



rf_param_grid_best = {"max_depth": [None],

              "max_features": [3],

              "min_samples_split": [4],

              "min_samples_leaf": [5],

              "bootstrap": [False],

              "n_estimators" :[200],

              "criterion": ["gini"]}



gs_rf = GridSearchCV(RFC, param_grid = rf_param_grid_best, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)



gs_rf.fit(X_train, y_train)



rf_best = gs_rf.best_estimator_

RFC.fit(X_train, y_train)
print(f'RandomForest GridSearch best params: {gs_rf.best_params_}\n')

print(f'RandomForest GridSearch best score: {gs_rf.best_score_}')

print(f'RandomForest score:                 {RFC.score(X_train,y_train)}')
KNN = KNeighborsClassifier()



knn_param_grid = {'algorithm': ['auto'],

                 'weights': ['uniform', 'distance'], 

                 'leaf_size': [20, 25, 30], 

                 'n_neighbors': [12, 14, 16]}



knn_best_param_grid = {'algorithm': ['auto'],

                 'weights': ['uniform'], 

                 'leaf_size': [25], 

                 'n_neighbors': [14]}



gs_knn = GridSearchCV(KNN, param_grid = knn_best_param_grid, cv=kfold, scoring = "roc_auc", n_jobs= 4, verbose = 1)



gs_knn.fit(X_train, y_train)

KNN.fit(X_train, y_train)



knn_best = gs_knn.best_estimator_
print(f'KNN GridSearch best params: {gs_knn.best_params_}')

print()

print(f'KNN GridSearch best score: {gs_knn.best_score_}')

print(f'KNN score:                 {KNN.score(X_train,y_train)}')
knn1 = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 

                           metric_params=None, n_jobs=1, n_neighbors=6, p=2, 

                           weights='uniform')



knn1.fit(X_train, y_train)
print(f'KNN score - 2nd model:           {knn1.score(X_train, y_train)}')
GB = GradientBoostingClassifier()



gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [1000],

              'learning_rate': [0.02, 0.05],

              'min_samples_split': [15, 20, 25],

              'max_depth': [4, 6],

              'min_samples_leaf': [50, 60],

              'max_features': ["sqrt"] 

              }



gb_param_grid_best = {'loss' : ["deviance"],

              'n_estimators' : [1000],

              'learning_rate': [0.02],

              'min_samples_split': [25],

              'max_depth': [4],

              'min_samples_leaf': [60],

              'max_features': ["sqrt"] 

              }



gs_gb = GridSearchCV(GB, param_grid = gb_param_grid_best, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)



gs_gb.fit(X_train,y_train)

GB.fit(X_train, y_train)



gb_best = gs_gb.best_estimator_
print(f'GradienBoost GridSearch best params: {gs_gb.best_params_}')

print()

print(f'GradienBoost GridSearch best score: {gs_gb.best_score_}')

print(f'GradienBoost score:                 {GB.score(X_train, y_train)}')
XGB = XGBClassifier()



xgb_param_grid = {'learning_rate':[0.05, 0.1], 

                  'reg_lambda':[0.3, 0.5],

                  'gamma': [0.8, 1],

                  'subsample': [0.8, 1],

                  'max_depth': [2, 3],

                  'n_estimators': [200, 300]

              }



xgb_param_grid_best = {'learning_rate':[0.1], 

                  'reg_lambda':[0.3],

                  'gamma': [1],

                  'subsample': [0.8],

                  'max_depth': [2],

                  'n_estimators': [300]

              }



gs_xgb = GridSearchCV(XGB, param_grid = xgb_param_grid_best, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)



gs_xgb.fit(X_train,y_train)

XGB.fit(X_train, y_train)



xgb_best = gs_xgb.best_estimator_
print(f'XGB GridSearch best params: {gs_xgb.best_params_}')

print()

print(f'XGB GridSearch best score: {gs_xgb.best_score_}')

print(f'XGB score:                 {XGB.score(X_train, y_train)}')
def CVScore(classifiers):

    

    cv_score = []

    names = []

    

    for n_classifier in range(len(classifiers)):

        name = classifiers[n_classifier][0]

        model = classifiers[n_classifier][1]

        cv_score.append(cross_val_score(model, X_train, y_train, scoring = "roc_auc", cv = kfold, n_jobs=4))

        names.append(name)

        

    cv_means = []

    

    for cv_result in cv_score:

        cv_means.append(cv_result.mean())

        

    cv_res = pd.DataFrame({"Model":names,"CVMeans":cv_means})

    cv_res=cv_res.sort_values("CVMeans", axis = 0, ascending = False, inplace = False).reset_index(drop=True)

    print('\n-------------------------CrossVal Training scores-------------------------\n\n', cv_res)



clf_list = [("BestRandomForest", rf_best), ("BestGradientBoost", gb_best), ("BestKNN", knn_best), ("BestXGB", xgb_best), ("RandomForest", RFC), ("GradientBoost", GB), ("KNN Model 1", KNN), ("XGB", XGB), ("Best Model: KNN", knn1)]



CVScore(clf_list)
results=pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':knn1.predict(X_test)})

results.to_csv("Titanic_prediction.csv", index=False)



print('Done!')
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

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

                     train_scores_mean + train_scores_std, alpha=0.1)

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1)



    plt.plot(train_sizes, train_scores_mean, 'o-',

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-',

             label="Cross-validation score")



    plt.legend(loc="best")

    plt.show()



plot_learning_curve(rf_best,"RF learning curves", X_train, y_train, cv=kfold)

plot_learning_curve(xgb_best,"XGB learning curves", X_train, y_train, cv=kfold)

plot_learning_curve(gb_best,"Best GradientBoosting learning curves", X_train, y_train, cv=kfold)

plot_learning_curve(GB,"GradientBoosting learning curves", X_train, y_train, cv=kfold)

plot_learning_curve(knn1,"KNN: Winning Model learning curves", X_train, y_train, cv=kfold)