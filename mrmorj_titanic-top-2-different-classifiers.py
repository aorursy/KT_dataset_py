import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler, LabelEncoder



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier



from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, cross_val_score
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

testid = test['PassengerId']

train_len = len(train)



y = train['Survived']

X = pd.concat([train, test])
def remove_outlier(df_in, col_name):

    

    mean = df_in[col_name].mean()

    q1 = df_in[col_name].quantile(0.25)

    q3 = df_in[col_name].quantile(0.75)

    iqr = q3-q1

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    for ind in range(120):

        if (df_in[col_name].iloc[ind]>fence_high) or (df_in[col_name].iloc[ind]<fence_low):

            df_in[col_name].iloc[ind] = mean

            

    return df_in[col_name]



def graphics(df, y, features):

    for ind in features:

        plt.boxplot(x=df[ind], vert=False)

        plt.grid(True)

        plt.title(ind + " boxplot")

        plt.xlabel(ind+' value')

        plt.ylabel('Survived')

        plt.show()





#graphics(train, y, ["Pclass","SibSp","Fare"])



#train['Age'] = remove_outlier(train, "Age")

#train['SibSp'] = remove_outlier(train, "SibSp")

#train['Parch'] = remove_outlier(train, "Parch")

#train['Fare'] = remove_outlier(train, "Fare")
sns.heatmap(train[["Survived","Pclass","Age","SibSp","Parch","Fare"]].corr(), annot=True)
sns.FacetGrid(train, col='Survived').map(sns.distplot, "Age")
sns.catplot(x="SibSp", y="Survived", data=train, kind="bar")
sns.catplot(x="Pclass", y="Survived", data=train, kind="bar")
sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train, kind="bar")
sns.catplot(x="Parch", y="Survived", data=train, kind="bar")
sns.barplot(x="Sex", y="Survived", data=train)
sns.catplot(x="Embarked", y="Survived", data=train, kind="bar")
X['Title'] = X['Name']



for name_string in X['Name']:

    X['Title'] = X['Name'].str.extract('([A-Za-z]+)\.', expand=True)



mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}



X.replace({'Title': mapping}, inplace=True)

titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']



for title in titles:

    age_to_impute = X.groupby('Title')['Age'].median()[titles.index(title)]

    X.loc[(X['Age'].isnull()) & (X['Title'] == title), 'Age'] = age_to_impute



gr = sns.countplot(x="Title",data=X)

gr.set_xticklabels(gr.get_xticklabels(), rotation=75)

plt.show(gr)

sns.catplot(x="Title",y="Survived",data=X.iloc[:train_len],kind="bar")

plt.show(sns)    



X.drop('Title', axis = 1, inplace = True)
X['Family_Size'] = X['Parch'] + X['SibSp']
X['Last_Name'] = X['Name'].apply(lambda x: str.split(x, ",")[0])

X['Fare'].fillna(X['Fare'].mean(), inplace=True)



DEFAULT_SURVIVAL_VALUE = 0.5

X['Family_Survival'] = DEFAULT_SURVIVAL_VALUE



for grp, grp_df in X[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    

    if (len(grp_df) != 1):

        # A Family group is found.

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                X.loc[X['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin==0.0):

                X.loc[X['PassengerId'] == passID, 'Family_Survival'] = 0

                

for _, grp_df in X.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    X.loc[X['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin==0.0):

                    X.loc[X['PassengerId'] == passID, 'Family_Survival'] = 0

                        



sns.catplot(x="Family_Size",y="Survived",data = X.iloc[:train_len],kind="bar")
X['Fare'].fillna(X['Fare'].median(), inplace = True)



# Making Bins

X['FareBin'] = pd.qcut(X['Fare'], 5)



label = LabelEncoder()

X['FareBin_Code'] = label.fit_transform(X['FareBin'])



X.drop(['Fare'], 1, inplace=True)
X['AgeBin'] = pd.qcut(X['Age'], 4)



label = LabelEncoder()

X['AgeBin_Code'] = label.fit_transform(X['AgeBin'])



X.drop(['Age'], 1, inplace=True)
X['Sex'].replace(['male','female'],[0,1],inplace=True)





X.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',

               'Embarked', 'Last_Name', 'FareBin', 'AgeBin', 'Survived'], axis = 1, inplace = True)
X_train = X[:train_len]

X_test = X[train_len:]



y_train = y
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
kfold = StratifiedKFold(n_splits=8)
RFC = RandomForestClassifier()



rf_param_grid = {"max_depth": [None],

              "max_features": [3,"sqrt", "log2"],

              "min_samples_split": [2, 4],

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

RFC.fit(X_train, y_train)

rf_best = gs_rf.best_estimator_

print(f'RandomForest GridSearch best params: {gs_rf.best_params_}')

print(f'RandomForest GridSearch best score: {gs_rf.best_score_}')
KNN = KNeighborsClassifier()



knn_param_grid = {'algorithm': ['auto'],

                 'weights': ['uniform', 'distance'], 

                 'leaf_size': [20, 25, 30], 

                 'n_neighbors': [12, 14, 16]}

gs_knn = GridSearchCV(KNN, param_grid = knn_param_grid, cv=kfold, scoring = "roc_auc", n_jobs= 4, verbose = 1)



gs_knn.fit(X_train, y_train)

KNN.fit(X_train, y_train)



knn_best = gs_knn.best_estimator_

print(f'KNN GridSearch best params: {gs_knn.best_params_}')

print(f'KNN GridSearch best score: {gs_knn.best_score_}')
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

print(f'GradienBoost GridSearch best score: {gs_gb.best_score_}')
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

print(f'XGB GridSearch best score: {gs_xgb.best_score_}')
knn1 = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 

                           metric_params=None, n_jobs=1, n_neighbors=6, p=2, 

                           weights='uniform')



knn1.fit(X_train, y_train)
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

    cv_res.sort_values("CVMeans", axis = 0, ascending = True, inplace = True)



    print('----------CrossVal scores---------\n', cv_res)



    

best_class = [("RandomForest", rf_best), ("GradientBoost", gb_best), ("KNN", knn_best), ("XGB", xgb_best), ("KNN new", knn1)]

def_class = [("RandomForest", RFC), ("GradientBoost", GB), ("KNN", KNN), ("XGB", XGB), ("KNN new", knn1)]



#CVScore(def_class)

CVScore(best_class)
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

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    plt.show()



plot_learning_curve(rf_best,"RF learning curves", X_train, y_train, cv=kfold)

plot_learning_curve(xgb_best,"XGB learning curves", X_train, y_train, cv=kfold)

plot_learning_curve(gb_best,"GradientBoosting learning curves", X_train, y_train, cv=kfold)

plot_learning_curve(knn_best,"KNN learning curves", X_train, y_train, cv=kfold)

plot_learning_curve(knn1,"KNN (NEW) learning curves", X_train, y_train, cv=kfold)
y_pred = knn1.predict(X_test)



test_Survived = pd.Series(y_pred, name="Survived")

results = pd.concat([testid,test_Survived],axis=1)

results.to_csv("submit.csv",index=False)