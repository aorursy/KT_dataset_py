import seaborn as sns 

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.stats import zscore

from datetime import datetime



#Machine Learning Packages

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
df_gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
## Print the 5 first rows of the DataFrame

#df_train.head()

#df_test.head()



## Print the shape

#df_train.shape

#df_test.shape



## Print the datatypes

#df_train.dtypes

#df_test.dtypes



## Info of DataFrame

#df_train.info()

#df_test.info()



## Describe the DataFrame. Basic descriptive stats

#df_train.describe()

#df_test.describe()



## Null data amout

#print(pd.isnull(df_train).sum())

#print(pd.isnull(df_test).sum())

df_train['Sex'].replace(['female','male'],[0,1],inplace=True)

df_test['Sex'].replace(['female','male'],[0,1],inplace=True)
fig, ax = plt.subplots()

sns.countplot(x='Sex', hue ='Survived', data = df_train)

new_xtick = ['Females', 'Males']

ax.set_xticklabels(new_xtick)

new_legend = ['Deceased', 'Survivors']

plt.legend(new_legend)

plt.show()
df_train['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)

df_test['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)
fig, ax = plt.subplots()

sns.countplot(x='Embarked',hue='Survived',data=df_train)

new_xtick = ['Q','S', 'C']

ax.set_xticklabels(new_xtick)

new_legend = ['Deceased', 'Survivors']

plt.legend(new_legend)

plt.show()
avg_age = ((df_train["Age"].mean() * df_train["Age"].shape[0]) + (df_test["Age"].mean() * df_test["Age"].shape[0]))/ (df_train["Age"].shape[0] + df_test["Age"].shape[0])

avg_age = np.round(avg_age)



df_train['Age'] = df_train['Age'].replace(np.nan, avg_age)

df_test['Age'] = df_test['Age'].replace(np.nan, avg_age)
#Bands: (1) 0-7, (2) 8-15, (3) 16-25, (4) 26-32, (5) 33-40, (6) 41-60, (7) 61-100

bins = [0, 7, 15, 25, 32, 40, 60, 100]

names = ['1', '2', '3', '4', '5', '6', '7']

df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)

df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names)



df_train['Age'].groupby(df_train.Age).count()
fig, ax = plt.subplots()

sns.countplot(x='Age',hue='Survived',data=df_train)

new_xtick = ['0-7', '8-15', '16-25','26-32','33-40','41-60','61-100']

ax.set_xticklabels(new_xtick)

new_legend = ['Deceased', 'Survivors']

plt.legend(new_legend)

plt.show()
df_train.drop(['Cabin'], axis = 1, inplace=True)

df_test.drop(['Cabin'], axis = 1, inplace=True)

df_train = df_train.drop(['PassengerId','Name','Ticket'], axis=1)

df_test = df_test.drop(['Name','Ticket'], axis=1)
df_train.dropna(axis=0, how='any', inplace=True)

df_test.dropna(axis=0, how='any', inplace=True)
def column_dissociation(DataFrame, ColumnName):

    df = DataFrame

    uniques = df[ColumnName].drop_duplicates()#unique results of the column

    uniques = uniques.sort_values( ascending=True) #Sort the values

    new_column_name = ""

    for i, value in enumerate(uniques):

        new_column_name = ColumnName + '_' + str(value)

        df[new_column_name] = np.zeros(df.shape[0])

        df[new_column_name] = df[new_column_name].astype('int64')

        df.loc[df[ColumnName] == value, new_column_name] = 1

    df.drop([ColumnName], axis = 1, inplace=True)

    return df
def column_comparator(DataFrame1, DataFrame2):

    bool_test = False

    result = []

    for i, column1 in enumerate(DataFrame1):

        bool_test = False

        for j, column2 in enumerate(DataFrame2):

            if column1 == column2:

                bool_test = True 

                

        if not bool_test:

            result.append(column1)

    return result
def create_zeros_column(DataFrame, columns):

    df = DataFrame

    if len(columns) == 0:

        print('No columns added')

    elif isinstance(columns, str):

        df[columns] = np.zeros(df.shape[0])

        df[columns] = df[columns].astype('int64')

    else:

        for i, col in enumerate(columns):

            df[col] = np.zeros(df.shape[0])

            df[col] = df[col].astype('int64')

    return df
#- Age (Age Group)

df_train = column_dissociation(df_train, 'Age')

df_test = column_dissociation(df_test, 'Age')

#- Embarket

df_train = column_dissociation(df_train, 'Embarked')

df_test = column_dissociation(df_test, 'Embarked')

#- Pclass

df_train = column_dissociation(df_train, 'Pclass')

df_test = column_dissociation(df_test, 'Pclass')

#- SibSp

df_train = column_dissociation(df_train, 'SibSp')

df_test = column_dissociation(df_test, 'SibSp')

#- Parch

df_train = column_dissociation(df_train, 'Parch')

df_test = column_dissociation(df_test, 'Parch')



#Are there any column in a dataFrame that are not in the other

cc1 = column_comparator(df_test,df_train)

print(cc1)

cc2 = column_comparator(df_train, df_test)

print(cc2)

df_train = create_zeros_column(df_train, 'Parch_9')
# Correlation Heatmap

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

        annot_kws={'fontsize':6 }

    )

    

    plt.title('Heatmap Correlation', y=1.05, size=15)



correlation_heatmap(df_train)
X = np.array(df_train.drop(['Survived'], 1))

y = np.array(df_train['Survived'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
##Logistic Regression

logreg = LogisticRegression(max_iter = 100000)

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)



print('Logistic Regression')

print('Score: ' + str(logreg.score(X_train, y_train)))

print('Accuracy: '+ str(accuracy_score(y_test, Y_pred)))

print('Confusion Matrix:')

print(confusion_matrix(y_test, Y_pred))

print(classification_report(y_test, Y_pred))

print('------------------------------------')



##K neighbors

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)



print('K-Nearest Neighbors Classifier')

print('Score: ' + str(knn.score(X_train, y_train)))

print('Accuracy: '+ str(accuracy_score(y_test, Y_pred)))

print('Confusion Matrix:')

print(confusion_matrix(y_test, Y_pred))

print(classification_report(y_test, Y_pred))

print('------------------------------------')



##Support Vector Machines

svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)



print('Support Vector Machine Classifier')

print('Score: ' + str(svc.score(X_train, y_train)))

print('Accuracy: '+ str(accuracy_score(y_test, Y_pred)))

print('Confusion Matrix:')

print(confusion_matrix(y_test, Y_pred))

print(classification_report(y_test, Y_pred))

print('------------------------------------')



##Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, y_train)

Y_pred = perceptron.predict(X_test)



print('Perceptron Classifier')

print('Score: ' + str(perceptron.score(X_train, y_train)))

print('Accuracy: '+ str(accuracy_score(y_test, Y_pred)))

print('Confusion Matrix:')

print(confusion_matrix(y_test, Y_pred))

print(classification_report(y_test, Y_pred))

print('------------------------------------')



##XGBoost Classifier

xgboost = XGBClassifier(learning_rate=1.3, n_estimators=2000, max_depth=40, min_child_weight=40, 

                      gamma=0.4,nthread=10, subsample=0.8, colsample_bytree=.8, 

                      objective= 'binary:logistic',scale_pos_weight=10,seed=29)

xgboost.fit(X_train, y_train)

Y_pred = xgboost.predict(X_test)



print('XGBoost Classifier')

print('Score: ' + str(xgboost.score(X_train, y_train)))

print('Accuracy: '+ str(accuracy_score(y_test, Y_pred)))

print('Confusion Matrix:')

print(confusion_matrix(y_test, Y_pred))

print(classification_report(y_test, Y_pred))

print('------------------------------------')



##Random Forest

random_forest = RandomForestClassifier(n_estimators=1000, random_state = 0)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)



print('Random Forest Classifier')

print('Score: ' + str(random_forest.score(X_train, y_train)))

print('Accuracy: '+ str(accuracy_score(y_test, Y_pred)))

print('Confusion Matrix:')

print(confusion_matrix(y_test, Y_pred))

print(classification_report(y_test, Y_pred))

print('------------------------------------')



##Multi-Layer Perceptron Classifier

mlp_classifier = MLPClassifier(hidden_layer_sizes = 1000, alpha = 0.00001, learning_rate = 'adaptive', learning_rate_init = 0.001, random_state = 0, max_iter = 100000 )

mlp_classifier.fit(X_train, y_train)

Y_pred = mlp_classifier.predict(X_test)

mlp_classifier.score(X_train, y_train)



print('Multi-Layer Perceptron Classifier')

print('Score: ' + str(mlp_classifier.score(X_train, y_train)))

print('Accuracy: '+ str(accuracy_score(y_test, Y_pred)))

print('Confusion Matrix:')

print(confusion_matrix(y_test, Y_pred))

print(classification_report(y_test, Y_pred))

print('------------------------------------')





##AdaBoostClassifier

adaboost = AdaBoostClassifier()

adaboost.fit(X_train, y_train)

Y_pred = adaboost.predict(X_test)

adaboost.score(X_train, y_train)



print('AdaBoost Classifier')

print('Score: ' + str(adaboost.score(X_train, y_train)))

print('Accuracy: '+ str(accuracy_score(y_test, Y_pred)))

print('Confusion Matrix:')

print(confusion_matrix(y_test, Y_pred))

print(classification_report(y_test, Y_pred))

print('------------------------------------')



##Linear Discriminant Analysis

lineardiscriminant = LinearDiscriminantAnalysis()

lineardiscriminant.fit(X_train, y_train)

Y_pred = lineardiscriminant.predict(X_test)

lineardiscriminant.score(X_train, y_train)



print('Linear Discriminant Analysis')

print('Score: ' + str(lineardiscriminant.score(X_train, y_train)))

print('Accuracy: '+ str(accuracy_score(y_test, Y_pred)))

print('Confusion Matrix:')

print(confusion_matrix(y_test, Y_pred))

print(classification_report(y_test, Y_pred))

print('------------------------------------')



##Gradient Boosting Classifier

gradient_boosting = GradientBoostingClassifier()

gradient_boosting.fit(X_train, y_train)

Y_pred = gradient_boosting.predict(X_test)

gradient_boosting.score(X_train, y_train)



print('Gradient Boosting Classifier')

print('Score: ' + str(gradient_boosting.score(X_train, y_train)))

print('Accuracy: '+ str(accuracy_score(y_test, Y_pred)))

print('Confusion Matrix:')

print(confusion_matrix(y_test, Y_pred))

print(classification_report(y_test, Y_pred))

print('------------------------------------')





##Decision Tree Classifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

decision_tree.score(X_train, y_train)



print('Decision Tree Classifier')

print('Score: ' + str(decision_tree.score(X_train, y_train)))

print('Accuracy: '+ str(accuracy_score(y_test, Y_pred)))

print('Confusion Matrix:')

print(confusion_matrix(y_test, Y_pred))

print(classification_report(y_test, Y_pred))

print('------------------------------------')





##Decision Tree Classifier

extra_tree = DecisionTreeClassifier()

extra_tree.fit(X_train, y_train)

Y_pred = extra_tree.predict(X_test)

extra_tree.score(X_train, y_train)



print('Decision Tree Classifier')

print('Score: ' + str(extra_tree.score(X_train, y_train)))

print('Accuracy: '+ str(accuracy_score(y_test, Y_pred)))

print('Confusion Matrix:')

print(confusion_matrix(y_test, Y_pred))

print(classification_report(y_test, Y_pred))

print('------------------------------------')



# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=10)





# Modeling step Test differents algorithms 

random_state = 2

classifiers = []



classifiers.append(SVC(random_state=random_state))

classifiers.append(Perceptron(random_state=random_state))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(XGBClassifier())

classifiers.append(MLPClassifier(random_state=random_state))



classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(LinearDiscriminantAnalysis())

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","Perceptron",

"RandomForest","KNeighborsClassifier","LogisticRegression","XGBClassifier","MultipleLayerPerceptron", "AdaBoostClassifier", "LinearDiscriminantAnalysis", "GradientBoosting", "DecisionTree", "ExtraTreesClassifier"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g.set_xlim(xmin=0.6)

g = g.set_title("Cross validation scores")

# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300, 400, 500, 750, 1000],

              'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],

              'max_depth': [4, 8, 16, 32, 64, 128],

              'min_samples_leaf': [100,150, 200, 250],

              'max_features': [0.3, 0.1, 0.05, 0.01] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

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
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,y_train,cv=kfold)
## Choose the best algorithms (ensembled mode)

model = gsGBC



ids = df_test['PassengerId']



##Result

prediction = model.predict(df_test.drop('PassengerId', axis=1))

out_pred = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediction })

df_sup = pd.DataFrame({"PassengerId":[1044], "Survived":[0]}) 

out_pred = out_pred.append(df_sup)

out_pred = out_pred.sort_values(by='PassengerId', ascending=True)



out_pred.to_csv('Submission.csv', index = False)