import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



sns.set(style='white', context='notebook', palette='deep')



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/data-ghoul/train.csv")

test = pd.read_csv("/kaggle/input/data-ghoul/test.csv")

IDtest = test["id"]
## Outlier detection 



def detect_outliers(df,n,features):

    outlier_indices = []

    

    for col in features:

        Q1 = np.percentile(df[col], 25)

        Q3 = np.percentile(df[col],75)

        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



## Detect outliers from Age, SibSp , Parch and Fare (numerical features)

Outliers_to_drop = detect_outliers(train,2,["bone_length","rotting_flesh","hair_length","has_soul"])
# Show the outliers rows

train.loc[Outliers_to_drop]
## Join train and test data to obtain the same number of features during categorical conversion

train_len = len(train)

dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
dataset.head()
## Fill empty and NaNs values with NaN

dataset = dataset.fillna(np.nan)



## Check for Null values

dataset.isnull().sum()
train.isnull().sum()
train.dtypes
## Replace string to int

train["type_int"] = train["type"].replace({

    "Ghoul":1,

    "Goblin":2,

    "Ghost":3

})
## Correlation matrix

sns.heatmap(train[["type_int", "bone_length", "rotting_flesh", "hair_length", "has_soul"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
## Distribution plot

def f_dist_plot(col):

    g = sns.FacetGrid(train, col="type")

    g = g.map(sns.distplot, col)

    

f_dist_plot("bone_length")
## KDE Plot 

def f_kde_plot(col):

    g = sns.kdeplot(train[col][(train["type_int"] == 1) & (train[col].notnull())], color="Red", shade = True)

    g = sns.kdeplot(train[col][(train["type_int"] == 2) & (train[col].notnull())], ax =g, color="Blue", shade= True)

    g = sns.kdeplot(train[col][(train["type_int"] == 3) & (train[col].notnull())], ax =g, color="Green", shade= True)

    g.set_xlabel(col)

    g.set_ylabel("Frequency")

    g = g.legend(["Ghoul","Goblin","Ghost"])



f_kde_plot("bone_length")
## Distribution plot

f_dist_plot("rotting_flesh")
## KDE plot

f_kde_plot("rotting_flesh")
## Distribution plot

f_dist_plot("hair_length")
## KDE plot

f_kde_plot("hair_length")
## Distribution plot

f_dist_plot("has_soul")
## KDE plot

f_kde_plot("has_soul")
## Scatter plot has_soul vs hair_length

sns.scatterplot(x="hair_length", y="has_soul", data=train, hue="type")
## Plot color vs type

fig, axs = plt.subplots(3, 2, figsize=(10,8))



def plot_color(x, y, color, color_bar):

    df_color = train[train["color"] == color].groupby(["type"]).size()

    axs[x,y].bar(df_color.index.values, df_color.values, color=color_bar)

    axs[x,y].set_title(color)



plot_color(0,0,"clear", "beige")

plot_color(0,1,"green", "g")

plot_color(1,0,"black", "k")

plot_color(1,1,"white", "whitesmoke")

plot_color(2,0,"blue", "b")

plot_color(2,1,"blood", "r")

plt.tight_layout()
## Convert feature color to binary

dataset = pd.get_dummies(dataset, columns=["color"])
## Convert type from string to int

dataset["type"] = dataset["type"].replace({

    "Ghoul":1,

    "Goblin":2,

    "Ghost":3

})
dataset.head()
## Separate train dataset and test dataset

train = dataset[:train_len]

test = dataset[train_len:]



## Drop type and id label

test.drop(["type", "id"],axis = 1,inplace=True)
## Separate X and y label

train["type"] = train["type"].astype(int)



Y_train = train["type"]

X_train = train.drop(["type", "id"], axis = 1)
#KFold Stratified cross val

kfold = StratifiedKFold(n_splits=10)
## Modeling step Test differents algorithms 

random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})
## Plot score of cross validation models

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
cv_res
## Hyperparameter Cross Validation Models

classifiers
## LinearDiscriminantAnalysis

LDA = LinearDiscriminantAnalysis()





## Search grid for optimal parameters

ex_param_grid = {"n_components": [None, 1, 2, 3, 4]}





gsLDA = GridSearchCV(LDA,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsLDA.fit(X_train,Y_train)



LDA_best = gsLDA.best_estimator_



## Best score

gsLDA.best_score_
## Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train,Y_train)



GBC_best = gsGBC.best_estimator_



## Best score

gsGBC.best_score_
## MultipleLayerPerceptron

MLP = MLPClassifier()





## Search grid for optimal parameters

ex_param_grid = {

    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],

    'activation': ['tanh', 'relu'],

    'solver': ['sgd', 'adam'],

    'alpha': [0.0001, 0.05],

    'learning_rate': ['constant','adaptive'],

}





gsMLP = GridSearchCV(MLP, param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsMLP.fit(X_train,Y_train)



MLP_best = gsMLP.best_estimator_



## Best score

gsMLP.best_score_
## LogisticRegression

LRC = LogisticRegression()





## Search grid for optimal parameters

ex_param_grid = {

    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]

}





gsLRC = GridSearchCV(LRC, param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsLRC.fit(X_train,Y_train)



LRC_best = gsLRC.best_estimator_



## Best score

gsLRC.best_score_
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

    return plt
## Plot learning curves of best estimators

g = plot_learning_curve(gsLDA.best_estimator_,"Linear Discriminant Analysis learning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsGBC.best_estimator_,"Gradient Boosting learning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsMLP.best_estimator_,"MLP learning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsLRC.best_estimator_,"Logistic Regression learning curves",X_train,Y_train,cv=kfold)
## Plot Feature Importance of Gradiect Boosting



plt.figure(figsize=(10,10))

names_classifiers = [("GBC",GBC_best)]

nclassifier = 0

name = names_classifiers[nclassifier][0]

classifier = names_classifiers[nclassifier][1]

indices = np.argsort(classifier.feature_importances_)[::-1][:40]

g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h')

g.set_xlabel("Relative importance",fontsize=12)

g.set_ylabel("Features",fontsize=12)

g.tick_params(labelsize=9)

g.set_title(name + " feature importance")        
## Corealation between best models



test_type_LDA = pd.Series(LDA_best.predict(test), name="LDA")

test_type_GBC = pd.Series(GBC_best.predict(test), name="GBC")

test_type_MLP = pd.Series(MLP_best.predict(test), name="MLP")

test_type_LRC = pd.Series(LRC_best.predict(test), name="LRC")



concatenate_results = pd.concat([test_type_LDA,test_type_GBC,test_type_MLP,test_type_LRC],axis=1)



g= sns.heatmap(concatenate_results.corr(),annot=True)
concatenate_results.corr()
## Voting Classifier

votingC = VotingClassifier(estimators=[('lda', LDA_best), ('gbc', GBC_best),

('mlp', MLP_best), ('lrc',LRC_best)], voting='soft', n_jobs=4)



votingC = votingC.fit(X_train, Y_train)
## Voting Classifier Model Parameter

votingC
## Predict test data

test_type = pd.Series(votingC.predict(test), name="Type")

test_type = test_type.replace({

    1:"Ghoul",

    2:"Goblin",

    3:"Ghost"

})

results = pd.concat([IDtest,test_type],axis=1)

results.to_csv("voting_prediction.csv",index=False)