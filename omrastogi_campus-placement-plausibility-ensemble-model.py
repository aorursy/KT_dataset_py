# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

from matplotlib import pyplot as plt 

%matplotlib inline
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.head()
df.info()
df1 = df.drop(columns=['salary','sl_no'], axis=1)
dummy = pd.get_dummies(df['status'])

dummy.drop(columns=['Not Placed'],axis=1, inplace=True)

df1 = pd.concat([df1,dummy],axis =1)

df1.head()
df1['Placed'].value_counts().to_frame().style.bar()
ax = sns.heatmap(df1.corr(),annot=True)
g = sns.factorplot(x="gender",y="Placed",data= df1,kind="bar", size = 6)

g.despine(left=True)

g = g.set_ylabels("placement probability")
g = sns.factorplot(x="ssc_b",y="Placed",data= df1,kind="bar", size = 6)

g.despine(left=True)

g = g.set_ylabels("placement probability")
g = sns.factorplot(x="hsc_b",y="Placed",data= df1,kind="bar", size = 6)

g.despine(left=True)

g = g.set_ylabels("placement probability")
g = sns.factorplot(x="hsc_s",y="Placed",data= df1,kind="bar", size = 6)

g.despine(left=True)

g = g.set_ylabels("placement probability")
g = sns.factorplot(x="degree_t",y="Placed",data= df1,kind="bar", size = 6)

g.despine(left=True)

g = g.set_ylabels("placement probability")
g = sns.factorplot(x="workex",y="Placed",data= df1,kind="bar", size = 6)

g.despine(left=True)

g = g.set_ylabels("placement probability")
g = sns.factorplot(x="specialisation",y="Placed",data= df1,kind="bar", size = 6)

g.despine(left=True)

g = g.set_ylabels("placement probability")
g = sns.kdeplot(df1["ssc_p"][(df1["Placed"] == 0) & (df1["ssc_p"].notnull())], color="Red", shade = True)

g = sns.kdeplot(df1["ssc_p"][(df1["Placed"] == 1) & (df1["ssc_p"].notnull())], color="Blue", shade = True)

g.set_xlabel("Marks in secondary school")

g.set_ylabel("Placement")

g = g.legend(["Not Placed","Placed"])
g = sns.kdeplot(df1["hsc_p"][(df1["Placed"] == 0) & (df1["hsc_p"].notnull())], color="Red", shade = True)

g = sns.kdeplot(df1["hsc_p"][(df1["Placed"] == 1) & (df1["hsc_p"].notnull())], color="Blue", shade = True)

g.set_xlabel("Marks in high school")

g.set_ylabel("Placement")

g = g.legend(["Not Placed","Placed"])
g = sns.kdeplot(df1["degree_p"][(df1["Placed"] == 0) & (df1["degree_p"].notnull())], color="Red", shade = True)

g = sns.kdeplot(df1["degree_p"][(df1["Placed"] == 1) & (df1["degree_p"].notnull())], color="Blue", shade = True)

g.set_xlabel("Marks in secondary school")

g.set_ylabel("Placement")

g = g.legend(["Not Placed","Placed"])
g = sns.catplot(x="Placed",y= 'degree_p', order=[0, 1],data=df1)
g = sns.kdeplot(df1["etest_p"][(df1["Placed"] == 0) & (df1["etest_p"].notnull())], color="Red", shade = True)

g = sns.kdeplot(df1["etest_p"][(df1["Placed"] == 1) & (df1["etest_p"].notnull())], color="Blue", shade = True)

g.set_xlabel("Marks in secondary school")

g.set_ylabel("Placement")

g = g.legend(["Not Placed","Placed"])
g = sns.catplot(x="Placed",y= 'etest_p', order=[0, 1],data=df1)
g = sns.kdeplot(df1["mba_p"][(df1["Placed"] == 0) & (df1["mba_p"].notnull())], color="Red", shade = True)

g = sns.kdeplot(df1["mba_p"][(df1["Placed"] == 1) & (df1["mba_p"].notnull())], color="Blue", shade = True)

g.set_xlabel("Marks in secondary school")

g.set_ylabel("Placement")

g = g.legend(["Not Placed","Placed"])
g = sns.catplot(x="Placed",y= 'mba_p', order=[0, 1],data=df1)
df1.head()
df_feature = df1[['ssc_p','hsc_p','degree_p']]
df1["workex"] = df1["workex"].astype('category')

df_feature["workex"] = df1["workex"].cat.codes

#No=0, Yes=1
df1["specialisation"] = df1["specialisation"].astype('category')

df_feature["specialisation"] = df1["specialisation"].cat.codes

# Mkt&Fin=0 | Mkt&HR=1 
df1["gender"] = df1["gender"].astype('category')

df_feature["gender"] = df1["gender"].cat.codes

# F=0 | M=1 
hsc_dummy = pd.get_dummies(df1['hsc_s'], prefix='hsc')

df_feature = pd.concat([df_feature,hsc_dummy['hsc_Arts']],axis =1)

#hsc_Arts - Yes=1, No=0
deg_dummy = pd.get_dummies(df1['degree_t'], prefix='degree_in')

df_feature = pd.concat([df_feature,deg_dummy['degree_in_Others']],axis =1)

#degree_in_other - Yes=1, No=0
df_feature.head()
df_feature.dtypes
df_feature['ssc_p'] = df_feature['ssc_p']/df_feature['ssc_p'].max()

df_feature['hsc_p'] = df_feature['hsc_p']/df_feature['hsc_p'].max()

df_feature['degree_p'] = df_feature['degree_p']/df_feature['degree_p'].max()
df_feature.head()
df_feature['Placed'] = df1['Placed']

ax = sns.heatmap(df_feature.corr(),annot=True)

df_feature.drop(columns='Placed',axis=1,inplace=True)
df_feature.drop(columns=['gender','hsc_Arts'],axis=1,inplace=True)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

X_train, X_test, y_train, y_test = train_test_split(df_feature,df1['Placed'],test_size = 0.1,random_state=5)
clsf = LogisticRegression(C=3)

clsf.fit(X_train, y_train)

score = clsf.score(X_test,y_test)

print (score)
print(cross_val_score(clsf, df_feature, df1['Placed'], cv=10))

print (cross_val_score(clsf, df_feature, df1['Placed'], cv=10).mean())
accuracy = []

skf = StratifiedKFold(n_splits = 10,random_state=2)

score = cross_val_score(clsf, X_train, y = y_train, scoring = "accuracy", cv = skf, n_jobs=4)

print (score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier,VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
kfold = StratifiedKFold(n_splits = 10,random_state=2)

classifiers =[]

r=2



classifiers.append(LogisticRegression(random_state=r))

classifiers.append(DecisionTreeClassifier(random_state=r))

classifiers.append(MLPClassifier(random_state=r))

classifiers.append(RandomForestClassifier(random_state=r))

classifiers.append(AdaBoostClassifier(RandomForestClassifier(random_state=r),random_state=r,learning_rate=0.1))

classifiers.append(ExtraTreesClassifier(random_state=r))

classifiers.append(GradientBoostingClassifier(random_state=r))

classifiers.append(SVC(random_state=r))

classifiers.append(KNeighborsClassifier())
cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

    

cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["LogisticRegression","DecisionTree","MultipleLayerPerceptron",

                                                                                       "RandomForest","Adaboost","ExtraTree","GradientBoosting","SVC","KNeighboors"]})

cv_res.head()
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(X_train,y_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
adaRFC = AdaBoostClassifier(RFC_best, random_state=7)



ada_param_grid = {"algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaRFC = GridSearchCV(adaRFC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsadaRFC.fit(X_train,y_train)

adaRFC_best = gsadaRFC.best_estimator_

gsadaRFC.best_score_
ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsExtC.fit(X_train,y_train)



ExtC_best = gsExtC.best_estimator_



# Best score

gsExtC.best_score_
KNC = KNeighborsClassifier()

kn_param_grid = {'n_neighbors':[4,5,6,7,8,9],

                 'weights':["uniform", "distance"],

                 'algorithm':["auto", "ball_tree", "kd_tree", "brute"],

                 'metric':['euclidean','manhattan']

                }



gsKNC = GridSearchCV(KNC, param_grid= kn_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)



gsKNC.fit(X_train,y_train)



KNC_best = gsKNC.best_estimator_

gsKNC.best_score_
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train,y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
X = df_feature

y = df1['Placed']

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



g = plot_learning_curve(gsRFC.best_estimator_,"RF learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gsadaRFC.best_estimator_,"AdaBoost learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gsKNC.best_estimator_,"KNeighbor learning curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gsGBC.best_estimator_,"Gradient Boosting learning curves",X_train,y_train,cv=kfold)
nrows = 1

ncols = 3

fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(21,7))



names_classifiers = [("AdaBoosting", adaRFC_best),("RandomForest",RFC_best),("ExtraTrees",ExtC_best)]



nclassifier = 0

for row in range(nrows):

    for col in range(ncols):

        name = names_classifiers[nclassifier][0]

        classifier = names_classifiers[nclassifier][1]

        indices = np.argsort(classifier.feature_importances_)[::-1][:6]

        # [:6] is mentioned as there are only six feature, while the function can return more indexs than the feature. 

        print (name,indices)

        x_ =  classifier.feature_importances_[indices][:6]

        if name == "GradientBoosting": 

            print (x_)

        g = sns.barplot(y=X_train.columns[indices][:6],x = x_ , orient='h',ax=axes[col])



        g.set_xlabel("Relative importance",fontsize=12)

        g.set_ylabel("Features",fontsize=12)

        g.tick_params(labelsize=9)

        g.set_title(name + " feature importance")

        nclassifier += 1
g = plot_learning_curve(gsadaRFC.best_estimator_,"Adaboost RFC learning curves",X_train,y_train,cv=kfold)

print ("Score of the model",adaRFC_best.score(X,y))
Placement_predict = pd.Series(adaRFC_best.predict(X), name="Survived")



Pridiction = pd.concat([df[sl_no],Placement_predict],axis=1)



results.to_csv("python_ensemble_prediction.csv",index=False)