import sys

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn as sk



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn.metrics         import accuracy_score

from sklearn.metrics         import confusion_matrix

from sklearn.linear_model    import LogisticRegression

from sklearn.neighbors       import KNeighborsClassifier

from sklearn.tree            import DecisionTreeClassifier

from sklearn.naive_bayes     import GaussianNB

from sklearn.svm             import SVC

from sklearn.ensemble        import RandomForestClassifier

from sklearn.ensemble        import ExtraTreesClassifier

from sklearn.ensemble        import AdaBoostClassifier

from sklearn.ensemble        import GradientBoostingClassifier

from sklearn.ensemble        import VotingClassifier



print('*'*50)

#print('Python Version    : ', sys.version)

print('Pandas Version    : ', pd.__version__)

print('Numpy Version     : ', np.__version__)

print('Matplotlib Version: ', mpl.__version__)

print('Seaborn Version   : ', sns.__version__)

print('SKLearn Version   : ', sk.__version__)

print('*'*50)
sns.set_style('whitegrid')



pd.options.display.max_rows = 100

pd.options.display.max_columns = 100



#Reproducibility!

seed      = 42

v_size    = 0.33

num_folds = 10

scoring   = 'accuracy'



#random seeds

np.random.seed(seed)
def missingData(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    md = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    md = md[md["Percent"] > 0]

    plt.figure(figsize = (8, 4))

    plt.xticks(rotation='90')

    sns.barplot(md.index, md["Percent"],color="g",alpha=0.8)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)

    return md



def valueCounts(dataset, features):

    """Display the features value counts """

    for feature in features:

        vc = dataset[feature].value_counts()

        print(vc)



def correlationHeatmap(df, title):

    plt.figure(figsize =(20, 14))

    sns.heatmap(df.corr(),annot=True,cmap='coolwarm',linewidths=0.2)

    plt.title(title, size=25)

    plt.xticks(size=15)

    plt.yticks(size=15)

    plt.show()



#Spot-Check functions



def algoSpotCheck(models, X_train, y_train, num_folds, scoring, seed):

    """Makes a spot-check of the models"""

    results = []

    names   = []

    for name, model in models:

        kfold = KFold(n_splits=num_folds, random_state=seed)

        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

        results.append(cv_results)

        names.append(name)

        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

        print(msg)

        print("*"*60)

    return names, results



def boxplotCompare(names, results, title):

    """Generate a boxplot of the models"""

    fig = plt.figure(figsize=(20, 14)) 

    ax = fig.add_subplot(111)

    sns.boxplot(data=results)

    ax.set_xticklabels(names) 

    plt.title('Comparison between Algorithms', size = 40, color='k')

    plt.xlabel('Percentage',size = 20,color="k")

    plt.ylabel('Algorithm',size = 20,color="k")

    plt.xticks(size=15)

    plt.yticks(size=15)

    plt.show()



#Hyperparameter Tuning Function



def algoGridTune(model, param_grid, X_train, y_train, num_folds, scoring, seed):

    """Makes the hyperparameter tuning of the chosen model"""

    kfold = KFold(n_splits=num_folds, random_state=seed)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=-1)

    grid_result = grid.fit(X_train, y_train)

    best_estimator = grid_result.best_estimator_

    print("BestScore: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    print("*"*60)

    means = grid_result.cv_results_['mean_test_score']

    stds = grid_result.cv_results_['std_test_score']

    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):

        print("Score: %f (%f) with: %r" % (mean, stdev, param))

        print("*"*60)

    return best_estimator



def plotLearningCurve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure(figsize=(15,10))

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std  = np.std(train_scores, axis=1)

    test_scores_mean  = np.mean(test_scores, axis=1)

    test_scores_std   = np.std(test_scores, axis=1)

    

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")

    plt.show()



def plotFeatureImportance(best_model, datacols, title):

    plt.figure(figsize=(10,4))

    pd.Series(best_model.feature_importances_,datacols).sort_values(ascending=True).plot.barh(width=0.8)

    plt.title(title)

    plt.show()
path_train = '../input/train.csv'

path_test = '../input/test.csv'



df_train = pd.read_csv(path_train)

df_test  = pd.read_csv(path_test)
df_train.head(5)
df_test.head(5)
df_train.describe()
df_test.describe()
df_train.info()
df_test.info() #Since this is the test set, the Survivors class is missing
print('Datasets shapes: ')

print("Training shape: ", df_train.shape)

print("Test shape    : ", df_test.shape)
missingData(df_train)
missingData(df_test)
df_train.drop("Cabin", axis=1, inplace = True)

df_test.drop("Cabin", axis=1, inplace = True)
df_train["Age"].fillna(df_train["Age"].median(), inplace = True)

df_test["Age"].fillna(df_test["Age"].median(),  inplace = True)
df_test["Fare"].fillna(df_test['Fare'].median(), inplace = True)
df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])
def displayNanValues():

    print("Check the NaN value in train data")

    print(df_train.isnull().sum())

    print("---"*30)

    print("Check the NaN value in test data")

    print(df_test.isnull().sum())

    print("---"*30)
displayNanValues()
all_data = [df_train, df_test]
def featureExtraction(all_data):

    

    # Create new feature FamilySize as a combination of SibSp and Parch

    for dataset in all_data:

        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 #+1 because it indicates the person of the i-th row

        

    # Create bin for age features

    for dataset in all_data:

        dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,18,60,120], labels=['Children','Adult','Elder'])

        

    #Create a Title feature ...

    for dataset in all_data:

        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

    #And replaces old titles with new ones

    for dataset in all_data:

        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
featureExtraction(all_data)
dfTrain = df_train.copy()

dfTest  = df_test.copy()
traindf = pd.get_dummies(dfTrain, columns = ["Pclass","Title",'FamilySize',"Sex","Age_bin","Embarked"],

                             prefix=["Pclass","Title",'FamilySize',"Sex","Age_type","Em_type"])



testdf = pd.get_dummies(dfTest, columns = ["Pclass","Title",'FamilySize',"Sex","Age_bin","Embarked"],

                             prefix=["Pclass","Title",'FamilySize',"Sex","Age_type","Em_type"])
allData = [traindf, testdf]
traindf.head()
for dataset in allData:

    drop_column = ["Age","Fare","Name","Ticket","SibSp","Parch"]

    dataset.drop(drop_column, axis=1, inplace = True)



traindf.drop(["PassengerId"], axis=1, inplace = True)
correlationHeatmap(traindf, 'Pearson Correlation of Features')
#Random Shuffle

df_data_shuffled = traindf.iloc[np.random.permutation(len(traindf))]
df_data_shuffled.head()
df_data_shuffled.describe()
array = df_data_shuffled.values #convert into array the train set



features = array[:,1:].astype(float)

targeted = array[:,0].astype(float)
X_train,X_test,y_train,y_test = train_test_split(features,targeted,test_size=v_size,random_state=seed)



print('Data shapes: ')

print("X_train shape: ", X_train.shape)

print("X_test shape : ", X_test.shape)

print("y_train shape: ", y_train.shape)

print("y_test shape : ", y_test.shape)
# Spot-Check Algorithms



models = [('LR', LogisticRegression(solver='liblinear')),

          ('KNN', KNeighborsClassifier()),

          ('CART', DecisionTreeClassifier()),

          ('NB', GaussianNB()),

          ('SVM', SVC(gamma='auto')),     

         ]



names,results = algoSpotCheck(models,X_train,y_train,num_folds,scoring,seed)

boxplotCompare(names, results, 'Comparison_beetween_Algorithms0')
lr_clf = LogisticRegression()

lr_param_grid = {'solver' : ['liblinear', 'lbfgs'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

#lr_param_grid = {'solver' : ['liblinear'],'C': [1] }

best_lr = algoGridTune(lr_clf, lr_param_grid, X_train, y_train,num_folds, scoring, seed)
best_lr
knn_clf = KNeighborsClassifier()

knn_param_grid = {'n_neighbors':[3,5,7,9,11],

              'leaf_size':[1,2,3,5],

              'weights':['uniform', 'distance'],

              'algorithm':['auto', 'ball_tree','kd_tree','brute']

             }





best_knn = algoGridTune(knn_clf, knn_param_grid, X_train, y_train, num_folds, scoring, seed)
best_knn
dt_clf = DecisionTreeClassifier()

dt_param_grid = {'max_depth' : [3,4,5,6,7,8,9,10],

              'max_features': ['sqrt', 'log2'],

              'min_samples_split': [3,5,7,9,11], 

              'min_samples_leaf':[1,3,5,7,9,11]

             }



best_dt = algoGridTune(dt_clf, dt_param_grid, X_train, y_train,num_folds, scoring, seed)
best_dt
svc_clf = SVC()



svc_param_grid = [{"kernel": ["rbf"], 

                   "gamma": [10 ,1, 0.1, 1e-2, 1e-3],

                   "C": [0.1,1,10],

                   "random_state" : [seed]},

                  {"kernel": ["linear"], "C": [0.1,1,10,100]}

                 ]



best_SVC = algoGridTune(svc_clf, svc_param_grid, X_train, y_train, num_folds, scoring, seed)
best_SVC
# Spot-Check Algorithms



models = [('RFC', RandomForestClassifier(n_estimators=100)),

          ('ETC', ExtraTreesClassifier(n_estimators=100)),

          ('ABC', AdaBoostClassifier(n_estimators=100)),

          ('GBC', GradientBoostingClassifier(n_estimators=100))

           ]



names,results = algoSpotCheck(models,X_train,y_train,num_folds,scoring,seed)

boxplotCompare(names, results, 'Comparison_beetween_Algorithms1')
# Adaboost

DTC = DecisionTreeClassifier()

ABC_clf = AdaBoostClassifier(DTC, random_state=seed)



ABC_param_grid = {"base_estimator__criterion" : ["gini"],

                  "base_estimator__splitter" :   ["best"],

                  "algorithm" : ["SAMME"],

                  "n_estimators" :[30, 100],

                  "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}





best_ABC = algoGridTune(ABC_clf, ABC_param_grid, X_train, y_train, num_folds, scoring, seed)
best_ABC
#ExtraTrees 

ETC_clf = ExtraTreesClassifier()



 #Search grid for optimal parameters

ETC_param_grid = {"max_depth": [30],

                  "max_features": ['sqrt'],

                  "min_samples_split": [2, 3, 5],

                  "min_samples_leaf": [1, 3, 5],

                  "bootstrap": [True],

                  "n_estimators" :[300],

                  "criterion": ["gini"]}





best_ETC = algoGridTune(ETC_clf, ETC_param_grid, X_train, y_train, num_folds, scoring, seed)
best_ETC




GBC_clf = GradientBoostingClassifier()

GBC_param_grid = {'loss' : ["deviance"],

                  'n_estimators' : [100, 200, 300],

                  'learning_rate': [0.1, 0.05, 0.01],

                  'max_depth': [3, 5, 7],

                  'min_samples_leaf': [1, 5, 9],

                  'min_samples_split': [2, 6, 10],

                  'max_features': ['sqrt', 'log2'] 

                 }



best_GBC = algoGridTune(GBC_clf, GBC_param_grid, X_train, y_train, num_folds, scoring, seed)
best_GBC
RFC_clf = RandomForestClassifier()



RFC_param_grid = {"max_depth": [None],

                 "max_features": [1, 3, 10],

                 "min_samples_split": [2, 3, 10],

                 "min_samples_leaf": [1, 3, 10],

                 "bootstrap": [True, False],

                 "n_estimators" :[100,300],

                 "criterion": ["gini"]}







best_RFC = algoGridTune(RFC_clf, RFC_param_grid, X_train, y_train, num_folds, scoring, seed)
best_RFC

kfold = KFold(n_splits=num_folds, random_state=seed)
plotLearningCurve(best_lr,"LogisticRegression learning curves",X_train,y_train,cv=kfold)

plotLearningCurve(best_knn,"K-NearestNeighbor learning curves",X_train,y_train,cv=kfold)

plotLearningCurve(best_dt,"DecisionTree learning curves",X_train,y_train,cv=kfold)

plotLearningCurve(best_SVC,"SupportVectorClassifier learning curves",X_train,y_train,cv=kfold)
plotLearningCurve(best_RFC,"RandomForest learning curves",X_train,y_train,cv=kfold)

plotLearningCurve(best_ETC,"EtraTrees learning curves",X_train,y_train,cv=kfold)

plotLearningCurve(best_ABC,"AdaBoost learning curves",X_train,y_train,cv=kfold)

plotLearningCurve(best_GBC,"GradientBoosting learning curves",X_train,y_train,cv=kfold)
datacols = list(traindf.drop("Survived", axis=1))
plotFeatureImportance(best_RFC,datacols, 'Feature Importance for Best_RFC')
plotFeatureImportance(best_ETC,datacols, 'Feature Importance for Best_ETC')
plotFeatureImportance(best_ABC,datacols, 'Feature Importance for Best_ABC')
plotFeatureImportance(best_GBC,datacols, 'Feature Importance for Best_GBC')
passengerIds = testdf["PassengerId"].copy()

testdf.drop(["PassengerId"], axis=1, inplace = True)
test = testdf.values
test_Survived_lr = pd.Series(best_lr.predict(test), name="LR")

test_Survived_knn = pd.Series(best_knn.predict(test), name="KNN")

test_Survived_dt = pd.Series(best_dt.predict(test), name="DT")

test_Survived_SVC = pd.Series(best_SVC.predict(test), name="SVC")

test_Survived_ABC = pd.Series(best_ABC.predict(test), name="ABC")

test_Survived_RFC = pd.Series(best_RFC.predict(test), name="RFC")

test_Survived_ETC = pd.Series(best_ETC.predict(test), name="ETC")

test_Survived_GBC = pd.Series(best_GBC.predict(test), name="GBC")



# Concatenate all classifier results



ensemble_results = pd.concat([test_Survived_lr, test_Survived_knn, 

                              test_Survived_dt, test_Survived_SVC,

                              test_Survived_ABC, test_Survived_RFC,

                              test_Survived_ETC, test_Survived_GBC],axis=1)





correlationHeatmap(ensemble_results, 'Correlation beetween models results')
votingC = VotingClassifier(estimators=[('lr', best_lr),

                                       ('knn', best_knn),

                                       ('dt', best_dt),

                                       ('svc', best_SVC),

                                       ('abc', best_ABC),

                                       ('rfc', best_RFC),

                                       ('etc', best_ETC),

                                       ('gbc', best_GBC)], 

                           voting='hard', n_jobs=-1)



votingC = votingC.fit(X_train, y_train)
votingC
predictions = votingC.predict(test)

test_Survived = pd.Series(votingC.predict(test), name="Survived")

test_Survived = test_Survived.apply(int)

results = pd.concat([passengerIds,test_Survived],axis=1)

results.to_csv('submission.csv', columns=['PassengerId', 'Survived'], index=False)
results.head()