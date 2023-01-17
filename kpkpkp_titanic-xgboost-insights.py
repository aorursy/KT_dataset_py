# Data preprocessing
import pandas as pd
import numpy as np
import math
from sklearn.neighbors.kde import KernelDensity
from matplotlib import pyplot as plt
%matplotlib inline 
import seaborn as sns
# XGboost
from xgboost import DMatrix, XGBClassifier, cv
from xgboost import plot_importance, plot_tree
# Testing
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#Else
import datetime

def find_unique(table_name):
    # Find unique values
    unique = {}
    print("-"*90)
    print('{:<15s}{:10s}{:10s}{:<s}'.format("COLUMN","IS NAN","DISPLAYED","UNIQUE VALUES"))
    print("-"*90)
    for label in table_name.axes[1]:
        unique[label] = table_name[label].unique()
        isnan = ""
        for a in unique[label]:
            try:
                if math.isnan(a):
                    isnan = "HAS NAN"
            except:
                pass
        if len(unique[label]) > 10:
            displayed = "[10/" + str(len(unique[label])) + "]"
            unique_data = str(unique[label][0:10])
        else:
            displayed = "[" + str(len(unique[label])) + "/" + str(len(unique[label])) +"]"
            unique_data = str(unique[label])
        print('{:<15s}{:10s}{:10s}{:<s}'.format(str(label),isnan,displayed,unique_data))
PREDICTORS = ['Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TARGET = 'Survived'
ID_COLUMN = 'PassengerId'

download_data_train = pd.read_csv("../input/train.csv")
download_data_test = pd.read_csv("../input/test.csv")

data_train_labels = pd.DataFrame(download_data_train, columns = [ID_COLUMN, TARGET])

# create new matrices for more convenient data manipulation
download_data_train["trainOrTest"] = "train"
download_data_test["trainOrTest"] = "test"
download_data_train_sub = download_data_train.drop([TARGET], axis=1)
data_all = pd.concat([download_data_train_sub, download_data_test])
find_unique(data_all) #data overview 1
data_all.describe() #data overview 2
data_all.describe(include=['O']) #data overview 3
data_all.head(5) #data overview 4
for column in ['Pclass','Sex', 'Parch', 'SibSp', 'Embarked']:
        print('-'*30)
        print(download_data_train[[column, 'Survived']].groupby(column, as_index=False).mean().sort_values(by='Survived', ascending=False))
data_all["Sex"] = data_all["Sex"].apply(lambda x: 1 if x == "male" else 0)
data_all["Fare"] = data_all["Fare"].apply(lambda x: x if not math.isnan(x) else np.mean(data_all['Fare']))
data_all["Embarked"] = data_all["Embarked"].apply(lambda x: 2 if x == "C" else (1 if x == 'Q' else 0))
data_all['Title'] = data_all.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(data_all['Title'], data_all['Sex'])
data_all['Title'] = data_all['Title'].replace('Mlle', 'Miss')
data_all['Title'] = data_all['Title'].replace('Ms', 'Miss')
data_all['Title'] = data_all['Title'].replace('Mme', 'Mrs')
data_test_title = data_all[[ID_COLUMN,'Title']].merge(data_train_labels)
data_test_title[['Title', TARGET]].groupby(['Title'], as_index=False).mean()
title_map = {'Mrs': 4, 'Miss': 3, 'Master': 2, 'Mr' : 1, 'Rare' : 0}
data_all['Title'] = data_all['Title'].apply(lambda x: x if x in ['Master', 'Miss', 'Mr', 'Mrs'] else 'Rare')
data_all['Title'] = data_all['Title'].map(title_map)
data_all = data_all.drop(["Name"], axis=1)
corr = data_all.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
# new data has examples with non-nan Age
new_data = pd.DataFrame(data_all, columns = ['Pclass','Age'])
new_data["Age"] = new_data["Age"].apply(lambda x: 0 if math.isnan(x) else x)
new_data = new_data[new_data["Age"] != 0]

class1 = new_data[new_data['Pclass'] == 1]
class2 = new_data[new_data['Pclass'] == 2]
class3 = new_data[new_data['Pclass'] == 3]

age = sns.FacetGrid(new_data, col='Pclass')
age.map(plt.hist, 'Age', bins=int(180/7))
tt = np.random.rand(len(class1)) < 0.8
class1_train = class1[tt]
class1_test = class1[~tt]

tt = np.random.rand(len(class2)) < 0.8
class2_train = class2[tt]
class2_test = class2[~tt]

tt = np.random.rand(len(class3)) < 0.8
class3_train = class3[tt]
class3_test = class3[~tt]

prob_sets = [[class1_train,class1_test],[class2_train,class2_test],[class3_train,class3_test]]
prob_kernels = ['gaussian','tophat']
kde = [0,0,0]
i = 0
for class_prob in prob_sets:
    act_error = 10e6
    for kernel in prob_kernels:
        new_kde = KernelDensity(kernel=kernel, bandwidth=0.2)
        new_kde.fit(class_prob[0])
        new_error = np.sum(np.matmul(new_kde.score_samples(class_prob[1]),new_kde.score_samples(class_prob[1])))
        if new_error < act_error:
            kde[i] = new_kde
            act_error = new_error
    print('Best kde for' + str(i+1) + 'class id' + str(kde[i]) + ' with average error rate: ' + str(np.sqrt(act_error)/len(class_prob[1])))        
    i += 1
i = 0
#data_all["Age"] = data_all["Age"].apply(lambda x: 0 if math.isnan(x) else x)
#data_all["Age"] = data_all.apply(lambda row: kde[row["Pclass"]-1].sample()[0][1] if row["Age"] == 0 else row["Age"], axis = 1)
i = 0
for class_prob in [class1, class2, class3]:
    act_error = 10e6
    new_kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
    new_kde.fit(class_prob)
    kde[i] = new_kde
    i += 1
data_all["Age"] = data_all["Age"].apply(lambda x: 0 if math.isnan(x) else x)
data_all["Age"] = data_all.apply(lambda row: kde[row["Pclass"]-1].sample()[0][1] if row["Age"] == 0 else row["Age"], axis = 1)
check_cl_1 = [];check_cl_2 = []; check_cl_3 = []
for i in range(5000):
    check_cl_1.append(kde[0].sample()[0][1])
    check_cl_2.append(kde[1].sample()[0][1])
    check_cl_3.append(kde[2].sample()[0][1])
fig, ax = plt.subplots(3,2)
fig.set_size_inches(25, 20)
plt.subplot(3, 2, 1)
plt.title("Probability density [Pclass=1][downloaded data]")
plt.xlabel("Age")
plt.hist(check_cl_1, bins = int(180/7), color = 'r', normed=True)
plt.subplot(3, 2, 2)
plt.title("Probability density [Pclass=1][kde data]")
plt.xlabel("Age")
plt.hist(class1["Age"], bins = int(180/7), color = 'g', normed=True)
plt.subplot(3, 2, 3)
plt.title("Probability density [Pclass=2][downloaded data]")
plt.xlabel("Age")
plt.hist(check_cl_2, bins = int(180/7), color = 'r', normed=True)
plt.subplot(3, 2, 4)
plt.title("Probability density [Pclass=2][generated data]")
plt.xlabel("Age")
plt.hist(class2["Age"], bins = int(180/7), color = 'g', normed=True)
plt.subplot(3, 2, 5)
plt.title("Probability density [Pclass=2][downloaded data]")
plt.xlabel("Age")
plt.hist(check_cl_3, bins = int(180/7), color = 'r', normed=True)
plt.subplot(3, 2, 6)
plt.title("Probability density [Pclass=3][generated data]")
plt.xlabel("Age")
plt.hist(class3["Age"], bins = int(180/7), color = 'g', normed=True)
plt.show()
data_all = data_all.drop(["Cabin"], axis=1)
data_all = data_all.drop(["Ticket"], axis=1)
test_data = data_all[data_all["trainOrTest"] == 'test']
train_data = data_all[data_all["trainOrTest"] == 'train']
train_data = pd.merge(train_data, data_train_labels, on=[ID_COLUMN, ID_COLUMN])
test_data = test_data.drop(["trainOrTest"], axis=1)
train_data = train_data.drop(["trainOrTest"], axis=1)
test_data.describe()
train_data.describe()
def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy (Train): %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

class XgbMod():
    def __init__(self, params=None):
        if params is None:
            self.params = {"n_estimators": 20, "seed": 41, 'max_depth': 8}
            self.classifier = XGBClassifier(self.params)
        else:
            self.params = params
            self.classifier = XGBClassifier(**self.params)

    def trainModel(self, data_train):
        modelfit(self.classifier, data_train, PREDICTORS, TARGET, useTrainCV=True, cv_folds=5, early_stopping_rounds=50)

    def predict(self, data_test_X, data_test_y=None):
        self.data_test_X = data_test_X
        self.Y_pred = self.classifier.predict(data_test_X[PREDICTORS])
        self.Y_pred = pd.DataFrame(data={'Survived': self.Y_pred})
        if data_test_y is not None:
            print("Classifier accuracy score on TEST SET - XGboost - ", str(metrics.accuracy_score(self.Y_pred, data_test_y)))
        return self.Y_pred

    def plotTraining(self):
        plot_importance(self.classifier)
        plot_tree(self.classifier)
        plt.show()

    def savePredictionCSV(self, out_dir):
        time_stamp = datetime.datetime.now().strftime('__%Y_%m_%d__%H_%M_%S.csv')
        out = self.data_test_X.join(self.Y_pred)
        out = pd.DataFrame(out, columns=[ID_COLUMN, TARGET])
        out.to_csv(out_dir + time_stamp, sep=',', encoding='utf-8', index=False)

    def makeCVsearch(self, params, data_train):
        cvsearch = GridSearchCV(estimator=xgbm.classifier, param_grid=params, scoring='roc_auc', n_jobs=4, iid=False,
                             cv=5)
        cvsearch.fit(data_train[PREDICTORS], data_train[TARGET])
        for score in cvsearch.grid_scores_:
            print(score)
        print('-' * 30)
        print(cvsearch.best_params_)
        print('-' * 30)
        print(cvsearch.best_score_)
params = {"n_estimators": 1000,
          "seed": 42}
xgbm = XgbMod(params)
xgbm.trainModel(train_data)
#xgbm.makeCVsearch(param_test_1,data_train_X, data_train_y)
xgbm.plotTraining()
    #param_test_1 = {'max_depth': list(range(6, 12, 1)), 'min_child_weight': list(range(6, 10, 1))}
    #param_test_2 = {'gamma':[i/20.0 for i in range(2,8)]}
    #param_test_3 = {'subsample': [i / 10.0 for i in range(1, 10)], 'colsample_bytree': [i / 10.0 for i in range(1, 10)]}
    #param_test_4 = {'subsample': [i / 100.0 for i in range(25, 35)], 'colsample_bytree': [i / 100.0 for i in range(45, 55)]}
    #param_test_5 = {'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100]}
    #param_test_6 = {'reg_alpha': [0, 1e-10, 1e-6, 1e-5, 1e-4, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
    #param_test_7 = {'learning_rate': [0.01,0.03,0.01,0.03]}
    
    #EXAMPLE
    # params = {"n_estimators": 1000,
    #      "seed": 42}
    # param_test_1 = {'max_depth': list(range(2, 12, 1)), 
    #            'min_child_weight': list(range(6, 10, 1))}
    # xgbm_tuning = XgbMod(params)
    # xgbm_tuning.makeCVsearch(param_test_1,train_data)
params = {"n_estimators": 1000,
          "seed": 27,
          "max_depth": 2,
          "min_child_weight": 2,
          "gamma": 0.3,
          "colsample_bytree": 0.65,
          "subsample": 0.4,
          "reg_alpha": 0
        }
xgbm_tunned = XgbMod(params)
xgbm_tunned.trainModel(train_data)
#xgbm.makeCVsearch(param_test_1,data_train_X, data_train_y)
xgbm_tunned.plotTraining()