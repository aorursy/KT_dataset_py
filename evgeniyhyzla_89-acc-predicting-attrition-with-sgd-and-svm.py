import numpy as np

import pandas as pd

import scipy.stats as st



from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score, roc_auc_score, cohen_kappa_score

 

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.tree import ExtraTreeClassifier

from sklearn.ensemble import BaggingClassifier

from xgboost import XGBClassifier



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from os import path



pd.set_option('display.max_columns', None) # for displaying all columns

np.random.seed(0) # for reproducibility
data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

data.head()
data.describe()
data.info()
fig = plt.figure(figsize=(5, 5))

y = ["No", "Yes"]

ax = sns.categorical.barplot(y, np.array(data.Attrition.value_counts(normalize=True)), saturation=1)

ax.set_xticklabels(y)

ax.set_title("Attrition")

ax.set_xlabel("")

ax.set_ylabel("Frequency")

ax.set_ylim([0,1])

plt.show()
#sns.boxplot(data=data.YearsAtCompany)

#sns.distplot(data.MonthlyIncome)

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(1,1,1)

corr_data = data.select_dtypes(["number"]).corr()

sns.heatmap(corr_data, ax=ax)

ax.tick_params(axis='both', which='major', labelsize=20)

ax.set_title("Pearson correlation map")

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(10, 52))

cols = 3

target_column = "Attrition"

rows = np.ceil(float(data.shape[1] / cols))

for i, column in enumerate(data.columns):

    if target_column == column:

        continue

    ax = fig.add_subplot(rows, cols, i+1)

    ax.set_title(column)

    if data.dtypes[column] == np.object:

        cts = data[[target_column, column]]

        cts = cts.groupby([target_column, column]).size()

        cts.unstack().T.plot(kind="bar", ax=ax, stacked=True, alpha=1)

    else:

        cts = data[[target_column, column]]

        #(xmin, xmax) = (min(cts[column].tolist()), max(cts[column].tolist()))

        cts.groupby(target_column)[column].plot(

            bins=16,

            kind="hist",

            stacked=True,

            alpha=1,

            legend=True,

            ax=ax,

            #range=[xmin, max]

        )

plt.tight_layout()
target_label = "Attrition"



def plot_num(label, data, ax, bins=16):

    d = data[[label, target_label]].sort_values(label).reset_index(drop=True)

    t = np.linspace(data[[label]].min()[label], data[[label]].max()[label], bins)

    

    m = pd.DataFrame({"BINS":np.round((d[[label]].values >= t) * t, 2).max(axis=1)})

    p = pd.concat([d, m], axis=1).groupby(["BINS", target_label]).count().unstack().fillna(0)



    b = p[label]["Yes"] / p[label]["No"]

    ax.bar(b.index, b, width=(t[1] - t[0]) *0.8)

    ax.set_title(label)

    ax.set_ylabel("YES / NO")

    return ax



def plot_cat(label, data, ax, bins=16):

    t = data[[label, target_label]]

    c = pd.DataFrame({"Count":np.ones(len(t), dtype=np.bool)})

    t = pd.concat([t, c], axis=1)

    b = t.groupby([label, target_label]).count().unstack()

    r = b["Count"]["Yes"] / b["Count"]["No"]

    ax.bar(np.arange(len(r)), r, tick_label=r.index)

    ax.set_ylabel("YES / NO")

    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation="vertical")

    ax.set_title(label)

    return ax
fig = plt.figure(figsize=(10, 40))



cols = 3

target_column = "Attrition"

rows = np.ceil(float(data.shape[1] / cols))

for i, column in enumerate(data.columns):

    if target_column == column or column=="Over18":

        continue

    ax = fig.add_subplot(rows, cols, i+1)

    ax.set_title(column)

    plot_func = None

    if data.dtypes[column] == np.object or  len(np.unique(data[column])) <= 16:

        plot_func = plot_cat

    else:

        plot_func = plot_num

    ax = plot_func(column, data, ax)

        

plt.tight_layout()

plt.show()

fig.autofmt_xdate()
fig = plt.figure(2, figsize=(10, 5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

ax1 = plot_num("TotalWorkingYears", data, ax1, 16)

ax2 = plot_num("YearsAtCompany", data, ax2)

plt.tight_layout()

#plt.savefig("YearsAtCompanyVSYearsInCurrentRole.png")
fig = plt.figure(2, figsize=(10, 5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

ax1 = plot_num("MonthlyIncome", data, ax1, 16)

ax2 = plot_cat("JobLevel", data, ax2)

plt.tight_layout()

#plt.savefig("MonthlyIncomeVSJobLevel.png")
fig = plt.figure(2, figsize=(10, 5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

ax1 = plot_cat("PerformanceRating", data, ax1, 16)

ax2 = plot_num("PercentSalaryHike", data, ax2, 16)

#ax1.set_title("MonthlyIncome vs TotalWorkingYears")

#ax1.set_xlabel("PercentSalaryHike")

#ax1.set_ylabel("PerformanceRating")

#ax1.scatter(data["PercentSalaryHike"], data["PerformanceRating"])

plt.tight_layout()

#plt.savefig("PerformanceRatingVSPercentSalaryHike.png")
fig = plt.figure(2, figsize=(10, 5))

ax = fig.add_subplot(1,1,1)

ax.set_title("TotalWorkingYears vs MonthlyIncome")

ax.set_xlabel("TotalWorkingYears")

ax.set_ylabel("MonthlyIncome")

ax.scatter(data["TotalWorkingYears"], data["MonthlyIncome"], c=data[["Attrition"]].eq(["Yes"]).mul(1), cmap=plt.cm.autumn)

#ax1 = fig.add_subplot(1,2,1)

#ax2 = fig.add_subplot(1,2,2)

#ax1 = plot_num("TotalWorkingYears", data, ax1, 16)

#ax2 = plot_num("MonthlyIncome", data, ax2)

plt.tight_layout()

#plt.savefig("TotalWorkingYearsVSMonthlyIncome.png")


fig = plt.figure(2, figsize=(10, 5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)

ax1 = plot_cat("YearsWithCurrManager", data, ax1, 16)

ax2 = plot_num("YearsAtCompany", data, ax2)

plt.tight_layout()

#plt.savefig("YearsWithCurrManagerVYearsAtCompany.png")
fig = plt.figure(2, figsize=(7, 5))

ax1 = fig.add_subplot(1,1,1)

ax1 = plot_num("MonthlyRate", data, ax1, 10)

plt.tight_layout()

#plt.savefig("MonthlyRate.png")
fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(1, 4, 1)

ax1 = plot_cat("OverTime", data, ax1)

ax2 = fig.add_subplot(1, 4, 2)

ax2 = plot_cat("MaritalStatus", data, ax2)

ax3 = fig.add_subplot(1, 4, 3)

ax3 = plot_cat("Gender", data, ax3)

ax4 = fig.add_subplot(1, 4, 4)

ax4 = plot_cat("BusinessTravel", data, ax4)

ax4.set_xticklabels(["No", "Frequently", "Rarely"])



ax1.tick_params(axis='both', which='major', labelsize=15)

ax2.tick_params(axis='both', which='major', labelsize=15)

ax3.tick_params(axis='both', which='major', labelsize=15)

ax4.tick_params(axis='both', which='major', labelsize=15)

plt.tight_layout()

#plt.savefig("Categorial.png")
uniq = data.apply(lambda x: len(np.unique(np.array(x))))

uniq
no_inf = uniq.index[uniq==1]

print(no_inf)

data.drop(labels=no_inf, axis=1, inplace=True)
unuseful_label = ["DailyRate", "EmployeeNumber", "HourlyRate", "MonthlyRate", "PercentSalaryHike", 

                  "PerformanceRating", "TrainingTimesLastYear" , "YearsSinceLastPromotion", "Gender"]

data.drop(unuseful_label, axis=1, inplace=True)
two_val = uniq.index[(uniq==2) & (data.dtypes == "object")]

print(two_val)



data[two_val] = data[two_val].eq(["Yes", "Yes"]).mul(1)

data[two_val].head()
numerical_val = data.columns[data.dtypes != "object"]

data = pd.get_dummies(data, columns=data.columns.drop(numerical_val))

data.info()
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(1,1,1)

ax = data.corr().ix["Attrition"].drop("Attrition").sort_values().plot(kind="barh", figsize=(10, 12), ax=ax)

ax.tick_params(axis='y', which='major', labelsize=18)

ax.set_title("Attrititon Corelation")

plt.tight_layout()

#plt.savefig("AttritionCorelation.png")
Y = data[["Attrition"]].values.ravel()

X = data.drop("Attrition", axis=1).values.astype("float64")
scaler = StandardScaler()

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
def plot_confusion_matrix(y_test, y_pred, iters=500):

        fig = plt.figure(figsize=(7, 7))

        ax = sns.heatmap(confusion_matrix(y_test, y_pred), 

                         annot=True, 

                         cbar=False, 

                         linewidths=2, 

                         linecolor="k",

                         annot_kws={"size": 30},

                         fmt="")

        ax.tick_params(axis='y', which='major', labelsize=18)

        ax.set_title("Confusion matrix", fontdict={"size": 18})

        ax.set_ylabel("True label", fontdict={"size": 18})

        ax.set_xlabel("Predicted label")

        fig.tight_layout()

        return ax

    

def plot_learning_curve(train_sizes, train_scores, test_scores):

    fig = plt.figure(dpi=100)

    train_scores_mean = train_scores.mean(axis=1)

    train_scores_std = train_scores.std(axis=1)

    test_scores_mean = test_scores.mean(axis=1)

    test_scores_std = test_scores.std(axis=1)

    #fig.grid()

    ax1 = fig.add_subplot(1,1,1)

    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    ax1.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    ax1.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    ax1.legend(loc="best")

    fig.tight_layout()

    return ax1



def search_parameter(clf, params, X, y, cv=5, iters=500):

    

    model = RandomizedSearchCV(clf(), param_distributions=params, 

                               n_iter=iters, cv=cv, n_jobs=-1, scoring="f1", 

                               error_score=0, verbose=1)

    model.fit(X, y)

    return model
sgd_params = {'alpha': 0.0001,

              'class_weight': None,

              'l1_ratio': 1,

              'loss': 'log',

              'n_iter': 908,

              'penalty': 'elasticnet',

              'random_state': 1, 

              'shuffle': True}

clf = SGDClassifier(**sgd_params)

clf.fit(X_train, y_train)



print("Accuracy for train dataset: {}".format(clf.score(X_train, y_train)))

print("Accuracy for test dataset: {}".format(clf.score(X_test, y_test)))

print("F1: {}".format(f1_score(y_test, clf.predict(X_test))))

print("AUC ROC: {}".format(roc_auc_score(y_test, clf.predict(X_test))))

plot_confusion_matrix(y_test, clf.predict(X_test))

##plt.savefig("ConfusionSGD.png")
train_sizes, train_scores, valid_score = learning_curve(SGDClassifier(**sgd_params), X_train, y_train, train_sizes=[0.1, 0.3, 0.6, 0.9, 1], cv=5, scoring="f1")

plot_learning_curve(train_sizes, train_scores, valid_score)

#plt.savefig("LearningCurveSGD.png")
param = {

    "C":1,

    "kernel":"linear",

    "gamma":1,

    "random_state":0,

}

#param = svc_result.best_params_



clf = SVC(**param)

clf.fit(X_train, y_train)



print("Accuracy for train dataset: {}".format(clf.score(X_train, y_train)))

print("Accuracy for test dataset: {}".format(clf.score(X_test, y_test)))

print("F1: {}".format(f1_score(y_test, clf.predict(X_test))))

print("AUC ROC: {}".format(roc_auc_score(y_test, clf.predict(X_test))))

plot_confusion_matrix(y_test, clf.predict(X_test))

#plt.savefig("ConfusionSVM.png")
train_sizes, train_scores, valid_score = learning_curve(clf, X_train, y_train, train_sizes=np.linspace(0.01,1,10), cv=5, scoring="f1")

plot_learning_curve(train_sizes, train_scores, valid_score)

#plt.savefig("LerningCUrveSVM.png")