import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/cat-in-the-dat/train.csv")
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
print("Total number of train data is:", train.shape)
train.head(6)
train.drop(["id"], axis=1, inplace=True)
train.isna().sum()
train.describe()
#train.duplicated()
tar = train['target'].value_counts()

print("Number of cat ", tar[1], ", (", (tar[1]/(tar[1]+tar[0]))*100,"%)")

print("Number of non_cat ", tar[0], ", (", (tar[0]/(tar[1]+tar[0]))*100,"%)")

def bar_plot(feature):

    sns.set(style="darkgrid")

    ax = sns.countplot(x=feature , data=train)

    



    
bar_plot("target")
bar_plot("bin_0")
bar_plot("bin_1")
bar_plot("bin_2")
bar_plot("bin_3")
bar_plot("bin_4")
bar_plot("nom_0")
bar_plot("nom_1")
bar_plot("nom_2")
bar_plot("nom_3")
bar_plot("nom_4")
bar_plot("nom_5")
bar_plot("nom_6")
bar_plot("nom_7")
bar_plot("nom_8")
bar_plot("nom_9")
print("Total number of different category for nom_5 is:", train["nom_5"].value_counts().shape[0])

print("Total number of different category for nom_6 is:", train["nom_6"].value_counts().shape[0])

print("Total number of different category for nom_7 is:", train["nom_7"].value_counts().shape[0])

print("Total number of different category for nom_8 is:", train["nom_8"].value_counts().shape[0])

print("Total number of different category for nom_9 is:", train["nom_9"].value_counts().shape[0])
bar_plot("ord_0")
bar_plot("ord_1")
bar_plot("ord_2")
bar_plot("ord_3")
bar_plot("ord_4")
bar_plot("ord_5")
bar_plot("day")
bar_plot("month")
test = pd.read_csv("../input/cat-in-the-dat/test.csv")
test.shape
test.head(3)
test.drop(["id"], axis=1, inplace=True)
test["bin_0"].isin(train["bin_0"]).value_counts()
test["bin_1"].isin(train["bin_1"]).value_counts()
test["bin_2"].isin(train["bin_2"]).value_counts()
test["bin_3"].isin(train["bin_3"]).value_counts()
test["bin_4"].isin(train["bin_4"]).value_counts()
test["nom_0"].isin(train["nom_0"]).value_counts()
test["nom_1"].isin(train["nom_1"]).value_counts()
test["nom_2"].isin(train["nom_2"]).value_counts()
test["nom_3"].isin(train["nom_3"]).value_counts()
test["nom_4"].isin(train["nom_4"]).value_counts()
test["nom_5"].isin(train["nom_5"]).value_counts()
test["nom_6"].isin(train["nom_6"]).value_counts()
test["nom_7"].isin(train["nom_7"]).value_counts()
test["nom_8"].isin(train["nom_8"]).value_counts()
test["nom_9"].isin(train["nom_9"]).value_counts()
test["ord_0"].isin(train["ord_0"]).value_counts()
test["ord_1"].isin(train["ord_1"]).value_counts()
test["ord_2"].isin(train["ord_2"]).value_counts()
test["ord_3"].isin(train["ord_3"]).value_counts()
test["ord_4"].isin(train["ord_4"]).value_counts()
test["ord_5"].isin(train["ord_5"]).value_counts()
test["day"].isin(train["day"]).value_counts()
test["month"].isin(train["month"]).value_counts()
y = train["target"]

train.drop(["target"], inplace=True, axis=1)
X =train

T =test 
train = X

test = T
df = pd.concat([train, test])

dummies = pd.get_dummies(df, columns=df.columns, drop_first=True, sparse=True)

train = dummies.iloc[:train.shape[0], :]

test = dummies.iloc[train.shape[0]:, :]



print(train.shape)

print(test.shape)
train.head(2)
test.head(2)
def log_alpha(al):

    alpha=[]

    for i in al:

        a=np.log(i)

        alpha.append(a)

    return alpha    
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import roc_auc_score



svm = SGDClassifier(loss='log', class_weight='balanced')

alpha=alpha = [0.000001,0.000002, 0.000005, 0.00001, 0.00003, 0.00005, 0.00007]

parameters = {'alpha':alpha}

clf = RandomizedSearchCV(svm, parameters, cv=5, scoring='roc_auc', n_jobs=-1, return_train_score=True,)

clf.fit(train, y)



print("Model with best parameters :\n",clf.best_estimator_)



alpha = log_alpha(alpha)





best_alpha = clf.best_estimator_.alpha

best_penalty = clf.best_estimator_.penalty

#best_split = clf.best_estimator_.min_samples_split



print(best_alpha)

print(best_penalty)

#print(best_split)



train_auc= clf.cv_results_['mean_train_score']

train_auc_std= clf.cv_results_['std_train_score']

cv_auc = clf.cv_results_['mean_test_score'] 

cv_auc_std= clf.cv_results_['std_test_score']



plt.plot(alpha, train_auc, label='Train AUC')

# this code is copied from here: https://stackoverflow.com/a/48803361/4084039

plt.gca().fill_between(alpha,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')



plt.plot(alpha, cv_auc, label='CV AUC')

# this code is copied from here: https://stackoverflow.com/a/48803361/4084039

plt.gca().fill_between(alpha,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')



plt.scatter(alpha, train_auc, label='Train AUC points')

plt.scatter(alpha, cv_auc, label='CV AUC points')





plt.legend()

plt.xlabel("alpha and l1")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve

from sklearn.metrics import roc_curve, auc

from sklearn.calibration import CalibratedClassifierCV



svm = SGDClassifier(loss='log', alpha=best_alpha, penalty=best_penalty, class_weight="balanced")

#svm.fit(train_1, project_data_y_train)

# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

# not the predicted outputs



#sig_clf = CalibratedClassifierCV(svm, method="isotonic")

svm = svm.fit(train, y)





y_train_pred1 = svm.predict(train) 

y_test_pred1 = svm.predict(test)



train_fpr, train_tpr, tr_thresholds = roc_curve(y, y_train_pred1)

#test_fpr, test_tpr, te_thresholds = roc_curve(project_data_y_test, y_test_pred)



plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

#plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.legend()

plt.xlabel(" hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()
sub = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")
sub.head(2)
submission = pd.DataFrame({'id': sub["id"], 'target': y_test_pred1})

submission.to_csv('submission_log.csv', index=False)
