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
#Lets load useful packages for start

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import copy
# Loading dataset (pulsar classification)

df = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")

df.head()
# Let's see if we have some nan values in dataset

df.isnull().sum()
# Ok no nan values -> that's nice!

# I will treat features in the dataset as black box as i dont know what they mean. I will simply rename them as V1 to V8 and treat

# them as some given values

df.rename({"target_class":"Class", df.columns[0]:"V1", df.columns[1]:"V2", df.columns[2]:"V3", df.columns[3]:"V4", df.columns[4]:"V5",

                  df.columns[5]:"V6", df.columns[6]:"V7", df.columns[7]:"V8" }, axis=1, inplace=True) 
df.head()
# Let's look on balance between classes 0 and 1

sns.countplot("Class", data=df)
# Let's look on percentage of classes

print("Percentage of 0 class: ", round(df["Class"].value_counts()[0]/len(df) * 100, 2), "%")

print("Percentage of 1 class: ", round(df["Class"].value_counts()[1]/len(df) * 100, 2), "%")
# As you can see, we have pretty unbalanced data so we need to take care of that later by undersampling

# Let's see if we need to scale all of the data

df.describe()
# Unfortunatly yes, we will choose RobustScaler for this task as its prone to outliers (there is a lot, we will see that soon)

from sklearn.preprocessing import RobustScaler



df_class = df["Class"]

df.drop("Class", axis=1, inplace=True)



scaler = RobustScaler()



for column in df.columns:

    df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))



df = pd.concat([df, df_class], axis=1)

df.head()
# Ok we now have scaled all features

# Now we need to make original train and test set, and then create undersampled set

# original test set will be final evaluation

from sklearn.model_selection import StratifiedKFold



sss = StratifiedKFold()

X = df.drop("Class", axis=1)

y = df["Class"]



for train_index, test_index in sss.split(X, y):

    original_X_train, original_X_test = X.iloc[train_index], X.iloc[test_index]

    original_y_train, original_y_test = y.iloc[train_index], y.iloc[test_index]



original_X_train = original_X_train.values

original_X_test = original_X_test.values#original data

original_y_train = original_y_train.values

original_y_test = original_y_test.values
#Now we need to create dataset with class ratio 50/50 (undersampled)



df = df.sample(frac=1) #shuffling first



pulsar_yes_df = df.loc[df["Class"] == 1]

pulsar_no_df = df.loc[df["Class"] == 0][:len(pulsar_yes_df)]



df_new = pd.concat([pulsar_yes_df, pulsar_no_df])



df_new = df_new.sample(frac=1, random_state=42)

df_new.head()
sns.countplot("Class", data=df_new) #perfectly balanced ..as all things should be (popculture joke haha)
# Let's look on correlation matrix between features and classes

f, ax1 = plt.subplots(1, 1, figsize=(8, 6))



corr = df_new.corr()

sns.heatmap(corr, cmap='coolwarm_r', annot=True, ax=ax1)

ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)



ax1.set_ylim(9, 0)

plt.show()
# The most negative are features V1 and V7 and the most positive are features V3 and V6 (we are looking on the last row)

# Lets make boxplots



# Negative Correlations

f, axes = plt.subplots(ncols=2, figsize=(20, 6))



sns.boxplot(x="Class", y="V1", data=df_new, ax=axes[0])

axes[0].set_title("V1 vs Class Negative corr")



sns.boxplot(x="Class", y="V7", data=df_new, ax=axes[1])

axes[1].set_title("V7 vs Class Negative corr")
# Positive Correlations

f, axes = plt.subplots(ncols=2, figsize=(20, 6))

sns.boxplot(x="Class", y="V3", data=df_new, ax=axes[0])

axes[0].set_title("V3 vs Class Positive corr")



sns.boxplot(x="Class", y="V6", data=df_new, ax=axes[1])

axes[1].set_title("V6 vs Class Positive corr")
# we can see that we have a lot of outliers! Let's take care of that. Ouliers removing will make our model better in performance.

# We just have to expand lower and upper quartile range



v1_pulsar = df_new["V1"].loc[df_new["Class"] == 1].values

q25, q75 = np.percentile(v1_pulsar, 25), np.percentile(v1_pulsar, 75)



v1_iqr = q75 - q25

v1_cutoff = v1_iqr * 2.75

v1_lower, v1_upper = q25 - v1_cutoff, q75 + v1_cutoff



df_past = df_new.copy()

df_new = df_new.drop(df_new[(df_new["V1"] > v1_upper) | (df_new["V1"] < v1_lower)].index)



outliers = len(df_past) - len(df_new)

print("Number of removed outliers: ", outliers)



#V7 outlier delete

v7_pulsar = df_new["V7"].loc[df_new["Class"] == 1].values 

q25, q75 = np.percentile(v7_pulsar, 25), np.percentile(v7_pulsar, 75)



v7_iqr = q75 - q25

v7_cutoff = v7_iqr * 2.75

v7_lower, v7_upper = q25 - v7_cutoff, q75 + v7_cutoff



df_past = df_new.copy()

df_new = df_new.drop(df_new[(df_new["V7"] > v7_upper) | (df_new["V7"] < v7_lower)].index)



outliers = len(df_past) - len(df_new)

print("Number of removed outliers: ", outliers)
#Features after outliers removal

f,  axes = plt.subplots(ncols=2, figsize=(20, 6))



sns.boxplot(x="Class", y="V1", data=df_new, ax=axes[0])

axes[0].set_title("V1 vs class negative corr after outliers removing")



sns.boxplot(x="Class", y="V7", data=df_new, ax=axes[1])

axes[1].set_title("V7 vs class negative corr after outliers removing")
#It's slightly better

sns.countplot("Class", data=df_new)
#Our undersampled dataset is still balanced

#Let's try clustering and dimension reduction via PCA and tsne just for higher perspective



from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



pca = PCA(n_components=2, random_state=42)

tsne = TSNE(n_components=2, random_state=42)

#df_new is from the random undersample data (fewer instances)

X = df_new.drop("Class", axis=1)

y = df_new["Class"]



X_reduced = pca.fit_transform(X.values)

X_reduced_tsne = tsne.fit_transform(X.values)
f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6))



ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Pulsars', linewidths=2)

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Pulsars', linewidths=2)

ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)



ax2.scatter(X_reduced[:,0], X_reduced[:,1], c=(y==0), cmap="coolwarm", label="No Pulsars", linewidth=2)

ax2.scatter(X_reduced[:,0], X_reduced[:,1], c=(y==1), cmap="coolwarm", label="Pulsars", linewidth=2)

ax2.grid(True)

ax2.set_title("PCA", fontsize=14)

plt.legend()

plt.show()



#Blue color is class 0
# We can see that pulsar and non-pulsar cases are kind of separated, so our models will perform well (and plots look cool)

# Let's split out undersampled data

X = df_new.drop("Class", axis=1)

y = df_new["Class"]



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Turning to arrays so we can feed this data to our ML algorithms

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
#importing four models

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
#lets now test multiple classifiers

classifiers = {

    "LogisticRegression":LogisticRegression(),

    "KNearest":KNeighborsClassifier(),

    "Support Vector Machine":SVC(),

    "Random Forest":RandomForestClassifier()

}
from sklearn.model_selection import cross_val_score



for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifier: ",classifier.__class__.__name__, "Has score: ", round(training_score.mean(), 4) * 100, "%")
#Ok we can see that Logistic Regression has achieved the best accuracy

#Let's now plot ROC curves for all classifers and see which one is the best 

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import cross_val_predict



log_reg = LogisticRegression()

svm = SVC()

knear = KNeighborsClassifier()

forest = RandomForestClassifier()



y_pred_log = cross_val_predict(log_reg, X_train, y_train, cv=5)

y_pred_svm = cross_val_predict(svm, X_train, y_train, cv=5)

y_pred_knear = cross_val_predict(knear, X_train, y_train, cv=5)

y_pred_forest = cross_val_predict(forest, X_train, y_train, cv=5)



log_fpr, log_tpr, log_thresold = roc_curve(y_train, y_pred_log)

knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, y_pred_knear)

svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, y_pred_svm)

tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, y_pred_forest)



def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):

    plt.figure(figsize=(16,8))

    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)

    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, y_pred_log)))

    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, y_pred_svm)))

    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, y_pred_svm)))

    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, y_pred_forest)))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.01, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),

                arrowprops=dict(facecolor='#6E726D', shrink=0.05),

                )

    plt.legend()



graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)

plt.show()
# Okey, Logistic Regression is still the best! We will try to optimize this model

from sklearn.model_selection import GridSearchCV



log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)



log_reg = grid_log_reg.best_estimator_

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=10)



print("Accuracy of logistic regression with optimal parameters is: ", 

      round(log_reg_score.mean(), 4) * 100, "%")
# We need to get sure if our model is overfitting or underfitting

# Let's plot learning curve of logistic regression model

from sklearn.model_selection import learning_curve, ShuffleSplit





def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

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



cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(log_reg, "Logistic Regression with optimal parametets", X_train, y_train, cv=cv, n_jobs=4)

plt.show()
# Ok that's promising

# I would like to look at precision/recall tradeoff curve as well

from sklearn.metrics import precision_recall_curve



y_pred = cross_val_predict(log_reg, X_train, y_train, cv=5, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train, y_pred)



def plot_prec_recall_vs_thresh(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')

    plt.plot(thresholds, recalls[:-1], 'g--', label = 'recall')

    plt.xlabel('Threshold')

    plt.legend(loc='upper left')

    plt.ylim([0,1])



plot_prec_recall_vs_thresh(precisions, recalls, thresholds)

plt.legend()

plt.show()
# Not great not terrible (joking it looks just fine)

#-------------------------------------------ErrorEvaluation----------------------------------------

# Let's make confusion matrix on undersampled data

from sklearn.metrics import confusion_matrix



y_test_under_pred = cross_val_predict(log_reg, X_test, y_test, cv=5)



f, ax = plt.subplots(1, 1, figsize=(10, 6))



conf = confusion_matrix(y_test, y_test_under_pred)

sns.heatmap(conf, cmap="Greys", fmt="d", annot=True,  linewidths=5, annot_kws={"size": 15}, ax=ax)

ax.set_ylim(2, 0)

plt.show()
#Okey, that's great

#Last thing we need to do is to classify original test data and look for accuracy



f, ax = plt.subplots(1, 1, figsize=(10, 8))



y_test_pred_original = cross_val_predict(log_reg, original_X_test, original_y_test, cv=5)

conf = confusion_matrix(original_y_test, y_test_pred_original)

sns.heatmap(conf, cmap="Greys", annot=True, annot_kws={"size":15}, ax=ax, fmt="g")

ax.set_ylim(2, 0)

plt.show()
#Dont panic, only one black square means that our original data are very unbalanced

# Let's look on classification_report

from sklearn.metrics import classification_report



print(classification_report(original_y_test, y_test_pred_original))
#And last thing is our optimized model's accuracy:

from sklearn.metrics import accuracy_score



final_accuracy = accuracy_score(original_y_test, y_test_pred_original)

print("Our model's final accuracy is: {} %".format(round(final_accuracy, 3)*100))