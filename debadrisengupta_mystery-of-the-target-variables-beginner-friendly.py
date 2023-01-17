

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline 



## Models

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier





## Model evaluators

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve;
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv") # 'DataFrame' shortened to 'df'

df.shape # (rows, columns)
df.head()


df.head(10)
# No. of positive and negative patients in our samples

df.target.value_counts()
# Normalized value counts

df.target.value_counts(normalize=True)
# Plot the value counts with a bar graph

df.target.value_counts().plot(kind="bar", color=["purple", "magenta"]);
df.info()
df.describe()
df.sex.value_counts()
pd.crosstab(df.target, df.age)
# Compare target column with sex column

pd.crosstab(df.target, df.sex)


pd.crosstab(df.target, df.sex).plot(kind="bar", figsize=(10,6), color=["salmon", "lightblue"])



plt.title("Heart Disease Frequency for Sex")

plt.xlabel("0 = Disease, 1 = No Disease")

plt.ylabel("Amount")

plt.legend(["Female", "Male"])

plt.xticks(rotation=0);


plt.figure(figsize=(10,6))



# For positve examples

plt.scatter(df.age[df.target==0], 

            df.thalach[df.target==0], 

            c="salmon") # define it as a scatter figure



# Now for negative examples, 

plt.scatter(df.age[df.target==1], 

            df.thalach[df.target==1], 

            c="lightblue") 





plt.title("Heart Disease in function of Age and Max Heart Rate")

plt.xlabel("Age")

plt.legend(["Disease", "No Disease"])

plt.ylabel("Max Heart Rate");
# Histograms to check age distribution 

df.age.plot.hist();
pd.crosstab(df.cp, df.target)


pd.crosstab(df.cp, df.target).plot(kind="bar", 

                                   figsize=(10,6), 

                                   color=["lightblue", "salmon"])





plt.title("Heart Disease Frequency Per Chest Pain Type")

plt.xlabel("Chest Pain Type")

plt.ylabel("Frequency")

plt.legend(["Disease", "No disease"])

plt.xticks(rotation = 0);
# Find the correlation between our independent variables

corr_matrix = df.corr()

corr_matrix 


corr_matrix = df.corr()

fig, ax=plt.subplots(figsize=(15, 15))

ax=sns.heatmap(corr_matrix, 

            annot=True, 

            linewidths=0.5, 

            fmt= ".2f", 

            cmap="YlGnBu");

bottom, top=ax.get_ylim()

ax.set_ylim(bottom+0.5, top-0.5)
df.head()
# Everything except target variable

X = df.drop("target", axis=1)



# Target variable

y = df.target.values
# Independent variables (no target column)

X.head()
# Targets

y


np.random.seed(42)





X_train, X_test, y_train, y_test = train_test_split(X,  

                                                    y, 

                                                    test_size = 0.2) 
X_train.head()
y_train, len(y_train)
X_test.head()
y_test, len(y_test)
# Put models in a dictionary

models = {"KNN": KNeighborsClassifier(),

          "Logistic Regression": LogisticRegression(), 

          "Random Forest": RandomForestClassifier(), "Decision Tree":DecisionTreeClassifier()}



# Create function to fit and score models

def fit_and_score(models, X_train, X_test, y_train, y_test):



    

    np.random.seed(42)

    

    model_scores = {}

    

    for name, model in models.items():

        

        model.fit(X_train, y_train)

        

        model_scores[name] = model.score(X_test, y_test)*100

    return model_scores
model_scores = fit_and_score(models=models,

                             X_train=X_train,

                             X_test=X_test,

                             y_train=y_train,

                             y_test=y_test)

model_scores
model_compare = pd.DataFrame(model_scores, index=['accuracy'])

model_compare.plot.bar();
# Different LogisticRegression hyperparameters

log_reg_grid = {"C": np.logspace(-4, 4, 20),

                "solver": ["liblinear"]}



# Different RandomForestClassifier hyperparameters

rf_grid = {"n_estimators": np.arange(10, 1000, 50),

           "max_depth": [None, 3, 5, 10],

           "min_samples_split": np.arange(2, 20, 2),

           "min_samples_leaf": np.arange(1, 20, 2)}


np.random.seed(42)





rs_log_reg = RandomizedSearchCV(LogisticRegression(),

                                param_distributions=log_reg_grid,

                                cv=5,

                                n_iter=20,

                                verbose=True)





rs_log_reg.fit(X_train, y_train);
rs_log_reg.best_params_
rs_log_reg.score(X_test, y_test)
# Setup random seed

np.random.seed(42)



# Setup random hyperparameter search for RandomForestClassifier

rs_rf = RandomizedSearchCV(RandomForestClassifier(),

                           param_distributions=rf_grid,

                           cv=5,

                           n_iter=20,

                           verbose=True)



# Fit random hyperparameter search model

rs_rf.fit(X_train, y_train);


rs_rf.best_params_


rs_rf.score(X_test, y_test)


log_reg_grid = {"penalty" :['l2'],

"C":np.logspace(-4,4,30),

"class_weight":[{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],

"solver": ['liblinear', 'saga','sag','newton-cg','lbfgs'],"max_iter":[10] }







gs_log_reg = GridSearchCV(LogisticRegression(),

                          param_grid=log_reg_grid,

                          cv=5,

                          verbose=True)





gs_log_reg.fit(X_train, y_train);
# Check the best parameters

gs_log_reg.best_params_
# Evaluate the model

gs_log_reg.score(X_test, y_test)
from sklearn.ensemble import AdaBoostClassifier

adaboost=AdaBoostClassifier(base_estimator=LogisticRegression(C= 2.592943797404667,

 class_weight= {1: 0.5, 0: 0.5},

 max_iter= 10,

 penalty= 'l2',

 solver= 'liblinear'),n_estimators=100)

adaboost.fit(X_train,y_train)

adaboost.score(X_test,y_test)



from sklearn.metrics import roc_curve



y_preds = gs_log_reg.predict(X_test)
y_preds
y_test
y_probs=gs_log_reg.predict_proba(X_test)

y_probs_positive=y_probs[:,1]

y_probs_positive


from sklearn.metrics import auc



fpr, tpr, thresholds= roc_curve(y_test, y_probs_positive)

def plot_roc_curve(fpr,tpr):

 plt.plot(fpr, tpr, color="orange",label="ROC")

 plt.plot([0,1],[0,1],color="darkblue",linestyle="--",label="Guessing")

 plt.xlabel("False positive rate")

 plt.ylabel("True positive rate")

 plt.title("Receiver Operating Characterisitics curve")

 plt.legend()

 plt.show()

plot_roc_curve(fpr,tpr)

roc_auc=auc(fpr,tpr)

roc_auc
# Display confusion matrix

print(confusion_matrix(y_test, y_preds))


import seaborn as sns

sns.set(font_scale=1.5) 

def plot_conf_mat(y_test, y_preds):

    

    fig, ax = plt.subplots(figsize=(3, 3))

    ax = sns.heatmap(confusion_matrix(y_test, y_preds),

                     annot=True, # Annotate the boxes

                     cbar=True)

    plt.xlabel("true label")

    plt.ylabel("predicted label")

    bottom,top=ax.get_ylim()

    ax.set_ylim(bottom+0.5, top-0.5)

    

plot_conf_mat(y_test, y_preds)
prec=precision_score(y_test,y_preds)

prec
rec=recall_score(y_test,y_preds)

rec
# Show classification report

print(classification_report(y_test, y_preds))
rs_rf.best_params_

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

rf=RandomForestClassifier(n_estimators=560,

 min_samples_split=12,

 min_samples_leaf=15,

 max_depth=3)

cv_acc=np.mean(cross_val_score(rf,X,y,cv=5,scoring="accuracy"))

cv_prec=np.mean(cross_val_score(rf,X,y,cv=5,scoring="precision"))

cv_recall=np.mean(cross_val_score(rf,X,y,cv=5,scoring="recall"))

cv_f1=np.mean(cross_val_score(rf,X,y,cv=5,scoring="f1"))

cv_acc,cv_prec,cv_recall,cv_f1



cv_metrics = pd.DataFrame({"Accuracy": cv_acc,

                            "Precision": cv_prec,

                            "Recall": cv_recall,

                            "F1": cv_f1},

                          index=[0])

cv_metrics.T.plot.bar(title="Random Forest Cross-Validated Metrics", legend=False);


gs_log_reg.best_params_


from sklearn.model_selection import cross_val_score



clf = LogisticRegression(C=0.23357214690901212,

                         solver="liblinear")


cv_acc = cross_val_score(clf,

                         X,

                         y,

                         cv=5, 

                         scoring="accuracy")
cv_acc = np.mean(cv_acc)

cv_acc
# Cross-validated precision score

cv_precision = np.mean(cross_val_score(clf,

                                       X,

                                       y,

                                       cv=5,

                                       scoring="precision")) 

cv_precision


cv_recall = np.mean(cross_val_score(clf,

                                    X,

                                    y,

                                    cv=5, 

                                    scoring="recall")) 

cv_recall


cv_f1 = np.mean(cross_val_score(clf,

                                X,

                                y,

                                cv=5, 

                                scoring="f1")) 

cv_f1
# Visualizing cross-validated metrics

cv_metrics = pd.DataFrame({"Accuracy": cv_acc,

                            "Precision": cv_precision,

                            "Recall": cv_recall,

                            "F1": cv_f1},

                          index=[0])

cv_metrics.T.plot.bar(title="Logistic Regression Cross-Validated Metrics", legend=False);


clf.fit(X_train, y_train);
# Check feature importance

clf.coef_
# Match features to columns

features_dict = dict(zip(df.columns, list(clf.coef_[0])))

features_dict
# Visualize feature importance

features_df = pd.DataFrame(features_dict, index=[0])

features_df.T.plot.bar(title="Feature Importance", legend=False);