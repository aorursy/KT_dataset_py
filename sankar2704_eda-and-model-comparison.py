import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import precision_score,f1_score

from sklearn.metrics import plot_roc_curve
## Load Data

df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
df.shape
df['target'].value_counts().plot(kind='bar');
df.info()
##Compare target with Sex

pd.crosstab(df.target,df.sex)
plt.figure(figsize=(10,6))



#Plot for patients who have heart disease

plt.scatter(df.age[df.target == 1],

           df.thalach[df.target == 1]

           );



#Plot for patients who don't have heart disease

plt.scatter(df.age[df.target == 0],

           df.thalach[df.target == 0]

           );



plt.xlabel("Age")

plt.ylabel("Max Heart Rate")

plt.title("Heart Disease againt Age and Max Heart Rate")

plt.legend(["Disease","No disease"]);
#Checking the distribution of age

df.age.plot.hist();
pd.crosstab(df.cp,df.target)
#Plot for patients who have heart disease

pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(10,6));

plt.xlabel("Chest Pain Type");

plt.ylabel("Amount");

plt.title("Heart Disease againt Age and Max Heart Rate");

plt.legend(["No Disease","Disease"]);

plt.xticks(rotation=0);
### Check correlation between dependent and independent variables
corr = df.corr()
fig,ax = plt.subplots(figsize=(15,10))

ax = sns.heatmap(corr,annot=True,linewidth=0.5,fmt=".2f")
X = df.drop("target",axis=1)

Y = df["target"]
np.random.seed(2)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
models = {"Logistic" : LogisticRegression(),

         "KNN": KNeighborsClassifier(),

         "RF": RandomForestClassifier()}



def fit_score(models,X_train,X_test,Y_train,Y_test):

    np.random.seed(2)

    model_scores = {}

    for name, model in models.items():

        model.fit(X_train,Y_train)

        model_scores[name] = model.score(X_test,Y_test)

        

    return model_scores
scores = fit_score(models,X_train,X_test,Y_train,Y_test)



scores
model_scores = pd.DataFrame(scores,index=["accuracy"])

model_scores.T.plot.bar();
## Tunning KNN

train_scores = []

test_scores = []



# Create a list of KNN

neighbors = range(1,21)



knn = KNeighborsClassifier()



for i in neighbors:

    knn.set_params(n_neighbors=i)

    

    knn.fit(X_train,Y_train)

    

    train_scores.append(knn.score(X_train,Y_train))

    

    test_scores.append(knn.score(X_test,Y_test))
train_scores
test_scores
plt.plot(neighbors,train_scores,label="Train Scores")

plt.plot(neighbors,test_scores,label="Test Scores")



plt.xlabel("Number of Neighbors")

plt.ylabel("Model Score")

plt.legend();



print(f"MaxNN score: {max(test_scores)*100:.2f}%")
# Create a hyperparameter grid for LogisticRegression



log_reg_grid = {"C" : np.logspace(-4,4,20),

                "solver" : ["liblinear"]}



 # Create a hyperparameter grid for RandomForestClssifier

rf_grid = {"n_estimators" : np.arange(10,1000,50),

          "max_depth" : [None,3,5,10],

          "min_samples_split" : np.arange(2,20,2),

          "min_samples_leaf" : np.arange(1,20,2)}
# Tune LogisticRegression



np.random.seed(2)



rs_log_reg = RandomizedSearchCV(LogisticRegression(),

                               cv=5,

                               param_distributions=log_reg_grid,

                               n_iter=20,

                               verbose=True)



#Fit the model

rs_log_reg.fit(X_train,Y_train)
rs_log_reg.best_params_
rs_log_reg.score(X_test,Y_test)
# Tune RandomForestClassifier



np.random.seed(2)



rs_rf = RandomizedSearchCV(RandomForestClassifier(),

                               cv=5,

                               param_distributions=rf_grid,

                               n_iter=20,

                               verbose=True)



#Fit the model

rs_rf.fit(X_train,Y_train)
rs_rf.best_params_
rs_rf.score(X_test,Y_test)
#Our Original Model shows that default LogisticRegession

#performs well even before tunning

model_scores
# Create a hyperparameter grid for LogisticRegression



log_reg_grid = {"C" : np.logspace(-4,4,30),

                "solver" : ["liblinear"]}



#Setup gird hyperparameter search for LogisticRegression



gs_log_reg = GridSearchCV(LogisticRegression(),

                               cv=5,

                               param_grid=log_reg_grid,

                               verbose=True)



# Fit the model

gs_log_reg.fit(X_train,Y_train)
gs_log_reg.best_params_
#Evaluate GrdiSearch LogisticRegression

gs_log_reg.score(X_test,Y_test)

#Just a bit more than normal LogisticRegressions
#Original model scores without tuning

model_scores
#Making predicitions

y_preds = gs_log_reg.predict(X_test)

y_preds
# ROC Curve

plot_roc_curve(gs_log_reg,X_test,Y_test)
#Confusion Matrix

fig,ax = plt.subplots(figsize=(3,3))

ax = sns.heatmap(confusion_matrix(Y_test,y_preds),annot=True,cbar=False)

plt.xlabel("True label")

plt.ylabel("Predicted label")
print(classification_report(Y_test,y_preds))

# these metrics are calculated on one split only

# Precision is the ratio of correctly predicted positive observations to the total predicted positive observations

# Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes

# F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.

# we may need to calculate the metrics based on the cross-validation

gs_log_reg.best_params_
# Create a new classifier with best params

clf = LogisticRegression(C=0.1082636733874054,solver="liblinear")
# Cross validated Accuracy

# we are passing entire X,Y as we are going through the cross validation, 

# which will do the splits automatically

cv_acc = cross_val_score(clf,X,Y,cv=5,scoring="accuracy")

cv_acc = np.mean(cv_acc)

cv_acc
# Cross validated Precisions

cv_pre = cross_val_score(clf,X,Y,cv=5,scoring="precision")

cv_pre = np.mean(cv_pre)

cv_pre
# Cross validated Recall

cv_rec = cross_val_score(clf,X,Y,cv=5,scoring="recall")

cv_rec = np.mean(cv_rec)

cv_rec
# Cross validated F1

cv_f1 = cross_val_score(clf,X,Y,cv=5,scoring="f1")

cv_f1 = np.mean(cv_rec)

cv_f1
# Visualize cross validated metrics

cv_metrics = pd.DataFrame({"Accuracy" : cv_acc,

                          "Precision" : cv_pre,

                          "Recall" : cv_rec,

                          "F1-Score" : cv_f1}, index=[0])

cv_metrics.T.plot.bar(title="Cross-validatd classfication metrics", legend=False);
clf.fit(X_train,Y_train)
clf.coef_
feature_dict = dict(zip(df.columns,list(clf.coef_[0])))

feature_dict
# These values come form model buildng finding is a sort of co-relation

# this can be compared to Correleation matrix

feature_df = pd.DataFrame(feature_dict,index=[0])

feature_df.T.plot.bar(title = "Feature Importance",legend=False);
# Implementing catboos or XGboost

# tuning further

# Repeating the above process agin with differnt set of analysis