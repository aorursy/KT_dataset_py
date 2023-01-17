from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 

import os

import numpy as np

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import jaccard_similarity_score

from sklearn import metrics

from sklearn import preprocessing

import itertools

from sklearn.model_selection import validation_curve

from xgboost import XGBClassifier

import xgboost as xgb

from xgboost import plot_importance

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, auc

from sklearn.inspection.partial_dependence import partial_dependence, plot_partial_dependence

from sklearn.ensemble import GradientBoostingClassifier

import eli5

from eli5.sklearn import PermutationImportance

import shap

from mpl_toolkits.mplot3d import Axes3D
print(os.listdir('../input'))
filename = "../input/heart.csv"

df = pd.read_csv(filename)

df.head()
df_target = df.groupby("target").size()

df_target
plt.pie(df_target.values, labels = ["target 0", "target 1"], autopct='%1.1f%%', radius = 1.5, textprops = {"fontsize" : 16}) 

plt.show()
df_sex = df.groupby(["sex","target"]).size()

df_sex
plt.pie(df_sex.values, labels = ["sex_0,target_0", "sex_0,target_1", "sex_1,target_0", "sex_1,target_1"],autopct='%1.1f%%',radius = 1.5, textprops = {"fontsize" : 16})

plt.show()
plt.hist([df[df.target==0].age, df[df.target==1].age], bins = 20, alpha = 0.5, label = ["no_heart_disease","with heart disease"])

plt.xlabel("age")

plt.ylabel("percentage")

plt.legend()

plt.show()
plt.hist([df[df.target==0].chol, df[df.target==1].chol], bins = 20, alpha = 0.5, label = ["no_heart_disease","with heart disease"])

plt.xlabel("chol")

plt.ylabel("percentage")

plt.legend()

plt.show()
plt.hist([df[df.target==0].trestbps, df[df.target==1].trestbps], bins = 20, alpha = 0.5, label = ["no_heart_disease","with heart disease"])

plt.xlabel("trestbps")

plt.ylabel("percentage")

plt.legend()

plt.show()
plt.hist([df[df.target==0].thalach, df[df.target==1].thalach], bins = 20, alpha = 0.5, label = ["no_heart_disease","with heart disease"])

plt.xlabel("thalach")

plt.ylabel("percentage")

plt.legend()

plt.show()
df_1 = df[["age", "trestbps", "chol", "thalach", "oldpeak"]]

df_1.describe()
for item in df_1.columns:

    plt.subplot(3,2,list(df_1.columns).index(item)+1)

    plt.boxplot(df_1[item], patch_artist=True, labels = [item])

    plt.ylabel("value")

plt.tight_layout()

plt.show()
infor = df.describe()



df2 = df[df.trestbps < infor.loc["mean", "trestbps"] + 3 * infor.loc["std", "trestbps"]]

df3 = df2[df.chol < infor.loc["mean", "chol"] + 3 * infor.loc["std", "chol"]]

df4 = df3[df.thalach > infor.loc["mean", "thalach"] - 3 * infor.loc["std", "thalach"]]

df_new = df4[df.oldpeak < infor.loc["mean", "oldpeak"] + 3 * infor.loc["std", "oldpeak"]]

df_new.head()
df_new.cp = df_new.cp.map({0:"asymptomatic", 1: "typical angina", 2:"atypical angina", 3:"non-anginal pain"})

df_new.sex = df_new.sex.map({0:"Female", 1:"Male"}) 

df_new.exang = df_new.exang.map({0:"exercise did not induce angina", 1:"exercise induced angina"})

df_new.slope = df_new.slope.map({1:"upsloping", 2:"flat", 3:"downsloping"})

df_new.thal = df_new.thal.map({1:"normal",2:"fixed defect", 3:"reversable defect"})

df_new = pd.get_dummies(df_new, drop_first = True)

df_new.head(10)
X = df_new.drop("target", 1).values

y = df_new["target"].astype("int").values

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=10)
model = XGBClassifier()

param = dict(model_max_depth=[3,5,7], model_learning_rate=[0.001,0.01, 0.1], model_n_estimators=[100,500])



cv = GridSearchCV(model, param_grid=param, cv=10, iid = True)

cv
cv.fit(X_train, y_train)

test_pred = cv.predict(X_test)

cv.best_estimator_
max_depth_of_model = cv.best_estimator_.max_depth

best_learning_rate = cv.best_estimator_.learning_rate

best_estimator = cv.best_estimator_.n_estimators

best_reg_lambda = cv.best_estimator_.reg_lambda



model = XGBClassifier(max_depth=max_depth_of_model, learning_rate=best_learning_rate, n_estimators= best_estimator,n_jobs=1,)

model.fit(X_train, y_train)

yhat = model.predict(X_test)

y_proba = model.predict_proba(X_test)[:, 1]

accuracy_score(yhat,y_test)
importances = model.feature_importances_

importances
inducies = np.argsort(importances)[::-1]

inducies
feature_dict = dict()

for idx in inducies:

    feature_dict[list(df_new.drop("target",1).columns)[idx]] = float(importances[idx])

feature_dict
y_pos = np.arange(len(feature_dict.keys()))

plt.bar(y_pos, list(feature_dict.values()), align = "center",color = "lightgreen")

plt.xticks(y_pos, list(feature_dict.keys()), rotation = 90)

plt.xlabel("feature")

plt.ylabel("ratio")

plt.title("feature importances")

plt.show()
tp,fn,fp,tn = confusion_matrix(y_test, yhat, labels=[1,0]).ravel()

tp,tn,fp,fn
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()

    return ""
cmx = confusion_matrix(y_test, yhat, labels=[1,0])

print(plot_confusion_matrix(cmx, classes=['target=1','target=0'],normalize= False,  title='Confusion matrix'))
precision_rate = tp / (tp + fp)

recall_rate = tp / (tp + fn)

print("The precision rate is: ", precision_rate)

print("The recall rate is: ", recall_rate)
fpr, tpr, threshold =roc_curve(y_test, y_proba)

fig, ax = plt.subplots()

ax.plot(fpr,tpr)

ax.plot([0,1], [0,1], transform = ax.transAxes, ls="--", c="0.3")

plt.xlim(0.0, 1.0)

plt.ylim(0.0, 1.0)

plt.xlabel("FPR or 1-Specificity")

plt.ylabel("TPR or Sensitivity")

plt.rcParams["font.size"] = 10

plt.grid(True)

plt.show()
auc(fpr, tpr)
perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = df_new.drop("target", 1).columns.tolist())
clf = GradientBoostingClassifier(learning_rate = best_learning_rate, n_estimators = best_estimator,

                                 max_depth = max_depth_of_model).fit(X_train, y_train)



feature_dict = dict(enumerate(df_new.drop("target", 1).columns))

feature_dict
my_plots = plot_partial_dependence(clf, features = [0], X = X_test, 

                                   feature_names = df_new.drop("target", 1).columns[:1], grid_resolution=5)     



plt.subplots_adjust(top=1.0, right=1.0)

my_plots
my_plots = plot_partial_dependence(clf, features = [7], X = X_test, 

                                   feature_names = df_new.drop("target", 1).columns[:8], grid_resolution=5)     



plt.subplots_adjust(top=1.0, right=1.0)

my_plots
my_plots = plot_partial_dependence(clf, features = [5], X = X_test, 

                                   feature_names = df_new.drop("target", 1).columns[:6], grid_resolution=5)     



plt.subplots_adjust(top=1.0, right=1.0)

my_plots
my_plots = plot_partial_dependence(clf, features = [1], X = X_test, 

                                   feature_names = df_new.drop("target", 1).columns[:2], grid_resolution=5)     



plt.subplots_adjust(top=1.0, right=1.0)

my_plots
my_plots = plot_partial_dependence(clf, features = [6], X = X_test, 

                                   feature_names = df_new.drop("target", 1).columns[:7], grid_resolution=5)     



plt.subplots_adjust(top=1.0, right=1.0)

my_plots
fig = plt.figure()

target_feature = (6, 13)

pdp, axes = partial_dependence(clf, X_train, target_feature,  grid_resolution=50)

XX, YY = np.meshgrid(axes[0], axes[1])

Z = pdp[0].reshape(list(map(np.size, axes))).T

ax = Axes3D(fig)

surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor='k')

ax.set_xlabel(feature_dict[target_feature[0]])

ax.set_ylabel(feature_dict[target_feature[1]])

ax.set_zlabel('Partial dependence')



ax.view_init(elev=20, azim=72)

plt.colorbar(surf)

plt.subplots_adjust(top=0.9)

plt.show()
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names = list(feature_dict.values()), plot_type = "bar", color = "lightblue")
shap.summary_plot(shap_values, X_test, feature_names = list(feature_dict.values()))
def shap_force_plot_of_data(model, dataset):

    explainer = shap.TreeExplainer(model)

    shap_value_for_sample = explainer.shap_values(dataset)

    shap.initjs()

    drivein_force = shap.force_plot(explainer.expected_value, shap_value_for_sample, dataset)

    return drivein_force



person_is_monitored = pd.DataFrame(X_test, columns = list(feature_dict.values()))

shap_force_plot_of_data(model, person_is_monitored[person_is_monitored["sex_Male"]==1])
