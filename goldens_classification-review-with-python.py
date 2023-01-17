#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder



from sklearn.pipeline import make_pipeline

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb



from sklearn.metrics import confusion_matrix, classification_report



from pdpbox import pdp, get_dataset, info_plots

import shap



from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve, ClassificationReport

from yellowbrick.model_selection import LearningCurve, ValidationCurve, learning_curve





import warnings

warnings.filterwarnings(action="ignore")
df=pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')

print(df.shape)
#checking for duplicated rows and missing values

print(df.duplicated().sum())

print(df.isnull().sum().sum())
#checking types

df.dtypes.sort_values()
df.describe()
for col in df.select_dtypes(include='object').columns:

    print(col)

    print(df[col].unique())
df.drop("duration",axis=1, inplace=True)
#checking class balance

df.deposit.value_counts()/df.deposit.count()
#I'm going to use StratifiedShuffleSplit to preserve the class proportions.

sss=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

for train_index, test_index in sss.split(df.drop("deposit",axis=1), df.deposit):

    traindf=df.loc[train_index] #to select only rows (with all columns) we dont need comma and colon.

    testdf= df.loc[test_index]
# setting my own palette

mypalette = ['seagreen', 'indianred']

sns.set_palette(mypalette)

sns.palplot(sns.color_palette())
# Scatterplots to search for linear and non-linear relationships and histograms.

sns.pairplot(traindf, diag_kind='hist',  hue= 'deposit', height=1.5, 

             diag_kws={"edgecolor":"k", "alpha":0.5},

             plot_kws={"alpha":0.5})

#Pearson’s Correlations, which measures the strength of a linear relationship

sns.heatmap(traindf.corr(method='pearson'), cmap="Greys", annot=True)
#barplots showing the frequency of each category separated by label

plt.figure(figsize=[12,14])

features=["marital", "education", "contact", "default", "housing", "loan", "poutcome", "month"]

n=1

for f in features:

    plt.subplot(4,2,n)

    sns.countplot(x=f, hue='deposit', edgecolor="black", alpha=0.7, data=traindf)

    sns.despine()

    plt.title("Countplot of {}  by deposit".format(f))

    n=n+1

plt.tight_layout()

plt.show()





    

plt.figure(figsize=[14,4])

sns.countplot(x='job', hue='deposit',edgecolor="black", alpha=0.7, data=traindf)

sns.despine()

plt.title("Countplot of job by deposit")

plt.show()
#encoding target label

LE=LabelEncoder()

df['deposit']=LE.fit_transform(df.deposit.values)



#encoding categorical features

df=pd.get_dummies(df)
#partitioning again

for train_index, test_index in sss.split(df.drop("deposit",axis=1), df.deposit):

    traindf=df.loc[train_index]

    testdf= df.loc[test_index]
#partition x/y

xtrain=traindf.drop('deposit', axis=1)

ytrain=traindf.deposit



xtest=testdf.drop('deposit', axis=1)

ytest=testdf.deposit
# pipeline combining transformers and estimator

pipe_knn= make_pipeline(StandardScaler(), KNeighborsClassifier())

 

# grid searh to choose the best (combination of) hyperparameters

gs_knn=GridSearchCV(estimator= pipe_knn,

               param_grid={'kneighborsclassifier__n_neighbors':[4,5,6,7]},

               scoring='accuracy',

               cv=10)



# nested cross validation combining grid search (inner loop) and k-fold cv (outter loop)

gs_knn_scores = cross_val_score(gs_knn, X=xtrain, y=ytrain, cv=5,scoring='accuracy', n_jobs=-1)



# fit, and fit with best estimator

gs_knn.fit(xtrain, ytrain)

gs_knn_best=gs_knn.best_estimator_

gs_knn_best.fit(xtrain, ytrain)



print('Train Accuracy:   {0:.1f}%'.format(gs_knn.score(xtrain, ytrain)*100))

print('CV Mean Accuracy: {0:.1f}%'.format(np.mean(gs_knn_scores)*100))

print('Test Accuracy:    {0:.1f}%'.format(gs_knn.score(xtest, ytest)*100))
# pipeline combining transformers and estimator

pipe_svm= make_pipeline(StandardScaler(), SVC(random_state=1))



# grid searh to choose the best (combination of) hyperparameters

r=[0.1,1,10]

pg_svm=[{'svc__C':r, 'svc__kernel':['linear']},

        {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]



gs_svm=GridSearchCV(estimator= pipe_svm,

               param_grid= pg_svm,

               scoring='accuracy',

               cv=2)



# nested cross validation combining grid search (inner loop) and k-fold cv (outter loop)

gs_svm_scores = cross_val_score(gs_svm, X=xtrain, y=ytrain, cv=5,scoring='accuracy', n_jobs=-1)



# fit, and fit with best estimator

gs_svm.fit(xtrain, ytrain)

gs_svm_best=gs_svm.best_estimator_

gs_svm_best.fit(xtrain, ytrain)



print('Train Accuracy:   {0:.1f}%'.format(gs_svm.score(xtrain, ytrain)*100))

print('CV Mean Accuracy: {0:.1f}%'.format(np.mean(gs_svm_scores)*100))

print('Test Accuracy:    {0:.1f}%'.format(gs_svm.score(xtest, ytest)*100))
rf= RandomForestClassifier(random_state=1)



# grid searh to choose the best (combination of) hyperparameters

pg_rf={'n_estimators': [100,200,400],'max_depth': [20,40,50,60]}



gs_rf=GridSearchCV(estimator= rf,

               param_grid= pg_rf,

               scoring='accuracy',

               cv=2)



# nested cross validation combining grid search (inner loop) and k-fold cv (outter loop)

gs_rf_scores = cross_val_score(gs_rf, X=xtrain, y=ytrain, cv=5,scoring='accuracy', n_jobs=-1)



# fit, and fit with best estimator

gs_rf.fit(xtrain, ytrain)

gs_rf_best=gs_rf.best_estimator_

gs_rf_best.fit(xtrain, ytrain)



print('Train Accuracy:   {0:.1f}%'.format(gs_rf.score(xtrain, ytrain)*100))

print('CV Mean Accuracy: {0:.1f}%'.format(np.mean(gs_rf_scores)*100))

print('Test Accuracy:    {0:.1f}%'.format(gs_rf.score(xtest, ytest)*100))
# estimator

xb= xgb.XGBClassifier(random_state=1)



# grid searh to choose the best (combination of) hyperparameters

pg_xb={'n_estimators':[100,200,400], 'max_depth':[20,40,50]}



gs_xb=GridSearchCV(estimator= xb,

               param_grid= pg_xb,

               scoring='accuracy',

               cv=2)



# nested cross validation combining grid search (inner loop) and k-fold cv (outter loop)

gs_xb_scores = cross_val_score(gs_xb, X=xtrain, y=ytrain, cv=5,scoring='accuracy', n_jobs=-1)



# fit, and fit with best estimator

gs_xb.fit(xtrain, ytrain)

gs_xb_best=gs_xb.best_estimator_

gs_xb_best.fit(xtrain, ytrain)



print('Train Accuracy:   {0:.1f}%'.format(gs_xb.score(xtrain, ytrain)*100))

print('CV Mean Accuracy: {0:.1f}%'.format(np.mean(gs_xb_scores)*100))

print('Test Accuracy:    {0:.1f}%'.format(gs_xb.score(xtest, ytest)*100))
# using random forest results: confusion_matrix and classification report

ypreds=gs_rf_best.predict(xtest)

print(confusion_matrix(ypreds ,ytest))

print(classification_report(ypreds ,ytest))
# using random forest results: confusion_matrix and classification report (yellowbrick)



visualizer_cr = ClassificationReport(gs_rf_best, classes=["no", "yes"], support=True)

visualizer_cr.fit(xtrain, ytrain)

visualizer_cr.score(xtest, ytest)

visualizer_cr.show()
# using random forest results: precision recall curve

visualizer_pr = PrecisionRecallCurve(gs_rf_best)

visualizer_pr.fit(xtrain, ytrain)

visualizer_pr.score(xtest, ytest)

visualizer_pr.show()
# using random forest results: ROC curve

visualizer_roc = ROCAUC(gs_rf_best, classes=["no", "yes"])

visualizer_roc.fit(xtrain, ytrain)

visualizer_roc.score(xtest, ytest)

visualizer_roc.show()
# using random forest here to get feature importances

importances= gs_rf_best.feature_importances_

feature_importances= pd.Series(importances, index=xtrain.columns).sort_values(ascending=False)

sns.barplot(x=feature_importances[0:10], y=feature_importances.index[0:10], palette="rocket")

sns.despine()

plt.xlabel("Feature Importances")

plt.ylabel("Features")

plt.show()
# partial dependence plot of balance

pdp_data=pdp.pdp_isolate(model=gs_rf_best, dataset=xtrain, model_features=xtrain.columns, feature='balance')

pdp.pdp_plot(pdp_data, 'balance')

plt.show()
# partial dependence plot of age

pdp_data=pdp.pdp_isolate(model=gs_rf_best, dataset=xtrain, model_features=xtrain.columns, feature='age')

pdp.pdp_plot(pdp_data, 'age')

plt.show()
shap.initjs()
#high-speed exact algorithm for tree ensemble methods

explainer = shap.TreeExplainer(gs_xb_best)

shap_values = explainer.shap_values(xtrain)
# first instance, feature values and its effects on prediction

shap.force_plot(explainer.expected_value, shap_values[0,:], xtrain.iloc[0,:],matplotlib=True)
# all instances, feature values and its effects on prediction

# shap.force_plot(explainer.expected_value, shap_values, xtrain)
shap.summary_plot(shap_values, xtrain)
shap.dependence_plot("balance", shap_values, xtrain)
shap.dependence_plot("age", shap_values, xtrain)