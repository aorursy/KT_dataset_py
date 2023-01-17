# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
traindat = pd.read_csv("../input/train.csv")
testdat = pd.read_csv("../input/test.csv")
##Initial exploration of data
traindat.head()
traindat.info()
#Age,Cabin, and Embarked have missing data points
traindat.isnull().sum()
#Missing data: age
x=traindat["Age"]
x= x[~np.isnan(x)]
sns.distplot(x) #Skewd distribution -- use median rather than mean while imputing data
#Missing data: Cabin
x = traindat["Cabin"]
y = []
for val in x:
    if pd.isna(val):
        y.append("Missing")
    else:
        y.append("Not missing")
sns.countplot(y)
##Too much data missing--Ignore?
#Missing data: Embarked
x = traindat["Embarked"]
x = x[~pd.isna(x)]
sns.countplot(y=x, palette="Spectral")
#There were only 2 values with missing data for this variable, use 'S' as it is the most abundant.
traindat.describe()#Quick look at numerical variables
traindat["Sex"].value_counts()
traindat["Ticket"].value_counts()
traindat["Cabin"].value_counts()
traindat["Embarked"].value_counts()
#S = Southhampton, C = Cherbourg, Q = Queenstown
traindat["Age"].median()
##Impute missing data
traindat["Age"].fillna(traindat["Age"].median(),inplace=True)
traindat["Embarked"].fillna("S",inplace=True)
traindat.head()
##Feature engineering
#The data dictionary mentions the following regarding the "sibsp" and "parch" variables
#sibsp: The dataset defines family relations in this way...
#Sibling = brother, sister, stepbrother, stepsister
#Spouse = husband, wife (mistresses and fiancés were ignored)
#parch: The dataset defines family relations in this way...
#Parent = mother, father
#Child = daughter, son, stepdaughter, stepson
#Some children travelled only with a nanny, therefore parch=0 for them.

#To sumarize these two by adding them together as as Group Size (gpsz)
traindat["gpsz"] = traindat["SibSp"] + traindat["Parch"] + 1
# The '+1' accounts for the person itself, otherwise the group size would be 0
sns.countplot(x=traindat["gpsz"])
#There seems to be a fairly large number of people traviling on their own (gpzs=0)
#I'll keep this data with three categories, 0 for those traveling alone, 1 for those with one
#person another person, and 2 for those groups with 2 or more people in them
traindat["gpsz"][traindat["gpsz"]>3]=3
sns.countplot(x="gpsz",data=traindat)
##Exploratory analysis
traindat.info()
#Age vs Survival
sns.violinplot(x="Survived", y="Age", data=traindat,
              split=True, inner="quart")
#Age vs Group size
x = pd.to_numeric(testdat["Age"])
pt = sns.violinplot(x="gpsz", y= "Age", data=traindat,
              split=True, inner="quart")
pt.set(xlabel="Group size")
##Survival vs Class
pt = sns.barplot("Pclass","Survived",data=traindat, palette="dark")
pt.set(xlabel="Class")
#There seems to be a correlation in between Class an survival
##Survival vs Class
pt = sns.barplot("gpsz","Survived",data=traindat, palette="dark")
pt.set(xlabel="Group size")
#People traveling alone survived less, could be because most of the people on their own were men?
##Survival vs Class
pt = sns.countplot(x="gpsz",hue="Sex",data=traindat, palette="dark")
pt.set(xlabel="Group size")
#It seems like men were traveling on their own much more than women
##Survival per port
pt = sns.barplot("Embarked","Survived",data=traindat, palette="dark")
pt.set(xlabel="Port")
#People embarked at Cherbourg were more likely to survive, there might be a correlation with any of the previous variables?
##Sex per port
pt = sns.countplot(x="Embarked",hue="Sex",data=traindat, palette="dark")
pt.set(xlabel="Group size")
#Lots of men got in at Southhampton
##Class per port
pt = sns.countplot(x="Embarked",hue="Pclass",data=traindat, palette="dark")
pt.set(xlabel="Group size")
#And lots of people in third class, which also seems to not be great for survival.
##Group sizes per port
pt = sns.countplot(x="Embarked",hue="gpsz",data=traindat, palette="dark")
pt.set(xlabel="Group size")
#And lots of people in third class, which also seems to not be great for survival.
##So far it looks like there are some variables that are quite associated with survival: Sex and Class. 
#Port embarked and group size may also be, although there seem to be heavily correlated to Sex.
##Data clean-up before modeling
#Train set
traindat.drop("SibSp",axis=1,inplace=True)
traindat.drop("Parch", axis=1, inplace=True)
traindat.drop("Name", axis=1, inplace=True)#No use for this
traindat.drop("PassengerId", axis=1, inplace=True)#No use for this
traindat.drop("Ticket", axis=1, inplace=True)#Won't really tell much, probably correlated with class
traindat.drop("Fare", axis=1, inplace=True)#Same as above
traindat.drop("Cabin", axis=1, inplace=True)#Same as above

traindatSurv = traindat["Survived"]

traindat.drop("Survived",axis=1,inplace=True)
traindat.info()
#Get dumy varaibles for categorical variables (Pclass,Sex,Embarked,gpsz)
trainmodel= pd.get_dummies(traindat,columns=["Pclass","Sex","Embarked","gpsz"],drop_first=True)
trainmodel.head()
trainmodel.info()
#Repeat for test data set
testdat.info()
#Add Group Size
testdat_pID = testdat["PassengerId"]
testdat["Age"].fillna(traindat["Age"].median(),inplace=True)
testdat["gpsz"] = testdat["SibSp"] + testdat["Parch"] + 1
testdat["gpsz"][testdat["gpsz"]>3]=3

testdat.drop("PassengerId",axis=1,inplace=True)
testdat.drop("Name",axis=1,inplace=True)
testdat.drop("SibSp",axis=1,inplace=True)
testdat.drop("Parch",axis=1,inplace=True)
testdat.drop("Ticket",axis=1,inplace=True)
testdat.drop("Fare",axis=1,inplace=True)
testdat.drop("Cabin",axis=1,inplace=True)
testdat.info()
testfn = pd.get_dummies(testdat,columns=["Pclass","Sex","Embarked","gpsz"],drop_first=True)
testfn.head()
#–#–#–#–#–#–#–#
#Try logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
logreg = LogisticRegression()
##Quick look at this model using a statistical approach
import statsmodels.api as sm
log_stats = sm.Logit(traindatSurv,trainmodel)
log_res = log_stats.fit()
log_res.summary2()
##Seems like all variables selected could be correlated with survival
#Use GridSearch to find optimal model for predicton
log_paramgrid = {'C':[10**i for i in range(-6,6)],
                'class_weight':[None,'balanced'],
                }
grid_logreg = GridSearchCV(estimator=LogisticRegression(),param_grid=log_paramgrid,scoring="roc_auc")
grid_logreg.fit(X=trainmodel,y=traindatSurv)
grid_logreg.best_estimator_
#Most of these seem to be default values
#Compare scores of optimized and 'out of the box' model
grid_logreg.score(trainmodel,traindatSurv)
logreg_simple = logreg.fit(trainmodel,traindatSurv)
logreg_simple.score(trainmodel,traindatSurv)
y_pred = grid_logreg.predict_proba(trainmodel)[:,1]
#Test cross-validation and compare scores
from sklearn.model_selection import cross_val_score


scores = cross_val_score(grid_logreg,X=trainmodel,y=traindatSurv,cv=3,scoring='f1')
scores
print("Mean score: ",scores.mean())
print("StDev: ",scores.std())
#Seems fairly consistent
#Plot ROC curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import auc
fpr, tpr, thresh = roc_curve(y_true=traindatSurv,y_score=y_pred)
auc = auc(fpr,tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve | logistic regression')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.show()
#Plot learning curve
from sklearn.model_selection import learning_curve
train_size, train_scores, cv_scores = learning_curve(grid_logreg, X=trainmodel, y=traindatSurv,cv=3,scoring="f1")
#Taken from https://datascience.stackexchange.com/questions/29520/how-to-plot-learning-curve-and-validation-curve-while-using-pipeline
def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')

    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of training points')
    plt.ylabel('F1-measure')
    plt.grid(ls='--')
    plt.legend(loc='best')
    plt.show()
plot_learning_curve(train_sizes=train_size, train_scores=train_scores, test_scores=cv_scores,title="Learning curve | logistic regression")
#The train set gives good results, but the test set seems quite variable, maybe this method is overfitting the data
#Test Decision trees and Random forests
from sklearn.tree import DecisionTreeClassifier
dectre_paramgrid = {'max_depth':[1,2,3,4,5,6],
                    "min_samples_split":[5,10,15,20,25,30,40,50],
                    "min_samples_leaf":[5,10,15,20,25,30,40,50],
                    'criterion':["gini","entropy"]
                }
grid_dectree = GridSearchCV(estimator=DecisionTreeClassifier(),param_grid=dectre_paramgrid,scoring="roc_auc")
dectre = grid_dectree.fit(trainmodel,traindatSurv)
dectre.best_estimator_
dectre.fit_params
import graphviz
from sklearn.tree import export_graphviz
features = list(trainmodel.columns.values)
classes = np.where(traindatSurv==0,"No","Yes")


treegraph = export_graphviz(decision_tree=dectre.best_estimator_,out_file=None,
                            filled=True,feature_names=features,class_names=classes)
treegraph = graphviz.Source(treegraph)
treegraph
##Test with cross validation
scores = cross_val_score(grid_dectree,X=trainmodel,y=traindatSurv,cv=3,scoring="f1")
scores
print("Mean score: ",scores.mean())
print("StDev: ",scores.std())
#Slightly better than loggistic regression
#It is not possible to obtain ROC curves for decission trees stragight away, so let's assess this model with other paramers
from sklearn.metrics import confusion_matrix, accuracy_score
y_predtre = grid_dectree.predict(trainmodel)
confmat_tre = confusion_matrix(traindatSurv,y_predtre)
print("Accuracy: {0:.3f}".format(accuracy_score(traindatSurv,y_predtre)))
print("Sensitivity: {0:.3f}".format(confmat_tre[1][1]/confmat_tre[1,:].sum()))
print("Specificity: {0:.3f}".format(confmat_tre[0][0]/confmat_tre[:,0].sum()))
#The sensitivity, or TPR, is fairly low, meaning that there is relatively large proportion of positive cases being mislabeled.
##Try a random forest
from sklearn.ensemble import RandomForestClassifier
RandomForestClassifier()
paramgrid_forest = {
                "min_samples_split":[2,5,10,20,25],
                "min_samples_leaf":[2,5,10,15,20,25],
               'n_estimators':[int(10*i) for i in range(1,11)]
              }
from sklearn.grid_search import GridSearchCV
grid_forest = GridSearchCV(RandomForestClassifier(),param_grid=paramgrid_forest,scoring='roc_auc')
grid_forest.fit(trainmodel,traindatSurv)
grid_forest.best_estimator_
y_predforest = grid_forest.predict_proba(trainmodel)[:,1]
##Cross-validations scores
scores = cross_val_score(grid_forest,X=trainmodel,y=traindatSurv,cv=3,scoring="f1")
print("Mean score: ",scores.mean())
print("StDev: ",scores.std())
#Seems fairly consistent
#Roc curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
fpr, tpr, thresh = roc_curve(y_true=traindatSurv,y_score=y_predforest)
auc = auc(fpr,tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % auc)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve | Random Forest')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.show()
#Plot learning curve
from sklearn.model_selection import learning_curve
train_size, train_scores, cv_scores = learning_curve(grid_forest, X=trainmodel, y=traindatSurv,cv=3,scoring="f1")
plot_learning_curve(train_sizes=train_size, train_scores=train_scores, test_scores=cv_scores,title="Learning curve | Random Forest Classifier")
##Important features in Random Forest
feature = trainmodel.columns
score = grid_forest.best_estimator_.feature_importances_
feat_imp = pd.DataFrame({'feature':feature,'score':score})
feat_imp.sort_values(by=["score"],inplace=True)

pt = sns.barplot(x="feature",y="score",data=feat_imp)
pt.set_xticklabels(pt.get_xticklabels(),rotation=45,fontdict={'verticalalignment':'baseline',"horizontalalignment":'right'})
#Of the variables used, Sex is the most important, followed by being in the third class, and age.
#Try a support vector machine (SVM)
from sklearn.svm import SVC
SVC()
#GridSearch is a bit slow when implemented with SVC, so I will try it using the rbf (Gaussian) and linear kernel
SVClin = SVC(kernel='linear',probability=True)
SVCrbf = SVC(kernel='rbf',probability=True)
SVClin.fit(trainmodel,traindatSurv)
SVCrbf.fit(trainmodel,traindatSurv)
y_predSVClin = SVClin.predict_proba(trainmodel)[:,1]
y_predSVCrbf = SVCrbf.predict_proba(trainmodel)[:,1]
#Roc curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
fpr_lin, tpr_lin, thresh_lin = roc_curve(y_true=traindatSurv,y_score=y_predSVClin)
auc_lin = auc(fpr_lin,tpr_lin)
fpr_rbf, tpr_rbf, thresh_rbf = roc_curve(y_true=traindatSurv,y_score=y_predSVCrbf)
auc_rbf = auc(fpr_rbf,tpr_rbf)
plt.plot(fpr_lin, tpr_lin, label='ROC curve | linear (area = %0.3f)' % auc_lin)
plt.plot(fpr_rbf, tpr_rbf, label='ROC curve | rbf (area = %0.3f)' % auc_rbf)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve | SVCs')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.show()
#Plot learning curve
from sklearn.model_selection import learning_curve
train_size, train_scores, cv_scores = learning_curve(SVClin, X=trainmodel, y=traindatSurv,cv=3,scoring="f1")
plot_learning_curve(train_sizes=train_size, train_scores=train_scores, test_scores=cv_scores,title="Learning curve | logistic regression")
train_size, train_scores, cv_scores = learning_curve(SVCrbf, X=trainmodel, y=traindatSurv,cv=3, scoring="f1")
plot_learning_curve(train_sizes=train_size, train_scores=train_scores, test_scores=cv_scores,title="Learning curve | logistic regression")
##Of all the methods teste, Random forest algorithm used here seems to provide the best results for this dataset
y_final = grid_forest.predict(testfn)
submit = pd.DataFrame({
    "PassengerId": testdat_pID,
    "Survived": y_final
})
submit.to_csv('submission.csv.csv', index=False)
