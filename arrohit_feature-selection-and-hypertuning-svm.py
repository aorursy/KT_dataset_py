# Required Libraries are imported
import warnings  
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics import confusion_matrix
import time
import scikitplot as skplt
import itertools
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')
# Reading the Dataset
df_train=pd.read_csv("../input/train.csv")
df_test=pd.read_csv("../input/test.csv")

# Check null values
df_train.isnull().values.any()
df_test.isnull().values.any()
# No null values in train and test data

# Top 5 rows
df_train.head()
# Subject col not usefull hence dropped
if('subject' in df_train.columns):
    df_train.drop('subject', axis =1, inplace=True)
if('subject' in df_test.columns):
    df_test.drop('subject', axis =1, inplace=True)

# Encoding target - converting non-num to num variable
le = preprocessing.LabelEncoder()
for x in [df_train,df_test]:
    x['Activity'] = le.fit_transform(x.Activity)

# Split into features and class
df_traindata, df_trainlabel = df_train.iloc[:, 0:len(df_train.columns) - 1], df_train.iloc[:, -1]
df_testdata, df_testlabel = df_test.iloc[:, 0:len(df_test.columns) -1], df_test.iloc[:, -1]

warnings.filterwarnings('ignore')
# Baseline - comparing model accuracy using all features across classifiers 
classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    GaussianNB(),
    LogisticRegression()
    ]


# Naive Train Accuracy
algo = []
scores = []
for clf in classifiers:
    algo.append(clf.__class__.__name__)
    scores.append(cross_val_score(clf,df_traindata,df_trainlabel, cv=5).mean())
warnings.filterwarnings('ignore')
Naivescore_df_Train = pd.DataFrame({'Algorithm': algo, 'Score': scores}).set_index('Algorithm')


# Naive Test Accuracy

algo = []
scores = []

for clf in classifiers:
    clf = clf.fit(df_traindata, df_trainlabel)
    y_pred = clf.predict(df_testdata)
    algo.append(clf.__class__.__name__)
    scores.append(accuracy_score(y_pred, df_testlabel))
warnings.filterwarnings('ignore')
Naivescore_df_Test  = pd.DataFrame({'Algorithm': algo, 'Score': scores}).set_index('Algorithm')

# Bar plot between Train and Test Accuracy
fig = plt.figure(figsize=(5,5)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
width = .3

Naivescore_df_Train.Score.plot(kind='bar',color='green',ax=ax,width=width, position=0)
Naivescore_df_Test.Score.plot(kind='bar',color='red', ax=ax2,width = width,position=1)

ax.grid(None, axis=1)
ax2.grid(None)

ax.set_ylabel('Train')
ax2.set_ylabel('Test')

ax.set_xlim(-1,7)
plt.show()



# Feature selection using Random Forest Classifier

# Bagged decision trees for feature importance- embedded method
Rtree_clf = RandomForestClassifier()
Rtree_clf = Rtree_clf.fit(df_traindata,df_trainlabel)
model = SelectFromModel(Rtree_clf, prefit=True)
RF_tree_featuresTrain=df_traindata.loc[:, model.get_support()]
RF_tree_featuresTest = df_testdata.loc[:, model.get_support()]
warnings.filterwarnings('ignore')

# Based on Feature Selection only 87 features were selected

# Feature Importance

# Important scores
# for name, importance in zip(df_traindata, Rtree_clf.feature_importances_):
#     print(name, "=", importance)

importances = Rtree_clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in Rtree_clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
indices.shape
indices = indices[:200]
# Feature Ranking
#print("Feature ranking:")
#for f in range(200):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plots feature importances

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.xlabel("# of Features ")
plt.ylabel("Importance Score")
plt.bar(range(200), importances[indices],color="r", yerr=std[indices], align="center")
plt.xlim([0, 200])
plt.show()




skplt.estimators.plot_learning_curve(Rtree_clf,RF_tree_featuresTrain,df_trainlabel)

# Applying RFE Cross validation to find number of features
# The "accuracy" scoring is proportional to the number of correct classifications

# Before we apply RFE we need to know the optimal number of features. Hence RFECV crossvalidation technique is used to find 
# the optimal number of features based on the accuracy score in the training set. 

# Applying RFECV with svm classifier
svc=SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), # Stratified fold inorder to reduce bias
              scoring='accuracy')
rfetrain=rfecv.fit(RF_tree_featuresTrain, df_trainlabel)
print('Optimal number of features :', rfecv.n_features_)


# Plot showing the Cross Validation score
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()



# Applying RFE with optimal number of features
rfe = RFE(estimator=svc, n_features_to_select=rfecv.n_features_, step=1)
rfe = rfe.fit(RF_tree_featuresTrain, df_trainlabel)

rfe_train=RF_tree_featuresTrain.loc[:, rfe.get_support()]
rfe_test=RF_tree_featuresTest.loc[:, rfe.get_support()]


# Checking the Accuracy after rfe
# Train Accuracy
print("Train Accuracy:",cross_val_score(svc,rfe_train,df_trainlabel, cv=5).mean())
# Test Accuracy
scv = svc.fit(rfe_train, df_trainlabel)
y_pred = scv.predict(rfe_test)
print("Test Accuracy:",accuracy_score(y_pred, df_testlabel))


# Variance threshold
selector = VarianceThreshold(0.95*(1-.95))
varsel=selector.fit(rfe_train)
rfe_train.loc[:, varsel.get_support()].shape
# 55
vartrain=rfe_train.loc[:, varsel.get_support()]
vartest=rfe_test.loc[:, varsel.get_support()]

# Checking the Accuracy after Variance threshold
# Train Accuracy
print("Train Accuracy:",cross_val_score(svc,vartrain,df_trainlabel, cv=5).mean())

# Test Accuracy
scv = svc.fit(vartrain, df_trainlabel)
y_pred = scv.predict(vartest)
print("Test Accuracy:",accuracy_score(y_pred, df_testlabel))


# PCA
pca = PCA(n_components = len(vartrain.columns))
pca_traindata = pca.fit(vartrain)

pca_traindata.explained_variance_
pca_traindata.n_components_
pcatrain = pca_traindata.transform(vartrain)
pcatest = pca_traindata.transform(vartest)
cum_ratio = (np.cumsum(pca_traindata.explained_variance_ratio_))


# Visualize PCA result
plt.plot(np.cumsum(pca_traindata.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

# 21 features - constant after that
pca = PCA(n_components = 21)
pca_traindata = pca.fit(vartrain)

pca_traindata.explained_variance_
pca_traindata.n_components_
pcatrain = pca_traindata.transform(vartrain)
pcatest = pca_traindata.transform(vartest)
(np.cumsum(pca_traindata.explained_variance_ratio_))

# PCA in 2D projection
 
skplt.decomposition.plot_pca_2d_projection(pca, vartrain, df_trainlabel)

# Checking Accuracy after applying PCA

# Train Accuracy
print("Train Accuracy:",cross_val_score(svc,pcatrain,df_trainlabel, cv=5).mean())

# Test Accuracy
scv = svc.fit(pcatrain, df_trainlabel)
y_pred = scv.predict(pcatest)
ac_score = accuracy_score(y_pred, df_testlabel)
print("Test Accuracy:",accuracy_score(y_pred, df_testlabel))
# Confusion Matrix

cf_mat = confusion_matrix(df_testlabel, y_pred)
print("Accuracy: %f" %ac_score) 
activities = le.classes_

# Plotting Confusion matrix heatmap
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plot_confusion_matrix(cf_mat, classes=activities,title="Confusion Matrix for Test data")

# Parameter Tuning 

# Perfromance tuning using GridScore
param_grid = [
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svr = SVC()
clf = GridSearchCV(svr, param_grid,cv=5)
clf.fit(pcatrain,df_trainlabel)
print(clf.best_params_)




# Train Accuracy 
svr = SVC(kernel="rbf",C=1000,gamma=0.001)
print("Train Accuracy:",cross_val_score(svr,pcatrain,df_trainlabel, cv=5).mean())
# Test Accuracy
scv = svr.fit(pcatrain, df_trainlabel)
y_pred = scv.predict(pcatest)
print("Test Accuracy:",accuracy_score(y_pred, df_testlabel))
 