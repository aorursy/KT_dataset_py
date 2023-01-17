# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/table2/data.csv")
data
del data["diagnostics_Versions_PyRadiomics"]
del data["diagnostics_Versions_SimpleITK"]
del data["diagnostics_Configuration_Settings"]
del data["diagnostics_Configuration_EnabledImageTypes"]
del data["diagnostics_Image-original_Hash"]
del data["diagnostics_Image-original_Dimensionality"]
del data["diagnostics_Versions_Numpy"]
del data["diagnostics_Versions_PyWavelet"]
del data["diagnostics_Versions_Python"]
del data["Unnamed: 0"]
del data["diagnostics_Image-original_Spacing"]
del data["diagnostics_Image-original_Size"]
del data["diagnostics_Mask-original_Hash"]
del data["diagnostics_Mask-original_Spacing"]
del data["diagnostics_Mask-original_Size"]
del data["diagnostics_Mask-original_BoundingBox"]
del data["diagnostics_Mask-corrected_Size"]
del data["diagnostics_Mask-corrected_Spacing"]
del data["diagnostics_Mask-original_CenterOfMassIndex"]
del data["diagnostics_Mask-original_CenterOfMass"]
del data["diagnostics_Mask-corrected_BoundingBox"]
del data["diagnostics_Mask-corrected_CenterOfMassIndex"]
del data["diagnostics_Mask-corrected_CenterOfMass"]
data=data.drop([36],axis=0)
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from scipy.stats import ttest_ind
column_list= data.columns
df_out=pd.DataFrame()
c=0
d=0
p_r_list = []
p_p_list = []
while c< 118:
    while d<118:
        g1=data[column_list[c]]
        g2=data[column_list[d]]
        p_r, p_p = ttest_ind (g1, g2)
        p_r_list.append(p_r)
        p_p_list.append(p_p)
        d=d+1
    c=c+1
df_out['Columns'] = column_list
df_out['p_p_values'] = p_p_list
df_out['p_r_values'] = p_r_list
df_out.to_csv('indttestTABLE2.csv', index=False)
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal

column_list= data.columns
df_out=pd.DataFrame()
c=0
d=0
p_r_list = []
p_p_list = []
while c< 118:
    while d<118:
        g1=data[column_list[c]]
        g2=data[column_list[d]]
        p_r, p_p = pearsonr (g1, g2)
        p_r_list.append(p_r)
        p_p_list.append(p_p)
        d=d+1
    c=c+1
df_out['Columns'] = column_list
df_out['p_p_values'] = p_p_list
df_out['p_r_values'] = p_r_list
df_out.to_csv('pearsortable2.csv', index=False)
x_data = data.drop(["CovidORnot"],axis=1)
y = data.CovidORnot

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib._cm import datad
from matplotlib._cm_listed import cmaps as cmaps_listed
sns.clustermap(data,
               metric="correlation",
               standard_scale=1,cmap="summer")
color_dict=dict(zip(np.unique(data.CovidORnot),np.array(['g','skyblue'])))
target_df = pd.DataFrame({"CovidORnot":data.CovidORnot})
row_colors = target_df.CovidORnot.map(color_dict)

                                            


sns.clustermap(data,
               metric="correlation",
               standard_scale=1,
               row_colors=row_colors,cmap="summer")
species=data['CovidORnot']
lut = dict(zip(species.unique(), "rbg"))
row_colors = species.map(lut)

patient_subtype = species
g = sns.clustermap(data,z_score=1, row_colors=row_colors,figsize=(10, 10),cmap="plasma")
#correlation map
f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(data.corr(), annot=False, linewidths=.5, fmt= '.1f',ax=ax,cmap="Blues")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

# split data train 80 % and test 20 %
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#x_train_2 = select_feature.transform(x_train)
#x_test_2 = select_feature.transform(x_test)
#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier()      
clr_rf_2 = clf_rf_2.fit(x_train,y_train)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test))
print('Accuracy is: ',ac_2)
cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test))
sns.heatmap(cm_2,annot=True,fmt="d")
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
#x,y = data1.loc[:,data1.columns != 'CovidORnot'], data1.loc[:,'CovidORnot']
knn.fit(x,y)
prediction = knn.predict(x)
print('Prediction: {}'.format(prediction))
# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 2)
#x,y = data1.loc[:,data1.columns != 'CovidORnot'], data1.loc[:,'CovidORnot']
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
#print('Prediction: {}'.format(prediction))
print('With KNN (K=2) accuracy is: ',knn.score(x_test,y_test)) # accuracy
# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))
# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state = 1)
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(x_train,y_train)

y_pred = cv.predict(x_test)

print("Accuracy: {}".format(cv.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
# generate 2 class dataset
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=1)
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]
# fit a model
#model = LogisticRegression(solver='lbfgs')
#model.fit(trainX, trainy)
# Learn to predict each class against the other
model = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,random_state=2))
model.fit(trainX, trainy).decision_function(testX)
# predict probabilities
lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='AUC = %0.2f' % lr_auc)
plt.legend(loc = 'lower left')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
plt.title('ROC Curve of SVM')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=1)

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,y_train)

y_scores = knn.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'KNN_AUC = %0.2f' % roc_auc)
plt.plot(lr_fpr, lr_tpr, 'g', label = 'SVM_AUC = %0.2f' % lr_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curves of kNN and SVM')
plt.show()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features
data1=x.iloc[:,[33,88,48,37,2,20,44,82,112,80]]
# split data train 80 % and test 20 %
x_train1, x_test1, y_train1, y_test1 = train_test_split(data1, y, test_size=0.2, random_state=1)


#x_train_2 = select_feature.transform(x_train)
#x_test_2 = select_feature.transform(x_test)
#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier()      
clr_rf_2 = clf_rf_2.fit(x_train1,y_train1)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test1))
print('Accuracy is: ',ac_2)
cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test1))
sns.heatmap(cm_2,annot=True,fmt="d")

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
#x,y = data1.loc[:,data1.columns != 'CovidORnot'], data1.loc[:,'CovidORnot']
knn.fit(data1,y)
prediction = knn.predict(data1)
print('Prediction: {}'.format(prediction))
# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data1,y,test_size = 0.2,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 2)
#x,y = data1.loc[:,data1.columns != 'CovidORnot'], data1.loc[:,'CovidORnot']
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
#print('Prediction: {}'.format(prediction))
print('With KNN (K=2) accuracy is: ',knn.score(x_test,y_test)) # accuracy





# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))
# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
x_train, x_test, y_train, y_test = train_test_split(data1,y,test_size=0.3,random_state = 1)
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(x_train,y_train)

y_pred = cv.predict(x_test)

print("Accuracy: {}".format(cv.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
# split data train 80 % and test 20 %
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])
data2=x.loc[:,['diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum',
       'diagnostics_Image-original_Maximum',
       'diagnostics_Mask-original_VolumeNum',
       'diagnostics_Mask-corrected_VolumeNum',
       'diagnostics_Mask-corrected_Mean', 'diagnostics_Mask-corrected_Minimum',
       'diagnostics_Mask-corrected_Maximum', 'original_shape_VoxelVolume',
       'original_shape_Maximum3DDiameter', 'original_shape_MeshVolume',
       'original_shape_MajorAxisLength', 'original_shape_Sphericity',
       'original_shape_LeastAxisLength', 'original_shape_Elongation',
       'original_shape_Maximum2DDiameterSlice', 'original_shape_SurfaceArea',
       'original_shape_Maximum2DDiameterColumn',
       'original_shape_Maximum2DDiameterRow',
       'original_gldm_GrayLevelVariance', 'original_gldm_DependenceEntropy',
       'original_gldm_SmallDependenceHighGrayLevelEmphasis',
       'original_gldm_LargeDependenceLowGrayLevelEmphasis',
       'original_glcm_JointAverage', 'original_glcm_ClusterShade',
       'original_glcm_Contrast', 'original_glcm_DifferenceEntropy',
       'original_glcm_InverseVariance', 'original_glcm_DifferenceVariance',
       'original_glcm_Idm', 'original_glcm_Correlation', 'original_glcm_MCC',
       'original_glcm_SumSquares', 'original_glcm_DifferenceAverage',
       'original_glcm_Id', 'original_glcm_ClusterTendency',
       'original_firstorder_Skewness', 'original_firstorder_Uniformity',
       'original_firstorder_Energy', 'original_firstorder_TotalEnergy',
       'original_firstorder_Maximum', 'original_firstorder_Minimum',
       'original_firstorder_Entropy', 'original_firstorder_Variance',
       'original_glrlm_GrayLevelNonUniformityNormalized',
       'original_glrlm_GrayLevelNonUniformity',
       'original_glrlm_ShortRunHighGrayLevelEmphasis',
       'original_glrlm_RunLengthNonUniformity',
       'original_glrlm_ShortRunEmphasis',
       'original_glrlm_LongRunLowGrayLevelEmphasis',
       'original_glszm_GrayLevelNonUniformityNormalized',
       'original_glszm_SizeZoneNonUniformity',
       'original_glszm_GrayLevelNonUniformity',
       'original_glszm_SmallAreaHighGrayLevelEmphasis',
       'original_glszm_ZonePercentage', 'original_glszm_ZoneEntropy',
       'original_glszm_SmallAreaLowGrayLevelEmphasis',
       'original_ngtdm_Coarseness', 'original_ngtdm_Busyness']]
data2
x_train, x_test, y_train, y_test = train_test_split(data2, y, test_size=0.2, random_state=1)
clf_rf_2 = RandomForestClassifier()      
clr_rf_2 = clf_rf_2.fit(x_train,y_train)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test))
print('Accuracy is: ',ac_2)
cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test))
sns.heatmap(cm_2,annot=True,fmt="d")
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
#x,y = data1.loc[:,data1.columns != 'CovidORnot'], data1.loc[:,'CovidORnot']
knn.fit(data2,y)
prediction = knn.predict(data2)
print('Prediction: {}'.format(prediction))
# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data2,y,test_size = 0.2,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 2)
#x,y = data1.loc[:,data1.columns != 'CovidORnot'], data1.loc[:,'CovidORnot']
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
#print('Prediction: {}'.format(prediction))
print('With KNN (K=2) accuracy is: ',knn.score(x_test,y_test)) # accuracy


# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))
# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))




from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state = 1)
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(x_train,y_train)

y_pred = cv.predict(x_test)

print("Accuracy: {}".format(cv.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)
clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(x_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()
# split data train 80 % and test 20 %
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
#normalization
#x_train_N = (x_train-x_train.mean())/(x_train.max()-x_train.min())
#x_test_N = (x_test-x_test.mean())/(x_test.max()-x_test.min())

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(x_train)

plt.figure(1, figsize=(14, 13))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')