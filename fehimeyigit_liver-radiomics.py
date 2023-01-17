# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
from subprocess import check_output
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data = pd.DataFrame(pd.read_excel("/kaggle/input/servantz/liver_radiomics.xlsx"))

data
#del data["Kontrast(arteryal,kortikomedüller,geç)"]
#del data["Ad-soyad"]
#del data["tc kimlik no"]
#del data['Cinsiyet']
#del data['Kalsifikasyon (Yok:0, Var:1)']
#del data['Hemoraji']
#del data['Nekroz']
#del data['Yaş']
#del data['Lokalizasyon']
#del data['Boyut(en uzun aks cm)']
#del data['Heterojenite']
#del data['aynı taraf sürrenal invazyon']
#del data['Pararenal invazyon']
#del data['renal arter-ven invazyonu']
#del data['lenf nodu']
#del data['uzak metastaz']
del data['fuhrman grade']

data.columns[data.isnull().any()]
data.isnull().sum()
data.dropna(inplace=True)
data.rename(columns = {"Patolojik subtip":"Patolojik_subtip"}, inplace = True) 
#data.Patolojik_subtip.replace(to_replace=dict(clear_cell=1, papiller=0), inplace=True)
# pandas pivot with multiple variables

df = data.loc[:,['FOInterquartileRange', 'Skewness',
       'Uniformity', 'Median', 'Energy', 'RobustMeanAbsoluteDeviation',
       'MeanAbsoluteDeviation', 'TotalEnergy', 'Maximum', 'RootMeanSquared',
       '90Percentile', 'Minimum', 'Entropy', 'Range', 'Variance',
       '10Percentile', 'Kurtosis', 'Mean', 'GLCM JointAverage', 'SumAverage',
       'JointEntropy', 'ClusterShade', 'MaximumProbability', 'Idmn',
       'JointEnergy', 'Contrast', 'DifferenceEntropy', 'InverseVariance',
       'DifferenceVariance', 'Idn', 'Idm', 'Correlation', 'Autocorrelation',
       'SumEntropy', 'MCC', 'SumSquares', 'ClusterProminence', 'Imc2', 'Imc1',
       'DifferenceAverage', 'Id', 'ClusterTendency',
       'GRLM ShortRunLowGrayLevelEmphasis', 'GrayLevelVariance',
       'LowGrayLevelRunEmphasis', 'GrayLevelNonUniformityNormalized',
       'RunVariance', 'GrayLevelNonUniformity', 'LongRunEmphasis',
       'ShortRunHighGrayLevelEmphasis', 'RunLengthNonUniformity',
       'ShortRunEmphasis', 'LongRunHighGrayLevelEmphasis', 'RunPercentage',
       'LongRunLowGrayLevelEmphasis', 'RunEntropy', 'HighGrayLevelRunEmphasis',
       'RunLengthNonUniformityNormalized']]
df1 = data.Patolojik_subtip
x = dict(zip(df1.unique(),"rgb"))
row_colors = df1.map(x)
cg = sns.clustermap(df,row_colors=row_colors,figsize=(12, 12),metric="correlation")
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(),rotation = 0,size =8)
plt.show()
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
color_dict=dict(zip(np.unique(data.Patolojik_subtip),np.array(['g','skyblue'])))
target_df = pd.DataFrame({"Patolojik_subtip":data.Patolojik_subtip})
row_colors = target_df.Patolojik_subtip.map(color_dict)

                                            


sns.clustermap(data,
               metric="correlation",
               standard_scale=1,
               row_colors=row_colors,cmap="summer")


species=data['Patolojik_subtip']

lut = dict(zip(species.unique(), "rbg"))
row_colors = species.map(lut)

patient_subtype = species
g = sns.clustermap(data,z_score=1, row_colors=row_colors,figsize=(10, 10),cmap="plasma")


sns.clustermap(data,z_score=1, cmap="plasma",figsize=(10, 10))
#correlation map
f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(data.corr(), annot=False, linewidths=.5, fmt= '.1f',ax=ax,cmap="Blues")
y = data.Patolojik_subtip                          
list = ['Patolojik_subtip']
x = data.drop(list,axis = 1 )
x.head()

a=1
b=0
ax = sns.countplot(y,label="Count")       
a,b= y.value_counts()
print('Clear Cell: ',a)
print('NonClear Cell : ',b)
data
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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions
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
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 1)
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(x_train,y_train)

y_pred = cv.predict(x_test)

print("Accuracy: {}".format(cv.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


x=abs(x)
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features
data1=x.iloc[:,[4,7,50,47,52,36,32,14,49]]
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
x_train, x_test, y_train, y_test = train_test_split(data1,y,test_size=0.2,random_state = 1)
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(x_train,y_train)

y_pred = cv.predict(x_test)

print("Accuracy: {}".format(cv.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
y1 = data.Patolojik_subtip                          
list = ['Patolojik_subtip']
x1 = data.drop(list,axis = 1 )
x1.head()
# split data train 80 % and test 20 %
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.2, random_state=1)
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(x_train1, y_train1)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train1.columns[rfecv.support_])
data2=x.loc[:,['TotalEnergy', 'RunLengthNonUniformity', 'GrayLevelNonUniformity',
       'ClusterProminence', 'Autocorrelation', 'Variance',
       'ShortRunHighGrayLevelEmphasis'
      ]]
data2
x_train2, x_test2, y_train2, y_test2 = train_test_split(data2, y1, test_size=0.2, random_state=1)
clf_rf_2 = RandomForestClassifier()      
clr_rf_2 = clf_rf_2.fit(x_train2,y_train2)
ac_2 = accuracy_score(y_test2,clf_rf_2.predict(x_test2))
print('Accuracy is: ',ac_2)
cm_2 = confusion_matrix(y_test2,clf_rf_2.predict(x_test2))
sns.heatmap(cm_2,annot=True,fmt="d")
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
#x,y = data1.loc[:,data1.columns != 'CovidORnot'], data1.loc[:,'CovidORnot']
knn.fit(data2,y1)
prediction = knn.predict(data2)
print('Prediction: {}'.format(prediction))
# train test split
from sklearn.model_selection import train_test_split
x_train3,x_test3,y_train3,y_test3 = train_test_split(data2,y1,test_size = 0.2,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 2)
#x,y = data1.loc[:,data1.columns != 'CovidORnot'], data1.loc[:,'CovidORnot']
knn.fit(x_train3,y_train3)
prediction = knn.predict(x_test3)
#print('Prediction: {}'.format(prediction))
print('With KNN (K=2) accuracy is: ',knn.score(x_test3,y_test3)) # accuracy


# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train3,y_train3)
    #train accuracy
    train_accuracy.append(knn.score(x_train3, y_train3))
    # test accuracy
    test_accuracy.append(knn.score(x_test3, y_test3))
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
x_train4, x_test4, y_train4, y_test4 = train_test_split(data2,y1,test_size=0.2,random_state = 1)
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(x_train4,y_train4)

y_pred = cv.predict(x_test4)

print("Accuracy: {}".format(cv.score(x_test4, y_test4)))
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