# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import AxesGrid

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/kag_risk_factors_cervical_cancer.csv")
df.head()
df_nan = df.replace("?", np.nan)
df_nan.head()
df1 = df_nan.convert_objects(convert_numeric=True)
df1.hist()

fig = plt.gcf()

fig.set_size_inches(25,17)
H = df1['Hinselmann'].T.sum()

S = df1['Schiller'].T.sum()

C = df1['Citology'].T.sum()

B = df1['Biopsy'].T.sum()

H+S+C+B
sns.heatmap(df1.isnull(), cbar=False)
df1.columns = df1.columns.str.replace(' ', '')  #deleting spaces for ease of use
df1.drop(['STDs:Timesincefirstdiagnosis','STDs:Timesincelastdiagnosis','STDs:cervicalcondylomatosis','STDs:AIDS'],inplace=True,axis=1)
df1.isnull().T.any().T.sum()
sns.heatmap(df1.isnull(), cbar=False)

df.shape
df = df1[df1.isnull().sum(axis=1) < 10]
df.shape
sns.heatmap(df.isnull(), cbar=False)
numerical_df = ['Age', 'Numberofsexualpartners', 'Firstsexualintercourse','Numofpregnancies', 'Smokes(years)',

                'Smokes(packs/year)','HormonalContraceptives(years)','IUD(years)','STDs(number)']

categorical_df = ['Smokes','HormonalContraceptives','IUD','STDs','STDs:condylomatosis',

                  'STDs:vulvo-perinealcondylomatosis', 'STDs:syphilis','STDs:pelvicinflammatorydisease', 'STDs:genitalherpes',

                  'STDs:molluscumcontagiosum','STDs:HIV','STDs:HepatitisB', 'STDs:HPV', 'STDs:Numberofdiagnosis',

                  'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy']
for feature in numerical_df:

    print(feature,'',df[feature].convert_objects(convert_numeric=True).mean())

    feature_mean = round(df[feature].convert_objects(convert_numeric=True).mean(),1)

    df[feature] = df[feature].fillna(feature_mean)
(df['Age'] == 0).astype(int).sum() # checking if any 0 values in a column that could not contain such values
for feature in categorical_df:

    

    df[feature] = df[feature].convert_objects(convert_numeric=True).fillna(0.0)

    

#Filling binominal values with 0, with the assumption that if present, feature would have been recorded
df5 = df.copy()
df5['YAFSI'] = df5['Age'] - df5['Firstsexualintercourse']

df5['CNT'] = df.astype(bool).sum(axis=1)

df5['SEX'] = (df5['Numofpregnancies']+1) * (df5['Numberofsexualpartners']+1)

df5['FirstSexZ'] = (((df5['Firstsexualintercourse']+1) - (df5.loc[:,'Firstsexualintercourse'].mean())+1) / (df5.loc[:,'Firstsexualintercourse'].var()+1)*100)

df5['SexZ'] = (((df5['Numberofsexualpartners']+1) - (df5.loc[:,'Numberofsexualpartners'].mean())+1) / (df5.loc[:,'Numberofsexualpartners'].var()+1)*100)

df5['PILL'] = (((df5['HormonalContraceptives(years)']+1) - (df5.loc[:,'HormonalContraceptives(years)'].mean())+1) / (df5.loc[:,'HormonalContraceptives(years)'].var()+1)*100)

df5['SSY'] = df5['Age'] - df5['Smokes(years)']

df5['SPYP'] = df5['Numberofsexualpartners'] / df5['YAFSI']

df5['SP'] = df5['Smokes(years)'] / df5['Age']

df5['HCP'] = df5['HormonalContraceptives(years)'] / df5['Age']

df5['STDP'] = df5['STDs(number)'] / df5['Age']

df5['IUDP'] = df5['IUD(years)'] / df5['Age']

df5['TSP'] = df5['Smokes(packs/year)'] * df5['Smokes(years)']

df5['NPP'] = df5['Numofpregnancies'] / df5['Age']

df5['NSPP'] = df5['Numberofsexualpartners'] / df5['Age']

df5['NDP'] = df5['STDs:Numberofdiagnosis'] / df5['Age']

df5['YAHC'] = df5['Age'] - df5['HormonalContraceptives(years)']

df5['YAIUD'] = df5['Age'] - df5['IUD(years)']

df5['NPSP'] = df5['Numofpregnancies'] / df5['Numberofsexualpartners']

df5['IUDSY'] = df5['IUD(years)'] / df5['YAFSI']

df5['HCSY'] = df5['HormonalContraceptives(years)'] / df5['YAFSI']
df5.replace([np.inf, -np.inf], np.nan, inplace = True) #deleting extreme values caused by calculations
df = df5.copy()
numerical_df = ['Age', 'Numberofsexualpartners', 'Firstsexualintercourse','Numofpregnancies', 'Smokes(years)',

                'Smokes(packs/year)','HormonalContraceptives(years)','IUD(years)','STDs(number)', 'YAFSI', 'CNT',

                'FirstSexZ', 'SexZ', 'PILL','SSY','SPYP', 'SP', 'HCP', 'STDP', 'IUDP', 'TSP', 'NPP', 'NSPP', 'NDP',

                'YAHC', 'YAIUD', 'NPSP', 'IUDSY', 'HCSY']
#Adding in our newly created values to the NA filter

for feature in numerical_df:

    print(feature,'',df[feature].convert_objects(convert_numeric=True).mean())

    feature_mean = round(df[feature].convert_objects(convert_numeric=True).mean(),1)

    df[feature] = df[feature].fillna(feature_mean)
sns.heatmap(df.isnull(), cbar=False)
df.columns[df.isna().any()].tolist()
figure = plt.figure(figsize=(6,9), dpi=100);    

graph = figure.add_subplot(111);



dfN = df['Age']

freq = pd.value_counts(dfN)

bins = freq.index

x=graph.bar(bins, freq.values) #gives the graph without NaN



plt.ylabel('Frequency')

plt.xlabel('Age')

figure.show()
dfN.eq(0).any().any()
df[df['Age'] > 58]
df['Age'] = np.clip(df['Age'], a_max=58, a_min=None)
figure = plt.figure(figsize=(6,9), dpi=100);    

graph = figure.add_subplot(111);



dfN = df['Age']

freq = pd.value_counts(dfN)

bins = freq.index

x=graph.bar(bins, freq.values) #gives the graph without NaN



plt.ylabel('Frequency')

plt.xlabel('Age')

figure.show()
category_df = ['Hinselmann', 'Schiller','Citology', 'Biopsy']
for feature in categorical_df:

   sns.factorplot(feature,data=df,size=3,kind='count')
corrmat = df.corr()
k = 30 #number of variables for heatmap

cols = corrmat.nlargest(k, 'HormonalContraceptives')['HormonalContraceptives'].index



cm = df[cols].corr()



plt.figure(figsize=(20,20))



sns.set(font_scale=1.5)

hm = sns.heatmap(cm, cbar=True, cmap='Set1' ,annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 12},

                 yticklabels = cols.values, xticklabels = cols.values)

plt.show()
df = df.round()
target = df['Hinselmann'] | df['Schiller'] | df['Citology'] | df['Biopsy'] 
df = df.drop(columns=['Hinselmann', 'Schiller', 'Citology', 'Biopsy', 'Dx', 'Dx:Cancer', 'Smokes(years)', 'Smokes(packs/year)'])
from sklearn.model_selection import train_test_split

from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.4, random_state=1) # 60% training and 40% test

X_tr, X_te, y_tr, y_te = train_test_split(df, target, test_size=0.2, random_state=1) # 60% training 20% test for SMOTE()
figure = plt.figure(figsize=(2,4), dpi=100);    

graph = figure.add_subplot(111);



df5 = y_train

freq = pd.value_counts(y_train)

bins = freq.index

x=graph.bar(bins, freq.values) #gives the graph without NaN



plt.ylabel('Frequency')

plt.xlabel('Level of Cancer Test Output in Training Set')

figure.show()
## oversampling

from imblearn.over_sampling import SMOTE, ADASYN

X_trOVR, y_trOVR = SMOTE(random_state=2).fit_sample(X_tr, y_tr)
figure = plt.figure(figsize=(2,4), dpi=100);    

graph = figure.add_subplot(111);



df5 = y_trOVR

freq = pd.value_counts(y_trOVR)

bins = freq.index

x=graph.bar(bins, freq.values) #gives the graph without NaN



plt.ylabel('Frequency')

plt.xlabel('Level of Cancer Test Output in Oversampled Set')

figure.show()
#starting with a simple decision tree, attempting to classify the origional data into cancer test or no

from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy of Decision Tree",metrics.accuracy_score(y_test, y_pred))
import scikitplot as skplt

preds = clf.predict(X_test)

skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=preds)

plt.show()
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import tree

from sklearn.datasets import load_wine

from IPython.display import SVG

from graphviz import Source

from IPython.display import display

import graphviz 

dot_data = tree.export_graphviz(clf, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("Tree") 

dot_data = tree.export_graphviz(clf, out_file=None, filled=True,

                                feature_names=X_test.columns, class_names=["0", "1"])  

graph = graphviz.Source(dot_data)  

graph
importances = clf.feature_importances_

indices = np.argsort(importances)



plt.figure(1)

plt.figure(figsize=(15,20), dpi=100); 

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), X_train.columns)

plt.xlabel('Relative Importance')
clf = tree.DecisionTreeClassifier()

clf2 = clf.fit(X_trOVR, y_trOVR)
y_pred2 = clf2.predict(X_te)
print("Accuracy of Oversampled Decision Tree:",metrics.accuracy_score(y_te, y_pred2))
import graphviz 

dot_data = tree.export_graphviz(clf2, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("Tree") 

dot_data = tree.export_graphviz(clf2, out_file=None, filled=True,

                                feature_names=X_test.columns, class_names=["0", "1"])  

graph = graphviz.Source(dot_data)  

graph 
preds2 = clf2.predict(X_te)

skplt.metrics.plot_confusion_matrix(y_true=y_te, y_pred=preds2)

plt.show()
#print('Value counts of each target variable:',target.value_counts())

#cancer_df_label = target.astype(int)

#cancer_df_label = cancer_df_label.values.ravel()



#print('Final feature vector shape:',df.shape)

#print('Final target vector shape',cancer_df_label.shape)
importances = clf2.feature_importances_

indices = np.argsort(importances)



plt.figure(1)

plt.figure(figsize=(13,17), dpi=100); 

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), X_train.columns)

plt.xlabel('Relative Importance')
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))

rand = sel.fit(X_train, y_train)

#rand2 = sel.fit(X_trainOVR, y_trainOVR)
selected_feat= X_train.columns[(sel.get_support())]

len(selected_feat)
print(selected_feat)
k = 16 #number of variables for heatmap

cols = corrmat.nlargest(k, 'HCSY')['HCSY'].index



cm = df[selected_feat].corr()



plt.figure(figsize=(16,16))



sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, cmap='Set1' ,annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 10},

                 yticklabels = cols.values, xticklabels = cols.values)

plt.show()
dfFOR = df[selected_feat]
dfFOR.shape
X_train2, X_test2, y_train2, y_test2 = train_test_split(dfFOR, target, test_size=0.4, random_state=1) # 60% training and 40% test

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(dfFOR, target, test_size=0.2, random_state=1) # 80% training and 20% test
## oversampling

X_trOVR2, y_trOVR2 = SMOTE(random_state=2).fit_sample(X_tr2, y_tr2)
clf3 = clf.fit(X_train2, y_train2)

clf4 = clf.fit(X_trOVR2, y_trOVR2)
y_pred3 = clf3.predict(X_test2)

y_pred4 = clf4.predict(X_te2)

print("Accuracy of Trimmed Decision Tree:",metrics.accuracy_score(y_test2, y_pred3))

print("Accuracy of Trimmed Oversampled Decision Tree:",metrics.accuracy_score(y_te2, y_pred4))

print("Recall of Trimmed Decision Tree:",metrics.recall_score(y_test2, y_pred3))

print("Recall of Trimmed Oversampled Decision Tree:",metrics.recall_score(y_te2, y_pred4))
preds3 = clf3.predict(X_test2)

skplt.metrics.plot_confusion_matrix(y_true=y_test2, y_pred=preds3)

plt.title('Trimmed DT')

plt.show()

preds4 = clf4.predict(X_te2)

skplt.metrics.plot_confusion_matrix(y_true=y_te2, y_pred=preds4)

plt.title('Trimmed and Oversampled DT')

plt.show()
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier



clfAB = AdaBoostClassifier(n_estimators=100)

scores = cross_val_score(clfAB, X_train2, y_train2, cv=5)

scores.mean()  
clfABA = clfAB.fit(X_train, y_train)

y_predAB = clfAB.predict(X_test)

print("Accuracy of Origional AdaBoost:",metrics.accuracy_score(y_test, y_predAB))



clfAB2 = clfAB.fit(X_train2, y_train2)

y_predAB2 = clfAB2.predict(X_test2)

print("Accuracy of Trimmed AdaBoost:",metrics.accuracy_score(y_test2, y_predAB2))



clfAB3 = clfAB.fit(X_trOVR2, y_trOVR2)

y_predAB2 = clfAB2.predict(X_te2)

print("Accuracy of Oversampled and Trimmed AdaBoost:",metrics.accuracy_score(y_te2, y_predAB2))
skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=y_predAB)

plt.title('Origional Adaboost')

plt.show()



predAB2 = clfAB2.predict(X_test2)

skplt.metrics.plot_confusion_matrix(y_true=y_test2, y_pred=predAB2)

plt.title('Trimmed Adaboost')

plt.show()



#predAB3 = clfAB3.predict(X_test)

skplt.metrics.plot_confusion_matrix(y_true=y_te2, y_pred=y_predAB2)

plt.title('Adaboost Oversampled and Trimmed')

plt.show()
from sklearn.svm import SVC

clfs = SVC(gamma='auto')

svm = clfs.fit(X_train, y_train) 

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',

    max_iter=-1, probability=False, random_state=None, shrinking=True,

    tol=0.001, verbose=False)

sv = svm.predict(X_test)
from sklearn.svm import SVC

clfs = SVC(gamma='auto')

svm = clfs.fit(X_train2, y_train2) 

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',

    max_iter=-1, probability=False, random_state=None, shrinking=True,

    tol=0.001, verbose=False)

svv = svm.predict(X_test2)
clfs = SVC(gamma='auto')

svm = clfs.fit(X_trOVR2, y_trOVR2) 

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',

    max_iter=-1, probability=False, random_state=None, shrinking=True,

    tol=0.001, verbose=False)

svvv = svm.predict(X_te2)
print("Accuracy of SVM",metrics.accuracy_score(y_test, sv))

print("Accuracy of SVM Trimmed",metrics.accuracy_score(y_test2, svv))

print("Accuracy of SVM Trimmed",metrics.accuracy_score(y_te2, svvv))
skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=sv)

plt.title('SVM Origional')

plt.show()



skplt.metrics.plot_confusion_matrix(y_true=y_test2, y_pred=svv)

plt.title('SVM Trimmed')

plt.show()



skplt.metrics.plot_confusion_matrix(y_true=y_te2, y_pred=svvv)

plt.title('SVM Trimmed and Oversampled')

plt.show()
from sklearn.utils import resample

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from matplotlib import pyplot

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

import statistics



# load dataset

data = df

values = data.values

# configure bootstrap

n_iterations = 100

n_size = int(len(data) * 0.50)

# run bootstrap

stats = list()

stats2 = list()

for i in range(n_iterations):

	# prepare train and test sets

	train = resample(values, n_samples=n_size)

	test = np.array([x for x in values if x.tolist() not in train.tolist()])

	# fit model

	model = DecisionTreeClassifier()

	model.fit(train[:,:-1], train[:,-1])

	# evaluate model

	predictions = model.predict(test[:,:-1])

	score = accuracy_score(test[:,-1], predictions)

	score2 = recall_score(test[:,-1], predictions, average = 'macro')

	#print(score)

	stats.append(score)

	#print(score2)

	stats2.append(score2)

# plot scores

pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerDT = max(0.0, np.percentile(stats, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperDT = min(1.0, np.percentile(stats, p))

meanDT = statistics.mean(stats)

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanDT, lowerDT*100, upperDT*100))



# plot scores

pyplot.hist(stats2)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerDT2 = max(0.0, np.percentile(stats2, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperDT2 = min(1.0, np.percentile(stats2, p))

meanDT2 = statistics.mean(stats2)

print('%.3f average recall, confidence interval %.2f%% and %.2f%%' % (meanDT2, lowerDT2*100, upperDT2*100))
# load dataset

data = df

values = data.values

# configure bootstrap

n_iterations = 100

n_size = int(len(data) * 0.50)

# run bootstrap

stats = list()

stats2 = list()

for i in range(n_iterations):

	# prepare train and test sets

	train = resample(values, n_samples=n_size)

	test = np.array([x for x in values if x.tolist() not in train.tolist()])

	# fit model

	model = RandomForestClassifier()

	model.fit(train[:,:-1], train[:,-1])

	# evaluate model

	predictions = model.predict(test[:,:-1])

	score = accuracy_score(test[:,-1], predictions)

	score2 = recall_score(test[:,-1], predictions, average = 'macro')

	#print(score)

	stats.append(score)

	#print(score2)

	stats2.append(score2)

# plot scores

pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerRF = max(0.0, np.percentile(stats, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperRF = min(1.0, np.percentile(stats, p))

meanRF = statistics.mean(stats)

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanRF, lowerRF*100, upperRF*100))



# plot scores

pyplot.hist(stats2)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerRF2 = max(0.0, np.percentile(stats2, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperRF2 = min(1.0, np.percentile(stats2, p))

meanRF2 = statistics.mean(stats2)

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanRF2, lowerRF2*100, upperRF2*100))
# load dataset

data = dfFOR

values = data.values

# configure bootstrap

n_iterations = 100

n_size = int(len(data) * 0.50)

# run bootstrap

stats = list()

stats2 = list()

for i in range(n_iterations):

	# prepare train and test sets

	train = resample(values, n_samples=n_size)

	test = np.array([x for x in values if x.tolist() not in train.tolist()])

	# fit model

	model = DecisionTreeClassifier()

	model.fit(train[:,:-1], train[:,-1])

	# evaluate model

	predictions = model.predict(test[:,:-1])

	score = accuracy_score(test[:,-1], predictions)

	score2 = recall_score(test[:,-1], predictions, average = 'macro')

	#print(score)

	stats.append(score)

	#print(score2)

	stats2.append(score2)

# plot scores

pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerDTT = max(0.0, np.percentile(stats, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperDTT = min(1.0, np.percentile(stats, p))

meanDTT = statistics.mean(stats)

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanDTT, lowerDTT*100, upperDTT*100))



# plot scores

pyplot.hist(stats2)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerDTT2 = max(0.0, np.percentile(stats2, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperDTT2 = min(1.0, np.percentile(stats2, p))

meanDTT2 = statistics.mean(stats2)

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanDTT2, lowerDTT2*100, upperDTT2*100))
from sklearn.linear_model import LogisticRegression

# load dataset

data = dfFOR

values = data.values

# configure bootstrap

n_iterations = 100

n_size = int(len(data) * 0.50)

# run bootstrap

stats = list()

stats2 = list()

for i in range(n_iterations):

	# prepare train and test sets

	train = resample(values, n_samples=n_size)

	test = np.array([x for x in values if x.tolist() not in train.tolist()])

	# fit model

	model = LogisticRegression(random_state=0, solver='lbfgs',class_weight='balanced')

	model.fit(train[:,:-1], train[:,-1])

	# evaluate model

	predictions = model.predict(test[:,:-1])

	score = accuracy_score(test[:,-1], predictions)

	score2 = recall_score(test[:,-1], predictions, average = 'macro')

	#print(score)

	stats.append(score)

	#print(score2)

	stats2.append(score2)

# plot scores

pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerLRT = max(0.0, np.percentile(stats, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperLRT = min(1.0, np.percentile(stats, p))

meanLRT = statistics.mean(stats)

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanLRT, lowerLRT*100, upperLRT*100))



# plot scores

pyplot.hist(stats2)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerLRT2 = max(0.0, np.percentile(stats2, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperLRT2 = min(1.0, np.percentile(stats2, p))

meanLRT2 = statistics.mean(stats2)

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanLRT2, lowerLRT2*100, upperLRT2*100))
from sklearn.linear_model import LogisticRegression

# load dataset

data = df

values = data.values

# configure bootstrap

n_iterations = 100

n_size = int(len(data) * 0.50)

# run bootstrap

stats = list()

stats2 = list()

for i in range(n_iterations):

	# prepare train and test sets

	train = resample(values, n_samples=n_size)

	test = np.array([x for x in values if x.tolist() not in train.tolist()])

	# fit model

	model = LogisticRegression(random_state=0, solver='lbfgs',class_weight='balanced')

	model.fit(train[:,:-1], train[:,-1])

	# evaluate model

	predictions = model.predict(test[:,:-1])

	score = accuracy_score(test[:,-1], predictions)

	score2 = recall_score(test[:,-1], predictions, average = 'macro')

	#print(score)

	stats.append(score)

	#print(score2)

	stats2.append(score2)

# plot scores

pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerLR = max(0.0, np.percentile(stats, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperLR = min(1.0, np.percentile(stats, p))

meanLR = statistics.mean(stats)

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanLR, lowerLR*100, upperLR*100))



# plot scores

pyplot.hist(stats2)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerLR2 = max(0.0, np.percentile(stats2, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperLR2 = min(1.0, np.percentile(stats2, p))

meanLR2 = statistics.mean(stats2)

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanLR2, lowerLR2*100, upperLR2*100))
from sklearn.svm import LinearSVC

# load dataset

data = df

values = data.values

# configure bootstrap

n_iterations = 100

n_size = int(len(data) * 0.50)

# run bootstrap

stats = list()

stats2 = list()

for i in range(n_iterations):

	# prepare train and test sets

	train = resample(values, n_samples=n_size)

	test = np.array([x for x in values if x.tolist() not in train.tolist()])

	# fit model

	model = LinearSVC(random_state=0, tol=1e-5, class_weight='balanced')

	model.fit(train[:,:-1], train[:,-1])

	# evaluate model

	predictions = model.predict(test[:,:-1])

	score = accuracy_score(test[:,-1], predictions)

	score2 = recall_score(test[:,-1], predictions, average = 'macro')

	#print(score)

	stats.append(score)

	#print(score2)

	stats2.append(score2)

# plot scores

pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerS = max(0.0, np.percentile(stats, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperS = min(1.0, np.percentile(stats, p))

meanSVC = statistics.mean(stats)

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanSVC, lowerS*100, upperS*100))



# plot scores

pyplot.hist(stats2)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerS2 = max(0.0, np.percentile(stats2, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperS2 = min(1.0, np.percentile(stats2, p))

meanSVC2 = statistics.mean(stats2)

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanSVC2, lowerS2*100, upperS2*100))
from sklearn.svm import LinearSVC

# load dataset

data = dfFOR

values = data.values

# configure bootstrap

n_iterations = 100

n_size = int(len(data) * 0.50)

# run bootstrap

stats = list()

stats2 = list()

for i in range(n_iterations):

	# prepare train and test sets

	train = resample(values, n_samples=n_size)

	test = np.array([x for x in values if x.tolist() not in train.tolist()])

	# fit model

	model = LinearSVC(random_state=0, tol=1e-5, class_weight='balanced')

	model.fit(train[:,:-1], train[:,-1])

	# evaluate model

	predictions = model.predict(test[:,:-1])

	score = accuracy_score(test[:,-1], predictions)

	score2 = recall_score(test[:,-1], predictions, average = 'macro')

	#print(score)

	stats.append(score)

	#print(score2)

	stats2.append(score2)

# plot scores

pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerST = max(0.0, np.percentile(stats, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperST = min(1.0, np.percentile(stats, p))

meanSVCT = statistics.mean(stats)

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanSVCT, lowerST*100, upperST*100))



# plot scores

pyplot.hist(stats2)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerST2 = max(0.0, np.percentile(stats2, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperST2 = min(1.0, np.percentile(stats2, p))

meanSVCT2 = statistics.mean(stats2)

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanSVCT2, lowerST2*100, upperST2*100))
from sklearn.ensemble import GradientBoostingClassifier

# load dataset

data = df

values = data.values

# configure bootstrap

n_iterations = 100

n_size = int(len(data) * 0.50)

# run bootstrap

stats = list()

stats2 = list()

for i in range(n_iterations):

	# prepare train and test sets

	train = resample(values, n_samples=n_size)

	test = np.array([x for x in values if x.tolist() not in train.tolist()])

	# fit model

	model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)

	model.fit(train[:,:-1], train[:,-1])

	# evaluate model

	predictions = model.predict(test[:,:-1])

	score = accuracy_score(test[:,-1], predictions)

	score2 = recall_score(test[:,-1], predictions, average = 'macro')

	#print(score)

	stats.append(score)

	#print(score2)

	stats2.append(score2)

# plot scores

pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerG = max(0.0, np.percentile(stats, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperG = min(1.0, np.percentile(stats, p))

meanG = statistics.mean(stats)

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanG, lowerG*100, upperG*100))



# plot scores

pyplot.hist(stats2)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerG2 = max(0.0, np.percentile(stats2, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperG2 = min(1.0, np.percentile(stats2, p))

meanG2 = statistics.mean(stats2)

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanG2, lowerG2*100, upperG2*100))
from sklearn.ensemble import GradientBoostingClassifier

# load dataset

data = dfFOR

values = data.values

# configure bootstrap

n_iterations = 100

n_size = int(len(data) * 0.50)

# run bootstrap

stats = list()

stats2 = list()

for i in range(n_iterations):

	# prepare train and test sets

	train = resample(values, n_samples=n_size)

	test = np.array([x for x in values if x.tolist() not in train.tolist()])

	# fit model

	model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)

	model.fit(train[:,:-1], train[:,-1])

	# evaluate model

	predictions = model.predict(test[:,:-1])

	score = accuracy_score(test[:,-1], predictions)

	score2 = recall_score(test[:,-1], predictions, average = 'macro')

	#print(score)

	stats.append(score)

	#print(score2)

	stats2.append(score2)

# plot scores

pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerGT = max(0.0, np.percentile(stats, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperGT = min(1.0, np.percentile(stats, p))

meanGT = statistics.mean(stats)

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanGT, lowerGT*100, upperGT*100))



# plot scores

pyplot.hist(stats2)

pyplot.show()

# confidence intervals

alpha = 0.9

p = ((1.0-alpha)/2.0) * 100

lowerGT2 = max(0.0, np.percentile(stats2, p))

p = (alpha+((1.0-alpha)/2.0)) * 100

upperGT2 = min(1.0, np.percentile(stats2, p))

meanGT2 = statistics.mean(stats2)

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanGT2, lowerGT2*100, upperGT2*100))
print('Decision Tree')

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanDT, lowerDT*100, upperDT*100))

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanDT2, lowerDT2*100, upperDT2*100))

print('Decision Tree Trimmed')

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanDTT, lowerDTT*100, upperDTT*100))

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanDTT2, lowerDTT2*100, upperDTT2*100))

print('Random Forest Classifier')

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanRF, lowerRF*100, upperRF*100))

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanRF2, lowerRF2*100, upperRF2*100))

print('Random Forest Classifier Trimmed')

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanDT, lowerDT*100, upperDT*100))

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanDT2, lowerDT2*100, upperDT2*100))

print('Logistic Regression')

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanLR, lowerLR*100, upperLR*100))

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanLR2, lowerLR2*100, upperLR2*100))

print('Logistic Regression Trimmed')

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanLRT, lowerLRT*100, upperLRT*100))

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanLRT2, lowerLRT2*100, upperLRT2*100))

print('SVC')

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanSVC, lowerS*100, upperS*100))

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanSVC2, lowerS2*100, upperS2*100))

print('SVC Trimmed')

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanSVCT, lowerST*100, upperST*100))

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanSVCT2, lowerST2*100, upperST2*100))

print('Gradient Boosting')

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanG, lowerG*100, upperG*100))

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanG2, lowerG2*100, upperG2*100))

print('Gradient Boosting Trimmed')

print('%.3f average Accuracy with confidence interval %.1f%% and %.1f%%' % (meanGT, lowerGT*100, upperGT*100))

print('%.3f average recall with confidence interval %.1f%% and %.1f%%' % (meanGT2, lowerGT2*100, upperGT2*100))

clf = tree.DecisionTreeClassifier()

clf2 = clf.fit(X_train2, y_train2)



importances = clf2.feature_importances_

indices = np.argsort(importances)



plt.figure(1)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), dfFOR.columns)

plt.xlabel('Relative Importance')
import eli5

from eli5.sklearn import PermutationImportance



permumtation_impor = PermutationImportance(clf2, random_state=2019).fit(X_test2, y_test2)

eli5.show_weights(permumtation_impor, feature_names = X_test2.columns.tolist())
from pdpbox import pdp, get_dataset, info_plots

random_forest = RandomForestClassifier(n_estimators=500, random_state=2019).fit(X_test2, y_test2)

def pdpplot( feature_to_plot, pdp_model = clf2, pdp_dataset = X_test2, pdp_model_features = X_test2.columns):

    pdp_cancer = pdp.pdp_isolate(model=pdp_model, dataset=pdp_dataset, model_features=pdp_model_features, feature=feature_to_plot)

    fig, axes = pdp.pdp_plot(pdp_cancer, feature_to_plot, figsize = (13,6),plot_params={})

     #_ = axes['pdp_ax'].set_ylabel('Probability of Cancer')

    

pdpplot('CNT')

pdpplot('PILL')

pdpplot('TSP')

pdpplot('YAFSI')

pdpplot('YAHC')

pdpplot('SSY')

pdpplot('SEX')

pdpplot('SexZ')

pdpplot('HormonalContraceptives(years)')

pdpplot('YAIUD')

pdpplot('FirstSexZ')

pdpplot('Numofpregnancies')

pdpplot('YAIUD')



plt.show()

#Remember the trimmed data set

#X_train2, X_test2, y_train2, y_test2 = train_test_split(dfFOR, target, test_size=0.4, random_state=1) # 60% training and 40% test



clf = tree.DecisionTreeClassifier()



clfFINAL = clf.fit(X_train2, y_train2)

y_pred = clfFINAL.predict(X_test2)



import graphviz 

dot_data = tree.export_graphviz(clfFINAL, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("Tree") 

dot_data = tree.export_graphviz(clfFINAL, out_file=None, filled=True,

                                feature_names=X_test2.columns, class_names=["0", "1"])  

graph = graphviz.Source(dot_data)  

graph
predicted_probas = clfFINAL.predict_proba(X_te2)

skplt.metrics.plot_cumulative_gain(y_te2, predicted_probas)

plt.show()

print("Accuracy of Decision Tree",metrics.accuracy_score(y_test2, y_pred))

print("Recall of Decision Tree",metrics.recall_score(y_test2, y_pred))