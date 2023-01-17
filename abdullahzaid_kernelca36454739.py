# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv('/kaggle/input/dont-overfit-ii/train.csv')
test = pd.read_csv('/kaggle/input/dont-overfit-ii/test.csv')
labels = train.columns.drop(['id', 'target'])
train.shape

train.head()
test.head()
train[train.columns[2:]].std().plot(kind = 'hist')
plt.title('Standard Deviation')
train[train.columns[2:]].mean().plot(kind = 'hist')
plt.title('Mean')

train[train.columns[2:32]].std()

train[train.columns[2:32]].mean()
train['target'].value_counts()

#Logistic Regression Model
sns.countplot(x = 'target', data = train, palette = 'hls')
plt.show
plt.savefig('count')

train.groupby('target').mean()
sns.set(style = "white")
corr = train.corr()

f, ax = plt.subplots(figsize= (25,25))

cmap = sns.diverging_palette(220, 10, as_cmap = True)

sns.heatmap(corr[['target']].sort_values('target').tail(10), cmap = cmap, vmax = 3, 
            center = 0, square = True, linewidth = 0.5, cbar_kws = {"shrink" : 0.5})

#logistic regression model
X = train.drop(['id','target'],axis = 1)
Y = train['target']
X_eval = test.drop(['id'], axis = 1)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)


model = LogisticRegression(solver = 'liblinear',C = 0.1, penalty = 'l1')
model.fit(x_train, y_train)
x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)
model.score(x_train, y_train)
model.score(x_test, y_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority', n_jobs=-1)
X_sm, y_sm = smote.fit_resample(X,Y)

df = pd.DataFrame(X_sm, columns = labels)
df['target'] = y_sm

sns.countplot(x = 'target', data = df, palette = 'hls')
plt.show
plt.savefig('count')
model = LogisticRegression(solver = 'liblinear',C = 1, penalty = 'l2')
normX = df.drop(['target'], axis = 1)
normY = df['target']

x_train, x_test, y_train, y_test = train_test_split(normX, normY, test_size=0.25, random_state=0)
model.fit(x_train, y_train)
x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)
model.score(x_train, y_train)
model.score(x_test, y_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
class_names = [0,1]

fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap ='YlGnBu', fmt = 'g')
plt.xlabel('Prediction')
plt.ylabel('Actual')
rfe = RFE(model)
rfe.fit(X,Y)
print('selected features:')
print(labels[rfe.support_].tolist())
X_fs = rfe.transform(normX)
X_fs_eval = rfe.transform(X_eval)

model.fit(X_fs, normY)

pred = model.predict_proba(X_fs_eval)[:,1]
pred

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
x_train_rfecv = rfecv.transform(x_train)
x_test_rfecv = rfecv.transform(x_test)
model.fit(x_train_rfecv, y_train)
y_pred = model.predict(x_test_rfecv)
model.score(x_train_rfecv, y_train)
model.score(x_test_rfecv, y_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
class_names = [0,1]

fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap ='YlGnBu', fmt = 'g')
plt.xlabel('Prediction')
plt.ylabel('Actual')


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

eli5.show_weights(model, top = 50)

model.predict_proba(X_eval)[:,1]
perm = PermutationImportance(model, random_state = 1).fit(x_train, y_train)
eli5.show_weights(perm, top = 50)
sel = SelectFromModel(perm, threshold = 0.05, prefit = True)
X_trans = sel.transform(x_train)
X_test = sel.transform(x_test)
model.fit(X_trans, y_train)
y_pred = model.predict(X_test)
model.score(X_trans, y_train)
model.score(X_test, y_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
class_names = [0,1]

fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap ='YlGnBu', fmt = 'g')
plt.xlabel('Prediction')
plt.ylabel('Actual')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

model = GaussianNB()
model.fit(x_train, y_train)
#x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)
model.score(x_train, y_train)
model.score(x_test, y_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
class_names = [0,1]

fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap ='YlGnBu', fmt = 'g')
plt.xlabel('Prediction')
plt.ylabel('Actual')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
model =  KNeighborsClassifier(n_neighbors=2)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
model.score(x_train, y_train)
model.score(x_test, y_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
class_names = [0,1]

fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap ='YlGnBu', fmt = 'g')
plt.xlabel('Prediction')
plt.ylabel('Actual')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
model = DecisionTreeClassifier(random_state = 0, max_depth=3, min_samples_leaf = 3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
model.score(x_train, y_train)
model.score(x_test, y_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
class_names = [0,1]

fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap ='YlGnBu', fmt = 'g')
plt.xlabel('Prediction')
plt.ylabel('Actual')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
model = svm.SVC(kernel ='linear', gamma='scale')
model.fit(x_train, y_train)
#x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)
model.score(x_train, y_train)
model.score(x_test, y_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
class_names = [0,1]

fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap ='YlGnBu', fmt = 'g')
plt.xlabel('Prediction')
plt.ylabel('Actual')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
from xgboost import XGBClassifier
model = XGBClassifier(max_depth = 2, gamma = 2)
model.fit(x_train, y_train)
#x_test = scaler.transform(x_test)
y_pred = model.predict(x_test)
model.score(x_train, y_train)
model.score(x_test, y_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
class_names = [0,1]

fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap ='YlGnBu', fmt = 'g')
plt.xlabel('Prediction')
plt.ylabel('Actual')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
#combining models
#Logistic Regression with feature Selection
#Decision tree with hyper parameters
#X = training set
#Y= target for training set
#X_eval = test set

modelLR = LogisticRegression(solver = 'liblinear',C = 0.3, penalty = 'l1', class_weight ='balanced')
modelDT = DecisionTreeClassifier(random_state = 0, max_depth=3, min_samples_leaf = 5, min_samples_split = 2, max_features=200 )
XGBClassifier(max_depth = 2, gamma = 2, eta = 0.8, reg_alpha = 0.5, reg_lambda = 0.5)
modelSVM = svm.SVC(kernel ='linear', gamma='scale')
modelKNN = KNeighborsClassifier(n_neighbors=5)
modelGNB = GaussianNB()
scaler = StandardScaler()

X = scaler.fit_transform(X)

X_eval = scaler.fit_transform(X_eval)

modelLR.fit(X, Y)
Y_pred_LR = modelLR.predict_proba(X_eval)

modelDT.fit(X,Y)
Y_pred_DT = modelDT.predict_proba(X_eval)

modelXGB.fit(X,Y)
Y_pred_XGB = modelXGB.predict_proba(X_eval)

modelSVM.fit(X,Y)
Y_pred_SVM= modelSVM.predict(X_eval)

modelKNN.fit(X,Y)
Y_pred_KNN= modelKNN.predict_proba(X_eval)

modelGNB.fit(X,Y)
Y_pred_GNB= modelGNB.predict_proba(X_eval)
#logistic regression model
X = train.drop(['id','target'],axis = 1)
Y = train['target']
X_eval = test.drop(['id'], axis = 1)

modelLR.fit(X,Y)
modelLR.predict_proba(X_eval)[:,1]
modelXGB = XGBClassifier(max_depth = 2, gamma = 2, eta = 0.8, reg_alpha = 0.5, reg_lambda = 0.5)
modelXGB.fit(X,Y)
modelXGB.predict_proba(X_eval)[:,1]
isf = IsolationForest(n_jobs = -1, random_state =1)
isf.fit(X,Y)

print(isf.score_samples(X))
isf.predict(X)
rfe = RFE(XGBClassifier(max_depth = 2, gamma = 2, eta = 0.8, reg_alpha = 0.5, reg_lambda = 0.5))
rfe.fit(X,Y)
print('selected features:')
print(labels[rfe.support_].tolist())
#Balance dataset

sns.countplot(x = 'target', data = train, palette = 'hls')
plt.show
plt.savefig('count')

df.head()
train.head()
rfe = RFE(XGBClassifier(max_depth = 2, gamma = 2, eta = 0.8, reg_alpha = 0.5, reg_lambda = 0.5))
normX = df.drop(['target'], axis = 1)
normY = df['target']
rfe.fit(normX,normY)
print('selected features:')
print(labels[rfe.support_].tolist())
X_fs = rfe.transform(normX)
X_fs_eval = rfe.transform(X_eval)

modelXGB.fit(X_fs, normY)
modelLR.fit(X_fs, normY)
Y_pred_LR = modelLR.predict_proba(X_fs_eval)

Y_pred_XGB = modelXGB.predict_proba(X_fs_eval)
#Weight of Logistic Regression can be higher as it has some feature selection
#Weight of Decision trees can be a bit lower as it has a lot of noise
weights = [1,4]
final_pred_prob = (Y_pred_DT * weights[0] + Y_pred_LR * weights[1])/sum(weights)
final_pred = final_pred_prob[:,1]
final_pred
from mlxtend.classifier import StackingClassifier
m = StackingClassifier(
    classifiers=[
        modelLR,
        modelXGB
    ],
    use_probas=True,
    meta_classifier= modelLR
)

m.fit(X_fs, normY)

pred = m.predict_proba(X_fs_eval)[:,1]
pred
submission = pd.read_csv('/kaggle/input/dont-overfit-ii/sample_submission.csv')

submission['target'] = pred
submission.to_csv('sample_submission.csv', index = False)
