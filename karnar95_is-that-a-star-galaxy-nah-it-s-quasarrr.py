import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
%matplotlib inline
data = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
data = pd.DataFrame(data)
datanew = data
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
datanew['class'] = lab.fit_transform(datanew['class'])
 
datanew.head()
for col in datanew.columns:
    print("    {} \n ---------- \n".format(col),np.unique(np.asarray(datanew[col])),"\n")
from collections import Counter

count = Counter(datanew['class'])
count
Y = datanew['class']
Y = pd.DataFrame(Y)
Y.head()
X=datanew
X = X.drop(columns=['class','objid','rerun'])
X.head()
from sklearn import preprocessing

X_scaled = preprocessing.scale(X)

X_scaled = pd.DataFrame(data=X_scaled,columns=X.columns)
datanew_scaled = X_scaled.copy()
datanew_scaled['class']=Y
finaldata = datanew_scaled.copy()
datanew_scaled.head()
X_scaled.shape
import seaborn as sns
sns.pairplot(datanew_scaled,kind='scatter',hue='class',palette="Set2")
import scipy as sp
def corrfunc(x, y, **kws):
    r, _ = sp.stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("{:.2f}".format(r), xy=(.1, .5), xycoords=ax.transAxes, size=50)
g = sns.PairGrid(datanew_scaled)
g = g.map_lower(plt.scatter)
g = g.map_diag(plt.hist, edgecolor="w")
g = g.map_upper(corrfunc)
datanew_scaled.corr()
plt.figure(figsize=(15,30), dpi=100)
plt.subplot(7,2,1)
plt.scatter('g','u', data=datanew_scaled)
plt.xlabel('g')
plt.ylabel('u')
plt.subplot(7,2,2)
plt.scatter('g','r',data=datanew_scaled)
plt.xlabel('g')
plt.ylabel('r')
plt.subplot(7,2,3)
plt.scatter('i','r',data=datanew_scaled)
plt.xlabel('i')
plt.ylabel('r')
plt.subplot(7,2,4)
plt.scatter('z','i',data=datanew_scaled)
plt.xlabel('z')
plt.ylabel('i')
plt.subplot(7,2,5)
plt.scatter('mjd','plate',data=datanew_scaled)
plt.xlabel('mjd')
plt.ylabel('plate')
plt.subplot(7,2,6)
plt.scatter('specobjid','plate',data=datanew_scaled)
plt.xlabel('specobjid')
plt.ylabel('plate')
plt.subplot(7,2,7)
plt.scatter('specobjid','mjd',data=datanew_scaled)
plt.xlabel('specobjid')
plt.ylabel('mjd')
plt.ylabel('plate')
plt.subplot(7,2,8)
plt.scatter('g','i',data=datanew_scaled)
plt.xlabel('g')
plt.ylabel('i')

plt.show()
fig,ax = plt.subplots()
fig.set_size_inches(15,8)
sns.boxplot(data=X_scaled)
datanew_scaled.describe()
Q1 = X_scaled.quantile(0.25)
Q3 = X_scaled.quantile(0.75)
IQR = Q3 - Q1
((X_scaled < (Q1 - 1.5 * IQR)) | (X_scaled > (Q3 + 1.5 * IQR))).sum()
mask = (X_scaled < (Q1 - 1.5 * IQR)) | (X_scaled > (Q3 + 1.5 * IQR))
datanew_scaled[mask] = np.nan
datanew_scaled = datanew_scaled.dropna()
datanew_scaled.shape
datanew_scaled = datanew_scaled.reset_index(drop=True)
from collections import Counter

count = Counter(datanew_scaled['class'])
count
fig,ax = plt.subplots()
fig.set_size_inches(15,8)
sns.boxplot(data=datanew_scaled)
finaldata.shape
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
X = finaldata.iloc[:,0:14]
Y = finaldata.iloc[:,15]

Y = np.asarray(Y)
Y = Y.astype('int')

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=66)
kf = KFold(n_splits=5)

for train, valid in kf.split(X_train):
	print('train: %s, valid: %s' % (train, valid))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

for k in range(1,21,2):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X_train,y_train,cv=5,scoring="accuracy")
    print(scores.mean())   
knn=KNeighborsClassifier(n_neighbors=5)

training_start = time.perf_counter()
knnfit = knn.fit(X_train,y_train)
training_end = time.perf_counter()
total_time = training_end-training_start
print("Training Accuracy:       ",knnfit.score(X_train,y_train))
scores = cross_val_score(knn,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy ", scores.mean())
print("\nTime consumed for training %5.4f" % (total_time))
a=[None]*6
a[0]=scores.mean()
from sklearn.tree import DecisionTreeClassifier
t = DecisionTreeClassifier(max_depth=5)

training_start = time.perf_counter()
tfit = t.fit(X_train,y_train)
training_end = time.perf_counter()

total_time = training_end-training_start
print("Training Accuracy:        ",tfit.score(X_train,y_train))
scores = cross_val_score(t,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean())   
print("\nTime consumed for training %5.4f" % (total_time))
a[1]=scores.mean()
from sklearn.naive_bayes import GaussianNB
Gnb = GaussianNB(priors=None)


training_start = time.perf_counter()
Gnbfit = Gnb.fit(X_train,y_train)
training_end = time.perf_counter()

total_time = training_end-training_start

print("Training Accuracy:        ",Gnbfit.score(X_train,y_train))
scores = cross_val_score(Gnb,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %5.4f" % (total_time))
a[2]=scores.mean()
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes = (1000,1000),max_iter = 1000)
training_start = time.perf_counter()
MLPfit = MLP.fit(X_train,y_train)
training_end = time.perf_counter()

total_time = training_end-training_start
print("Training Accuracy:        ",MLPfit.score(X_train,y_train))
scores = cross_val_score(MLP,X_train,y_train,cv=5,scoring="accuracy")

print("Cross Validation Accuracy:", scores.mean())
print("\nTime consumed for training %5.4f" % (total_time))
a[3]=scores.mean()
from sklearn.svm import LinearSVC
SVC = LinearSVC(penalty='l2',C=10.0,max_iter = 100000)
training_start = time.perf_counter()
SVCfit = SVC.fit(X_train,y_train)
training_end = time.perf_counter()


total_time = training_end-training_start

print("Training Accuracy:        ",SVCfit.score(X_train,y_train))
scores = cross_val_score(SVC,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %5.4f" % (total_time))
a[4]=scores.mean()
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=10,max_depth=10)

training_start = time.perf_counter()
RFCfit = RFC.fit(X_train,y_train)
training_end = time.perf_counter()

total_time = training_end-training_start
print("Training Accuracy:        ",RFCfit.score(X_train,y_train))
scores = cross_val_score(RFC,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %5.4f" % (total_time))
a[5]=scores.mean()
d1 = pd.DataFrame(a)
x=['KNN','Decision Tree','Naive Bayes','Neural Network','SVM','Random Forest']

fig,ax = plt.subplots()
fig.set_size_inches(15,8)
bottom, top = ax.set_ylim(0.85, 1)
plt.bar(x,a)

prediction_start = time.perf_counter()
knnpred = knnfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",knnfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime ))
b = [None]*6
b[0]=knnfit.score(X_test,y_test)
knnpred

prediction_start = time.perf_counter()
tpred = tfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",tfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))
b[1]=tfit.score(X_test,y_test)
tpred
prediction_start = time.perf_counter()
Gnbpred = Gnbfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",Gnbfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))
b[2]=Gnbfit.score(X_test,y_test)
Gnbpred
prediction_start = time.perf_counter()
MLPpred = MLPfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",MLPfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))

b[3]=MLPfit.score(X_test,y_test)
MLPpred
prediction_start = time.perf_counter()
SVCpred = SVCfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",SVCfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))

b[4]=SVCfit.score(X_test,y_test)
SVCpred
prediction_start = time.perf_counter()
RFCpred = RFCfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",RFCfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))

b[5]=RFCfit.score(X_test,y_test)
RFCpred

x=['KNN','Decision Tree','Naive Bayes','Neural Network','SVM','Random Forest']

fig,ax = plt.subplots()
fig.set_size_inches(15,8)
bottom, top = ax.set_ylim(0.85, 1)
plt.bar(x,b)

fig,ax = plt.subplots()
fig.set_size_inches(20,8)
plt.subplot(1,2,1)
bottom, top = ax.set_ylim(0.8, 1)
plt.bar(x,a)
plt.title("Training set")
plt.ylabel("Accuracy")

plt.subplot(1,2,2)
bottom, top = ax.set_ylim(0.8, 1)
plt.bar(x,b)
plt.title("Test set")
plt.ylabel("Accuracy")


import itertools
from sklearn.metrics import confusion_matrix

class_names =['GALAXY','QUASAR','STAR']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
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


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, RFCpred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
from sklearn.model_selection import GridSearchCV

grid_param = {
    'n_estimators' : [50,100,300,500,800,1000],
    'criterion' : ['gini','entropy'],
    'bootstrap' : [True, False],
    'max_depth' : [10,20,50,100]
}
RFCbest = GridSearchCV(RFC,param_grid=grid_param,scoring = "accuracy",cv=5,n_jobs=-1)
RFCbest.fit(X_train,y_train)
print(RFCbest.best_estimator_)
best_params = RFCbest.best_params_
print(best_params)
from sklearn.ensemble import RandomForestClassifier
RFCa = RandomForestClassifier(bootstrap=False,criterion='gini',max_depth=50,n_estimators=1000)

training_start = time.perf_counter()
RFCbestfit = RFCa.fit(X_train,y_train)
training_end = time.perf_counter()

total_time = training_end-training_start

print("Training Accuracy:        ",RFCbestfit.score(X_train,y_train))
scores = cross_val_score(RFCa,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %6.5f" % (total_time))

a[5]=scores.mean()
prediction_start = time.perf_counter()
RFCpred = RFCbestfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",RFCbestfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))
b[5]=RFCbestfit.score(X_test,y_test)
RFCpred
un = np.unique(np.asarray(finaldata['class']).astype('int'))
from collections import Counter
from imblearn.over_sampling import SMOTE

for i,k in enumerate(un):
    print("Before Oversampling", k,"  ",list(Counter(y_train).values())[i]) # counts the elements' frequency

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(bootstrap=False,criterion='gini',max_depth=50,n_estimators=1000)

training_start = time.perf_counter()
RFCfit = RFC.fit(X_train_res,y_train_res)
training_end = time.perf_counter()

total_time = training_end-training_start

print("Training Accuracy:        ",RFCfit.score(X_train_res,y_train_res))
scores = cross_val_score(RFC,X_train_res,y_train_res,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
a[5]=scores.mean()

prediction_start = time.perf_counter()
RFCpred = RFCfit.predict(X_test)
prediction_end = time.perf_counter()


total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",RFCfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))

b[5]=RFCfit.score(X_test,y_test)
RFCpred
from sklearn.decomposition import PCA
pca_d = PCA()
pca_d.fit(X)
cumsum = np.cumsum(pca_d.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
d
fig,ax = plt.subplots()
plt.plot(cumsum)
plt.grid()
plt.axvline(d,c='r',linestyle='--')
pca = PCA(n_components = 7)
d_reduced = pca.fit_transform(X)
d_reducedt = pca.inverse_transform(d_reduced)

print(d_reduced.shape,d_reducedt.shape)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

X_train,X_test,y_train,y_test = train_test_split(d_reduced,Y,test_size=0.33,random_state=66)
kf = KFold(n_splits=5)

for train, valid in kf.split(X_train):
	print('train: %s, valid: %s' % (train, valid))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
RFCa = RandomForestClassifier(bootstrap=False,criterion='gini',max_depth=50,n_estimators=1000)

training_start = time.perf_counter()
RFCbestfit = RFCa.fit(X_train,y_train)
training_end = time.perf_counter()

total_time = training_end-training_start

print("Training Accuracy:        ",RFCbestfit.score(X_train,y_train))
scores = cross_val_score(RFCa,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %6.5f" % (total_time))

prediction_start = time.perf_counter()
RFCpred = RFCbestfit.predict(X_test)
prediction_end = time.perf_counter()


total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",RFCbestfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))

from collections import Counter
from imblearn.over_sampling import SMOTE
for i,k in enumerate(un):
    print("Before Oversampling", k,"  ",list(Counter(y_train).values())[i]) # counts the elements' frequency

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(bootstrap=False,criterion='gini',max_depth=50,n_estimators=1000)

training_start = time.perf_counter()
RFCfit = RFC.fit(X_train_res,y_train_res)
training_end = time.perf_counter()

total_time = training_end-training_start

print("Training Accuracy:        ",RFCfit.score(X_train_res,y_train_res))
scores = cross_val_score(RFC,X_train_res,y_train_res,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %6.5f" % (total_time))
prediction_start = time.perf_counter()
RFCpred = RFCfit.predict(X_test)
prediction_end = time.perf_counter()


total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",RFCfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=66)
kf = KFold(n_splits=5)

for train, valid in kf.split(X_train):
	print('train: %s, valid: %s' % (train, valid))
# Fitting the model to get feature importances after training
model = RandomForestClassifier()
model.fit(X_train , y_train)

# Draw feature importances
imp = model.feature_importances_
f = X.columns
# Sort by importance descending
f_sorted = f[np.argsort(imp)[::-1]]
fig,ax = plt.subplots(figsize=(15,8))
sns.barplot(x=f,y = imp, order = f_sorted)


plt.title("Features importances")
plt.ylabel("Importance")
plt.show()
finaldatanew = finaldata[['redshift','specobjid','mjd','z','plate','i','r','g','u']]
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

X = finaldatanew
Y = finaldata.iloc[:,15]

Y = np.asarray(Y)
Y = Y.astype('int')

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=66)
kf = KFold(n_splits=5)

for train, valid in kf.split(X_train):
	print('train: %s, valid: %s' % (train, valid))
from collections import Counter
from imblearn.over_sampling import SMOTE
for i,k in enumerate(un):
    print("Before Oversampling", k,"  ",list(Counter(y_train).values())[i]) # counts the elements' frequency

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(bootstrap=False,criterion='gini',max_depth=50,n_estimators=1000)

training_start = time.perf_counter()
RFCfit = RFC.fit(X_train_res,y_train_res)
training_end = time.perf_counter()

total_time = training_end-training_start

print("Training Accuracy:        ",RFCfit.score(X_train_res,y_train_res))
scores = cross_val_score(RFC,X_train_res,y_train_res,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %6.5f" % (total_time))
prediction_start = time.perf_counter()
RFCpred = RFCfit.predict(X_test)
prediction_end = time.perf_counter()


total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",RFCfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))
