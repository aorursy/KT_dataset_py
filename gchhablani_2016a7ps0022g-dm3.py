import numpy as np

import sys

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report,make_scorer

from sklearn.model_selection import cross_validate,GridSearchCV,train_test_split



from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier,RandomForestClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier

from sklearn.svm import SVC



from keras.models import Sequential

from keras.layers import Dense,Dropout

import tensorflow as tf

from keras import backend as K





np.random.seed(42)



from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN

from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster



from tqdm import tqdm_notebook as tqdm
benign = pd.read_csv("../input/opcode_frequency_benign.csv")

malware = pd.read_csv("../input/opcode_frequency_malware.csv")

test = pd.read_csv("../input/Test_data.csv")
benign['FileName'].unique()
malware['FileName'].unique()
test['Unnamed: 1809'].unique()
test.head()
test.drop(['Unnamed: 1809'],axis=1,inplace=True)
benign['Malware'] = np.zeros(shape=benign.shape[0])

malware['Malware'] = np.ones(shape = malware.shape[0])

benign.drop(['FileName'],axis=1,inplace=True)

malware.drop(['FileName'],axis=1,inplace=True)

#test.drop(['FileName'],axis=1,inplace=True)
benign.duplicated().sum()
malware.duplicated().sum()
benign.drop_duplicates(inplace=True)

malware.drop_duplicates(inplace=True)
train = benign.append(malware,ignore_index=False)
train['Malware']=train['Malware'].apply(lambda x: int(x))
np.array(train[train.drop(['Malware'],axis=1).duplicated()].drop(['Malware'],axis=1)).sum()
type(train.isnull().sum(axis = 0)>0)
train.isnull().values.any()
pd.set_option('display.max_columns', 60)

train.head()
zero_cols = []

for i in train.columns:

    if(train[i].unique().tolist()==[0]):

        zero_cols.append(i)
print(len(zero_cols))
no_zero_test = []

for i in zero_cols:

    if (test[i].unique().tolist() != [0]):

        no_zero_test.append(i)

        #print(test[i].unique())

print(len(no_zero_test))

for i in no_zero_test:

    zero_cols.remove(i)
len(zero_cols)
ntrain = train.drop(zero_cols,axis=1)

ntest = test.drop(zero_cols,axis = 1)
ntest.shape
ntrain.shape
ntrain.head()
# Scaling

num = list(ntrain.columns)

num.remove('Malware')

ss = StandardScaler()

sntrain = ntrain.copy()

sntest = ntest.copy()

sntrain[num]=ss.fit_transform(ntrain[num])

sntest[num]=ss.transform(ntest[num])
sntrain['Malware'].value_counts()
todrop = []

# #Drop via Correlation between them

# corr = sntrain.corr()

# for i in range(len(corr.columns)):

#     for j in range(i):

#         if(abs(corr[corr.columns[i]][j])>=0.95) and (corr.columns[i] not in todrop) and corr.columns[i]!='Malware':

#             todrop.append(corr.columns[i])

# len(todrop)
ntodrop = []

# for i in range(len(corr.columns)):

#     if(abs(corr['Malware'][i])<0.01 and  (corr.columns[i] not in todrop)):

#         ntodrop.append(corr.columns[i])

# print(len(ntodrop))

# print(ntodrop)
todrop +=ntodrop
ftrain = sntrain.drop(todrop,axis=1)

ftest = sntest.drop(todrop,axis=1)
ftrain.shape
ftest.shape
#Shuffle

ftrain = ftrain.sample(frac=1).reset_index(drop=True)
model=PCA(n_components=2)

model_data = model.fit(ftrain.drop('Malware',axis=1)).transform(ftrain.drop('Malware',axis=1))
#print(plt.colormaps())
plt.figure(figsize=(8,6))

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Fig 1. PCA Representation of Given Classes')

plt.scatter(model_data[:,0],model_data[:,1],c=ftrain['Malware'],cmap = plt.get_cmap('RdBu'))

plt.show()
# #Plotting Elbow Method Graph for KMeans

# wcss = []

# for i in range(2,30):

#     kmean = KMeans(n_clusters = i, random_state = 42)

#     kmean.fit(ftrain.drop(['Malware'],axis=1))

#     wcss.append(kmean.inertia_)

# plt.figure(figsize=(10,8))

# plt.plot(range(2,30),wcss)

# plt.title('Fig 3. The Elbow Method')

# plt.xlabel('Number of clusters')

# plt.ylabel('WCSS')

# plt.show()
# for i in range(2, 18):

#     kmean = KMeans(n_clusters = i, random_state = 42)

#     kmean.fit(ftrain.drop(['Malware'],axis=1))

#     #TO copy predictions on non-duplicated data to duplicated data

#     pred = kmean.predict(ftrain.drop(['Malware'],axis=1))

#     plt.subplot(2,8, i - 1)

#     plt.title(str(i)+" clusters")

#     plt.scatter(model_data[:, 0], model_data[:, 1], c=pred)

# fig = plt.gcf()

# fig.set_size_inches((20,10))

# fig.suptitle("Fig 4. Multiple K Clusters")

# plt.show()
# #The elbow can be seen somewhere around 13.

# #Choose K = 15

# colors = ['red','green','blue','yellow','purple','pink','palegreen','violet','cyan','blue']

# colors = colors + colors

# plt.figure(figsize=(16, 8))

# kmean = KMeans(n_clusters = 13, random_state = 42)

# kmean.fit(ftrain.drop(['Malware'],axis=1))

# #TO copy predictions on non-duplicated data to duplicated data

# pred = kmean.predict(ftrain.drop(['Malware'],axis=1))

# pred_pd = pd.DataFrame(pred)

# arr = pred_pd[0].unique()

# plt.title('Fig 5. K-Means Results')

# for i in arr:

#     meanx = 0

#     meany = 0

#     count = 0

#     for j in range(len(pred)):

#           if i == pred[j]:

#             count+=1

#             meanx+=model_data[j,0]

#             meany+=model_data[j,1]

#             plt.scatter(model_data[j, 0], model_data[j, 1], c=colors[i])

#     meanx = meanx/count

#     meany = meany/count

#     plt.annotate(i,(meanx, meany),size=30, weight='bold', color='black')

# ftrain['predK'] = kmean.predict(ftrain.drop(['Malware'],axis=1))

# ftest['predK']=kmean.predict(ftest.drop(['FileName'],axis=1))
# model=PCA(n_components=50)

# model_train_data = model.fit(ftrain.drop(['Malware','predK'],axis=1)).transform(ftrain.drop(['Malware','predK'],axis=1))

# model_test_data = model.fit(ftrain.drop(['Malware','predK'],axis=1)).transform(ftest.drop(['FileName','predK'],axis=1))
train_df = ftrain.copy()

test_df = ftest.copy()

# train_df[['pca_'+str(i) for i in range(50)]] = pd.DataFrame(model_train_data)

# test_df[['pca_'+str(i) for i in range(50)]] = pd.DataFrame(model_test_data)
train_df.shape
X = train_df

X = X.drop('Malware',axis=1)

y = train_df['Malware']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)
#Using XGBClassifier for feature selection

#plt.figure(figsize=(60,60))

xgb = XGBClassifier(max_depth=30,random_state=42)

xgb = xgb.fit(X,y)

features = X_train.columns

importances = xgb.feature_importances_

impfeatures_index = np.argsort(importances)

#print([features[i] for i in impfeatures_index])

# sns.barplot(x = [importances[i] for i in impfeatures_index], y = [features[i] for i in impfeatures_index])

# plt.xlabel('value', fontsize=32)

# plt.ylabel('parameter', fontsize=32)

# plt.tick_params(axis='both', which='major', labelsize=25)

# plt.tick_params(axis='both', which='minor', labelsize=25)

# plt.show()
importances
len(importances)
importance = importances

dictionary = dict((key,value) for (key,value) in zip(X.columns,importance) if (value!=0))
sorted_importances=(sorted(dictionary,key=lambda k:dictionary[k],reverse=True))
print(len(sorted_importances))
cols = ['472', '568', '446', '494', '31', '718', '1790', '655', '11', '460', '1029', '161', '463', '516', '5', '692', '569', '743', '852', '650', '978', '1', '912', '1788', '93', '457', '443', '641', '734', '1022', '1091', '432', '475', '645', '918', '950', '154', '668', '693', '723', '949', '92', '858', '1081', '234', '487', '13', '218', '390', '418', '497', '1113', '99', '123', '213', '466', '208', '210', '490', '738', '1000', '83', '467', '689', '733', '129', '215', '430', '438', '484', '683', '1050', '1085', '33', '124', '162', '175', '284', '481', '688', '732']
# scores = []

# maxscore = 0

# bestfeat = 0

# for i in tqdm(range(70,91,1)):

#     numfeat= i

#     cols = sorted_importances[:numfeat]

#     temp_xgb = XGBClassifier()

#     X = train_df[cols]

#     y = train_df['Malware']

#     scorer = make_scorer(roc_auc_score)

#     cv_results = cross_validate(temp_xgb, X, y, cv=10, scoring=(scorer), return_train_score=True)

#     score = np.mean(cv_results['test_score'])

#     if(score>maxscore):

#         bestfeat = numfeat

#     print(i,score)

#     scores.append(score)
#plt.plot(range(70,91,1),scores,marker='o')
#numfeat= 81
X = train_df[cols]

y = train_df['Malware']



# test_fin_df = test_df[cols]

# test_fin_df['FileName'] = test['FileName']
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.30, random_state=42)
model=PCA(n_components=2)

model_data = model.fit(X).transform(X)

plt.figure(figsize=(8,6))

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Fig 2. PCA Representation of Given Classes')

plt.scatter(model_data[:,0],model_data[:,1],c=ftrain['Malware'],cmap = plt.get_cmap('RdBu'))

plt.show()
tempcols = sorted_importances[:5]

sns.pairplot(train_df[tempcols+['Malware',]])

plt.title('Fig 3. Pairplot of Top 5 Important Features and Malware Class')

plt.show()
X.shape
y.shape
nb = GaussianNB()

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(nb, X, y, cv=10, scoring=(scorer), return_train_score=True)

print(cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
nb.fit(X_train,y_train)

y_pred = nb.predict(X_val)

print(classification_report(y_val, y_pred))
train_roc = []

test_roc = []

for i in range(1,21):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    roc_auc_train = roc_auc_score(y_train,knn.predict(X_train))

    train_roc.append(roc_auc_train)

    roc_auc_test = roc_auc_score(y_val,knn.predict(X_val))

    test_roc.append(roc_auc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,21),train_roc,color='blue', linestyle='dashed', marker='o')

test_score,=plt.plot(range(1,21),test_roc,color='red',linestyle='dashed', marker='o')

plt.legend( [train_score, test_score],["Train ROC_AUC", "Test ROC_AUC"])

plt.title('Fig 3. ROC_AUC_Score vs K neighbors')

plt.xlabel('K neighbors')

plt.ylabel('ROC_AUC_Score')
knn = KNeighborsClassifier(n_neighbors=3)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(knn, X, y, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
train_roc = []

test_roc = []

Cval = [0.01,0.1,1,10]

for i in Cval:

    lg = LogisticRegression(C=i)

    lg.fit(X_train,y_train)

    roc_auc_train = roc_auc_score(y_train,lg.predict(X_train))

    train_roc.append(roc_auc_train)

    roc_auc_test = roc_auc_score(y_val,lg.predict(X_val))

    test_roc.append(roc_auc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(Cval,train_roc,color='blue', linestyle='dashed', marker='o')

test_score,=plt.plot(Cval,test_roc,color='red',linestyle='dashed', marker='o')

plt.legend( [train_score, test_score],["Train ROC_AUC", "Test ROC_AUC"])

plt.title('Fig 4. ROC_AUC_Score vs C')

plt.xlabel('C')

plt.ylabel('ROC_AUC_Score')
lg = LogisticRegression(C=1)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(lg, X, y, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
train_roc = []

test_roc = []

for i in range(1,21):

    dt = DecisionTreeClassifier(max_depth=i,random_state=42)

    dt.fit(X_train,y_train)

    roc_auc_train = roc_auc_score(y_train,dt.predict(X_train))

    train_roc.append(roc_auc_train)

    roc_auc_test = roc_auc_score(y_val,dt.predict(X_val))

    test_roc.append(roc_auc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,21),train_roc,color='blue', linestyle='dashed', marker='o')

test_score,=plt.plot(range(1,21),test_roc,color='red',linestyle='dashed', marker='o')

plt.legend( [train_score, test_score],["Train ROC_AUC", "Test ROC_AUC"])

plt.title('Fig 5. ROC_AUC_Score vs max_depth')

plt.xlabel('max_depth')

plt.ylabel('ROC_AUC_Score')
train_roc = []

test_roc = []

for i in range(2,31):

    dt = DecisionTreeClassifier(max_depth=13,min_samples_split=i,random_state=42)

    dt.fit(X_train,y_train)

    roc_auc_train = roc_auc_score(y_train,dt.predict(X_train))

    train_roc.append(roc_auc_train)

    roc_auc_test = roc_auc_score(y_val,dt.predict(X_val))

    test_roc.append(roc_auc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(2,31),train_roc,color='blue', linestyle='dashed', marker='o')

test_score,=plt.plot(range(2,31),test_roc,color='red',linestyle='dashed', marker='o')

plt.legend( [train_score, test_score],["Train ROC_AUC", "Test ROC_AUC"])

plt.title('Fig 6. ROC_AUC_Score vs min_samples_split')

plt.xlabel('max_samples_split')

plt.ylabel('ROC_AUC_Score')
# dt_temp = DecisionTreeClassifier(random_state=42) #Initialize the classifier object

# parameters = {'max_depth':[8,13,15],'min_samples_split':[2,5,10]} #Dictionary o

# scorer = make_scorer(roc_auc_score) #Initialize the scorer using make_scorer

# grid_obj = GridSearchCV(dt_temp, parameters, scoring=scorer) #Initialize a GridSearchC

# grid_fit = grid_obj.fit(X, y) #Fit the gridsearch object with X_train,y_tr

# best_rf = grid_fit.best_estimator_ #Get the best estimator. For this, check documentat

# print(grid_fit.best_params_)
dt = DecisionTreeClassifier(max_depth=13,min_samples_split=2,random_state=42)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(dt, X, y, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
# rf_temp = RandomForestClassifier(n_estimators = 220,random_state=42) #Initialize the classifier object

# parameters = {'max_depth':[15,25,30,40],'min_samples_split':[2, 3, 4, 5]} #Dictiona

# scorer = make_scorer(roc_auc_score) #Initialize the scorer using make_scorer

# grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer) #Initialize a GridSearchC

# grid_fit = grid_obj.fit(X, y) #Fit the gridsearch object with X_train,y_tr

# best_rf = grid_fit.best_estimator_ #Get the best estimator. For this, check documentat

# print(grid_fit.best_params_)
rf = RandomForestClassifier(n_estimators = 220,max_depth=15,min_samples_split=3,random_state=42)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(rf, X, y, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
# adb_temp = AdaBoostClassifier(random_state=0) #Initialize the classifier object

# parameters = {'n_estimators':[220,240,250]} #Dictionary of parameters

# scorer = make_scorer(roc_auc_score) #Initialize the scorer using make_scorer

# grid_obj = GridSearchCV(adb_temp, parameters, scoring=scorer) #Initialize a GridSearch

# grid_fit = grid_obj.fit(X, y) #Fit the gridsearch object with X_train,y_tr

# best_rf = grid_fit.best_estimator_ #Get the best estimator. For this, check documentat

# print(grid_fit.best_params_)
adb = AdaBoostClassifier(n_estimators = 220,random_state=0)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(adb, X, y, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
# #Code to toggle warnings

# from IPython.display import HTML

# HTML('''<script>

# code_show_err=false; 

# function code_toggle_err() {

#  if (code_show_err){

#  $('div.output_stderr').hide();

#  } else {

#  $('div.output_stderr').show();

#  }

#  code_show_err = !code_show_err

# } 

# $( document ).ready(code_toggle_err);

# </script>

# To toggle on/off output_stderr, click <a href="javascript:code_toggle_err()">here</a>.''')
# bestscore = 0

# bestModel = None

# for n_estimators in tqdm([220,230,250,260,270,300]):

#     for max_depth in tqdm([3,5,7,9,10,15]):

#         xgb_temp = XGBClassifier(random_state=42,n_estimators= n_estimators,max_depth=max_depth)

#         scorer = make_scorer(roc_auc_score)

#         cv_results = cross_validate(xgb_temp, X, y, cv=10, scoring=(scorer), return_train_score=True)

#         score = np.mean(cv_results['test_score'])

#         print(n_estimators,max_depth,score)

#         if(score>bestscore):

#             bestscore = score

#             bestModel = xgb_temp

xgb = XGBClassifier(max_depth=5,n_estimators = 1500,random_state=220)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(xgb, X, y, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
# svm_temp = SVC(random_state=42) #Initialize the classifier object

# parameters = {'C':[0.01,0.1,1,10],'kernel':['rbf','poly','linear'],'degree':[1,3,4,5],'gamma':[0.001,0.01,0.1,1]} #Dictionary of parameters

# scorer = make_scorer(roc_auc_score) #Initialize the scorer using make_scorer

# grid_obj = GridSearchCV(svm_temp, parameters, scoring=scorer) #Initialize a GridSearchCV object with above parameters,scorer and classifier

# grid_fit = grid_obj.fit(X,y) #Fit the gridsearch object with X_train,y_train

# best_rf = grid_fit.best_estimator_ #Get the best estimator. For this, check documentation of GridSearchCV object

# print(grid_fit.best_params_)
svm = SVC(C=0.1,degree=3,gamma=0.1,kernel='poly',random_state=42)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(svm, X, y, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
# gdb_temp = GradientBoostingClassifier(n_estimators=220,random_state=42)

# parameters = {'min_samples_split':[2,3,4,5],'max_depth':[2,4,6,7,10]}

# scorer = make_scorer(roc_auc_score) #Initialize the scorer using make_scorer

# grid_obj = GridSearchCV(gdb_temp, parameters, scoring=scorer) #Initialize a GridSearchCV object with above parameters,scorer and classifier

# grid_fit = grid_obj.fit(np.array(X_train), np.array(y_train)) #Fit the gridsearch object with X_train,y_train

# best_rf = grid_fit.best_estimator_ #Get the best estimator. For this, check documentation of GridSearchCV object

# print(grid_fit.best_params_)
gdb = GradientBoostingClassifier(n_estimators=220,max_depth=4,min_samples_split=2,random_state=42)

scorer = make_scorer(roc_auc_score)

cv_results = cross_validate(gdb, X, y, cv=10, scoring=(scorer), return_train_score=True)

print (cv_results.keys())

print("Train ROC_AUC for 10 folds= ",np.mean(cv_results['train_score']))

print("Validation ROC_AUC for 10 folds = ",np.mean(cv_results['test_score']))
cb=CatBoostClassifier(iterations=1420, depth=7, learning_rate=0.01)

scorer = make_scorer(roc_auc_score)

cb.fit(X_train,y_train, eval_set=(X_val,y_val),verbose=False)

roc_auc_score(y_val,cb.predict(X_val))
lgbm = LGBMClassifier(n_estimators=1000,num_leaves =50)

lgbm.fit(X_train,y_train)

roc_auc_score(y_val,lgbm.predict(X_val))
nncols = sorted_importances[:500]

X_nn = X_train

y_nn = y_train
X_train_nn,X_val_nn,y_train_nn,y_val_nn = train_test_split(X_nn,y_nn,test_size=0.3,random_state=42)
y_train_oh =pd.get_dummies(y_train_nn)

y_val_oh = pd.get_dummies(y_val_nn)
def classification_model():

    model = Sequential()

    model.add(Dense(256, input_dim=81, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(rate=0.3))

    model.add(Dense(512, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(rate=0.3))

    model.add(Dense(512, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(rate=0.3))

    model.add(Dense(128, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(rate =0.3))

    model.add(Dense(64,kernel_initializer='normal', activation='relu'))

    model.add(Dropout(rate = 0.3))

    model.add(Dense(16, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(rate= 0.3))

    model.add(Dense(8,kernel_initializer='normal', activation='relu'))

    model.add(Dropout(rate =0.3))

    model.add(Dense(4, kernel_initializer='normal', activation='relu'))

    model.add(Dropout(rate =0.3))

    model.add(Dense(2, kernel_initializer='normal', activation='softmax'))

    model.summary()

    return model
def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
nn = classification_model()

nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = nn.fit(X_train_nn,y_train_oh,validation_data=(X_val_nn,y_val_oh),epochs=100,batch_size=32,verbose=0)

predictions = [np.argmax(ans) for ans in nn.predict(X_val_nn)]
history.history.keys()
# plt.plot(history.history['acc'], label='train')

# plt.plot(history.history['val_acc'], label='test')

# plt.legend()

# plt.show()
roc_auc_score(y_val_nn,predictions)
final_clf = xgb
X.columns
test_df[cols].columns
# final_clf = CatBoostClassifier(iterations=1420, depth=7, learning_rate=0.01)

final_clf.fit(X,y)

preds = final_clf.predict(test_df.drop('FileName',axis=1)[cols])
sub_df = test_df.copy()

sub_df['Class'] = preds

sub_csv = sub_df[['FileName','Class']]

sub_csv['Class']=sub_csv['Class'].astype(int)
sub_csv.shape
sub_csv.to_csv('Final Submission.csv',index=False)
# rf.fit(X,y)

# preds = rf.predict(test_df.drop('FileName',axis=1)[cols])

# sub_df = test_df.copy()

# sub_df['Class'] = preds

# sub_csv = sub_df[['FileName','Class']]

# sub_csv['Class']=sub_csv['Class'].astype(int)

# sub_csv.to_csv('submission-19.csv',index=False)
# xgb.fit(X,y)

# preds = xgb.predict(test_df.drop('FileName',axis=1)[cols])

# sub_df = test_df.copy()

# sub_df['Class'] = preds

# sub_csv = sub_df[['FileName','Class']]

# sub_csv['Class']=sub_csv['Class'].astype(int)

# sub_csv.to_csv('submission-16.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html='<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title} </a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub_csv)