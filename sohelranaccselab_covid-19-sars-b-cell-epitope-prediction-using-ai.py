# ****-------Notebook Summary----***

#Data Science, Machine Learning,Deep Learning(Artificial Neural Networks(ANN))

#Data Visualization,EDA Analysis, Data Pre-processing,Data Manipulation,Data Cleaning,Data Split
#-------------------------------------------------------------------------------------------------
#Machine Learning Algorithm:

#Apply into Model or Classifiar:

#Part1= Decision Tree; Accuracy=()

#Part2 =Random forest;Accuracy=()

#Part3 =XGBoost Classifier;Accuracy=()

#Part4 =Logistic Regression;Accuracy=()

#Part5 =k-nearest neighbors algorithm (k-NN);Accuracy=()

##(UnSupervised Machine Learning Algorithm)

#Part6=K-means Clustering or Partition clustering

#Part7=Hierarchical Clustering or Agglomerative clustering.

#---------------
#Deep learning -> Artificial Nueral Networks(ANN)

#Part8 =ANN model;Accuracy=()

#Visualize output at graph, CM Matrix,Accuracy Report, #Predication into sample Data
#Environment Setup:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Plotting data 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Basic Python lib
import numpy
import pandas

#Keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

#tf 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#ML
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV,train_test_split,cross_val_score
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.ensemble import ExtraTreesClassifier

from sklearn import metrics

from sklearn.metrics import classification_report,confusion_matrix


#Data Read
file_path = '../input/epitope-prediction'
bcell_df = pd.read_csv(f'{file_path}/input_bcell.csv')
covid_df = pd.read_csv(f'{file_path}/input_covid.csv')
sars_df = pd.read_csv(f'{file_path}/input_sars.csv')
#Data Visualization,EDA Analysis, Data Pre-processing,Data Manipulation,Data Cleaning,Data Split
bcell_df.head()
df_bellsars = pd.concat([bcell_df,sars_df],axis = 0)
df_bellsars.head()
df_bellsars = df_bellsars.sample(frac=1).reset_index(drop=True)
df_bellsars.head()
df_bellsars = df_bellsars.drop(['parent_protein_id','protein_seq','peptide_seq'],axis = 1)
df_bellsars.head()
df=df_bellsars.copy()
df.describe()
df.info()
import seaborn; seaborn.set()
df.plot();
df.corr()
def correlation_matrix(d):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('COVID-19/SARS B-cell Epitope Prediction dataset features correlation\n',fontsize=15)
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=9)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])
    plt.show()

correlation_matrix(df)

plt.figure(figsize=(20,14))
sns.heatmap(df.corr(),annot=True,linecolor='green',linewidths=3,cmap = 'plasma')
i=1
plt.figure(figsize=(25,20))
for c in df.describe().columns[:]:
    plt.subplot(5,3,i)
    plt.title(f"Histogram of {c}",fontsize=10)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.hist(df[c],bins=20,color='blue',edgecolor='k')
    i+=1
plt.show()
df.apply(lambda x: sum(x.isnull()),axis=0)
i=1
plt.figure(figsize=(25,15))
for c in df.columns[:-1]:
    plt.subplot(5,3,i)
    plt.title(f"Boxplot of {c}",fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    sns.boxplot(y=df[c],x=df['target'])
    i+=1
plt.show()
idx_train = df['target'].astype("bool").values
fig, axes = plt.subplots(2, 3,figsize=(16,8))
sns.set_style('darkgrid')
axes = [x for a in axes for x in a]
for i,name in enumerate(["isoelectric_point", "aromaticity", "hydrophobicity", "stability", "parker", "emini"]):
    value = df[name]
    sns.distplot(value[~idx_train],ax = axes[i], color='red')
    sns.distplot(value[idx_train],ax = axes[i], color = 'blue')
    axes[i].set_xlabel(name,fontsize=12)
    fig.legend(labels = ["target 0","target 1"],loc="right",fontsize=12)
#checking the target variable countplot
sns.countplot(data=df,x = 'target',palette='plasma')
sns.set()
fig = plt.figure(figsize = [15,20])
cols = ['isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability', 'chou_fasman', 'start_position', 'end_position']
cnt = 1
for col in cols :
    plt.subplot(4,3,cnt)
    sns.distplot(df[col],hist_kws=dict(edgecolor="k", linewidth=1,color='green'),color='red')
    cnt+=1
plt.show() 
sns.pairplot(df)
plt.show()
sns.set()
fig = plt.figure(figsize = [15,20])
cols = ['isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability', 'chou_fasman', 'start_position', 'end_position']
cnt = 1
for col in cols :
    plt.subplot(4,3,cnt)
    sns.violinplot(x="target", y=col, data=df)
    cnt+=1
plt.show()

#Feature Extraction & Splitting 
y= df['target']

X = df.drop(['target'],axis = 1)
forest_clf = ExtraTreesClassifier(n_estimators=1000, random_state=42)
forest_clf.fit(X,y)
imp_features = forest_clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest_clf.estimators_], axis = 0)
 
plt.figure(figsize = (15,8))
plt.bar(X.columns, std, color = 'red') 
plt.xlabel('Feature Labels') 
plt.ylabel('Feature Importances') 
plt.title('Comparison of different Feature Importances') 
plt.show()
ec = ExtraTreesClassifier()
ec.fit(X,y)
ec_series = pd.Series(ec.feature_importances_,index=X.columns)
ec_series.plot(kind = 'barh',color = 'green')
#train_test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
dims = X_train.shape[1]
print(dims, 'dims')
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(y_train)
#Using RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
cr = classification_report(y_test,rfc_pred)
print(cr)
#Models performance Analysis with scaling(standard Scaler)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
list_models=[]
list_scores=[]
x_train=sc.fit_transform(X_train)
lr=LogisticRegression(max_iter=10000)
lr.fit(X_train,y_train)
pred_1=lr.predict(sc.transform(X_test))
score_1=accuracy_score(y_test,pred_1)
list_scores.append(score_1)
list_models.append('LogisticRegression')
score_1
from sklearn.neighbors import KNeighborsClassifier
list_1=[]
for i in range(1,50):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    preds=knn.predict(sc.transform(X_test))
    scores=accuracy_score(y_test,preds)
    list_1.append(scores)
    
list_scores.append(max(list_1))
list_models.append('KNeighbors Classifier')
print(max(list_1))
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
pred_2=rfc.predict(sc.transform(X_test))
score_2=accuracy_score(y_test,pred_2)
list_models.append('Randomforest Classifier')
list_scores.append(score_2)
score_2
from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train,y_train)
pred_3=svm.predict(sc.transform(X_test))
score_3=accuracy_score(y_test,pred_3)
list_scores.append(score_3)
list_models.append('Support vector machines')
score_3
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
pred_4=xgb.predict(sc.transform(X_test))
score_4=accuracy_score(y_test,pred_4)
list_models.append('XGboost')
list_scores.append(score_4)
score_4
plt.figure(figsize=(12,5))
plt.bar(list_models,list_scores)
plt.xlabel('classifiers')
plt.ylabel('accuracy scores')
plt.show()
#Additional Part-1

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
#DT
clf = DecisionTreeClassifier(random_state=7)
clf.fit(X_train,y_train)
path = clf.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas = path.ccp_alphas
alpha_list = []
for i in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=7,ccp_alpha=i)
    clf.fit(X_train,y_train)
    alpha_list.append(clf)
train_score = [clf.score(X_train,y_train) for clf in alpha_list]
test_score = [clf.score(X_test,y_test) for clf in alpha_list]

plt.plot(ccp_alphas,train_score,label = 'Training',color = 'red',marker = 'o',drawstyle = 'steps-post')
plt.plot(ccp_alphas,test_score,label = 'Testing',color = 'green',marker = '+',drawstyle = 'steps-post')
plt.legend()
plt.show()
#Random forest, Parameter tuning
params = {
    'RandomForest':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators' : [int(x) for x in np.linspace(100,1200,10)],
            'max_depth': [int(x) for x in np.linspace(1,50,10)],
            'min_samples_split': [1,2,5,10],
            'min_samples_leaf': [1,2,5,10],
            'ccp_alpha':[0.0025,0.0030,0.0045,0.005],
            'criterion':['gini','entropy'],
        }
    },
}
score = []
for model_name,mp in params.items():
    clf = RandomizedSearchCV(mp['model'],param_distributions=mp['params'],cv = 5,n_iter=10,scoring='accuracy',verbose=2)
    clf.fit(X_train,y_train)
    score.append({
        'model_name':model_name,
        'best_score':clf.best_score_,
        'best_estimator':clf.best_estimator_,
    })
score_df = pd.DataFrame(score,columns=['model_name','best_score','best_estimator'])
score_df
for i in score_df['best_estimator']:
    print(i)
    print("="*100)
dt = DecisionTreeClassifier(ccp_alpha=0.0025)
dt.fit(X_train,y_train)
plt.figure(figsize=(16,9))
tree.plot_tree(dt,filled=True,feature_names=X.columns,class_names=['has','does not have'])
rf = RandomForestClassifier(ccp_alpha=0.0025, criterion='entropy', max_depth=39,
                       min_samples_leaf=2, n_estimators=1000)
rf.fit(X_train,y_train)
rf.score(X_test,y_test)
xgb = XGBClassifier(base_score=0.5, booster='dart', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0.7795918367346939,
              learning_rate=0.325, max_delta_step=0, max_depth=22,
              min_child_weight=1, missing=None, n_estimators=833, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0.25, reg_lambda=2, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
xgb.fit(X_train,y_train)
xgb.score(X_train,y_train)
xgb.score(X_test,y_test)
y_predxgb = xgb.predict(X_test)
y_predxgb = np.array(y_predxgb)
y_testxgb = np.array(y_test)
xgb_actual = pd.DataFrame(y_testxgb)
xgb_predicted = pd.DataFrame(y_predxgb)
xgb_df = pd.concat([xgb_actual,xgb_predicted],axis = 1)
xgb_df.columns = ['Actual','Predicted']
xgb_df
import seaborn as sn
for i in xgb_df.columns:
    print(f' count of <{i}> ia {xgb_df[i].value_counts()}')
    print("="*100)
sn.countplot(data = xgb_df,x = 'Predicted',palette='plasma')
sn.countplot(data = xgb_df,x = 'Actual',palette='plasma')
metrics.plot_confusion_matrix(xgb,X_test,y_test,cmap='inferno',display_labels=['Covid -ve','Covid +ve'])
#Rf
y_predrf = rf.predict(X_test)
y_predrf = np.array(y_predrf)
y_testrf = np.array(y_test)
rf_actual = pd.DataFrame(y_testrf)
rf_predict = pd.DataFrame(y_predrf)
rf_df = pd.concat([rf_actual,rf_predict],axis = 1)
rf_df.columns = ['Actual','Predicted']
for i in rf_df.columns:
    print(f' count of <{i}> ia {rf_df[i].value_counts()}')
    print("="*100)
sn.countplot(data = rf_df,x = 'Predicted',palette='plasma')
sn.countplot(data = rf_df,x = 'Actual',palette='plasma')
metrics.plot_confusion_matrix(rf,X_test,y_test,cmap = 'plasma',display_labels=['Covid +ve','Covid -ve'])
rf_report = metrics.classification_report(y_test,y_predrf)
xgb_report = metrics.classification_report(y_test,y_predxgb)
print(f' report of RandomForest is {rf_report}\n\n report of XGBmodel is {xgb_report}')
#XGboost has performed better than Random Forest overall,well lets see the Neural Network Implementation
mm = MinMaxScaler()
X_train_mm = mm.fit_transform(X_train)
X_test_mm = mm.transform(X_test)
model = Sequential()
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

r_mm = model.fit(X_train_mm,y_train,epochs = 100,validation_data=(X_test_mm,y_test))
model.summary()
plt.plot(r_mm.history['loss'],label = 'loss',color = 'red')
plt.plot(r_mm.history['val_loss'],label = 'validation_loss',color = 'blue')
plt.legend()
plt.show()
plt.plot(r_mm.history['accuracy'],label = 'loss',color = 'red')
plt.plot(r_mm.history['val_accuracy'],label = 'validation_accuarcy',color = 'blue')
plt.legend()
plt.show()
prediction = model.predict(X_test_mm)
prediction = (prediction > 0.5)
print(metrics.classification_report(y_test, prediction, target_names = ['Covid_Negative','Covid_Positive']))
cn = metrics.confusion_matrix(y_test,prediction)
sn.heatmap(cn,annot=True,xticklabels=['Covid -ve','Covid +ve'],yticklabels=['Covid -ve','Covid +ve'],cmap = 'plasma')
df_covid = covid_df.drop(['parent_protein_id', 'protein_seq','peptide_seq'],axis = 1)
df_covid.head(3)
df_covid = mm.transform(df_covid)
prediction_class = model.predict_classes(df_covid)
prediction_class
predictions_covid = pd.DataFrame(prediction_class,columns=['predicted_class'])
predictions_covid.head()
cn = metrics.confusion_matrix(y_test,prediction)
sn.heatmap(cn,annot=True,linecolor='red',linewidths=3,xticklabels=['Covid -ve','Covid +ve'],yticklabels=['Covid -ve','Covid +ve'],cmap='plasma')
sn.countplot(data = predictions_covid,x = 'predicted_class',palette='plasma')
predictions_covid['predicted_class'].value_counts()
print(metrics.classification_report(y_test, prediction, target_names = ['Covid_Negative','Covid_Positive']))
X = df.iloc[:, 1:10].values
y = df.iloc[:, 10].values
X
y
seed = 123
numpy.random.seed(seed)
#Initializing Artificial Neural Network
model = Sequential()

#Adding input layer
model.add(Dense(128, input_dim=9, kernel_initializer='normal', activation='relu'))

# Adding the second hidden layer
model.add(Dense(256, kernel_initializer='uniform', activation='relu'))

# Adding the Third hidden layer
model.add(Dense(512, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling Neural Network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.33, random_state=seed)

fBestModel = 'best_model.h5' 
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1) 
best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)

model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs=100, 
          batch_size=32, verbose=True, callbacks=[best_model, early_stop])
score = model.evaluate(X_val, Y_val, verbose=1)
print('Accuracy: ', score[1]*100)
print( 'loss:', score[0]*100)
#Final Artificial Neural Networks for Prediction
y= df['target']

X = df.drop(['target'],axis = 1)
#train_test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train.shape
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model = Sequential()
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=64,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

#Early stopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train, 
          y=y_train, 
          epochs=150,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
#predictions
predictions = model.predict_classes(X_test)
print(classification_report(y_test, predictions))
print(classification_report(y_test, predictions, target_names = ['Covid_Negative','Covid_Positive']))
#confusion matrix
plt.figure(figsize = (10,10))
cm = confusion_matrix(y_test,predictions)
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['Covid_Negative','Covid_Positive'] , yticklabels = ['Covid_Negative','Covid_Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
#Applying PCA
pca = PCA(n_components = 2)

projected = pca.fit_transform(df[['isoelectric_point', 'aromaticity', 
                                             'start_position', 'end_position', 
                                             'stability', 'hydrophobicity', 
                                             'emini', 'parker']])
plt.figure(figsize=(8,8))
plt.scatter(projected[:, 0], projected[:, 1],
            c=df.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('coolwarm', 2))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();
#Prediction for Covid dataset
covid_df_Pred = covid_df.drop(['parent_protein_id', 'protein_seq', 'peptide_seq'], axis = 1)
#transform data
covid_df_Pred = sc.transform(covid_df_Pred)
predictions_covid = model.predict_classes(covid_df_Pred)
predictions_covid
predictions_covid = pd.DataFrame(predictions_covid, columns = ['Predictions'])
#predictions_covid.head()
frames = [covid_df, predictions_covid]
output = pd.concat(frames, axis = 1)
output.head(10)
#Unsupervised Machine Learning
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

np.random.seed(5)
standard_scalar = StandardScaler()
data_scaled = standard_scalar.fit_transform(df)
df = pd.DataFrame(data_scaled, columns=df.columns)
df.head()
from sklearn.cluster import KMeans

km = KMeans(init="random", n_clusters=2)
km.fit(df)
km.labels_
km.cluster_centers_
# k-means determine k
distortions = []
K = range(1, 20)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)
    
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('No of clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
estimators = [('k_means_5', KMeans(n_clusters=5, init='k-means++')),
              ('k_means_2', KMeans(n_clusters=2, init='k-means++')),
              ('k_means_bad_init', KMeans(n_clusters=2, n_init=1, init='random'))]

fignum = 1
titles = ['5 clusters', '2 clusters', '2 clusters, bad initialization']

for name, est in estimators:
    fig = plt.figure(fignum, figsize=(8, 6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(df)
    labels = est.labels_

    ax.scatter(df.values[:, 3], df.values[:, 0], df.values[:, 2], c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('start_position')
    ax.set_ylabel('end_position')
    ax.set_zlabel('chou_fasman')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1
#Hierarchical Clustering or Agglomerative clustering.
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering().fit(df)
clustering
clustering.labels_
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(df)

plt.figure(fignum, figsize=(10, 6))
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
