#Importing Libraries
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import time
from BorutaShap import BorutaShap
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,plot_confusion_matrix,accuracy_score
from sklearn.feature_selection import SelectKBest,chi2,RFECV
from sklearn.decomposition import PCA
#Loading Data
data = pd.read_csv("../input/breastcancer-dataset/data.csv")
data.info()
data.head()
#Separating target from features
col = data.columns
y = data['diagnosis']
col_drop = ['id','diagnosis','Unnamed: 32']
x = data.drop(col_drop,axis=1)
#Plot Diagnosis Distribution
ax = sns.countplot(y,label="Count",palette="RdBu_r")
B,M = y.value_counts()
print("Number of Benign Tumors: ",B)
print("Number of Malign Tumors: ",M)
#Features Statistics
x.describe()
#Normalizing Dataset
data = x
data_std = (data - data.mean())/data.std()
#Violin Plot
fig,ax = plt.subplots(figsize = (15,35))

plt.subplot(3,1,1)
data = pd.concat([y,data_std.iloc[:,:10]],axis=1)
data = pd.melt(data,id_vars='diagnosis',var_name='features',value_name='value')
sns.violinplot(data=data,hue='diagnosis',x='features',y='value',split=True,inner='quartile')
plt.xticks(rotation=45)

plt.subplot(3,1,2)
data = pd.concat([y,data_std.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars='diagnosis',var_name='features',value_name='value')
sns.violinplot(data=data,hue='diagnosis',x='features',y='value',split=True,inner='quartile')
plt.xticks(rotation=45)

plt.subplot(3,1,3)
data = pd.concat([y,data_std.iloc[:,20:30]],axis=1)
data = pd.melt(data,id_vars='diagnosis',var_name='features',value_name='value')
sns.violinplot(data=data,hue='diagnosis',x='features',y='value',split=True,inner='quartile')
plt.xticks(rotation=45)

#Jointplots for correlation
sns.jointplot(x['concavity_mean'],x['concave points_mean'],kind="regg")
sns.jointplot(x['concavity_worst'],x['concave points_worst'],kind="regg")
#SwarmPlots - To determine class separability
fig,ax = plt.subplots(figsize = (15,40))

plt.subplot(3,1,1)
data = pd.concat([y,data_std.iloc[:,:10]],axis=1)
data = pd.melt(data,id_vars='diagnosis',var_name='features',value_name='value')
sns.swarmplot(data=data,hue='diagnosis',x='features',y='value',palette="cubehelix")
plt.xticks(rotation=45)

plt.subplot(3,1,2)
data = pd.concat([y,data_std.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars='diagnosis',var_name='features',value_name='value')
sns.swarmplot(data=data,hue='diagnosis',x='features',y='value',palette="cubehelix")
plt.xticks(rotation=45)

plt.subplot(3,1,3)
data = pd.concat([y,data_std.iloc[:,20:30]],axis=1)
data = pd.melt(data,id_vars='diagnosis',var_name='features',value_name='value')
sns.swarmplot(data=data,hue='diagnosis',x='features',y='value',palette="cubehelix")
plt.xticks(rotation=45)
#Heatmap
fig,ax = plt.subplots(figsize=(20,20))
sns.heatmap(x.corr(),annot=True,linewidth=0.5,fmt=".1f",ax=ax)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
# Removing cols by looking at heatmap (Minimal Feature selection)
drop_col = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean',
            'radius_se','perimeter_se','compactness_se','concave points_se',
            'radius_worst','perimeter_worst','area_worst','texture_worst','compactness_worst','concave points_worst']
df = x.drop(drop_col,axis=1)
x_train1,x_test1,y_train1,y_test1 = train_test_split(df,y,test_size=0.3,random_state=42)
fig,ax = plt.subplots(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,linewidth=0.5,fmt=".1f",ax=ax)
#Univariate feature selection
select_feature = SelectKBest(chi2,k=10).fit(x_train1,y_train1)
x_train2 = select_feature.transform(x_train1)
x_test2 = select_feature.transform(x_test1)
#Principal Component Analysis
x_train_norm = (x_train - x_train.mean())/(x_train.max()-x_train.min())
x_test_norm = (x_test - x_test.mean())/(x_test.max()-x_test.min())
pca = PCA()
pca.fit(x_train_norm)
plt.figure(figsize=(10,8))
sns.lineplot(data=np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("No of components")
plt.ylabel("Cumulative explained variance")
#XGBoost using Minimal feature selection
model1 = xgb.XGBClassifier(random_state=42)
model1.fit(x_train1,y_train1)
y_pred1 = model1.predict(x_test1)
print("Accuracy: ",accuracy_score(y_test1,y_pred1))
plot_confusion_matrix(model1,x_test1,y_test1,cmap=plt.cm.Blues)
plt.grid(False)
#XGBost using Univariate feature selection
model2 = xgb.XGBClassifier(random_state=42)
model2.fit(x_train2,y_train1)
y_pred2 = model2.predict(x_test2)
print("Accuracy: ",accuracy_score(y_test1,y_pred2))
plot_confusion_matrix(model2,x_test2,y_test1,cmap=plt.cm.Blues)
plt.grid(False)
#XGBost using Recursive feature selection with cross validation
model3 = xgb.XGBClassifier()
rfecv = RFECV(estimator=model3,step=1,cv=5,n_jobs=-1,min_features_to_select=10,scoring='accuracy').fit(x_train,y_train)
y_pred3 = rfecv.predict(x_test)
print("Accuracy: ",accuracy_score(y_test,y_pred3))
plot_confusion_matrix(rfecv,x_test,y_test,cmap=plt.cm.Blues)
plt.grid(False)
num_features = [i for i in range(1,len(rfecv.grid_scores_)+1)]
cv_scores = rfecv.grid_scores_
sns.lineplot(num_features,cv_scores)
print(cv_scores)
#Random Forest using Boruta Shap
feature_selector = BorutaShap(importance_measure='shap',classification=True)
feature_selector.fit(x_train,y_train,random_state=0,n_trials=100)
feature_selector.plot(which_features='all')
x_train_boruta = feature_selector.Subset()
x_test_boruta = x_test[['concave points_mean', 'texture_mean', 'radius_se', 'texture_worst', 'symmetry_worst', 'perimeter_worst', 'concavity_worst', 'radius_mean', 'concavity_mean', 'area_worst', 'smoothness_worst', 'area_se', 'area_mean', 'compactness_worst', 'radius_worst', 'perimeter_se', 'concave points_worst', 'fractal_dimension_worst', 'compactness_mean', 'perimeter_mean']]

model7 = xgb.XGBClassifier(random_state=42,learning_rate=0.01,n_estimators=1000)
model7.fit(x_train_boruta,y_train)
y_pred7 = model7.predict(x_test_boruta)
print("Accuracy: ",accuracy_score(y_test,y_pred7))
plot_confusion_matrix(model7,x_test_boruta,y_test,cmap=plt.cm.Blues)
plt.grid(False)
#Logistic Regression using Univariate feature selection
model4 = LogisticRegression(C=500,penalty='l2',max_iter=500)
model4.fit(x_train2,y_train1)
y_pred4 = model4.predict(x_test2)
print("Accuracy: ",accuracy_score(y_test1,y_pred4))
plot_confusion_matrix(model4,x_test2,y_test1,cmap=plt.cm.Blues)
plt.grid(False)

#SVM using Univariate feature selection
model5 = SVC(C=500,gamma='scale',kernel='poly',degree=1)
model5.fit(x_train2,y_train1)
y_pred5 = model5.predict(x_test2)
print("Accuracy: ",accuracy_score(y_test1,y_pred5))
plot_confusion_matrix(model5,x_test2,y_test1,cmap=plt.cm.Blues)
plt.grid(False)
#Random Forest Classifier using Univariate Linear Regression
model6 = RandomForestClassifier(n_estimators=1000,max_depth=20,criterion='entropy',bootstrap=True)
model6.fit(x_train2,y_train1)
y_pred6 = model6.predict(x_test2)
print("Accuracy: ",accuracy_score(y_test1,y_pred6))
plot_confusion_matrix(model6,x_test2,y_test1,cmap=plt.cm.Blues)
plt.grid(False)
#Naive Bayes Classifier using Univariate feature selection
model8 = GaussianNB()
model8.fit(x_train2,y_train1)
y_pred8 = model8.predict(x_test2)
print("Accuracy: ",accuracy_score(y_test1,y_pred8))
plot_confusion_matrix(model8,x_test2,y_test1,cmap=plt.cm.Blues)
plt.grid(False)
#K-Nearest Neighbors using Univariate feature selection
model9 = KNeighborsClassifier(n_neighbors=8)
model9.fit(x_train2,y_train1)
y_pred9 = model9.predict(x_test2)
print("Accuracy: ",accuracy_score(y_test1,y_pred9))
plot_confusion_matrix(model9,x_test2,y_test1,cmap=plt.cm.Blues)
plt.grid(False)