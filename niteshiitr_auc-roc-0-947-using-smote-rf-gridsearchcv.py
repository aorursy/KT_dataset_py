import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from statsmodels.formula.api import ols

pd.set_option('precision', 3)

import seaborn as sns
wine = pd.read_csv("../input/winequality-red.csv")

wine.head()
wine[pd.isnull(wine).any(axis=1)]
# Dropping the null rows

wine.dropna(axis=0,inplace=True)
wine.describe()
wine.quality.value_counts()

#plt.hist(wine.quality)
plt.figure(figsize=(10,5))



cor = wine.corr()

mask = np.zeros_like(cor, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(cor,mask=mask,annot=True)
wine.groupby('quality').mean()
plt.figure(figsize=(25,35))

i=1

for col in wine.columns[:-1] :

    plt.subplot(4,3,i)

    sns.boxplot(x='quality',y=col,data=wine)

    i = i+1
from sklearn.model_selection import train_test_split

x = wine.drop('quality',axis=1)

y = wine.quality
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)



from sklearn.preprocessing import StandardScaler

scale = StandardScaler().fit(x_train)

x_train_std = scale.transform(x_train)

x_test_std = scale.transform(x_test)



from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=0)

x_train_std_os,y_train_os = sm.fit_sample(x_train_std,y_train)
import mord

quality = np.array([3,4,5,6,7,8])



ord_model_IT = mord.LogisticIT(alpha=0).fit(x_train_std,y_train.astype('int'))

y_pred_IT = ord_model_IT.predict_proba(x_test_std)

ord_model_AT = mord.LogisticAT(alpha=0).fit(x_train_std,y_train.astype('int'))

y_pred_AT = ord_model_AT.predict_proba(x_test_std)



order_IT = np.argmax(y_pred_IT,axis=1)

predicted_IT = np.zeros((order_IT.size))

for i in range(len(predicted_IT)):

    predicted_IT[i] = order_IT[i]+3

    

order_AT = np.argmax(y_pred_AT,axis=1)

predicted_AT = np.zeros((order_AT.size))

for i in range(len(predicted_AT)):

    predicted_AT[i] = order_AT[i]+3
from sklearn.metrics import mean_absolute_error,mean_squared_error,precision_score,f1_score



df1 = pd.DataFrame(y_test)

df2 = pd.DataFrame(predicted_IT,columns=['predicted'])



res_ord = pd.concat([df1.reset_index(),df2.reset_index()],axis=1).drop(['index'],axis=1)

print ('Macro precision = ',precision_score(res_ord.quality,res_ord.predicted,average='macro'))

print ('Micro precision = ',precision_score(res_ord.quality,res_ord.predicted,average='micro'))



print ('Macro f1 score = ',f1_score(res_ord.quality,res_ord.predicted,average='macro'))

print ('Micro f1_score = ',f1_score(res_ord.quality,res_ord.predicted,average='micro'))
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=0).fit(x_train_std_os,y_train_os)

y_pred_RF = rf_model.predict(x_test_std)
df1 = pd.DataFrame(y_test)

df2 = pd.DataFrame(y_pred_RF,columns=['predicted'])



res_rf = pd.concat([df1.reset_index(),df2.reset_index()],axis=1).drop(['index'],axis=1)

print ('Macro precision = ',precision_score(res_rf.quality,res_rf.predicted,average='macro'))

print ('Micro precision = ',precision_score(res_rf.quality,res_rf.predicted,average='micro'))



print ('Macro f1 score = ',f1_score(res_rf.quality,res_rf.predicted,average='macro'))

print ('Micro f1_score = ',f1_score(res_rf.quality,res_rf.predicted,average='micro'))
good = y.apply(lambda x: int(x/7))

good.value_counts()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,good,test_size=0.2,random_state=0)



from sklearn.preprocessing import StandardScaler

scale = StandardScaler().fit(x_train)

x_train_std = scale.transform(x_train)

x_test_std = scale.transform(x_test)



from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=0)

x_train_std_os,y_train_os = sm.fit_sample(x_train_std,y_train)
# Logistic regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0).fit(x_train_std_os,y_train_os)

lr_pred = lr.predict_proba(x_test_std)



# Decision tree

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0).fit(x_train_std_os,y_train_os)

tree_pred = tree.predict_proba(x_test_std)



# Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0).fit(x_train_std_os,y_train_os)

rf_pred = rf.predict_proba(x_test_std)



# KNN classifier

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier().fit(x_train_std_os,y_train_os)

knn_pred = knn.predict_proba(x_test_std)



# SVC : Linear kernel

from sklearn.svm import SVC

sv_lin = SVC(kernel='linear',random_state=0,probability=True).fit(x_train_std_os,y_train_os)

sv_lin_pred = sv_lin.predict_proba(x_test_std)



# SVC : RBF kernel

from sklearn.svm import SVC

sv_rbf = SVC(kernel='rbf',random_state=0,probability=True).fit(x_train_std_os,y_train_os)

sv_rbf_pred = sv_rbf.predict_proba(x_test_std)
from sklearn.metrics import precision_recall_curve,average_precision_score,auc,roc_auc_score



models = ['LR','Decision Tree','Random Forest','KNN','Linear kernel SVM','RBF kernel SVM']

preds = [lr_pred,tree_pred,rf_pred,knn_pred,sv_lin_pred,sv_rbf_pred]

for pred,model in zip(preds,models):

    precision,recall,thresholds = precision_recall_curve(y_test,pred[:,1])

    print ('Area under precision recall curve for %s model = '%(model),round(auc(recall,precision),3))

    print ('Area under ROC curve for %s model = '%(model),round(roc_auc_score(y_test,pred[:,1]),3),'\n') 
from sklearn.model_selection import cross_val_score

print ('Train set AUC for Random Forest model : ',roc_auc_score(y_train,rf.predict_proba(x_train_std)[:,1]))

print ('Cross validation AUC for Random Forest model : ',np.mean(cross_val_score(rf,x_train_std,y_train,scoring='roc_auc',cv=10)))

print ('Test set AUC for Random Forest model : ',roc_auc_score(y_test,rf.predict_proba(x_test_std)[:,1]))
# We will first use RandomizedSearchCV to narrow down the sample space for GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [i for i in range(100,1100,100)]

max_depth = [i for i in range(10,110,10)]

max_depth.append(None)

max_features = ['auto','sqrt']



random_grid = {'n_estimators' : n_estimators,

               'max_depth' : max_depth,

               'max_features' : max_features}



rf_new = RandomForestClassifier(random_state=0)

rf_rand = RandomizedSearchCV(rf_new,random_grid,n_iter=100,cv=10,verbose=0,random_state=0,n_jobs=-1)

rf_rand.fit(x_train_std_os,y_train_os)
print ('Best params obtained via random search : ',rf_rand.best_params_)

print ('AUC achieved using the base model : ',round(roc_auc_score(y_test,rf.predict_proba(x_test_std)[:,1]),3))

best_rand = rf_rand.best_estimator_

print ('AUC achieved using the best params achieved in Randomized search : ',round(roc_auc_score(y_test,best_rand.predict_proba(x_test_std)[:,1]),3))
from sklearn.model_selection import GridSearchCV



param_grid = {'n_estimators' : [100,200,300,400,500,600,1000],

              'max_depth' : [10,20,30,40,50,60,None],

              'max_features' : ['auto','sqrt']}



rf_new = RandomForestClassifier(random_state = 0)

rf_grid = GridSearchCV(rf_new,param_grid,verbose=0,n_jobs=-1,cv=10)

rf_grid.fit(x_train_std_os,y_train_os)
print ('Best params achieved via GridSearch : ',rf_grid.best_params_)

rf_best = rf_grid.best_estimator_

print ('Best AUC achieved using best params : ',round(roc_auc_score(y_test,rf_best.predict_proba(x_test_std)[:,1]),3))
feature_importances = pd.DataFrame(rf.feature_importances_,index = x_train.columns,

                                   columns=['importance']).sort_values('importance',ascending=True)  



plt.figure(figsize=(15,5))

plt.barh(feature_importances.index,feature_importances.importance)