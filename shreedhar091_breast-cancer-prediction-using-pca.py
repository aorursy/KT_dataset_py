import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.datasets import make_hastie_10_2

import os
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
os.chdir("../input")

os.listdir()
data = pd.read_csv("../input/data.csv")
data.shape
data.columns
data.dtypes
data.head()
data.shape
df=data.drop(['id','Unnamed: 32'],axis=1)
df.shape
df.head()
X = df.loc[: , 'radius_mean':'fractal_dimension_worst']

y = df.loc[:, 'diagnosis']
df['diagnosis'].replace('M',1,inplace=True)

df['diagnosis'].replace('B',0,inplace=True)
scale = ss()

X = scale.fit_transform(X)
pca = PCA()

out = pca.fit_transform(X)

out.shape 
pca.explained_variance_ratio_
pca.explained_variance_ratio_.cumsum()
final_data = out[:, :10]
final_data.shape
final_data[:5,:]
pcdf = pd.DataFrame( data =  final_data,

                    columns = ['pc1', 'pc2','pc3', 'pc4','pc5','pc6','pc7','pc8','pc9','pc10'])
pcdf['target'] = data['diagnosis'].map({'M': 1, "B" : 0 })
pcdf.head()
X = pcdf.loc[: , 'pc1':'pc10']

y = pcdf.loc[:,'target']
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size = 0.2,shuffle=True)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
dt = DecisionTreeClassifier()

rf = RandomForestClassifier(n_estimators=100)

xg = XGBClassifier(learning_rate=0.5,

                   reg_alpha= 5,

                   reg_lambda= 0.1

                   )

gbm = GradientBoostingClassifier()

et = ExtraTreesClassifier()

knn = KNeighborsClassifier()
dt1 = dt.fit(X_train,y_train)

rf1 = rf.fit(X_train,y_train)

xg1 = xg.fit(X_train,y_train)

gbm1 = gbm.fit(X_train,y_train)

et1 = et.fit(X_train,y_train)

knn1 = knn.fit(X_train,y_train)
y_pred_dt = dt1.predict(X_test)

y_pred_rf = rf1.predict(X_test)

y_pred_xg= xg1.predict(X_test)

y_pred_gbm= gbm1.predict(X_test)

y_pred_et= et1.predict(X_test)

y_pred_knn= knn1.predict(X_test)

y_pred_dt
y_pred_dt_prob = dt1.predict_proba(X_test)

y_pred_rf_prob = rf1.predict_proba(X_test)

y_pred_xg_prob = xg1.predict_proba(X_test)

y_pred_gbm_prob= gbm1.predict_proba(X_test)

y_pred_et_prob= et1.predict_proba(X_test)

y_pred_knn_prob= knn1.predict_proba(X_test)
print (accuracy_score(y_test,y_pred_dt))

print (accuracy_score(y_test,y_pred_rf))

print (accuracy_score(y_test,y_pred_xg))

print (accuracy_score(y_test,y_pred_gbm))

print (accuracy_score(y_test,y_pred_et))

print (accuracy_score(y_test,y_pred_knn))
confusion_matrix(y_test,y_pred_dt)

confusion_matrix(y_test,y_pred_rf)

confusion_matrix(y_test,y_pred_xg)

confusion_matrix(y_test,y_pred_gbm)

confusion_matrix(y_test,y_pred_et)

confusion_matrix(y_test,y_pred_knn)

tn,fp,fn,tp= confusion_matrix(y_test,y_pred_dt).flatten()
fpr_dt, tpr_dt, thresholds = roc_curve(y_test,

                                 y_pred_dt_prob[: , 1],

                                 pos_label= 1

                                 )
fpr_rf, tpr_rf, thresholds = roc_curve(y_test,

                                 y_pred_rf_prob[: , 1],

                                 pos_label= 1

                                 )
fpr_xg, tpr_xg, thresholds = roc_curve(y_test,

                                 y_pred_xg_prob[: , 1],

                                 pos_label= 1

                                 )
fpr_gbm, tpr_gbm,thresholds = roc_curve(y_test,

                                 y_pred_gbm_prob[: , 1],

                                 pos_label= 1

                                 )
fpr_et, tpr_et,thresholds = roc_curve(y_test,

                                 y_pred_et_prob[: , 1],

                                 pos_label= 1

                                 )
fpr_knn, tpr_knn,thresholds = roc_curve(y_test,

                                 y_pred_knn_prob[: , 1],

                                 pos_label= 1

                                 )
p_dt,r_dt,f_dt,_ = precision_recall_fscore_support(y_test,y_pred_dt)

p_rf,r_rf,f_rf,_ = precision_recall_fscore_support(y_test,y_pred_rf)

p_gbm,r_gbm,f_gbm,_ = precision_recall_fscore_support(y_test,y_pred_gbm)

p_xg,r_xg,f_xg,_ = precision_recall_fscore_support(y_test,y_pred_xg)

p_et,r_et,f_et,_ = precision_recall_fscore_support(y_test,y_pred_et)

p_knn,r_knn,f_knn,_ = precision_recall_fscore_support(y_test,y_pred_knn)

p_dt,r_dt,f_dt
print (auc(fpr_dt,tpr_dt))

print (auc(fpr_rf,tpr_rf))

print (auc(fpr_gbm,tpr_gbm))

print (auc(fpr_xg,tpr_xg))

print (auc(fpr_et,tpr_et))

print (auc(fpr_knn,tpr_knn))
fig = plt.figure(figsize=(12,10))          # Create window frame

ax = fig.add_subplot(111)   # Create axes

# 9.2 Also connect diagonals

ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line

# 9.3 Labels etc

ax.set_xlabel('False Positive Rate')  # Final plot decorations

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for multiple models')

# 9.4 Set graph limits

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])



# 9.5 Plot each graph now

ax.plot(fpr_dt, tpr_dt, label = "dt")

ax.plot(fpr_rf, tpr_rf, label = "rf")

ax.plot(fpr_xg, tpr_xg, label = "xg")

ax.plot(fpr_gbm, tpr_gbm, label = "gbm")

ax.plot(fpr_et, tpr_et, label = "et")

ax.plot(fpr_knn, tpr_knn, label = "knn")

# 9.6 Set legend and show plot

ax.legend(loc="lower right")

plt.show()