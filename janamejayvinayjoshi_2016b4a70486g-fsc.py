import numpy as np

import pandas as pd

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB as NB

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestRegressor
#Training data

data_orig = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')

data_tr = data_orig.copy()



#Test data

data_te = pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',')
#Identifying list of categorical, ordinal variables

cat_feat = ['col11','col37','col44','col2','col56']
#One-hot encoding the categorical variables for train and test data

for col in cat_feat:

    one_hot_tr = pd.get_dummies(data_tr[col], prefix=col, prefix_sep='_').copy()

    data_tr = data_tr.drop(col,axis = 1)

    data_tr = data_tr.join(one_hot_tr)

    

    one_hot_te = pd.get_dummies(data_te[col], prefix=col, prefix_sep='_').copy()

    data_te = data_te.drop(col,axis = 1)

    data_te = data_te.join(one_hot_te)
#Dropping ID, Class labels from train data

X_tr = data_tr.copy()

X_tr = X_tr.drop(['ID','Class'],axis=1).copy()



#Dropping ID label from test data

X_te = data_te.copy()

X_te = X_te.drop(['ID'],axis=1).copy()



#Target labels for training

Y_tr = data_tr['Class'].copy()
#Scaling train data's numerical features

stdsc = preprocessing.StandardScaler()

np_scaled = stdsc.fit_transform(X_tr)

X_tr_sc = pd.DataFrame(np_scaled)



#Scaling test data's numerical features

np_scaled_te = stdsc.transform(X_te)

X_te_sc = pd.DataFrame(np_scaled_te)
#Sampling from train, test data

X_tr_sc_samp = X_tr_sc.sample(n=150)

X_te_sc_samp = X_te_sc.sample(n=150)
#If sample from train data, Origin = 1

#If sample from test data, Origin = 0



train_orig = []

test_orig = []

for i in range(150):

    train_orig.append(1)

    test_orig.append(0)

    

X_tr_sc_samp['Origin'] = train_orig

X_te_sc_samp['Origin'] = test_orig



#Concatenate sample dataframes

Samp_df = pd.concat([X_tr_sc_samp, X_te_sc_samp]) 
target = Samp_df['Origin'].copy()

Samp_df = Samp_df.drop(['Origin'],axis=1)



#Train-Val split

s_train, s_val, sy_train, sy_val = train_test_split(Samp_df, target, test_size=0.20)



#Classification model for sample source

rf = RandomForestClassifier(n_estimators=100)

rf.fit(s_train, sy_train)

rf.score(s_val,sy_val)



#Prediction for val

pred = rf.predict(s_val)



#Confusion matrix for sample source classification

print(confusion_matrix(sy_val, pred))
#If feature contributes to identifying source of a sample, it is added to drop list

#i.e. if ROC-AUC score > 0.8 for a feature for source classification, it is added to drop list 

model = RandomForestClassifier(n_estimators = 100, max_depth = 5,min_samples_leaf = 5)

drop_list = []

for i in Samp_df.columns:

    score = cross_val_score(model,pd.DataFrame(Samp_df[i]),target,cv=2,scoring='roc_auc')

    if (np.mean(score) > 0.8):

        drop_list.append(i)
print(drop_list)
X_tr_wd = X_tr_sc.copy()

X_te_wd = X_te_sc.copy()



X_trwd, X_tewd, y_trwd, y_tewd = train_test_split(X_tr_wd, Y_tr, test_size=0.20)
#n_est = 100; plot accuracy vs max-depth

score_train_RFwd = []

score_test_RFwd = []



for i in range(5,35,1):

    rf = RandomForestClassifier(n_estimators = 100, max_depth=i)

    rf.fit(X_trwd, y_trwd)

    sc_train = rf.score(X_trwd,y_trwd)

    score_train_RFwd.append(sc_train)

    sc_test = rf.score(X_tewd,y_tewd)

    score_test_RFwd.append(sc_test)
#Accuracy vs Max-Depth

plt.figure(figsize=(10,6))

train_score,=plt.plot(range(5,35,1),score_train_RFwd,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(5,35,1),score_test_RFwd,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Score vs. Max-Depth (n_est = 100)')

plt.xlabel('Max-Depth')

plt.ylabel('Score')
rf_temp = RandomForestClassifier(n_estimators = 100)        

parameters = {'max_depth':[5, 14],'min_samples_split':[2, 3, 4, 5, 6, 7]}    

scorer = make_scorer(f1_score, average = 'micro')     #scorer: mean f1    

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         

grid_fit = grid_obj.fit(X_trwd, y_trwd)        

best_rf = grid_fit.best_estimator_         



print(grid_fit.best_params_) 
#Use max depth from graph

rf_best3 = RandomForestClassifier(n_estimators=100, max_depth = 13, min_samples_split = 2, min_samples_leaf = 1)

rf_best3.fit(X_trwd, y_trwd)

rf_best3.score(X_tewd,y_tewd)
#Predict for test

rf_best3.fit(X_tr_wd,Y_tr)

rf_pred3 = rf_best3.predict(X_te_wd)
outid = data_te['ID'].copy()

outdf_rf = pd.DataFrame(columns=['ID','Class'])

outdf_rf['ID'] = outid

outdf_rf['Class'] = rf_pred3

outdf_rf.to_csv('rf_md13_mss2_msl1.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(outdf_rf)