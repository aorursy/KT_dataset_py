import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import time

from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder,MinMaxScaler

from sklearn.pipeline import FeatureUnion,Pipeline

from sklearn.base import BaseEstimator,TransformerMixin

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,LinearRegression

from sklearn.metrics import log_loss

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,StratifiedKFold,train_test_split

from sklearn.feature_selection import SelectKBest,chi2,f_classif, mutual_info_classif,f_regression,mutual_info_regression

from sklearn.decomposition import PCA

from sklearn.svm import LinearSVC, SVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import catboost
dfheart_train_X=pd.read_csv('../input/warm-up-machine-learning-with-a-heart/train_values.csv').set_index('patient_id')

dfheart_train_y=pd.read_csv('../input/warm-up-machine-learning-with-a-heart/train_labels.csv').set_index('patient_id')

dfheart_test_X=pd.read_csv('../input/warm-up-machine-learning-with-a-heart/test_values.csv').set_index('patient_id')

dftest_check=pd.read_csv('../input/heart-check/sub_check.csv').set_index('patient_id')
merged_file=dfheart_train_X.join(dfheart_train_y)
scaler=StandardScaler()

MinMax_scaler=MinMaxScaler(feature_range=(0, 1))
merged_file_encoded=pd.get_dummies(dfheart_train_X,columns=['thal','sex']).join(dfheart_train_y)

dfheart_train_X_encoded=pd.get_dummies(dfheart_train_X,columns=['thal','sex'])
#scaled and encoded train dataset

scaled_dfheart_train_X_encoded=pd.DataFrame(scaler.fit_transform(dfheart_train_X_encoded),columns=dfheart_train_X_encoded.columns,index=dfheart_train_X.index)
#Function for creating principal components dataframe

pca = PCA(n_components=2)

def pca_fit_transform(df,no_of_components):

    #pca = PCA(n_components=no_of_components)

    principalComponents = pca.fit_transform(df)

    columns=[]

    for i in range(1,no_of_components+1):

        columns.append('principal_component_%d'%i)

    return pd.DataFrame(data = principalComponents, columns = columns).set_index(df.index)

#Function for creating principal components dataframe

def pca_transform(df,no_of_components):

    principalComponents = pca.transform(df)

    columns=[]

    for i in range(1,no_of_components+1):

        columns.append('principal_component_%d'%i)

    return pd.DataFrame(data = principalComponents, columns = columns).set_index(df.index)
#Selecting the kbest Features with different scoring - chi2, f_classif, mutual_classif, f_regression, mutual_info_regression

X_kbest_features = SelectKBest(f_classif, k = 5).fit_transform(dfheart_train_X_encoded,dfheart_train_y['heart_disease_present'])
#select kbest columns with different scoring

chi2_cols=['num_major_vessels','oldpeak_eq_st_depression','max_heart_rate_achieved','exercise_induced_angina','thal_reversible_defect']

f_classif_cols=['chest_pain_type','num_major_vessels','exercise_induced_angina','thal_normal','thal_reversible_defect']

mutual_classif_cols=['chest_pain_type','num_major_vessels','exercise_induced_angina','thal_normal','thal_reversible_defect']

common_cols=['num_major_vessels','exercise_induced_angina','thal_reversible_defect']
# Training Dataframes of kBest features:

pca_df=pca_fit_transform(scaled_dfheart_train_X_encoded,2)

chi2_df=scaled_dfheart_train_X_encoded.copy()[chi2_cols]

f_classif_df=scaled_dfheart_train_X_encoded.copy()[f_classif_cols]

common_df=scaled_dfheart_train_X_encoded.copy()[common_cols]

#with PCA

pc_chi2_df=chi2_df.join(pca_df)

pc_f_classif_df=f_classif_df.join(pca_df)

pc_common_df=common_df.join(pca_df)
lda=LinearDiscriminantAnalysis()

lda.fit_transform(dfheart_train_X_encoded,dfheart_train_y['heart_disease_present'])

lda_df=pd.DataFrame(lda.fit_transform(pc_f_classif_df,dfheart_train_y))

lda_df.head()

lda_pc_f_classif_df=pc_f_classif_df.join(lda_df.set_index(f_classif_df.index))

#lda_f_classif_df=f_classif_df.join(lda_df.set_index(f_classif_df.index))
def Label_One_Hot(df,columns):

    return pd.get_dummies(df,columns=columns)
#Preparing Training Data - Not Used

Xt1=Label_One_Hot(dfheart_train_X.copy(),['thal','sex'])
#Function to Prepare Training Data

def test_data_prep(df,columns,pc=True,no_of_components=None): #df:test dataframe;pc:False if no PCA; no_of_components: PCA Components; columns:K_best column names

    temp=pd.get_dummies(df,columns=['thal','sex']).copy()

    scaled_temp=pd.DataFrame(scaler.transform(temp.copy()),columns=temp.columns,index=temp.index)

    kbest_df=scaled_temp.copy()[columns]

    if pc:

        pc_temp=pca_transform(scaled_temp,no_of_components)

        return kbest_df.join(pc_temp)

    return kbest_df
merged_file.info()
merged_file.describe()
#Custom Code to Display properties of each column

l=[]

for i in merged_file_encoded.columns:

    l.append([i,len(merged_file_encoded[i].unique()),max(merged_file_encoded[i].unique()),min(merged_file_encoded[i].unique()),merged_file_encoded[i].var(),merged_file_encoded[i].astype(bool).sum(axis=0),merged_file_encoded[i].count(),merged_file_encoded[i].unique()])

ldf=pd.DataFrame(l, columns=['Features', 'No_Unique_Values', 'Max_Value','Min_Value','Variance','Non-Zero','Total_Values','Unique_Values'])

ldf
#Heat Map Generation

copy_merged=merged_file_encoded.copy()

column=list(copy_merged.columns)

scaler = StandardScaler()

copy_merged = pd.DataFrame(scaler.fit_transform(copy_merged),columns=column,index=dfheart_train_X.index)



corr = copy_merged.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(corr, mask=mask, cmap="RdYlGn",annot=True, square=True, linewidths=.5, center=0, vmax=1);

fig=plt.gcf()

fig.set_size_inches(15,15)

plt.show()
sns.countplot(data=merged_file, x='heart_disease_present').set_title('Presence of Heart Disease')
sns.countplot(data=merged_file, x='sex',hue='heart_disease_present').set_title('Heart Attack w.r.t. Sex')
sns.countplot(data=merged_file, x='thal',hue='heart_disease_present').set_title('Effect of different thal types on Heart Disease')
sns.countplot(data=merged_file, x='slope_of_peak_exercise_st_segment',hue='heart_disease_present').set_title('Slope at Peak Exerxise')
sns.countplot(data=merged_file, hue='fasting_blood_sugar_gt_120_mg_per_dl' ,y='heart_disease_present').set_title('Fasting Blood Sugar')
sns.violinplot(data=merged_file, y='fasting_blood_sugar_gt_120_mg_per_dl', x='sex' ,hue='heart_disease_present').set_title('Fasting Blood Sugar')
sns.violinplot(data=merged_file, y='resting_blood_pressure',x='sex' ,hue='heart_disease_present', pallete='set3').set_title('Resting Blood Pressure')
sns.violinplot(data=merged_file, y='serum_cholesterol_mg_per_dl',x='sex' ,hue='heart_disease_present', pallete='set3').set_title('Cholestrol Serum')
sns.boxplot(data=merged_file,x='sex', y='serum_cholesterol_mg_per_dl', hue='heart_disease_present').set_title('Cholestrol Serum')
knn=KNeighborsClassifier()

knn.fit(lda_pc_f_classif_df,dfheart_train_y['heart_disease_present'])
xgb=XGBClassifier()

xgb.fit(lda_pc_f_classif_df,dfheart_train_y['heart_disease_present'])
params_tree={

        'booster':['gbtree'],

        'num_feature':range(1,9),

        'eta':list(np.linspace(0,1,50)),

        'max_depth':range(1,20),

        'subsample': list(np.linspace(0.01,1,50)),

        'learning rate': list(np.linspace(0,0.5,100)),

        'eval_metric':['logloss']

}

rscv_xgb_tree=RandomizedSearchCV(xgb,params_tree, cv=5,n_iter=500, n_jobs=-1, verbose=1,scoring='neg_log_loss')

rscv_xgb_tree.fit(lda_pc_f_classif_df,dfheart_train_y['heart_disease_present'])
rscv_xgb_tree.best_params_
gbc=GradientBoostingClassifier()

gbc.fit(lda_pc_f_classif_df,dfheart_train_y['heart_disease_present'])
linear_svc=LinearSVC()

cal_linear_svc=CalibratedClassifierCV(linear_svc)

cal_linear_svc.fit(lda_pc_f_classif_df,dfheart_train_y['heart_disease_present'])
#Extra Tree Classifier with randomised search for Classification

etc=ExtraTreesClassifier()

params_etc={

    'n_estimators' : range(10,1000), 

    'criterion': ['gini','entropy'], 

    'max_depth':range(1,50), 

    'min_samples_split': range(2,100), 

    'min_samples_leaf':range(1,100),

    'max_features':range(1,9)

}

etc_rscv=RandomizedSearchCV(etc,params_etc, verbose=1, cv=5,n_iter=300, n_jobs=3,scoring='neg_log_loss')

etc_rscv.fit(lda_pc_f_classif_df,dfheart_train_y['heart_disease_present'])
#Extra Class Classidiwe

etc_rscv.best_params_
#Random Forest Classifier with Hyperparameters

rf1_rscv=RandomForestClassifier()

rf1_rscv.fit(lda_pc_f_classif_df,dfheart_train_y['heart_disease_present'])
#Random Forest Classifier

rf1=RandomForestClassifier()

rf1.fit(lda_pc_f_classif_df,dfheart_train_y['heart_disease_present'])
#Logistic Regression with Hyperparameters

lr1_gscv=LogisticRegression(solver='lbfgs',max_iter=1000,fit_intercept= False,class_weight= None)

lr1_gscv.fit(lda_pc_f_classif_df,dfheart_train_y['heart_disease_present'])
#Logistic Regression

lr1=LogisticRegression()

#lr1.fit(pc_f_classif_df,dfheart_train_y['heart_disease_present'])

lr1.fit(lda_pc_f_classif_df,dfheart_train_y['heart_disease_present'])
#ForRandomForestClassifier

params_rf={'n_estimators':range(2,1000), #102 #32 #82

       'max_depth':range(2,100), #18 #8 #16

       'min_samples_leaf': range(1,50),

       'min_samples_split': range(2,40),

        'max_features':range(1,9),

       'criterion': ['gini','entropy'],} 

rs_rf=RandomizedSearchCV(rf1,params_rf,cv=5,n_iter=400, n_jobs=2,verbose=1,scoring='neg_log_loss')
rs_rf.fit(lda_pc_f_classif_df,dfheart_train_y['heart_disease_present'])
#LogisticRegression

params_lr={'solver':['newton-cg','lbfgs','liblinear','saga','sag'],

           'max_iter': range(100,1000,100),

           'class_weight': [None,'balanced'],

           'fit_intercept': [True,False] }

gs_lr=GridSearchCV(lr1,params_lr,cv=20, n_jobs=2,scoring='neg_log_loss')
gs_lr.fit(lda_pc_f_classif_df,dfheart_train_y['heart_disease_present'])
gs_lr.best_params_
print (log_loss(dfheart_train_y,rscv_xgb_tree.predict_proba(lda_pc_f_classif_df)))
#Data Preperation As Per Model

Xtest=test_data_prep(dfheart_test_X,f_classif_df.columns,True,2)

Xtest2=Xtest.join(pd.DataFrame(lda.transform(Xtest)).set_index(dfheart_test_X.index))

#lda_pc_f_classif_df=pc_f_classif_df.join(lda_df.set_index(f_classif_df.index))
#Test Prediction

test_preds=knn.predict_proba(Xtest2)[:,1]

output = pd.DataFrame({'heart_disease_present': test_preds}, index=dfheart_test_X.index)
print (log_loss(dftest_check,output))
print ('Accuracy : ', accuracy_score(dftest_check, lr1.predict(Xtest2)))

print ('Confusion Matrix : \n', confusion_matrix(dftest_check, lr1.predict(Xtest2)))

print ('Classification Report : \n', classification_report(dftest_check, lr1.predict(Xtest2)))
output.to_csv('submission-1.csv', index=True)
import os

#os.chdir(r'kaggle/working')

from IPython.display import FileLink

FileLink(r'submission-1.csv')
train_df=lda_pc_f_classif_df.copy()

val_df=dfheart_train_y

test_df=Xtest2



train_X,test_X,train_y,test_y=train_test_split(train_df,val_df, test_size=0.33,random_state=42)
#Model1-RandomForestClassifier

brf=RandomForestClassifier(n_estimators=20,min_samples_split=27,min_samples_leaf=15,max_features=5,max_depth=32)

brf.fit(train_X,train_y)
val_pred_1=pd.DataFrame(brf.predict(test_X),columns=['brf']).set_index(test_X.index)

test_pred_1=pd.DataFrame(brf.predict(test_df),columns=['brf']).set_index(test_df.index)
#Model2-XGBClassifier

bdt=XGBClassifier(subsample= 0.1610169491525424, num_feature= 7,max_depth=17,learning_rate=0.5064220183486238,eval_metric='logloss',eta=0.1864406779661017)

bdt.fit(train_X,train_y)
val_pred_2=pd.DataFrame(bdt.predict(test_X),columns=['bxgb']).set_index(test_X.index)

test_pred_2=pd.DataFrame(bdt.predict(test_df),columns=['bxgb']).set_index(test_df.index)
#Model3-kNN

bknn=KNeighborsClassifier(algorithm='brute',leaf_size=76, n_neighbors=19, weights='uniform')

bknn.fit(train_X,train_y)
val_pred_3=pd.DataFrame(bknn.predict(test_X),columns=['bknn']).set_index(test_X.index)

test_pred_3=pd.DataFrame(bknn.predict(test_df),columns=['bknn']).set_index(test_df.index)
df_val=pd.concat([test_X,val_pred_1,val_pred_2,val_pred_3], axis=1)

df_test=pd.concat([test_df,test_pred_1,test_pred_2,test_pred_3], axis=1)
model = LogisticRegression(C=0.08,fit_intercept=False)

model.fit(df_val,test_y)
model_svc=LinearSVC(C=0.7,fit_intercept=False)

cal_model=CalibratedClassifierCV(model_svc)

cal_model.fit(df_val,test_y)
from sklearn.linear_model import ElasticNet
log_loss(dftest_check,rblr.predict_proba(df_test)[:,1])
lr=LinearRegression()

lr.fit(df_val,test_y)

log_loss(dftest_check,lr.predict(df_test))
en=ElasticNet(alpha=0.14)

en.fit(df_val,test_y)

log_loss(dftest_check,en.predict(df_test))
params_tree={

        'booster':['gbtree'],

        'num_feature':range(1,9),

        'eta':list(np.linspace(0,1,50)),

        'max_depth':range(1,20),

        'subsample': list(np.linspace(0.01,1,50)),

        'learning rate': list(np.linspace(0,0.5,100)),

        'eval_metric':['logloss']

}

rmodel=RandomizedSearchCV(XGBClassifier(),params_tree, cv=3,n_iter=500, n_jobs=-1, verbose=1,scoring='neg_log_loss')

rmodel.fit(df_val,test_y)
log_loss(dftest_check,rblr.predict_proba(df_test)[:,1])
params_lr={

    'C':list(np.linspace(0.02,1,100)),

    'fit_intercept':[False],

    'max_iter':range(50,1000),

}
rblr=RandomizedSearchCV(LogisticRegression(),params_lr,scoring='neg_log_loss',n_iter=500,cv=3,verbose=1,n_jobs=3)

rblr.fit(df_val,test_y)

#log_loss(dftest_check,rblr.predict_proba(df_test)[:,1])


rblr=RandomizedSearchCV(KNeighborsClassifier(),scoring='neg_log_loss',n_iter=500,cv=5,verbose=1,n_jobs=3)

rblr.fit(df_val,test_y)

log_loss(dftest_check,rblr.predict_proba(df_test)[:,1])
rbknn.best_params_
confusion_matrix(test_y,brf.predict(test_X))

accuracy_score(test_y,brf.predict(test_X))

print (classification_report(test_y,brf.predict(test_X)))
log_loss(test_y,bknn.predict_proba(test_X)[:,1])
#Incomplete

def Stacking(model,train,y,test,n_fold):

    folds=StratifiedKFold(n_splits=n_fold,random_state=1)

    test_pred=np.empty((test.shape[0],1),float)

    train_pred=np.empty((0,1),float)

    for train_indices,val_indices in folds.split(train,y.values):

        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]

        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

    model.fit(X=x_train,y=y_train)

    train_pred=np.append(train_pred,model.predict(x_val))

    test_pred=np.append(test_pred,model.predict(test))

    return test_pred.reshape(-1,1),train_pred