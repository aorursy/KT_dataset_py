import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pck
warnings.filterwarnings('ignore')
%matplotlib inline
train_df = pd.read_csv("pulsar_data_train.csv")
test_df = pd.read_csv("pulsar_data_test.csv")
train_df.head(5)
test_df.head(5)
print(train_df.info())
print(test_df.info())
train_df.isnull().sum()
test_df.isnull().sum()
testdf = test_df.drop(['target_class'] ,axis=1)
testdf.tail(7)
def col_imputer(df):
    for col in df.columns:
        if df[col].isnull:
            df[col] = df[col].fillna(df[col].median())
    return df
col_imputer(train_df)
train_df.head(7)
col_imputer(testdf)
testdf.head(8)
print(train_df.isna().sum(),"\n\n")
print(testdf.isnull().sum())
train_df.hist(bins=50,figsize=(14,10))
plt.show()
corr = train_df.corr().abs()
plt.figure(figsize=(6,6))
sns.heatmap(corr ,annot=True ,cmap="coolwarm")
from sklearn.preprocessing import MinMaxScaler
nor_scale = MinMaxScaler()
train_sc_df = train_df.iloc[:,:-1]
train_sc_df.head(5)
train_tar = train_df.target_class
train_tar.head(3)
colset = train_sc_df.columns.to_list()
colset
scaled_train_feat = nor_scale.fit_transform(train_sc_df)
scaled_train_feat
dsf = pd.DataFrame(scaled_train_feat,columns=colset)
dsf.head(6)
colset_2 = testdf.columns.to_list()
scaled_test_feat = nor_scale.transform(testdf)
scaled_test_feat
dsftest = pd.DataFrame(scaled_test_feat ,columns=colset_2)
dsftest.head(5)

dsf.hist(bins=50,figsize=(14,12))
plt.show()
# Assigning "X" with Features .
X = dsf
#Assigning "y" with target labels 
y = train_tar
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10,random_state=0) #splitting feature and target into train &
                                                             #validation set
skf.get_n_splits(X,y)
for train_index,validation_index in skf.split(X,y):
    X_train , X_val = X.iloc[train_index] , X.iloc[validation_index]
    y_train , y_val = y.iloc[train_index] , y.iloc[validation_index]    
#creating matrixes or numpy arrays for features.
def conv_arr(data):          #AS NUMPY ARRAYS AND VECTORS WILL SPEED UP COMPUTATION
    return np.array(data)
X_train1 = conv_arr(X_train)
X_val1 = conv_arr(X_val)

#creating vectors for Target sets.
y_train1 = conv_arr(y_train).reshape(-1,1)
y_val1 = conv_arr(y_val).reshape(-1,1)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
Log_reg = LogisticRegression(solver='liblinear')    
dec_tree = DecisionTreeClassifier(random_state=42)
Ran_for = RandomForestClassifier(n_estimators=50,criterion='entropy',max_leaf_nodes=200,n_jobs=-1)
xgb_class = XGBClassifier(n_estimators=50, n_jobs=-1)
# we can use rest of the models after testing and hyperparameter tuning
#A list to append accuracy scores from classifiers in various n_splits of strata.
# Create function to check out various model algorithms.
def model_create(clf):
    clf.fit(X_train1,y_train1)
    pred = clf.predict(X_val1)
    score = accuracy_score(pred,y_val1)
    return score,pred
ranacc,predran = model_create(Ran_for) #Score of RandomForestClassifier model
ranacc,predran
logacc, predlog = model_create(Log_reg)  #Score of LogisticRegression model.
logacc
decacc, preddec = model_create(dec_tree) #Score of DecisionTreeClassifier model
decacc
xgb_acc, pred_xgb = model_create(xgb_class) #Score of XGBoostClassifier 
xgb_acc
#Hyoerparameter tuning with Randomized search CV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# model_params={
#     'ran_for':{
#         'model':RandomForestClassifier(n_jobs=-1),
#         'params':{
#             'n_estimators':[100,150,200,500,800,1000],
#             'criterion':['gini','entropy'],
#             'max_leaf_nodes':[50,55,60,100,200,500,800,2000],
#             'max_depth':[d for d in range(5,50,5)],
               
#         }
#      },
#     'log_reg':{
#         'model':LogisticRegression(),
#         'params':{
#             'solver':['liblinear','saga','sag','lbfgs'],
#             'multi_class':['auto','ovr','multinomial'],
#             'C':[1,3,5,7,10],
#             'tol':[1e-4,1e-3,2e-3,2e-4,3e-4]
            
#         }
    
#     },
#     'xgb_class':{
#         'model':XGBClassifier(),
#         'params':{
#             'booster':['gblinear','gbtree'],
#             'base_score':[0.5,0.6,0.7,0.8],
#             "min_child_weight":[i for i in range(0,50,5)],
#             'gamma':[i for i in range(0,30,6)],
#             'max_delta_step':[i for i in range(0,30,6)],
#             'validate_parameters':[1,2,3,4,5],
#             'max_depth':[i for i in range(6,42,6)],
#             'max_depth':[6,9,12,30,60,200,800,1000,2000]
            
#         }
#     }
# }
# score = []

# for mod,mp in model_params.items():
#     RanSrClf = RandomizedSearchCV(mp['model'],mp['params'],cv=5,
#                                      n_jobs=-1,
#                                   return_train_score=False,n_iter=12) 
#     RanSrClf.fit(X_train1,y_train1)
#     score.append({
#         'model':mod,
#         'best_score':RanSrClf.best_score_ ,
#         'best_params':RanSrClf.best_params_
#     })
# # Scoring    
# score
# for i in score:
#     print(i,'\n\n')
        
# with open ("model.pickle",'wb') as f:
#     pck.dump(RanSrClf,f)
RanSrmodel = pck.load(open("model.pickle",'rb'))
# clfdf = pd.DataFrame(score,columns = ['model','best_score','best_params'])
# clfdf

Ran_for_tuned = RandomForestClassifier({'n_estimators': 500, 'max_leaf_nodes': 2000, 
                                        'max_depth': 45, 'criterion': 'gini'})
Log_reg_tuned = LogisticRegression({'tol': 0.002, 'solver': 'saga', 
                                    'multi_class': 'multinomial', 'C': 5})
Xgb_cla_tuned = XGBClassifier(validate_parameters= 5, min_child_weight= 35, 
                               max_depth= 2000,
                               max_delta_step= 24, gamma= 0,
                               booster= 'gbtree', base_score= 0.5)


       
        
acc_xgbtune , pred_xgbtune = model_create(Xgb_cla_tuned)   
pred_xgbtune
repo_ = confusion_matrix(pred_xgbtune,y_val1)
print("confusion matrix  {}".format(repo_))

repo1_  = confusion_matrix(predran,y_val1)
repo1_
repo2 = confusion_matrix(pred_xgb,y_val1)
repo2
X_test = scaled_test_feat
finpred = Xgb_cla_tuned.predict(X_test)
print("Prediction of tuned XGBoost Classifier {}".format(finpred.reshape(-1,1)))
finpred_ran = Ran_for.predict(X_test)
print("Prediction of Random Forest Classifier {}".format(finpred_ran.reshape(-1,1)))
finpred_xgb_ = xgb_class.predict(X_test)
print("Predixction of untuned Xgboost classifier {}".format(finpred_xgb_.reshape(-1,1)))
submission = pd.DataFrame({'target_class':finpred_ran})
    
    
submission.to_csv('submission.csv')
test_df_pred = dsftest.copy()
test_df_pred['target_class'] = finpred_ran
test_df_pred.head(20)
val_pred_proba = Ran_for.predict_proba(X_val1)
val_pred_proba
y_pred_proba_threshold = Ran_for.predict_proba(X_val1)[:,1]
y_pred_proba_threshold
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score
plt.hist(y_pred_proba_threshold ,bins=12)
plt.xlim(0,0.5)
FPR ,TPR ,thresholds = roc_curve(y_val1,predran)

accuracy_ls = []
for thres in y_pred_proba_threshold:
    y_acc_review_pred = np.where(y_pred_proba_threshold > thres ,1,0)
    accuracy_ls.append(accuracy_score(y_val1,y_acc_review_pred ,normalize = True))
thres_series = pd.Series(y_pred_proba_threshold)
accuracy_series = pd.Series(accuracy_ls)    
fin_accuracy = pd.concat((thres_series,accuracy_series) ,axis=1 )
fin_accuracy.columns = ['threshold','accuracy']
fin_accuracy.sort_values(by='accuracy',ascending=False,inplace=True)
fin_accuracy
(y_pred_proba_threshold > 0.360038).astype(float)
#Instead of for loop we can use vactorized assistance of numpy library of python.
final_Roc_adj_prediction = np.where(y_pred_proba_threshold > 0.360038 ,1,0)
score_auc_roc = accuracy_score(final_Roc_adj_prediction,y_val1)
score_auc_roc
confusion_matrix(final_Roc_adj_prediction,y_val1)
final_test_data_pred = Ran_for.predict_proba(X_test)[:,1]
fin_adj_pred_test_data = (final_test_data_pred > 0.360038).astype(float)
fin_adj_pred_test_data
submission_auc_roc = pd.DataFrame({'target_class':fin_adj_pred_test_data})

submission_auc_roc.to_csv("submission_auc_roc.csv")
test_df_adj_pred = dsftest.copy()
test_df_adj_pred['target_class'] = fin_adj_pred_test_data
test_df_adj_pred.head(5)
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(20,input_dim=8,activation='relu'))
model.add(Dense(48,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train1,y_train1 ,epochs=100)
model.evaluate(X_val1,y_val1)
pred = model.predict(X_val1)
pred = pred.round()
accuracy = accuracy_score(pred,y_val1)
score = confusion_matrix(pred,y_val1)
print(f'confusion-matrix{score} accuracy{accuracy}')
final_test_pred = model.predict(X_test).round()
final_test_pred
testdf_NNPred = testdf.copy()
testdf_NNPred["target_class_NN"] = final_test_pred
testdf_NNPred

