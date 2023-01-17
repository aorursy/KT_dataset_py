import pandas as pd
import numpy as np
import seaborn as sns
df1 = pd.read_csv('data/mem_data.csv', parse_dates=['BIRTH_DT','RGST_DT','LAST_VST_DT']).sort_values('MEM_ID')
df1
df1['GENDER'].value_counts()
df2 = pd.read_csv('data/mem_transaction.csv', parse_dates=['SELL_DT','MEMP_DT']).sort_values('MEM_ID')
df2
df3 = pd.read_csv('data/store_info.csv')
df3
merged = df1.merge(df2,how='left')
merged = merged.merge(df3,how='left')
merged
merged = merged.fillna(0)
merged
merged.info()
merged.BIRTH_SL = (merged.BIRTH_SL=='S').astype(int)
merged.BIRTH_SL.value_counts()
merged.SMS = (merged.SMS=='Y').astype(int)
merged.SMS.value_counts()
merged.MEMP_STY = (merged.MEMP_STY=='O').astype(int)
merged.MEMP_STY.value_counts()
apply = np.where(merged['USED_PNT']==0,1,0)
local = merged.columns.get_loc('MEMP_TP')
merged.insert(loc=local, column='MEMP_TP_CODE', value=apply)

merged = merged.drop('MEMP_TP', axis=1)
merged
merged = merged.assign(RGSTyear=merged.RGST_DT.dt.year,
                      LASTyear=merged.LAST_VST_DT.dt.year,
                      SELLyear=merged.SELL_DT.dt.year,
                      MEMPyear=merged.MEMP_DT.dt.year)
merged
# Label encoding
'''from sklearn.preprocessing import LabelEncoder

cat_features = ['RGST_DT','LAST_VST_DT','MEMP_DT']
encoder = LabelEncoder()
encoded = merged[cat_features].apply(encoder.fit_transform)
encoded'''
cols = ['BIRTH_DT','ZIP_CD']
merged[cols] = merged[cols].apply(lambda x: x.astype('category').cat.codes)
merged
merged.info()
train = merged.GENDER!='UNKNOWN'
train_df = merged[train]
train_df.GENDER = (train_df.GENDER=='M').astype(int)
train_df
pred = merged.GENDER=='UNKNOWN'
pred_df = merged[pred].sort_values('MEM_ID')
pred_df
'''sns.heatmap(train_df[["GENDER","VISIT_CNT","SALES_AMT","BIRTH_SL_CODE","USABLE_PNT","USED_PNT","ACC_PNT",
                     "USABLE_INIT","SMS_CODE","STORE_ID","SELL_AMT"]]\
            .corr(),annot=True, fmt = ".2f", cmap = "coolwarm")'''
train_df.corr()
train_df.info()
from sklearn.model_selection import train_test_split
feature_data = train_df.drop(['GENDER','MEM_ID','RGST_DT','LAST_VST_DT','SELL_DT','MEMP_DT'],axis=1)
label_data = train_df.GENDER.values

X_train,X_test,y_train,y_test = train_test_split(feature_data,label_data,test_size=.25,random_state=0)

prediction_data = pred_df.drop(['GENDER','MEM_ID','RGST_DT','LAST_VST_DT','SELL_DT','MEMP_DT'],axis=1)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
prediction_data
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.under_sampling import *
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import glob

kfold = StratifiedKFold(n_splits=5) # 하이퍼 파라미터 지정
n_it = 12

params = {'max_features':list(np.arange(1, train_df.shape[1])), 'bootstrap':[False], 'n_estimators': [50], 'criterion':['gini','entropy']}
model = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params, n_iter=n_it, cv=kfold, scoring='roc_auc',n_jobs=-1, verbose=1)
print('MODELING.............................................................................')
model.fit(feature_data, label_data)
print('========BEST_AUC_SCORE = ', model.best_score_)
model = model.best_estimator_
pred_df.GENDER = model.predict_proba(prediction_data.values)[:,1]
print('COMPLETE')
y_model = model.predict(X_test)
print("Dummy model:"); print(confusion_matrix(y_test, pred_dummy))
print("LightGBM:"); print(confusion_matrix(y_test, y_model))
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
auc(fpr, tpr)
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
def plot_roc_curve(fpr, tpr, model, color=None) :
    model = model + ' (auc = %0.3f)' % auc(fpr, tpr)
    plt.plot(fpr, tpr, label=model, color=color)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.axis([0,1,0,1])
    plt.xlabel('FPR (1 - specificity)')
    plt.ylabel('TPR (recall)')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
fpr_dummy, tpr_dummy, _ = roc_curve(y_test, 
                                    dummy.predict_proba(X_test)[:,1])
plot_roc_curve(fpr_dummy, tpr_dummy, 'dummy model', 'hotpink')
fpr_tree, tpr_tree, _ = roc_curve(y_test, 
                                  model.predict_proba(X_test)[:,1])
plot_roc_curve(fpr_tree, tpr_tree, 'lightgbm', 'darkgreen')
from sklearn.metrics import precision_recall_curve

def plot_precision_recall_curve(precisions, recalls) :
    plt.plot(recalls, precisions, color='blue')
    plt.axis([0,1,0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
precisions, recalls, _ = precision_recall_curve(y_test, 
                                    model.predict_proba(X_test)[:,1])
plot_precision_recall_curve(precisions, recalls)
final_check = pred_df.copy()
final_check.GENDER = model.predict(prediction_data.values)
final_check['GENDER'].value_counts()
final_check = final_check.groupby(['MEM_ID','GENDER']).agg({'SELL_AMT':'count'}).reset_index()
final_check['GENDER'].value_counts()
pred_df.GENDER = model.predict_proba(prediction_data.values)[:,1]
pred_df
final = pred_df.groupby('MEM_ID').agg({'GENDER':'mean'}).reset_index()
final
final.to_csv('output_data2.csv',index=False)