import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from itertools import compress
from sklearn import model_selection, svm, metrics
import numpy as np
from sklearn.neural_network import MLPClassifier
df_old=pd.read_csv("../input/heart-disease-uci/heart.csv")

# analyze the dataset
dd=list(df_old) # take columnsof df
hist=df_old.hist(dd,figsize=(10,10))
# convert categorical variable to dummy variable
a = pd.get_dummies(df_old['cp'], prefix = "cp")
b = pd.get_dummies(df_old['thal'], prefix = "thal")
c = pd.get_dummies(df_old['slope'], prefix = "slope")
d = pd.get_dummies(df_old['ca'], prefix = "ca")
e = pd.get_dummies(df_old['exang'], prefix = "exang")
f = pd.get_dummies(df_old['fbs'], prefix = "fbs")
g = pd.get_dummies(df_old['restecg'], prefix = "restecg")
h = pd.get_dummies(df_old['sex'], prefix = "sex")

frames = [a, b, c, d, e, f, g, h, df_old]
df_old = pd.concat(frames, axis = 1)
df_old.head()
del a, b, c,d,e,f,g,h
df=df_old.drop(['cp','thal','slope','ca','exang','fbs','sex','restecg'],axis=1)

dd_hat=list(df) # take columnsof df
dd=dd_hat[0:30] # remove target column
del dd_hat

df.to_csv("train_dummy.csv",index=False) 
def fold_formation():
    df_frame=pd.read_csv("train_dummy.csv")
    df_frame["kfold"]=-1
    df_frame=df_frame.sample(frac=1).reset_index(drop=True) 
    kf=model_selection.StratifiedKFold(n_splits=5)
    
    for fold, (trn_,val_) in enumerate(kf.split(X=df_frame,y=df_frame['target'])):
        df_frame.loc[val_,'kfold']=fold
     
    df_frame.to_csv("train_fold.csv",index=False)  
    return df_frame
     
      
# perform 20x5 fold cross validation to obtain unbiased estimate of model
test_OUF=[]
all_metric=[]
train_OUF=[]
all_metric_OUF=[]
feat_all=[]
for ou_f in range(20):
    
   df_fr=fold_formation()  # call the function for forming folds
   
   for in_f in range(5):
        
       # for test_data
      flag=df_fr['kfold']==in_f        
      test_fold=df_fr[flag]
     
      # form train data
      train_fold=df_fr[~flag]
      del flag 
      
      # normalize train_data[excluding 'kfold' and 'target']
      train_data=train_fold.drop(['kfold','target'],axis=1)
      train_norm=train_data
      del train_data
      
      # normalize test data with mean and std of train data
      test_data=test_fold.drop(['kfold','target'],axis=1)
      test_norm=test_data
            
      fsel_col=dd
      train_matrix=train_norm[fsel_col] 
      del train_norm
      test_matrix=test_norm[fsel_col]
      del test_norm
      
      clf = svm.SVC(kernel='linear',gamma='auto',class_weight='balanced')
      clf.fit(train_matrix, train_fold['target'])
      train_pred = clf.predict(train_matrix)  # training accuracy
      tr_acc=metrics.accuracy_score(train_fold['target'],train_pred)
      tr_acc=tr_acc*100
      
      test_pred = clf.predict(test_matrix)  # training accuracy
      test_acc=metrics.accuracy_score(test_fold['target'],test_pred)
      test_acc=test_acc*100
      
      cm=metrics.confusion_matrix(test_fold['target'],test_pred)
      spec=(cm[0,0]*100)/(cm[0,1]+cm[0,0])
      sens=(cm[1,1]*100)/(cm[1,0]+cm[1,1])
      
      all_metric=([sens, spec, test_acc])
      all_metric_OUF.append(all_metric)
      train_OUF.append(tr_acc)
      feat_all.append(fsel_col)
      del fsel_col,train_pred, tr_acc,test_pred,test_acc,cm,spec,sens,all_metric
      del test_fold, train_fold,test_matrix, train_matrix
      
   del df_fr   
      
dm=np.asarray(all_metric_OUF)
dm_mean=np.mean(dm,axis=0)  

# sensitivity refers to rate of prediction of diseased subjects and specificity refers to rate of prediction of normal 
rslt=pd.DataFrame([dm_mean],columns=['sensitivity','specificity','accuracy'])
print(rslt)