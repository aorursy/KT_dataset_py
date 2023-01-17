from sklearn import model_selection,linear_model,metrics
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import decomposition,ensemble
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fmin
from functools import partial
import xgboost

import numpy as np 
import pandas as pd 
import glob
import time
import os

import warnings
warnings.filterwarnings("ignore")
## getting all the data files available
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def create_folds(df_path,num_folds):
    df  = pd.read_csv(df_path,sep='\t')
    df.loc[:,"kfold"] = -1
    
    ## for random shuffle of data, frac=1 will return all the records in shuffled manner.If you want to extract random subsample
    ##, change the frac paramter value
    
    df = df.sample(frac=1).reset_index(drop=True) 
    
    y = df.sentiment.values
    skf = model_selection.StratifiedKFold(n_splits=num_folds)
    
    ## t_ : indices for training, v_: indices for validation, f : fold number
    for f,(t_,v_) in enumerate(skf.split(X=df,y=y)):
        df.loc[v_,"kfold"] = f
        
    df.to_csv(f"train_folds_{num_folds}.csv",index=False)
    

create_folds("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip",5)
pd.read_csv("train_folds_5.csv").kfold.value_counts()
model_dict = {'lr':linear_model.LogisticRegression(),'rf':ensemble.RandomForestClassifier()}
feat_create_dict = {'tfidf':TfidfVectorizer(max_features=1000),'cntvec':CountVectorizer(),'svd':decomposition.TruncatedSVD(n_components=120)}
def run_training(path,fold,feat_create_mode,model_name):
    df = pd.read_csv(path)
    
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)

    if feat_create_mode!='svd':
        featvec = feat_create_dict[feat_create_mode]
        featvec.fit(df_train.review.values)

        xtrain = featvec.transform(df_train.review.values)
        xvalid = featvec.transform(df_valid.review.values)
    else :
        featvec = feat_create_dict['tfidf']
        featvec.fit(df_train.review.values)

        xtrain = featvec.transform(df_train.review.values)
        xvalid = featvec.transform(df_valid.review.values)
        
        svd = feat_create_dict[feat_create_mode]
        svd.fit(xtrain)
        
        xtrain = svd.transform(xtrain)
        xvalid = svd.transform(xvalid)
        
    
    ytrain = df_train.sentiment.values
    yvalid = df_valid.sentiment.values
    
    clf = model_dict[model_name]
    clf.fit(xtrain,ytrain)
    
    ypred = clf.predict_proba(xvalid)[:,1]
    auc = metrics.roc_auc_score(yvalid,ypred)
    
    print(f"fold = {fold} AUC = {auc}")
    
    df_valid.loc[:,f'{feat_create_mode}_{model_name}_pred'] = ypred
    
    return df_valid[['id','sentiment','kfold',f'{feat_create_mode}_{model_name}_pred']]
for key_feat,val_feat in  feat_create_dict.items():
    for key_model,val_model in model_dict.items():
        strt = time.time()
        print(f" Training model feature creation:{key_feat} and model:{key_model}")
        dfs = []
        for j in range(5):
            temp_df = run_training("train_folds_5.csv",j,key_feat,key_model)
            dfs.append(temp_df)
        fin_valid_df = pd.concat(dfs)
        fin_valid_df.to_csv(f"{key_feat}_{key_model}.csv",index=False)
        end = time.time()
        print(f"Time Taken: {end-strt}secs")
        print(fin_valid_df.shape)
files = glob.glob("*.csv")
files.remove("train_folds_5.csv")
df = pd.DataFrame()
for f in files :
    if len(df)<=0:
        df = pd.read_csv(f)
    else :
        temp_df = pd.read_csv(f).drop(columns = ['sentiment','kfold'])
        df = pd.merge(df,temp_df,on=['id'],how='left')
pred_cols = [col for col in df.columns if col.find("pred")>=0]
targets = df.sentiment.values

pred_dict = {col:df[col].values for col in pred_cols}
pred_rank_dict = {col:df[col].rank().values for col in pred_cols}

## Getting AUC for all models separately 
for col in pred_cols:
    auc = metrics.roc_auc_score(targets,df[col].values)
    print(f"pred_col = {col}; overall_auc ={auc}")
print("-------------------------------------------")
print("Blending Results")
print("-------------------------------------------")

print("average")
avg_pred = df[pred_cols].mean(axis=1).values
print(metrics.roc_auc_score(targets,avg_pred))

print("-------------------------------------------")
print("weighted average")
wt_dict = {col:1 for col in pred_cols}
wt_dict['tfidf_lr_pred'] = 3
print("weights used")
print(wt_dict)
avg_pred = np.sum(np.array([val*wt_dict[key] for key,val in pred_dict.items()]),axis=0)/sum(list(wt_dict.values()))
print(metrics.roc_auc_score(targets,avg_pred))

print("-------------------------------------------")
print("rank averaging")
avg_pred  = np.mean(np.array([val for key,val in pred_dict.items()]),axis=0)
print(metrics.roc_auc_score(targets,avg_pred))

print("-------------------------------------------")
print("weighted rank averaging")
wt_rank_dict = {col:1 for col in pred_cols}
wt_rank_dict['tfidf_lr_pred'] = 3
print("weights used")
print(wt_rank_dict)
avg_pred = np.sum(np.array([val*wt_rank_dict[key] for key,val in pred_rank_dict.items()]),axis=0)/sum(list(wt_rank_dict.values()))
print(metrics.roc_auc_score(targets,avg_pred))

print("-------------------------------------------")
print("weighted rank averaging")
wt_rank_dict = {col:1 for col in pred_cols}
wt_rank_dict['cntvec_lr_pred'] = 3
print("weights used")
print(wt_rank_dict)
avg_pred = np.sum(np.array([val*wt_rank_dict[key] for key,val in pred_rank_dict.items()]),axis=0)/sum(list(wt_rank_dict.values()))
print(metrics.roc_auc_score(targets,avg_pred))
## defining custom class for optimzation to get optimal weights
class OptimizeAUC():
    def __init__(self):
        self.coef_ = 0
    
    ## function to caluculate AUC for each fold while optimizing and 
    ## multiplying it with -1 , becasue we are using fmin (minimizing the metric) which in turn led to maximization of AUC
    def _auc(self,coef,X,y):
        X_coef = X*coef
        predictions = np.sum(X_coef,axis=1)
        auc_score = metrics.roc_auc_score(y,predictions)
        return -1.0*auc_score
    
    ## function for initiating optimization process.Here we are initializing the weights with dirichlet distribution
    ##, we can take any other values also for weight initialization
    def fit(self,X,y):
        partial_loss = partial(self._auc,X=X,y=y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        self.coef_ = fmin(partial_loss,init_coef,disp=True)
    
    ## function to make prediction using weights obtained while training
    def predict(self,X):
        x_coef = X*self.coef_
        predictions = np.sum(x_coef,axis=1)
        return predictions
## function for model development for optimal parameters
def run_training_wts(pred_df,fold,pred_cols,model_name,std=False):

    train_df = pred_df[pred_df.kfold!=fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold==fold].reset_index(drop=True)
    
    xtrain = train_df[pred_cols].values
    xvalid = valid_df[pred_cols].values
    
    if std:
        std_ = StandardScaler()
        std_.fit(xtrain)
        
        xtrain = std_.transform(xtrain)
        xvalid = std_.transform(xvalid)
    
    opt = model_dict[model_name]
    opt.fit(xtrain,train_df.sentiment.values)
    if model_name != 'lr':
        preds = opt.predict(xvalid)
    else :
        preds = opt.predict_proba(xvalid)[:,1]
    
    auc = metrics.roc_auc_score(valid_df.sentiment.values,preds)
    
    print(f"fold={fold} auc={auc}")

    return opt.coef_
model_dict = {'lr':linear_model.LogisticRegression(),'custom_opt':OptimizeAUC(),'linear':linear_model.LinearRegression()}
for key_model,val_model in model_dict.items():
    for std in [True,False]:
        print(f"model:{key_model}; Scaling:{std}")
        coefs = []
        for j in range(5):
            temp_df =  run_training_wts(df,j,pred_cols,key_model,std)
            coefs.append(temp_df)
        coefs  = np.mean(np.array(coefs),axis=0)
        print(coefs)
        if key_model!='lr':
            wt_avg = np.sum(np.array([coefs[idx]*df[col].values for idx,col in enumerate(pred_cols)]),axis=0)
        else :
            wt_avg = np.sum(np.array([coefs[0,idx]*df[col].values for idx,col in enumerate(pred_cols)]),axis=0)
            
        auc = metrics.roc_auc_score(targets,wt_avg)
        print(f"optimal coefs overall auc = {auc}")
        print("==================================")

def run_training_stack(pred_df,fold,pred_cols):

    train_df = pred_df[pred_df.kfold!=fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold==fold].reset_index(drop=True)
    
    xtrain = train_df[pred_cols].values
    xvalid = valid_df[pred_cols].values
    
    clf = xgboost.XGBClassifier()
    clf.fit(xtrain,train_df.sentiment.values)
    preds = clf.predict_proba(xvalid)[:,1]
    
    auc = metrics.roc_auc_score(valid_df.sentiment.values,preds)
    print(f"fold={fold} auc={auc}")

    valid_df.loc[:,"xgb_pred"] = preds
    
    return valid_df
dfs = []
for j in range(5):
    temp_df = run_training_stack(df,j,pred_cols)
    dfs.append(temp_df)
fin_valid_df = pd.concat(dfs)
fin_valid_df.to_csv("xgb.csv",index=False)
auc = metrics.roc_auc_score(targets,fin_valid_df.xgb_pred.values)
print(f"AUC using Xgboost : {auc}")
