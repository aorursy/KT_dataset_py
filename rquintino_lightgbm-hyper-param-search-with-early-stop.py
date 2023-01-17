import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore")

import os
print(os.listdir("../input"))


target="HasDetections"
submission_id_col="MachineIdentifier"

seed_split=1 
test_size=1/3
seed_train=100

df_kaggle_train = pd.read_hdf(
         '../input/save-hdf-full/train.hdf',
         key="train"
)
df_kaggle_test = pd.read_hdf(
         '../input/save-hdf-full/test.hdf',
         key="test"
)
df_kaggle_train.shape,df_kaggle_test.shape
X=df_kaggle_train.sample(10000)
from sklearn.base import BaseEstimator, TransformerMixin

# From other kernels available, frequency based rank encoding
def frequency_encoding(X,variable):
    t = X[variable].value_counts(normalize=True,dropna=False).reset_index()
    t = t.reset_index()
    t.loc[t[variable] == 1, 'level_0'] = np.nan
    t.set_index('index', inplace=True)
    max_label = t['level_0'].max() + 1
    t.fillna(max_label, inplace=True)
    
    #return t.to_dict()['level_0']
    return t.to_dict()[variable]


class FitState():
    def __init__(self):
        pass
    
class PrepPipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self,fillna=None,fix_smartscreen=False,fix_engineversion=False,cat_encode=False,copy=True,notes=None):
        self.notes=notes
        self.copy=copy
        self.fillna=fillna
        self.fix_smartscreen=fix_smartscreen
        self.fix_engineversion=fix_engineversion
        self.cat_encode=cat_encode
        
    def fit(self, X, y=None):
        self.fit_state=FitState()
        self.prepare(X=X,y=y,fit=True)
        return self

    def transform(self, X,y=None):
        assert isinstance(X, pd.DataFrame)
        return self.prepare(X=X,y=y,fit=False)
    
    def show_params(self):
        print("fit_state",vars(self.fit_state))
        print("params",self.get_params())
        
    # Experiment is reduce class overhead, bring related fit & transform closer, no models without pipelines
    def prepare(self,X,y=None,fit=False):
        
        fit_state=self.fit_state
        if (self.copy):
            X=X.copy()
        

        # SmartScreen fix
        if (self.fix_smartscreen):
            X.SmartScreen=X.SmartScreen.str.lower()
            X.SmartScreen.replace({"promt":"prompt",
                                    "promprt":"prompt",
                                    "00000000":"0",
                                    "enabled":"on",
                                    "of":"off" ,
                                    "deny":"0" , # just one
                                    "requiredadmin":"requireadmin"
                                   },inplace=True)
            X.SmartScreen=X.SmartScreen.astype("category")

        # Numeric missings
        if self.fillna is not None:        
            X.select_dtypes(include=[np.number]).fillna(self.fillna,inplace=True)
    
        if self.fix_engineversion:
            X[["EngineVersion_1","EngineVersion_2","EngineVersion_3","EngineVersion_4"]]=X.EngineVersion.str.split(".",expand=True).astype(int)
            X.drop(columns=["EngineVersion_1","EngineVersion_2"],inplace=True)
            X["EngineVersion_34"]=X.EngineVersion_3*10+X.EngineVersion_4
        
        
        if self.cat_encode:
            if fit:
                cat_encode_vars=['Census_OEMModelIdentifier', 'CityIdentifier', 'Census_FirmwareVersionIdentifier',
                                 'AvSigVersion','AVProductStatesIdentifier','CountryIdentifier','Census_ProcessorModelIdentifier',
                                 'Census_OSInstallTypeName','Census_FirmwareManufacturerIdentifier',
                                 'SmartScreen','AppVersion','EngineVersion',"SMode",
                                 'OsVer','OsSuite','OsPlatformSubRelease','SkuEdition',"Platform"
                                ]
                fit_state.categorical_columns=cat_encode_vars
        
                freq_enc_dict_dict = {}
                for variable in fit_state.categorical_columns:
                    freq_enc_dict_dict[variable] = frequency_encoding(X,variable)
                fit_state.freq_enc_dict_dict=freq_enc_dict_dict
            else:
                freq_enc_dict_dict = fit_state.freq_enc_dict_dict
                test_freq_enc_dict_dict={}
                for variable in fit_state.categorical_columns:
                    test_freq_enc_dict_dict[variable] = frequency_encoding(X,variable)
                    X[variable+"_freq"] = X[variable].map(lambda x: freq_enc_dict_dict[variable].get(x, np.nan)).astype("float64")
                    X[variable+"_freq_test"] = X[variable].map(lambda x: test_freq_enc_dict_dict[variable].get(x, np.nan)).astype("float64")
        
        X.drop(columns=["AvSigVersion","EngineVersion","SMode",
                        "AppVersion","Census_OSVersion"],inplace=True)
        return X


# Test
PrepPipeline(copy=True,cat_encode=True).fit_transform(X).head().T
TRAIN_ROWS=500000
df_train=df_kaggle_train.sample(TRAIN_ROWS,random_state=seed_split)
df_train.head()
from sklearn.model_selection import train_test_split

# Split X,y
y= df_train[target].values
df_train.drop(columns=target,inplace=True)
df_train.drop(columns=[submission_id_col],inplace=True)
# Split kaggle train, reserve internal hold out test set
X_train_all, X_test, y_train_all,y_test = train_test_split(df_train,y, 
                                                   test_size=test_size, random_state=seed_split,stratify =y)
# Split for eval df/early stopping

test_frac=.10
from sklearn.model_selection import train_test_split
X_train, X_eval, y_train,y_eval = train_test_split(X_train_all,y_train_all, 
                                                   test_size=test_frac, random_state=seed_split,stratify =y_train_all)
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

param_grid={"clf__boosting_type": ["gbdt","dart","goss","rf"],
           # "clf__class_weight":[None,"balanced"],
            #"clf__colsample_bytree": [.1,.2,.3,.4,.5,.6,.8,.9,1],
            "clf__subsample":[.4,.5,.75,.9,1],
            "clf__max_bin":[10,50,100,340,500,170],
            "clf__importance_type":['split'],
            "clf__num_leaves":[10,25,31,53,100,31,None],
           # "clf__min_split_gain":[.05,.025,.01,.1],
                     }
n_estimators=2000

pipeline=Pipeline([
    ('prep', PrepPipeline(notes="with cat encode",cat_encode=True)),
    ("clf",LGBMClassifier(random_state=seed_train,
                          n_jobs=1,
                          learning_rate=.3,
                          n_estimators=n_estimators ))])
         

prep_pipeline=pipeline.named_steps["prep"].fit(X_train,y_train)
eval_set=[(prep_pipeline.transform(X_eval),y_eval), 
          (prep_pipeline.transform(X_train), y_train)]
eval_set[0][0].shape
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,train_test_split

if __name__ == "__main__":
    
    from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold,RepeatedKFold

    random_state=43
    rskf = RepeatedKFold(n_splits=5, n_repeats=2,random_state=random_state)
    
    # run randomized search
    n_iter_search = 50
    search_njobs=-1
    model_search = RandomizedSearchCV(pipeline, param_distributions=param_grid,n_jobs=search_njobs,n_iter=n_iter_search,
                                      scoring="roc_auc",error_score=0,cv=rskf,random_state=seed_train,
                                     verbose=1)
    
    # Fit with early stopping
    early_stopping_round=50
    print(X_train.shape,y_train.shape) 
    model_search.fit(X_train,y_train,
           #clf__eval_metric = 'auc',
           clf__eval_set= eval_set,
           clf__eval_names=["eval","train"],
           clf__verbose=False,
           clf__early_stopping_rounds = early_stopping_round
            )

    
df_cv_results=pd.DataFrame(model_search.cv_results_).sort_values(by='rank_test_score')
df_cv_results["dif_test_train"]=df_cv_results.mean_train_score-df_cv_results.mean_test_score
df_cv_results.fillna("NA",inplace=True)

df_cv_results.drop("params",axis=1,inplace=True)

df_cv_results.sort_values("mean_test_score",ascending=False,inplace=True)

df_cv_results.to_csv(f"cv_results_{TRAIN_ROWS}_{n_iter_search}_{seed_train}.csv")
df_cv_results
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display as display, Markdown

eval_cv_metric="mean_test_score"
train_cv_metric="mean_train_score"

score_result=eval_cv_metric
score_result2="dif_test_train"
split_col=""

display(Markdown("## %s,%s vs parameters (numeric)"%(score_result,score_result2)))

all=df_cv_results

all["all"]=""

if not split_col in all.keys():
     split_col="all"

axis=0
for col in all.columns:
    if col.startswith("param_") and len(all[col].unique())>1:
        plt.figure(figsize=(12,6))

        sns.boxplot(x=col, y=score_result, hue=split_col,data=all)
        sns.swarmplot(x=col, y=score_result, color="red",data=all)
        plt.legend()
        ax2 = plt.twinx()
        sns.pointplot(x=col, y=score_result2,hue=split_col,ax=ax2, data=all)
