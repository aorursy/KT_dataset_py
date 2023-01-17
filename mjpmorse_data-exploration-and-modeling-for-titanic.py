%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt



plt.style.use('ggplot')

import matplotlib.cm as cm

import seaborn as sns



import pandas as pd

import pandas_profiling

import numpy as np

from numpy import percentile

from scipy import stats

from scipy.stats import skew

from scipy.special import boxcox1p



import os, sys

import calendar



from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_log_error, make_scorer

from sklearn.metrics.scorer import neg_mean_squared_error_scorer



from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, RidgeCV

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor









import warnings

warnings.filterwarnings('ignore')



plt.rc('font', size=18)        

plt.rc('axes', titlesize=22)      

plt.rc('axes', labelsize=18)      

plt.rc('xtick', labelsize=12)     

plt.rc('ytick', labelsize=12)     

plt.rc('legend', fontsize=12)   



plt.rcParams['font.sans-serif'] = ['Verdana']



# function that converts to thousands

# optimizes visual consistence if we plot several graphs on top of each other

def format_1000(value, tick_number):

    return int(value / 1_000)



pd.options.mode.chained_assignment = None

pd.options.display.max_seq_items = 500

pd.options.display.max_rows = 500

pd.set_option('display.float_format', lambda x: '%.5f' % x)



local = False

debug = False



if(local):

    BASE_PATH = "./"

else:

    BASE_PATH = "../input/titanic/"



df = pd.read_csv(f"{BASE_PATH}train.csv",index_col='PassengerId')

df_test = pd.read_csv(f"{BASE_PATH}test.csv",index_col='PassengerId')
df.info(verbose=True, null_counts=True)
missing = [(c, df[c].isna().mean()*100) for c in df]

missing = pd.DataFrame(missing, columns=["column_name", "percentage"])

missing = missing[missing.percentage > 0]

display(missing.sort_values("percentage", ascending=False))
def cabin_location(df):

    df['Cabin'].unique()

    df['Cabin'].fillna('Unknown',inplace = True)

    df['Cabin_Location'] = df['Cabin'].str[0]

    #Here we use the class data to determine the Fill Na

    df['Cabin_Location'] = np.where((df.Pclass==1) & (df['Cabin_Location']=='U'),'C',

                           np.where((df.Pclass==2) & (df['Cabin_Location']=='U'),'D',

                           np.where((df.Pclass==3) & (df['Cabin_Location']=='U'),'G',

                           np.where(df['Cabin_Location']=='T','C',df['Cabin_Location']))))

    #    cabin_map = {'A':6,'B':5,'C':4,'D':3,'E':2,'F':1,'G':0}

    #    df['Cabin_Location'] = df['Cabin_Location'].replace(cabin_map)

    return df



    

    

def title_extract(df):

    ## Mme is a french honorific for Madame: Similar to Mrs. 

    ## Mlle is a french honorific for Mademoiselle: Similar to Ms. 

    ## Miss = Ms

    ## Don, Mr, and Sir will be combined into one set

    ## Donna, Mrs and Lady will be combined into one set

    ## We will keep: Mr, Mrs, Mss, Dr, Master, Rev, Other 

    name_splice = df['Name'].str.split(pat=', ',n=-1,expand=True)

    df['Title'] = name_splice[1].str.split(pat='.',n=-1,expand=True)[0]

    title_correction_map = {'Mme':'Mrs','Mlle':'Mss','Miss':'Mss', 'Don':'Mr','Donna':'Mrs',

                           'Lady':'Mrs','Sir':'Mr'}

    title_list = {'Mr','Mrs','Mss','Dr','Master','Rev'}

    df['Title'].replace(title_list,inplace=True)

    df['Title'] = df['Title'].apply(lambda x:'Other' if (not any(x in title for title in title_list)) else x )

    return df

    

def age_fill(df):

    ## Since we will fill age based on title, we make sure title is called

    ## Must be careful not to use information from test set so we will call a clean train set

    ####

    ## FOR TITANIC TRAIN + TEST MAKES THE POPULATION SO WE HAVE ACCESS TO POPULATION MEANS

    ## WITHOUT THE RISK OF DATA LEAKING

    ####

    

    df_train = pd.read_csv(f"{BASE_PATH}train.csv",index_col='PassengerId')

    df_test = pd.read_csv(f"{BASE_PATH}test.csv",index_col='PassengerId')

    df_total = pd.concat([df_train, df_test]).copy()

    df_total = title_extract(df_total)

    

    df = title_extract(df)

    mean_age_by_title = np.round(df_total.groupby('Title').Age.mean()).astype('int64')

    df['Age'] = df.groupby('Title')['Age'].apply(lambda x: x.fillna(mean_age_by_title[x.name]))

    return df



def fare_fill(df):

    ## we will fill are with avg cost by class, again carefull not to leak data

    ####

    ## FOR TITANIC TRAIN + TEST MAKES THE POPULATION SO WE HAVE ACCESS TO POPULATION MEANS

    ## WITHOUT THE RISK OF DATA LEAKING

    ####

    

    df_train = pd.read_csv(f"{BASE_PATH}train.csv",index_col='PassengerId')

    df_test = pd.read_csv(f"{BASE_PATH}test.csv",index_col='PassengerId')

    df_total = pd.concat([df_train, df_test]).copy()

    

    

    mean_fare_by_class = df_total.groupby('Pclass').Fare.mean()

    df['Fare'] = df.groupby('Pclass')['Fare'].apply(lambda x: x.fillna(mean_fare_by_class[x.name]))

    ## we will replace the 0's with the avg fare of their respective class

    df['is_free'] = np.where(df['Fare']== 0,1,0)

    df['Fare'] = np.where((df.Pclass==1) & (df['Fare']== 0),mean_fare_by_class[1],

                           np.where((df.Pclass==2) & (df['Fare']== 0),mean_fare_by_class[2],

                           np.where((df.Pclass==3) & (df['Fare']== 0),mean_fare_by_class[3],

                           df['Fare'])))

    return df



def embarked_fill(df):

    ## we will fill are with avg cost by class, again carefull not to leak data

    ####

    ## FOR TITANIC TRAIN + TEST MAKES THE POPULATION SO WE HAVE ACCESS TO POPULATION MEANS

    ## WITHOUT THE RISK OF DATA LEAKING

    ####

    

    df_train = pd.read_csv(f"{BASE_PATH}train.csv",index_col='PassengerId')

    df_test = pd.read_csv(f"{BASE_PATH}test.csv",index_col='PassengerId')

    df_total = pd.concat([df_train, df_test]).copy()    

    



    mode_embarked = df_total['Embarked'].mode()

    df['Embarked'] = df['Embarked'].fillna(mode_embarked[0])

    return df



def totally_cleaned(df):

    df_copy = df.copy()

    df_copy = cabin_location(df)

    df_copy = title_extract(df)

    df_copy = age_fill(df)

    df_copy = fare_fill(df)

    df_copy = embarked_fill(df)

    return df_copy

    

    

 
def is_female(df):

    df['is_female']=df['Sex'].apply(lambda x: 1 if x=='female' else 0)

    return df



def traveling_party(df):

    df['traveling_party']=df['Parch'] + df['SibSp'] + 1 #Everyone travels with at least themself

    df['is_mother'] = np.where(( (df['Sex']=='female') & (df['Parch'] > 0) & (df['Age'] > 30.)),1,0)

    df['is_wife'] = np.where(( (df['Sex']=='female') & (df['Title'] == 'Mrs') & (df['Age'] > 30.)),1,0)

    df['is_alone'] = np.where(df['traveling_party'] == 1,1,0)

    return df   



def bin_data(df,columns,nbins,log=False):

    frames = []

    for cut_col in columns:

        if (log):

            tmp = pd.DataFrame(pd.cut(np.log1p(feat[cut_col]), bins=nbins, labels=np.arange(0,nbins)))

        else:

            tmp = pd.DataFrame(pd.cut(feat[cut_col], bins=nbins, labels=np.arange(0,nbins)))

        tmp.columns = [cut_col + "_binned"]

        frames.append(tmp)

        binned = pd.concat(frames, axis=1).astype(int)

        df = pd.concat([df, binned], axis=1)

    return df



def ticket_pre(df):

    ticket_prefix = []

    for i in list(df.Ticket):

        if not i.isdigit() :

            ticket_prefix.append((i.replace(".","").replace("/","").strip().split(' ')[0]).upper()) #Take prefix

        else:

            ticket_prefix.append("X")    

    df["Ticket_Pre"] = ticket_prefix

    return df

    

    



def feature_engineering(df):

    df = is_female(df)

    df = traveling_party(df)

    df = ticket_pre(df)

    return df

        
def catboost_encode(df,columns):

    import category_encoders as ce

    ## we will fill are with avg cost by class, again carefull not to leak data

    

    ####

    ## FOR TITANIC TRAIN + TEST MAKES THE POPULATION SO WE HAVE ACCESS TO POPULATION MEANS

    ## WITHOUT THE RISK OF DATA LEAKING

    ####

    

    df_train = pd.read_csv(f"{BASE_PATH}train.csv",index_col='PassengerId')

    df_test = pd.read_csv(f"{BASE_PATH}test.csv",index_col='PassengerId')

    df_total = pd.concat([df_train, df_test]).copy()

    

      

    df_total = totally_cleaned(df_total)

    df_total = feature_engineering(df_total)

    cb_enc = ce.CatBoostEncoder(cols=columns)

    count_encoded = cb_enc.fit(df_total[columns],df_total['Survived'])

    df = df.join(cb_enc.transform(df[columns]).add_suffix('_cb'))

    return df

    



def onehot_encode(df,columns):

    #df_numeric = df.select_dtypes(exclude=['object'])

    df_obj = df[columns].copy()



    cols = []

    for c in df_obj:

        dummies = pd.get_dummies(df_obj[c])

        dummies.columns = [c + "_" + str(x) for x in dummies.columns]

        cols.append(dummies)

    df_obj = pd.concat(cols, axis=1)



    df = pd.concat([df, df_obj], axis=1)

    df.reset_index(inplace=True, drop=True)

    return df    

 

    

def get_dummy(df,columns):

    for col in columns:  

        df = pd.get_dummies(df, columns = [col], prefix = col )

    return df



def drop_catagotical(df):

    catagorical_cols = [col for col in df.columns if df[col].dtype == "object"]

    df = df.drop(catagorical_cols,axis=1)

    return df



# Variable corrilation

def show_corrilations(df):

    corr = df.select_dtypes(include="number").corr()

    plt.subplots(figsize=(8,8));

    sns.heatmap(corr, cmap="RdBu", square=True, cbar_kws={"shrink": .7})

    plt.title("Correlation matrix of all numerical features\n")

    plt.tight_layout()

    plt.show()



# Variable corrilation with survived    

def corr_with_survived(df):

    corr = df.select_dtypes(include="number").corr()

    plt.figure(figsize=(8,8));

    corr["Survived"].sort_values(ascending=True)[:-1].plot(kind="barh")

    plt.title("Correlation of numerical features to Survival")

    plt.xlabel("Correlation to Survival")

    plt.tight_layout()

    plt.show()

    
df = pd.read_csv(f"{BASE_PATH}train.csv",index_col='PassengerId')

df_test = pd.read_csv(f"{BASE_PATH}test.csv",index_col='PassengerId')



feat = pd.concat([df, df_test]).copy()
df = totally_cleaned(df)

df = feature_engineering(df)

df = catboost_encode(df,['Title','Cabin_Location'])

df = bin_data(df,['Age'],5,False)

df = bin_data(df,['Fare'],10,True)

df = onehot_encode(df,['Embarked'])

df = get_dummy(df,['Ticket_Pre'])

#df.info(verbose=True, null_counts=True)
corr_with_survived(df)

show_corrilations(df)
df = pd.read_csv(f"{BASE_PATH}train.csv",index_col='PassengerId')

df_test = pd.read_csv(f"{BASE_PATH}test.csv",index_col='PassengerId')



feat = totally_cleaned(feat)

feat = feature_engineering(feat)

feat = catboost_encode(feat,['Title','Cabin_Location'])

feat = bin_data(feat,['Age'],5,False)

feat = bin_data(feat,['Fare'],10,True)

feat = onehot_encode(feat,['Embarked'])

feat = get_dummy(feat,['Ticket_Pre'])

feat = drop_catagotical(feat)



dtrain = feat[feat.Survived.notnull()].copy()

dtest  = feat[feat.Survived.isnull()].copy()

dtest = dtest.drop(['Survived'], axis=1)

dtest.index = df_test.index

print(f"Raw data shape   : {df.shape}  {df_test.shape}")

print(f"Clean data shape : {dtrain.shape} {dtest.shape}")
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold

X = dtrain[dtrain['Fare']<300].drop(["Survived"], axis=1)

y = dtrain[dtrain['Fare']<300].Survived



metric = 'accuracy'

rf = RandomForestClassifier()

kfold = KFold(n_splits=10, shuffle=True, random_state=1)



print(f"{cross_val_score(rf, X, y, cv=kfold, scoring=metric).mean()*100:.4f} % Accuracy")







from sklearn.linear_model import RidgeClassifier,Perceptron,SGDClassifier,LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,VotingClassifier,BaggingClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

import lightgbm as lgb



metric = 'accuracy'

#Cross Validate Scores

kfold = KFold(n_splits=10, shuffle=True)



#set up the classifiers

random_state = 1  

classifiers = [

               GaussianProcessClassifier(random_state = random_state),  

               Perceptron(random_state = random_state),

               RidgeClassifier(random_state = random_state),

               SGDClassifier(random_state = random_state),

               SVC(random_state = random_state),

               RandomForestClassifier(random_state = random_state),

               GradientBoostingClassifier(random_state = random_state),

               lgb.LGBMClassifier(random_state = random_state),

               xgb.XGBClassifier(objective="reg:squarederror",random_state = random_state),

               xgb.XGBClassifier(objective="reg:squarederror",booster='dart',random_state = random_state),

               AdaBoostClassifier(random_state = random_state),

 #              VotingClassifier(estimators=None),

               BaggingClassifier(random_state = random_state),

               KNeighborsClassifier(),

               LogisticRegression(random_state = random_state) 

]



clf_names = [

            "GaussianProcessClassifier",

            "Perceptron",

            "Ridge",

            "SGDClassifier",

            "SVC",

            "rndmforest", 

            "gbmC", 

            "lgbm", 

            "xgboost",

            "xgboost_dart",

            "AdaBoost",

#            "Voting",

            "Bagging",

            "KNeighbors",

            "Logistic"

]



cv_results = []

for clf in classifiers:

    cv_results.append(cross_val_score(clf, X, y, cv=kfold, scoring=metric,n_jobs=4))







#for clf_name, clf in zip(clf_names, classifiers):

#    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

#    print(f"{clf_name} {cross_val_score(clf, X, y, cv=kfold, scoring=metric).mean()*100:.4f}% Accuracy")

    

    

 





    
cv_mean = []

cv_std = []



for cv_result in cv_results:

    cv_mean.append(cv_result.mean())

    cv_std.append(cv_result.std())





cv_res = pd.DataFrame({'CrossValMeans':cv_mean,'CrossValStd':cv_std,'Classifiers':clf_names})





cross_val_score_plot = sns.barplot('CrossValMeans','Classifiers',data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

cross_val_score_plot.set_xlabel('Mean Accuracy')

cross_val_score_plot = cross_val_score_plot.set_title("Cross validation scores")

ridge = RidgeClassifier()



ridge_pg = {'alpha':[1,10,100],'tol':[0.001,0.0001,0.00001]}



gs_ridge = GridSearchCV(ridge,param_grid = ridge_pg, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gs_ridge.fit(X,y)



ridge_best = gs_ridge.best_estimator_
lrc = LogisticRegression()



lrc_pg = {'penalty':['l1','l2','elasticnet'],'max_iter':[100,500,1000],'warm_start':[True],'tol':[0.001,0.0001,0.00001],'solver':['saga']

          ,'l1_ratio':[0.5]}



gs_lrc = GridSearchCV(lrc,param_grid = lrc_pg, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gs_lrc.fit(X,y)



lrc_best = gs_lrc.best_estimator_
RFC = RandomForestClassifier()



rf_pg = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}



gsRFC = GridSearchCV(RFC,param_grid = rf_pg, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(X,y)



RFC_best = gsRFC.best_estimator_
XGBC =xgb.XGBClassifier(objective="reg:squarederror",random_state = random_state)





#Avg time to fit is 20 mins therefore to debug we dont fit

if(debug):

    XGBC_pg = {'max_depth':[2]}

else:

    XGBC_pg = {'max_depth':[2,4,100],'learning_rate':[0.0001,0.001,0.005],'n_estimators':[100,250,500,1000],

                 'reg_alpha':[0.00001,0.00005,0.0001,0.0005],'colsample_bytree':[.5,.6,.7,.8]}

gsXGBC = GridSearchCV(XGBC,param_grid = XGBC_pg, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)





gsXGBC.fit(X,y)



XGBC_best = gsXGBC.best_estimator_
XGBDC =xgb.XGBClassifier(objective="reg:squarederror",booster = 'dart',random_state = random_state)



#Avg time to fit is 18 mins

if(debug):

    XGBDC_pg = {'max_depth':[2]}

else:

    XGBDC_pg = {'max_depth':[2,4],'learning_rate':[0.0001,0.001,0.005],'n_estimators':[100,250,500],

                 'reg_alpha':[0.00001,0.00005,0.0001,0.0005],'colsample_bytree':[.5,.7,.8]}



gsXGBDC = GridSearchCV(XGBDC,param_grid = XGBDC_pg, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)





gsXGBDC.fit(X,y)



XGBDC_best = gsXGBDC.best_estimator_
BaggC = BaggingClassifier()



BaggC_pg = {'n_estimators':[5,10,25],'bootstrap':[False,True],'max_features':[.1,.4,.5,.7],

                 'max_samples':[.2,.5,.7,1]}



gsBaggC = GridSearchCV(BaggC,param_grid = BaggC_pg, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)





gsBaggC.fit(X,y)



BaggC_best = gsBaggC.best_estimator_
adac = AdaBoostClassifier()



if(debug):

    adac_pg = {'algorithm':['SAMME']}

else:

    adac_pg = {'algorithm':['SAMME','SAMME.R'],'learning_rate':[.001,.00001,.01,.5,1],'n_estimators':[25,50,100,150,200]}

    

gsADAC =  GridSearchCV(adac,param_grid = adac_pg, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsADAC.fit(X,y)



ADAC_best = gsADAC.best_estimator_
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
g = plot_learning_curve(BaggC_best,'Bagging Learning Curve',X,y,cv=kfold)

g = plot_learning_curve(XGBDC_best,"XGBoost Dart learning curves",X,y,cv=kfold)

g = plot_learning_curve(XGBC_best,"XGBoost Learning curves",X,y,cv=kfold)

g = plot_learning_curve(RFC_best,"Random Forest learning curves",X,y,cv=kfold)

g = plot_learning_curve(lrc_best,"Linear Regression learning curves",X,y,cv=kfold)

g = plot_learning_curve(ridge_best,"Ridge learning curves",X,y,cv=kfold)

g = plot_learning_curve(ADAC_best,"ADAC learning curves",X,y,cv=kfold)
def important_features(names_classifiers):

    ncols = 2

    nrows = int(np.ceil(len(names_classifiers)/ncols))

    nclassifier = 0

    for row in range(nrows):

        for col in range(ncols):

            if(nclassifier == len(names_classifiers)):

                break

            name = names_classifiers[nclassifier][0]

            classifier = names_classifiers[nclassifier][1]

            indices = np.argsort(classifier.feature_importances_)[::-1][:10]

            ## LightGBM does not normalize importance so we renormalie, for normalized classifiers this does not make a difference

            g = sns.barplot(y=X.columns[indices][:10],x = (classifier.feature_importances_/(classifier.feature_importances_.sum()))[indices][:10] , orient='h',ax=axes[row][col])

            g.set_xlabel("Relative importance",fontsize=10)

            g.set_ylabel("Features",fontsize=10)

            g.tick_params(labelsize=8)

            g.set_title(name + " feature importance")

            nclassifier += 1
nrows = 2

ncols = 2

fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(20,20))



names_classifiers = [("XGBoost",XGBC_best),("RandomForest",RFC_best),('ADAC',ADAC_best)]



important_features(names_classifiers)







test_RFC = pd.Series(RFC_best.predict(dtest), name="RFC")

test_lrc = pd.Series(lrc_best.predict(dtest), name="LRC")

test_XGBC = pd.Series(XGBC_best.predict(dtest), name="XGBoost")

test_XGBDC = pd.Series(XGBDC_best.predict(dtest), name="XGBoost_Dart")

test_BaggC = pd.Series(BaggC_best.predict(dtest), name="Bagging")

test_Ridge = pd.Series(ridge_best.predict(dtest), name="Ridge")

test_ADAC = pd.Series(ADAC_best.predict(dtest), name="ADAC")

# Concatenate all classifier results

ensemble_results = pd.concat([test_RFC,test_lrc,test_XGBC,test_XGBDC, test_BaggC,test_Ridge,test_ADAC],axis=1)



# We also need training fits

train_RFC = pd.Series(RFC_best.predict(X), name="RFC")

train_lrc = pd.Series(lrc_best.predict(X), name="LRC")

train_XGBC = pd.Series(XGBC_best.predict(X), name="XGBoost")

train_XGBDC = pd.Series(XGBDC_best.predict(X), name="XGBoost_Dart")

train_BaggC = pd.Series(BaggC_best.predict(X), name="Bagging")

train_Ridge = pd.Series(ridge_best.predict(X), name="Ridge")

train_ADAC = pd.Series(ADAC_best.predict(X), name="ADAC")

# Concatenate all classifier results

ensemble_results_train = pd.concat([train_RFC,train_lrc,train_XGBC,train_XGBDC, train_BaggC,train_Ridge,train_ADAC],axis=1)







g= sns.heatmap(ensemble_results.corr(),annot=True)



lgbmc = lgb.LGBMClassifier()



if(debug):

    lgbmc_pg = {'num_leaves':[30]} 

else:

    lgbmc_pg = {'num_leaves':[30,50,60],'max_depth':[2,3,7,-1],'learning_rate':[.001,.00001,.01,.5],'n_estimators':[100,150,200,500]}



gsLGBMC =  GridSearchCV(lgbmc,param_grid = lgbmc_pg, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



ensemble_results.head(3)

gsLGBMC.fit(ensemble_results_train,y)

LGBMC_best = gsLGBMC.best_estimator_



g = plot_learning_curve(LGBMC_best,"Emsemble LGBM learning curves",ensemble_results_train,y,cv=kfold)



test_Survived = pd.Series(LGBMC_best.predict(ensemble_results), name="Survived")

output = pd.DataFrame({'PassengerId': dtest.index,

                       'Survived': test_Survived.astype('int32')})

#display(output.head(10))

output.to_csv("ensemble_python_voting.csv",index=False)