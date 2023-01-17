%matplotlib inline



import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import pandas as pd

from scipy import stats



from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn import preprocessing

from sklearn.linear_model import Lasso, ElasticNet, ElasticNetCV

from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, AdaBoostRegressor



from sklearn import metrics

from sklearn.metrics import r2_score



import xgboost as xgb

from xgboost import XGBRegressor



from scipy.stats import boxcox

import time
#Two functions that create dummy variables

#Just needs the DataFrame to be encoded

def dummy(data):

    data = pd.get_dummies(data)

    return data



#All missing values for categorical features will be labelled as "None"

def cat_null_values(dat,feat):

    dat.loc[dat[feat].isnull(), feat] = "None"



#Replace numeric null values by their mean (or zero)

def numeric_null_values(dat,feat):

    dat.loc[dat[feat].isnull(),feat] = np.mean(dat[feat])

    #dat.loc[dat[feat].isnull(),feat] = 0



#Cross validation score for a model for selected features

def cross_val(model,data,features):

    scores = (cross_val_score(model,data[features],data["SalePrice"],cv=5,n_jobs=1)).mean()

    return scores



#Scoring method for this competition

#true = actual "SalePrice", pred = predicted "SalePrice" (using a model)

def RMSE_log(true,pred):

    RMSE = metrics.mean_squared_error(np.log(true),np.log(pred))**0.5

    return RMSE



#Optimise parameters with GridSearchCV

#data = données à utiliser, feat = features, target = SalePrice, params = dict of parameters to pass to GridSearch

#model = (Lasso, Elastic Net, etc), X_valid = data for prediction and submission to Kaggle

def GridSearch(data,feat,target,params,model,X_valid):

    X = data[feat]

    y = data[target]

    mod = model()

    mod_CV = GridSearchCV(estimator=mod, param_grid=params,cv=10,n_jobs=2,scoring='r2')

    mod_CV.fit(X,y)

    print("Best parameters :",mod_CV.best_params_)

    print("Best R2 score :",mod_CV.best_score_)

    pred = mod_CV.predict(X_valid)

    return mod_CV, pred



#data_valid=data for prediction and submission to Kaggle; pred = "SalePrice" predicted by the model

def writePredFile(data_valid, pred, fileName):

    answer = pd.DataFrame()

    answer["Id"] = (data_valid.Id).astype(int)

    answer["SalePrice"] = pred

    answer.to_csv(fileName,index=False)

    

#Constructing an ensemble regressor (simply calculates the mean)

def EnsembleRegressor(regressors,X_Validation):

    EnsemblePred = pd.DataFrame()

    for reg in regressors:

        colname = str(reg)[:4]

        EnsemblePred[colname] = reg.predict(X_Validation)

    EnsemblePred["Ensemble"] = EnsemblePred.apply(lambda x: np.mean(x), axis=1) #Mean scores better than median

    return EnsemblePred



def OptimiseParameters(model,X,y,parameters,n_jobs=1, scoring='r2', error_score=0):

    t_init = time.time()

    modelCV = model()

    CV_model = GridSearchCV(estimator=modelCV,param_grid=parameters,cv=10,n_jobs=n_jobs,scoring=scoring)

    CV_model.fit(X,y)

    t_final = time.time()

    bparams = CV_model.best_params_

    bscore = CV_model.best_score_

    print("Execution time: {}\n".format(t_final-t_init))

    print("Best parameters: {}\n ".format(CV_model.best_params_))

    print("Best R2 score: {}\n".format(CV_model.best_score_))

    print("Best estimator: {}\n".format(CV_model.best_estimator_))

    return CV_model, bscore,bparams  #returns the model, best score and best parameters



def plot_cat(data, x_axis, y_axis, hue):

    plt.figure()    

    sns.barplot(x=x_axis, y=y_axis, hue=hue, data=data)

    sns.set_context("notebook", font_scale=1.6)

    plt.legend(loc="upper right", fontsize="medium") 
data = pd.read_csv("../input/train.csv")

data_validation = pd.read_csv("../input/test.csv")

data_Id = data_validation.Id

sns.set()

cols = ['SalePrice', "LotArea","MSSubClass", "GrLivArea"]

sns.pairplot(data[cols], size = 2.5, kind='scatter', diag_kind="kde", dropna=True,diag_kws=dict(shade=True))

plt.show()
sns.factorplot('KitchenQual', 'SalePrice', estimator = np.mean, 

               size = 4.5, aspect = 1.4, data = data, order = ['Ex', 'Gd', 'TA', 'Fa'])
sns.factorplot('HeatingQC', 'SalePrice', hue = 'CentralAir', estimator = np.mean, data = data, 

             size = 4.5, aspect = 1.4)
sns.distplot(np.log1p(data.SalePrice),fit=stats.norm)
#saleprice correlation matrix

corrmat = data.corr()

k = 11 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(data[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
colnames = (data.select_dtypes(['object'])).columns

for c_ in colnames:

    plot_cat(data,c_,"SalePrice",None)
features_to_delete = ["GarageYrBlt", "MoSold","MasVnrArea"] #We'll keep this list for a later use

for fd in features_to_delete:

    data = data.drop(fd,axis=1)

    data_validation = data_validation.drop(fd,axis=1)
#"AgeTravaux" is french for "Number of years since last Remod" ;)

data["AgeTravaux"] = np.abs(data.YrSold-data.YearRemodAdd)

data_validation["AgeTravaux"] = np.abs(data_validation.YrSold-data_validation.YearRemodAdd)

features_to_remove = set()

features_to_remove.add("YearRemodAdd") #This feature is no longer needed
for dat in [data,data_validation]:

    #Let's fetch the names of all numeric features

    colnames_num = (dat.select_dtypes(exclude=[object])).columns

    #Numerical features

    for c_ in colnames_num:

        numeric_null_values(dat,c_)
#Liste of features to be "boxcox" tranformed (none in this example, but improves score when carefully selected)

boxcox_features = [] #Keep the list of transformed features for later use

for bf in boxcox_features:

    for dat in [data,data_validation]:

        colname = str(bf)

        box = boxcox(dat[colname])

        dat[colname] = box[0]
#List of features to be binned (keep it for later use)

features_bin=["YearBuilt", "AgeTravaux"]

data["YearBuilt"] = pd.cut(data["YearBuilt"],bins=[1872,1940,1960,1980,2000,2010], labels=[1,2,3,4,5]) #Values were taken almost randomly to have ~same number of values in each bin

data_validation["YearBuilt"] = pd.cut(data_validation["YearBuilt"],bins=[1872,1940,1960,1980,2000,2010], labels=[1,2,3,4,5])
#Liste of features that should be squared

squared_features = ["BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","Fireplaces"]

for sf in squared_features:

    for dat in [data,data_validation]:

        colname = str(sf)

        dat[colname] = dat[colname].apply(lambda x: x**2)
Poly = preprocessing.PolynomialFeatures(degree=2)



polynomial_features = ["GrLivArea"]

for pf in polynomial_features:

    for dat in [data,data_validation]:

        x_ = Poly.fit_transform(dat[pf].values.reshape(-1,1))

        for i in np.arange(1,3):

            colname = pf+"_x_%s"%i

            print(colname)

            dat[colname]=pd.Series(x_[:,i])

            features_to_remove.add(colname) #These are features I do not want to log transform later
features_transformed = set(polynomial_features)

features_transformed = features_transformed.union(squared_features,features_to_remove, features_bin, boxcox_features)

features_transformed
#The log should be applied only on numeric features, excluding those already transformed

numeric_features = set((data.select_dtypes(exclude=[object])).columns) #I'll select only numeric features (excluding "objects)



#Taking care not to modify the target or the Id. Not sure if it's a good Idea for "SalePrice", just try and compare scores

numeric_features.remove("SalePrice")

numeric_features.remove("Id")



#Let's keep features that have not been transformed

log_features = numeric_features.difference(features_transformed)

#Let's apply a log transform

for lf in log_features:

    for dat in [data,data_validation]:

        dat[lf]=dat[lf].apply(lambda x:np.log1p(x))
for dat in [data,data_validation]:

    #Names of categorical features

    colnames_cat = (dat.select_dtypes(['object'])).columns

    #For categorical features, I'll replace any null value by "None"

    for c_ in colnames_cat:

        cat_null_values(dat,c_)

#Let's create dummy data for both data sets

data = dummy(data)

data_validation = dummy(data_validation)
features = list(data.columns)

features = [feat for feat in features if feat not in polynomial_features]

#features = [feat for feat in features if feat not in squared_features]

features = [feat for feat in features if feat not in colnames_cat]

len(features)



#Features for validation data (i.e. data used for submission)

features_validation = [feat for feat in features if feat in data_validation.columns]

len(features_validation)



#Features that are not in both data sets

missing_features = list(set(features).difference(features_validation))

missing_features.remove("SalePrice")



features = [feat for feat in features if feat not in missing_features]

features.remove("YearRemodAdd")

features_validation = list(features)

features_validation.remove("SalePrice") #SalePrice is not included in the validation data set
data = data[features]

data_validation = data_validation[features_validation]

#data[features].to_csv("2-Prepared Data/data_prep.csv", index=None)

#data_validation[features_validation].to_csv("2-Prepared Data/data_validation_prep.csv",index=None)
#data = pd.read_csv("2-Prepared Data/data_prep.csv")

#data_validation = pd.read_csv("2-Prepared Data/data_validation_prep.csv")
features = list(data.columns)

features.remove("SalePrice")

target = "SalePrice"

data_Id = data_validation.Id #For later use when writing the final file for submission (see below)
train,test = train_test_split(data,test_size=0.2)

X_train = train[features]

y_train = train[target]

X_test = test[features]

y_test = test[target]
#ELASTIC NET

ElNet_model = ElasticNetCV(l1_ratio=np.arange(0.05,0.95,0.05),alphas=np.arange(1,100,10),cv=5, max_iter=1500)

ElNet_model.fit(X_train,y_train)

ElNet_pred_test = ElNet_model.predict(X_test)

print("***ELASTIC NET REGRESSOR***\nRMSE on test set : ",RMSE_log(y_test,ElNet_pred_test))

print("R2 for test set : ",r2_score(y_test,ElNet_pred_test))

print("Crossval score : {}".format(cross_val(ElNet_model,data,features)))

coef_EN = pd.Series(ElNet_model.coef_, index = X_train.columns)

print("Elastic net has selected {0} features and deleted {1} features\n".format(sum(coef_EN!=0),sum(coef_EN==0)))



#GBOOST REGRESSOR

GBoost_model = GradientBoostingRegressor(loss="ls",

                                         learning_rate=0.09,

                                         n_estimators=500,

                                        max_depth=4,

                                        alpha=0.08)

GBoost_model.fit(X_train,y_train)

GBoost_pred_test = GBoost_model.predict(X_test)

print("***GRADIENT BOOSTING REGRESSOR*** \nRMSE for test set : ",RMSE_log(y_test,GBoost_pred_test))

print("R2 for test set : {}".format(r2_score(y_test,GBoost_pred_test)))

print("Crossval score : {}\n".format(cross_val(GBoost_model,data,features)))





#ADABOOST REGRESSOR

AdaBoost_model = AdaBoostRegressor(n_estimators=100, learning_rate=0.7, loss="exponential")

AdaBoost_model.fit(X_train,y_train)

AdaBoost_pred_test = AdaBoost_model.predict(X_test)

print("***ADABOOST REGRESSOR***\nRMSE for test set : ",RMSE_log(y_test,AdaBoost_pred_test))

print("R2 pour le test set : {}".format(r2_score(y_test,AdaBoost_pred_test)))

print("Crossval score : {}\n".format(cross_val(AdaBoost_model,data,features)))



#XGBoost REGRSSOR

xgb_model = XGBRegressor(n_estimators=1000,max_depth=3,learning_rate=0.05)

xgb_model.fit(X_train,y_train)

xgb_pred_test = xgb_model.predict(X_test)

print("***XGBOOST REGRESSOR*** \nRMSE pour le test set : {}".format(RMSE_log(y_test,xgb_pred_test)))

print("R2 pour le test set : {}".format(r2_score(y_test,xgb_pred_test)))

print("Crossval score : {}\n".format(cross_val(xgb_model,data,features)))



lasso_model = Lasso(alpha=150,max_iter=3000)

lasso_model.fit(X_train,y_train)

lasso_pred = lasso_model.fit(X_test,y_test)

lasso_pred = lasso_model.predict(X_test)

print("***LASSO REGRESSOR***: {}".format(RMSE_log(pred=lasso_pred,true=y_test)))

print("LASSO has eliminated {0} features and kept {1}".format((lasso_model.coef_ == 0).sum(),(lasso_model.coef_ != 0).sum()))

print("Crossval score : {}".format(cross_val(lasso_model,data,features)))

coef = pd.Series(ElNet_model.coef_, index = X_train.columns)

imp_coef = pd.concat([coef.sort_values().head(15),

                     coef.sort_values().tail(15)])



plt.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Elastic Net Model")
coef = pd.Series(lasso_model.coef_, index = X_train.columns)

imp_coef = pd.concat([coef.sort_values().head(15),

                     coef.sort_values().tail(15)])



plt.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the LASSO Model")
#Prédiction on the test set for testing/checking purpose

#In the "EnsembleRegressor" function, simply pass any combination of models you want

predTestSet = EnsembleRegressor([GBoost_model, ElNet_model,lasso_model,xgb_model], X_test)

print("***ENSEMBLE REGRESSOR***\nRMSE for test set : {}".format(RMSE_log(pred=predTestSet.Ensemble,true=y_test)))



#Prédiction finale

Final_answer = EnsembleRegressor([GBoost_model, ElNet_model,lasso_model,xgb_model],data_validation[features])

Final_answer = Final_answer

data_validation["Id"] = data_Id #Yes, this is why I put aside "Id" at the beginning of the Notebook
Final_answer.head()
writePredFile(data_validation,Final_answer["Ensemble"],"filename.csv")
#I have put few parameters here to speed up the procedure but you should definitely use a larger grid

param_grid = { 

    'loss' : ["ls","huber"],

    'learning_rate': np.arange(0.02,0.22,0.1),

    'min_samples_leaf':[1,2],

    'n_estimators':[150],

    'max_depth':[3],

    'alpha':np.arange(0.02,0.12,0.1)

}

CV_model, bscore,bparams = OptimiseParameters(GradientBoostingRegressor,X_train,y_train,parameters=param_grid, n_jobs=2)## Optimisation des hyperparamètres pour Gradient Boosting
# 'contamination' is for controlling the cluster's size : the larger the cluster the 

clf = IsolationForest(contamination=0.05)

#clf.fit(data.select_dtypes(exclude=['object']))

#pred_outlier = clf.predict(data.select_dtypes(exclude=['object']))

X = pd.DataFrame()

X["LotArea"] = data.LotArea

X["SalePrice"] = data.SalePrice

clf.fit(X)

pred_outlier = clf.predict(X)

print("Number of outliers : ",(len(pred_outlier)-pred_outlier.sum())/2)

data["Outlier"] = pred_outlier
plt.scatter(data.LotArea, data.SalePrice, c=data.Outlier)
data = data.query('Outlier==1')

data = data.drop('Outlier', axis=1)
init_features = list(data.columns) #All features

init_features.remove("SalePrice")

alphas = [40,60,80]

GB_score = []

Lasso_score = []

for a in alphas:

    features = list(init_features)

    print("*** FEATURES SELECTION WITH LASSO ***")

    train,test = train_test_split(data,test_size=0.2)

    X_train = train[features]

    y_train = train[target]

    X_test = test[features]

    y_test = test[target]

    lasso_model = Lasso(a, max_iter=3000)

    lasso_model.fit(X_train,y_train)

    lasso_pred = lasso_model.fit(X_test,y_test)

    lasso_pred = lasso_model.predict(X_test)



    #print("Le score RMSE pour le lasso: {}".format(RMSE_log(pred=lasso_pred,true=y_test)))

    print("LASSO has eliminated {0} features and kept {1}\nAlpha={2}".format((lasso_model.coef_ == 0).sum(),(lasso_model.coef_ != 0).sum(),a))

    cscorelasso = cross_val(lasso_model,data,features)

    

    print("Crossval score LASSO: {}\n".format(cscorelasso))

    Lasso_score.append(cscorelasso)

    zero_coef=[]

    for i in range(len(features)):

        #print("{0}  \t Coef : {1}".format(features[i],lasso_model.coef_[i]))

        if lasso_model.coef_[i] != 0:

            zero_coef.append(features[i])

    len(zero_coef) #Ca correspond bien au nombre de features conservées par LASSO

    features = list(zero_coef)

    

    #On fit un modèle XGBOOST sur les différentes valeurs de alpha et avec de nouvelles features

    X_train = train[features]

    y_train = train[target]

    X_test = test[features]

    y_test = test[target]

   

    #GBOOST REGRESSOR

    GBoost_model = GradientBoostingRegressor(loss="ls",

                                         learning_rate=0.1,

                                         n_estimators=300,

                                        max_depth=4,

                                        alpha=0.15)

    GBoost_model.fit(X_train,y_train)

    GBoost_pred_test = GBoost_model.predict(X_test)

    print("***GRADIENT BOOSTING REGRESSOR*** \nRMSE for test set : ",RMSE_log(y_test,GBoost_pred_test))

    print("R2 pour le test set : {}".format(r2_score(y_test,GBoost_pred_test)))

    cscore_GB = cross_val(GBoost_model,data,features)

    print("Crossval score GBoost : {}\n".format(cscore_GB))

    GB_score.append(cscore_GB)



print("Max GBoost: {0}, max Lasso: {1}".format(max(GB_score), max(Lasso_score)))