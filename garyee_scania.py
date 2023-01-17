import numpy as np

import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.base import BaseEstimator, TransformerMixin



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def preprocess(df,eightbit=False):

    #data.replace('na',np.NaN,True)

    if(eightbit==True):

        df = df.apply(round)

        df.replace({-1:0},True)

        return df

    else:

        df['class'].replace('neg',0,True)

        df['class'].replace('pos',1,True)

        return df
eightBit=False

logging=True



if(eightBit):

    X_test = pd.read_csv("../input/aps-failure-at-scania-trucks-data-set/aps_failure_test_set_processed_8bit.csv",na_values=0)

    X_train = pd.read_csv("../input/aps-failure-at-scania-trucks-data-set/aps_failure_training_set_processed_8bit.csv",na_values=0)

else:

    X_test = pd.read_csv("../input/aps-failure-at-scania-trucks-data-set/aps_failure_test_set.csv",na_values='na')

    X_train = pd.read_csv("../input/aps-failure-at-scania-trucks-data-set/aps_failure_training_set.csv",na_values='na')





#X_train=preprocess(X_train,eightBit)

#X_test=preprocess(X_test,eightBit)



X_train['class'] = pd.Categorical(X_train['class']).codes

X_test['class'] = pd.Categorical(X_test['class']).codes



#print(X_train.head())



y_train = X_train['class']            

X_train.drop(['class'], axis=1, inplace=True)



#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, test_size=0.2, stratify=y_train)



y_test = X_test['class']            

X_test.drop(['class'], axis=1, inplace=True)
from sklearn.metrics.scorer import make_scorer

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, roc_auc_score,recall_score,precision_score





def scania_scorer(y_true,y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  

    total_cost = 10*fp + 500*fn

    return total_cost

    

def scania_score_tensor(y_true,y_pred):

    tf.print(y_true)

    tf.print(y_pred)

    mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred)

    print(mat)

    total_cost = 10*mat[0][1] + 500*mat[1][0]

    return total_cost

    

def print_all_scores(y_true, y_pred):

    global logging

    if(logging):

        accuracy = accuracy_score(y_true, y_pred)

        print("Accuracy: %.2f%%" % (accuracy * 100.0))



        score = f1_score(y_true, y_pred, average='binary')

        print('F-Measure: %.3f' % score)



        valid_score = roc_auc_score(y_true, y_pred)

        print(f"Validation AUC score: {valid_score:.4f}")



        conf_mat = confusion_matrix(y_true, y_pred)

        print('Confusion matrix:\n', conf_mat)



        print(scania_scorer(y_true, y_pred))



my_scania_scorer = make_scorer(scania_scorer, greater_is_better=False)



from sklearn.preprocessing import MinMaxScaler



def normalize(df):

    scaler = MinMaxScaler(random_state=1)

    scaler.fit(df)

    return pd.DataFrame(scaler.transform(df), columns=df.columns)



#X_train=normalize(X_train)

    
import collections

import re

#inspired by https://www.kaggle.com/percevalve/scania-dataset-eda-for-histograms



def getHistograms(df):

    stripedNameList=[col_name.split("_")[0] for col_name in df.columns if "_" in col_name]

    histStripedNameList=[item for item, count in collections.Counter(stripedNameList).items() if count > 1]

    histDict={k:[col for col in df.columns if k in col if re.match('^\w{2}_\d{3}$',col)] for k in histStripedNameList }

    return histDict



def addSystemAge(df):

    histDict=getHistograms(df)

    hists=histDict.keys();

    for hist in hists:

        df[f"{hist}_total"] = sum(df[col] for col in histDict[hist])

    df["system_age"] = df[[f"{hist}_total" for hist in hists]].min(axis=1)

    df.drop([f"{hist}_total" for hist in hists],axis=1,inplace=True)



def addAvg(df,replace=False,filterList=None):

    histDict=getHistograms(df)

    hists=histDict.keys();

    if(filterList is not None):

        hists=filterList

    for hist in hists:

        df[f"{hist}_avg"] = 0

        df[f"{hist}_total"] = sum(df[col] for col in histDict[hist])

        for col in histDict[hist]:

            df[f"{col}_density"] = df[col]/df[f"{hist}_total"]

            df.loc[df[f"{hist}_total"] == -10, f"{col}_density"] = -1

            df.loc[df[f"{hist}_total"] == 0, f"{col}_density"] = 0

            df[f"{hist}_avg"] += int(col[3:])*df[col]

        df[f"{hist}_avg"] = df[f"{hist}_avg"]/df.system_age

        df.loc[df[f"{hist}_total"] == 0, f"{hist}_avg"] = 0

        df.loc[df[f"{hist}_total"] == -1, f"{hist}_avg"] = 0



    df.drop([f"{hist}_total" for hist in hists],axis=1,inplace=True)

    df.drop([col for col in df.columns if "density" in col],axis=1,inplace=True)

    if(replace==True):

        deleteHistCols(df,filterList)

    

def deleteHistCols(df,colNames=None):

    columns=df.columns

    if(colNames is not None):

        columns=colNames

    for col in columns:

        if re.match('^\w{2}_\d{3}$',col):

            df.drop(col,axis=1,inplace=True)

            

def addBins(df):

    _, bins_for_total_feature = pd.qcut(df.system_age,3,retbins=True)

    bins_for_total_feature[3] = np.max(df.system_age)

    df["total_cat"] = pd.cut(df.system_age.replace(np.nan,-1),[-10.1] + list(bins_for_total_feature),labels=[-0.10,10.0,20.0, 30.0])

    df.total_cat= pd.to_numeric(df.total_cat)

    

#addSystemAge(X_train)

#addSystemAge(X_test)

#addAvg(X_train)

#addAvg(X_test)

#print(X_train.describe())

#print(X_train.columns)
def display_missing_values_table_chart(df,axis=1):

    df_null_pct = df.isna().mean(axis=axis).round(4) * 100

    df_null_pct.plot(kind='hist')

    plt.show()



def delete_missing_values_table(dataf,dataf2=None,nanThreshold=55,axis=0):

    global logging

    if(axis==1):

        indexListPosClass=dataf2.index[dataf2==1].tolist()

        rowsByMissingValue=dataf.isnull().sum(axis=1)

        rowsByMissingValue=rowsByMissingValue.drop(indexListPosClass)

        rowsByMissingValue=rowsByMissingValue.index[rowsByMissingValue>60].tolist()

        #print(rowsByMissingValue.value_counts().sort_index().to_string())

        dataf.drop(rowsByMissingValue,axis=0,inplace=True)

        dataf2.drop(rowsByMissingValue,axis=0,inplace=True)

        dataf.drop(['class'], axis=1, inplace=True)

        

        if(logging):

            #Display Missing Values per row with the nr of rows 

            print(X_train.isnull().sum(axis=1).value_counts().sort_index().to_string())

            print(dataf.shape)

    else:

        cols_with_nan = [cname for cname in dataf.columns if 100 * dataf[cname].isnull().sum()/ len(dataf[cname]) > nanThreshold]

        dataf.drop(cols_with_nan,axis='columns', inplace=True)

        if (len(cols_with_nan)>0 and logging):

            print('Deleted Columns: ',cols_with_nan,'because it/they had more than',nanThreshold,'% of null values')

        if(dataf2 is not None):

            dataf2.drop(cols_with_nan,axis='columns', inplace=True)



if(logging):

    display_missing_values_table_chart(X_train,1)

#delete_missing_values_table(X_train,X_test)
def dropConstantFeatures(df,df2=None,nanThreshold=98):

    global logging

    #constantFeatures=df.std()[(df.std() == 0)].index.to_list()

    constantFeatures = [cname for cname in df.columns if 100 * df[cname].value_counts().iloc[0]/len(df.index) > nanThreshold]

    df.drop(constantFeatures, axis=1, inplace=True)

    if(logging):

        if (len(constantFeatures)>0):

            print('Deleted Columns: ',constantFeatures,'because it/they where constant')

        else:

            print('No constant feature found!')

    if(df2 is not None):

        df2.drop(constantFeatures,axis='columns', inplace=True)

        

#dropConstantFeatures(X_train,X_test)
#did make the result worse the first tine I tried

# https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779

#idea use k-NN

# idea use linear regression for imputation?



from sklearn.impute import SimpleImputer

#from fancyimpute import KNN

#from fancyimpute import IterativeImputer

from sklearn.neighbors import KNeighborsRegressor

import missingno as msno



def display_missingness(df):

    missingdata_df = df.columns[df.isnull().any()].tolist()

    print('Count of Null\'s in the dataframe: ',len(missingdata_df))

    #msno.matrix(df[missingdata_df])

    #msno.heatmap(df[missingdata_df])



def impute(df,algo=None,method='mean'):

    my_imputer= None

#    if(algo=='knn'):

#        my_imputer = KNN()

#    if(algo=='mice'):

#        my_imputer = IterativeImputer(verbose=True,initial_strategy=method,estimator=KNeighborsRegressor(n_neighbors=5,weights='uniform',algorithm='ball_tree'))

#    else:

    my_imputer = SimpleImputer(strategy=method)

    df_imputed = pd.DataFrame(my_imputer.fit_transform(df))

    # Imputation removed column names; put them back

    df_imputed.columns = df.columns

    if(logging):

        print('Count of Null\'s in the dataframe after impution:',df_imputed.isna().sum().sum())

    return df_imputed



display_missingness(X_train)

#X_train=impute(X_train)
from imblearn.combine import SMOTETomek

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler,EditedNearestNeighbours,TomekLinks



def printClassCount(df,comment=''):

    count_classes=pd.value_counts(df,sort=True)

    print(comment,count_classes)



def showBalanced(df,eightBit=False):

    count_classes=pd.value_counts(df,sort=True)

    printClassCount(df)

    if(logging):

        if(eightBit):

            print('1 :',count_classes[-1]/count_classes[1])

        else:

            print('1 :',count_classes[0]/count_classes[1])

    count_classes.plot.bar(rot=0)

    

def resample(X,y,strategy='auto',algo=None,random_state=1):

    my_resampler= None

    if(algo=='SMOTETomek'):

        my_resampler = SMOTETomek(random_state=random_state)

    elif(algo=='Tomek'):

        my_resampler = TomekLinks(random_state=random_state)

    elif(algo=='ENN'):

        my_resampler = EditedNearestNeighbours(random_state=random_state)

    elif(algo=='SMOTE'):

        my_resampler = SMOTE(random_state=random_state,sampling_strategy = strategy)

    elif(algo=='Cluster'):

        my_resampler = ClusterCentroids(random_state=random_state,sampling_strategy = strategy)

    #elif(algo=='RUS'):

    #    my_imputer = 

    else:

        my_resampler = RandomUnderSampler(random_state=random_state,sampling_strategy = strategy,)

    

    X_resampled, y_resampled = my_resampler.fit_resample(X, y)

    X_resampled = pd.DataFrame(X_resampled)

    X_resampled.columns = X.columns

    y_resampled = pd.Series(y_resampled)

    if(logging):

        printClassCount(y_resampled,'resampled by '+ (algo or 'RUS')+'\n')

    return X_resampled,y_resampled

    

showBalanced(y_train,eightBit)

#X_train,y_train = resample(X_train,y_train,strategy=1)

from sklearn.feature_selection import SelectKBest, chi2

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

from sklearn.decomposition import PCA, FastICA



    

def showKmostImportant():

    selector = SelectKBest(score_func=chi2,k='all')

    fit = selector.fit(X_train, y_train)

    dfscores = pd.DataFrame(fit.scores_)

    dfcolumns = pd.DataFrame(X_train.columns)

    featureScores = pd.concat([dfcolumns,dfscores],axis=1)

    featureScores.columns = ['Name','Score'] 

    

    ax=featureScores.plot(kind='hist')

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    plt.xticks(rotation=45)

    plt.show()

    

def doSelection(X,y,X2,take=84,random_state=1):

    global logging

    my_selector= SelectKBest(score_func=chi2)

    

    fit = my_selector.fit(X, y)

    dfscores = pd.DataFrame(fit.scores_)

    dfscores.set_index(X.columns,inplace=True)

    dfscores.sort_values(by=0,ascending=False,inplace=True);

    #dfscores[0].le()

    

    #keeptFeaturesList=dfscores[dfscores>=5000000000].dropna().index

    keeptFeaturesList=X.iloc[:,:take]

    X_reduced=X.filter(keeptFeaturesList)

    if(logging):

        print(X_reduced.shape)

    if(X2 is not None):

        X2_reduced=X2.filter(keeptFeaturesList)

        return X_reduced,X2_reduced

    return X_reduced



def doPCA(df,df2=None):

    global logging

    pca = PCA(84)

    pca.fit(df)

    df_reduced = pca.transform(df)

    df_reduced = pd.DataFrame(df_reduced)

    if(logging):

        print(df_reduced.shape)

    if(df2 is not None):

        df2_reduced=pd.DataFrame(pca.transform(np.nan_to_num(df2)))

        return df_reduced,df2_reduced

    return df_reduced



def selectCorrelated(df,df2=None):

    corr_matrix = X_train.corr(method = "spearman").abs()

    # Select upper triangle of matrix

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95

    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    df.drop(to_drop, axis = 1, inplace= True)

    if(df2 is not None):

        df2.drop(to_drop, axis = 1, inplace= True)



def selectFeatures(X,X2,algo=None,y=None):

    if(algo=='corr'):

        selectCorrelated(X,X2)

    elif(algo=='pca'):

        doPCA(X,X2)

    else:

        doSelection(X,y,X2)

        

#selectCorrelated(X_train,X_test)

#X_train,X_test=doPCA(X_train,X_test)

#showKmostImportant()

#X_train,X_test=doSelection(X_train,y_train,X_test,84)

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from xgboost.sklearn import XGBClassifier

from sklearn import svm

from sklearn.model_selection import cross_val_score, GridSearchCV

from time import time





seed = 1

import random

random.seed(seed)

np.random.seed(seed)

import tensorflow as tf

tf.random.set_seed(seed)

import os

os.environ['PYTHONHASHSEED'] = str(seed)



def makeNnModel():

    global X_train

    print(X_train.shape[1])

    model = Sequential()

    model.add(Dense(X_train.shape[1]*5, input_dim=X_train.shape[1], activation='relu'))

    #model.add(Dense(30, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # Compile model

    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives()])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[scania_score_tensor])

    

    return model

        

        

def selectAndFitModel(X,y,model_type=None):

    print(model_type)

    myModel=None

    if(model_type=='svm'):

        params = [{'kernel': ['rbf'], 'gamma': [0.01], 'C': [0.001, 0.01, 0.1, 1, 10]}]

        grid_search = GridSearchCV(svm.SVC(C=1), params, cv=2, scoring=my_scania_scorer, verbose=10, n_jobs=-1)

        start = time()

        grid_search.fit(X, y)

        print(grid_search.best_params_)

        myModel = svm.SVC(**grid_search.best_params_,probability=True)

    elif(model_type=='nn'):

        myModel=KerasClassifier(makeNnModel, epochs=500, batch_size=50, verbose=1)

    elif(model_type=='randomForest'):

        myModel = RandomForestClassifier(criterion='entropy', max_depth=16, max_features=37, n_estimators=295,random_state=1)

    elif(model_type=='randomForestReg'):

        myModel = RandomForestRegressor(n_estimators=250, random_state=1, n_jobs=-1)

    else:

        myModel = XGBClassifier(learning_rate = 0.05, n_estimators=200, max_depth=4, random_state=1)

    myModel.fit(X, y)

    return myModel
from sklearn.feature_selection import SelectFromModel

from numpy import sort



def predict(model,X_train,X_test,y_train,y_test,regression=False):

    global logging

    if(logging):

        print(X_train.shape)

    y_pred = model.predict(X_test)

    if(regression):

        y_pred = np.round(y_pred)

    print_all_scores(y_test, y_pred)

    score=scania_scorer(y_test, y_pred)

    return score



#model=predict(X_train,X_test)
def trainWithFSelect(model,threshold,X_train,y_train,X_test,y_test):

    global logging



    

    selection = SelectFromModel(model, threshold=threshold, prefit=True)

    select_X_train = selection.transform(X_train)

    select_X_test = selection.transform(X_test)

    

    selection_model = model

    selection_model.fit(select_X_train, y_train)

    # eval model

    

    #select_X_test = selection.transform(np.nan_to_num(X_test,True))

    #select_X_test = X_test.loc[:,selection.get_support()].to_numpy()

    #select_X_test.columns=select_X_train.columns

    

    y_pred = selection_model.predict(select_X_test)

    #print(select_X_test.shape)

    predictions = [round(value) for value in y_pred]

    

    score=scania_scorer(y_test, y_pred)

    if(logging):

        print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

        print(">>>",score,"<<<")

    

    return selection_model,select_X_test,score

    #print_all_scores(y_test, y_pred)

#    print("Thresh=%.7f, n=%d, Score: %d" % (threshold, select_X_train.shape[1], scania_scorer(y_test, predictions)))

    

    

def showFeatureSelectionWithModelSelect(model,X_train,y_train,X_test,y_test):

    thresholds = sort(model.feature_importances_)

    for thresh in thresholds:

        trainWithFSelect(model,thresh,X_train,y_train,X_test,y_test)





#reduced_X_Model,reduced_X_test = trainWithFSelect(model,0.00141,X_train,y_train,X_test,y_test)

##print(X_train.columns)
from tqdm.notebook import tqdm



#https://www.kaggle.com/amirz79/random-forest-8390-vs-catboost-xgboost

#X&y test

def findBestProbabilties(trainedModel,X,y):

    global logging

    scores = trainedModel.predict_proba(X)[:,1]

    fpr, tpr, thresholds = roc_curve(y, scores)

    min_cost = np.inf

    best_threshold = 0.5

    costs = []

    innerThresholds=tqdm(thresholds)

    if(not logging):

        innerThresholds=thresholds

    for threshold in innerThresholds:

        y_pred_threshold = scores > threshold

        tn, fp, fn, tp = confusion_matrix(y, y_pred_threshold).ravel()

        cost = 10*fp + 500*fn

        costs.append(cost)

        if cost < min_cost:

            min_cost = cost

            best_threshold = threshold

    if(logging):

        print("Best threshold: {:.4f}".format(best_threshold))

        print("Min cost: {:.2f}".format(min_cost))

    else:

        return min_cost
#findBestProbabilties(reduced_X_Model,reduced_X_test)
from hyperopt import fmin, tpe, hp, anneal, Trials



def solve(doNormalize=False,

            histogramBasedFSelect=False,

            deleteMissingValCols=True,

            dropConstant=True,

            doImpute=True,

            doImputeTest=False,

            doResample=True,

            doFselection=False,

            doFselectWithModel=True,

            doThresholdMagic=True,

            model_type=None,

            regression_round_pred=False):

    global X_train,X_test,y_train,y_test

    result=float('inf')

    if(histogramBasedFSelect):

        addSystemAge(X_train)

        addSystemAge(X_test)

        addAvg(X_train)

        addAvg(X_test)

    if(deleteMissingValCols):

        delete_missing_values_table(X_train,X_test)

    if(dropConstant):

        dropConstantFeatures(X_train,X_test)

    if(doNormalize):

        X_train=normalize(X_train)

        X_test=normalize(X_test)

    if(doImpute):

        X_train=impute(X_train)

    if(doImputeTest):

        X_test=impute(X_test)

    else:

        X_test=pd.DataFrame(np.nan_to_num(X_test), columns=X_test.columns)

    if(doResample):

        X_train,y_train = resample(X_train,y_train,strategy=1)

    if(doFselection):

        selectFeatures(X_train,X_test,y=y_train)

    model=selectAndFitModel(X_train,y_train,model_type=model_type)

    result = predict(model,X_train,X_test,y_train,y_test,regression_round_pred)

    if(doFselectWithModel):

        reduced_X_Model,reduced_X_test,result = trainWithFSelect(model,0.00141,X_train,y_train,X_test,y_test)

    if(doThresholdMagic):

        if(doFselectWithModel):

            result =findBestProbabilties(reduced_X_Model,reduced_X_test,y_test)

        else:

            result =findBestProbabilties(model,X_test,y_test)

    return result



doNormalize=False

histogramBasedFSelect=False

deleteMissingValCols=True

dropConstant=True

doImpute=True

doImputeTest=False

doResample=True

doFselection=False

doFselectWithModel=True

doThresholdMagic=True

model_type='xgb'

regression_round_pred=False



logging=True



#X_train.replace('na','-1', inplace=True)

#X_test.replace('na','-1', inplace=True)

#X_train=pd.DataFrame(np.nan_to_num(X_train), columns=X_train.columns)

#X_test=pd.DataFrame(np.nan_to_num(X_test), columns=X_test.columns)

    

solve(doNormalize,

    histogramBasedFSelect,

    deleteMissingValCols,

    dropConstant,

    doImpute,

    doImputeTest,

    doResample,

    doFselection,

    doFselectWithModel,

    doThresholdMagic,

    model_type,

    regression_round_pred) 