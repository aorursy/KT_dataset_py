import numpy as np 
import pandas as pd 
import random
import datetime as dt
from pandas.core import datetools

pd.set_option('precision', 2)
import os
#print(os.listdir("../input/price-volume-data-for-all-us-stocks-etfs/Data/ETFs/"))

import time
from matplotlib import pyplot as pyplot

from sklearn import linear_model
from sklearn import tree
import csv
import os

## Preprocessign libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer

# Cross-Validation and Hyper-Parameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

## Evaluation Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc,roc_auc_score

## Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

## GGPLOT2-like library
from plotnine import *
import seaborn as sns



## INITIALIZE VARIABLES
numeric_columns = ['open','high','low','close','openint'] 
date_columns = [0,8,13,18]
closing_columns = ['date','ticker','close','vix_close','vxn_close','ndxt_close','delta_close_etf','delta_close_vix','delta_close_vxn','delta_close_ndxt']
X_Cols = ['ticker',  ### Only need delta and daily change because I will only be looking into Momentum (daily delta Diff) and Volatility (daily % delta change)
          'delta_close_etf','dailychange_close_etf',
          'delta_close_vix','dailychange_close_vix',
          'delta_close_vxn','dailychange_close_vxn',
          'delta_close_ndxt','dailychange_close_ndxt']
## Define Path where files are.
ETFPath = "../input/price-volume-data-for-all-us-stocks-etfs/Data/ETFs/"
filesETF = pd.Series(os.listdir(ETFPath))
#filesETF

## Define ETFs to be picked
## Selected 49 ETFs that are performing really well accoring to: (as of January 2018)
# http://etfdb.com/etfdb-category/technology-equities/%23etfs&sort_name=ytd_percent_return&sort_order=desc&page=1
Chosen_ETF = ['PNQI','ARKK','ROBO','FDN','IPAY','FINX','QQQC','ARKW','IGV','IGM','CQQQ','XTH','FXL','PRNT','RYT','SOCL','XITK','XT','XSW','FTEC','VGT','IXN','SMH','PSJ','QTEC','SKYY','JHMT','GAMR','IYW','PTF','SOXX','XWEB','TCHF','XLK','PSI','FTXL','ITEQ','HACK','TDIV','PXQ','CIBR','XSD','SNSR','PSCT','IGN','FCOM','DTEC','BLOK','BLCN']
Chosen_ETF_files = [x.lower()+'.us.txt' for x in Chosen_ETF]
Chosen_ETF_files = filesETF[filesETF.isin(Chosen_ETF_files)]

## Read Files and Combine them into one large file
start = time.time()
data_ETF = pd.DataFrame([])
filecount = 0
for f in Chosen_ETF_files:
    filecount += 1
    mid = pd.read_csv(ETFPath+f, index_col=False, skiprows = 0,header='infer', parse_dates=[0], infer_datetime_format  = True)
    mid['Ticker'] = f
    data_ETF = data_ETF.append(mid)
print(filecount,"- ETF files have been read and took ",time.time()-start," seconds")
   
## RENAME TICKER FIELD TO A MORE READABLE FORM
data_ETF['Ticker'] = data_ETF['Ticker'].str.replace('.us.txt','')
data_ETF['Ticker'] = data_ETF['Ticker'].str.upper()
data_ETF.columns = data_ETF.columns.str.strip().str.lower().str.replace(' ', '_')

#Backup ETF dataset
data_ETF_Backup = data_ETF

# First Look at the ETF dataset
print('ETF Columns')
print(data_ETF.columns.values)
print('ETF Shape')
print(data_ETF.shape)

data_ETF.head() 

####READ MARKET INDEX FILES
filenameVIX = '../input/vix-index-until-jan-202018/Jan20_vixcurrent_Jan20.csv'
dfVIX = pd.read_csv(filenameVIX,skiprows=1, parse_dates=[0])

filenameVXN = '../input/vxn-index-until-jan-202018/Jan20_vxncurrent_Jan20.csv'
dfVXN = pd.read_csv(filenameVXN,skiprows=2, parse_dates=[0])

filenameNDXT = '../input/ndxt-index-until-jan-202018/Jan20_NDXT.csv'
dfNDXT = pd.read_csv(filenameNDXT, parse_dates=[0])

# First Look at the Index datasets
print("VIX")
print(dfVIX.shape)
print("VXN")
print(dfVXN.shape)
print("NDXT")
print(dfNDXT.shape)

### Merge all index files with ETF dataframe
print("ETF Data set size", data_ETF.shape)
data_ETF = data_ETF.merge(dfVIX, left_on='date', right_on='Date', how='left')

data_ETF = data_ETF.merge(dfVXN, left_on='date', right_on='Date', how='left')
data_ETF = data_ETF.merge(dfNDXT, left_on='date', right_on='Date', how='left')
data_ETF.rename(columns={'Date':'Date_z'}, inplace=True)
data_ETF.columns = data_ETF.columns.str.strip().str.lower().str.replace(' ', '_')


#########################################################################
## Explore Original Dataset by focusing on Market Indexes available dates
#########################################################################

print("Total Unique Tickers")
print(data_ETF['ticker'].nunique())

print("Total Unique Tickers for Chosen ETFs") #FiltereddfETF['ticker'].value_counts()
TotalTickers = data_ETF['ticker'].loc[data_ETF['ticker'].isin(Chosen_ETF)].nunique()   
print(TotalTickers)
# If you wonder, in my original code I used to read all ETFs from the dataset and then filtered out the Chosen_ETFs. 
# To speed up this Kernel, I have started by reading only those ETFs I am interested in

## Filter full list of ETFs with Chosen ETFs. For this KErnel, they are the same.
FiltereddfETF = data_ETF.loc[data_ETF['ticker'].isin(Chosen_ETF)]


#Filter Dataset only for the ETFs of interest
TotalRowsPerTickers = FiltereddfETF.groupby('ticker')['ticker'].count()

# Obtain minimun date per ticker
TotalChosenRows = TotalRowsPerTickers/TotalRowsPerTickers.sum()*100

MinMarketIndex= np.empty((4), dtype='datetime64[D]')
i=0
for col in FiltereddfETF.iloc[:,[0,8,13,18] ]: ## LOOKS INTO DATE COLUMNS
    MinMarketIndex[i] = FiltereddfETF.loc[:,col].min()
    print("The Columns ", col, " has a minimum date of ",MinMarketIndex[i])
    i +=1

MinMarketIndex = MinMarketIndex.max()
print("The earliest Market Index to start on will be on ",MinMarketIndex)         

####### NDXT'S FIRST DATE IS ON 2006. IT IS THE THE WORST CASE FOR THE DATASET BECAUSE IT IS MISSING MOST DATES
### NEED TO FILTER TEH DATA SET TO INCLUDE ONLY DATES AVAILABLE FOR ALL INDEXES
FiltereddfETF = FiltereddfETF[FiltereddfETF.date>=MinMarketIndex]
print(FiltereddfETF.shape)
        
        

#########################################################################
## Explore  Dataset by focusing on TICKER  Available DATES
#########################################################################

MinTickerIndex = {}

for group, matrix in FiltereddfETF.groupby('ticker'):
    for col in matrix.iloc[:,date_columns]: ## LOOKS INTO DATA COLUMNS OF THE SUBMATRIX
        MinTickerIndex[group] = matrix.loc[:,col].min()
MinTickerIndex =  pd.DataFrame(data = MinTickerIndex, index = ['MinDates',])
# TRanspose Dataframe
MinTickerIndex = MinTickerIndex.T
MinTickerIndex.sort_index(inplace=True)
MinTickerIndex.sort_values('MinDates',inplace=True)

## From here it can be seen the first 13 Tickers have data from 2/22/2006 so these will be selected so training includes the 2008 crisis
Final_Chosen_ETFs =  MinTickerIndex.loc[MinTickerIndex.MinDates == '2006-02-22'].index
print("Chosen ETF due to full dates available:",Final_Chosen_ETFs.values)

## Filter Out other ETFs not on final list
print("Before Filtering:",FiltereddfETF.shape)
df_Final_Chosen_ETFs = FiltereddfETF.loc[FiltereddfETF['ticker'].isin(Final_Chosen_ETFs)]
print("After Filtering:",df_Final_Chosen_ETFs.shape)

## Confirm there are not missing dates or any fields in the dataset
print("Total of NAs in final dataset",df_Final_Chosen_ETFs.isnull().values.ravel().sum())
## Summary of dates for Final Dataset
df_Final_Chosen_ETFs.iloc[:,date_columns].describe()
#########################################################################
## CHECK/IDENTIFY WHICH ARE MISSING OR NON NUMERIC VALUES IN FINAL DATASET ON FOCUSED COLUMNS (CLOSE,vix_close,vXN_close,NDXT_close)
#########################################################################

tmp = ['ticker','date','close','vix_close','vxn_close','ndxt_close']
tmp = df_Final_Chosen_ETFs.loc[:,tmp]
tmp['close'] = pd.to_numeric(tmp['close'], errors='coerce')
tmp['vix_close'] = pd.to_numeric(tmp['vix_close'], errors='coerce')
tmp['vxn_close'] = pd.to_numeric(tmp['vxn_close'], errors='coerce')
tmp['ndxt_close'] = pd.to_numeric(tmp['ndxt_close'], errors='coerce')
print("Total of NAs in final dataset",tmp.isnull().values.ravel().sum())

tmp = tmp[tmp.isnull().any(axis=1)].index.values
AllNAs = tmp
#print(type(AllNAs))
print(AllNAs.shape)

# Add pre and post index/dates of the NULL value to the ALLNAs list
for ind in tmp:
    AllNAs = np.append(AllNAs,ind+1)
    AllNAs = np.append(AllNAs,ind-1)
    AllNAs = np.sort(AllNAs)


print( 'Sample of rows with missign NAs below')
#print(AllNAs)
#df_closed_prices = tmp
#print(type(df_closed_prices))
df_Final_Chosen_ETFs.loc[AllNAs].head(18)

#########################################################################
## FILL NAs VIA FORWARD IMPUTATION - Any recommendations on how to best impute on this type of exercise? (stock investing)
#########################################################################
print("Total of NAs before filling NAs ",df_Final_Chosen_ETFs[['ticker','date','close','vix_close','vxn_close','ndxt_close']].isnull().values.sum())

# Imputing Group by Group to ensure my imputation does not take values from other tickers by mistake
for group, matrix in df_Final_Chosen_ETFs.groupby('ticker'):
    tmp = matrix.index.values
    pd.to_numeric(df_Final_Chosen_ETFs.loc[tmp,'volume'], errors='coerce').fillna(method='ffill', inplace=True)

    ### Convert CLOSE columns to numeric
    pd.to_numeric(df_Final_Chosen_ETFs['close'], errors='coerce').fillna(method='ffill', inplace=True)
    pd.to_numeric(df_Final_Chosen_ETFs['vix_close'], errors='coerce').fillna(method='ffill', inplace=True)
    pd.to_numeric(df_Final_Chosen_ETFs['vxn_close'], errors='coerce').fillna(method='ffill', inplace=True)
    pd.to_numeric(df_Final_Chosen_ETFs['ndxt_close'], errors='coerce').fillna(method='ffill', inplace=True)

    ### Convert OPEN columns to numeric
    pd.to_numeric(df_Final_Chosen_ETFs['open'], errors='coerce').fillna(method='ffill', inplace=True)
    pd.to_numeric(df_Final_Chosen_ETFs['vix_open'], errors='coerce').fillna(method='ffill', inplace=True)
    pd.to_numeric(df_Final_Chosen_ETFs['vxn_open'], errors='coerce').fillna(method='ffill', inplace=True)
    pd.to_numeric(df_Final_Chosen_ETFs['ndxt_open'], errors='coerce').fillna(method='ffill', inplace=True)

print( df_Final_Chosen_ETFs.loc[AllNAs,['ticker','date','close','vix_close','vxn_close','ndxt_close']].head(10))

print("\nReconfirm the total of NAs in final dataset",df_Final_Chosen_ETFs[['ticker','date','close','vix_close','vxn_close','ndxt_close']].isnull().values.ravel().sum())
    
#########################################################################
## CREATE SHIFT OF COLUMNS TO CALCULATE DELTAS AMONG DAYS
#########################################################################

df_Final_Chosen_ETFs = df_Final_Chosen_ETFs.reindex(columns = df_Final_Chosen_ETFs.columns.tolist() + ['delta_close_etf','delta_open_etf','delta_volume_etf','delta_close_vix','delta_open_vix','delta_close_vxn','delta_open_vxn','delta_close_ndxt','delta_open_ndxt']) 

for group, matrix in df_Final_Chosen_ETFs.groupby('ticker'):
    tmp = matrix.index.values
    df2 = matrix.loc[tmp,:].shift(+1)
     #_etf
    df_Final_Chosen_ETFs.loc[tmp,'delta_close_etf'] = df_Final_Chosen_ETFs.loc[tmp,'close'] - df2['close']
    df_Final_Chosen_ETFs.loc[tmp,'delta_open_etf'] = df_Final_Chosen_ETFs.loc[tmp,'open'] - df2['open']
    df_Final_Chosen_ETFs.loc[tmp,'delta_volume_etf'] = df_Final_Chosen_ETFs.loc[tmp,'volume'] - df2['volume']
    df_Final_Chosen_ETFs.loc[tmp,'dailychange_close_etf'] = df_Final_Chosen_ETFs.loc[tmp,'close'] / df2['close'] - 1

    #_vix
    df_Final_Chosen_ETFs.loc[tmp,'delta_close_vix'] = df_Final_Chosen_ETFs.loc[tmp,'vix_close'] - df2['vix_close']
    df_Final_Chosen_ETFs.loc[tmp,'delta_open_vix'] = df_Final_Chosen_ETFs.loc[tmp,'vix_open'] - df2['vix_open']
    df_Final_Chosen_ETFs.loc[tmp,'dailychange_close_vix'] = df_Final_Chosen_ETFs.loc[tmp,'vix_close'] / df2['vix_close'] - 1

    #_vxn
    df_Final_Chosen_ETFs.loc[tmp,'delta_close_vxn'] = df_Final_Chosen_ETFs.loc[tmp,'vxn_close'] - df2['vxn_close']
    df_Final_Chosen_ETFs.loc[tmp,'delta_open_vxn'] = df_Final_Chosen_ETFs.loc[tmp,'vxn_open'] - df2['vxn_open']
    df_Final_Chosen_ETFs.loc[tmp,'dailychange_close_vxn'] = df_Final_Chosen_ETFs.loc[tmp,'vxn_close'] / df2['vxn_close'] - 1

    #_ndxt
    df_Final_Chosen_ETFs.loc[tmp,'delta_close_ndxt'] = df_Final_Chosen_ETFs.loc[tmp,'ndxt_close'] - df2['ndxt_close']
    df_Final_Chosen_ETFs.loc[tmp,'delta_open_ndxt'] = df_Final_Chosen_ETFs.loc[tmp,'ndxt_open'] - df2['ndxt_open']
    df_Final_Chosen_ETFs.loc[tmp,'dailychange_close_ndxt'] = df_Final_Chosen_ETFs.loc[tmp,'ndxt_close'] / df2['ndxt_close'] - 1

print(df_Final_Chosen_ETFs.shape)
#print(df_Final_Chosen_ETFs.columns.values)
df_Final_Chosen_ETFs.loc[:,closing_columns].describe(include = 'all')

#########################################################################
## STANDARIZE DELTA COLUMN RESULTS - 
#########################################################################
# Standardize data (0 mean, 1 stdev)


closing_columns = ['ticker','close','vix_close','vxn_close','ndxt_close','delta_close_etf','delta_close_vix','delta_close_vxn','delta_close_ndxt']
closing_columns2 = ['Std_'+ s for s in closing_columns[1:]]

dataframe = df_Final_Chosen_ETFs.reindex(columns = df_Final_Chosen_ETFs.columns.tolist() + closing_columns2) 

## SUBSET ONLY FOR THOSE DATES FOR WHICH ALL ETFs, AND INDEXES HAVE DATA AVAILABLE
dataframe = dataframe.loc[df_Final_Chosen_ETFs.date > MinMarketIndex]

print("before ",dataframe.shape)
for group, matrix in dataframe.groupby('ticker'):
    ind = matrix.index.values
    tmp=ind
    scaler = StandardScaler()
    dataframe.loc[ind,closing_columns2] = scaler.fit_transform(dataframe.loc[ind,closing_columns[1:]])

df_Final_Chosen_ETFs_Std = dataframe
del dataframe
df_Final_Chosen_ETFs_Std[X_Cols].describe()


#########################################################################
## GENERATE UP/DOWN/NO CHANGE IN DELTA CLOSE COLUMNS
#########################################################################

def setlabels(x,thresholdMin = 0,thresholdMax = 0):
    if x >= thresholdMax :
        return "Up"
    elif x < thresholdMin :
        return "Down"
    elif (x > thresholdMin) & (x < thresholdMax):
        return "No Change"
        

frq = pd.DataFrame(index = ['Down','No Change','Up','All'])        
        
tmp = df_Final_Chosen_ETFs_Std.columns[25:46]
for col in tmp: ## LOOKS INTO DATA COLUMNS OF THE SUBMATRIX
    df_Final_Chosen_ETFs_Std['Labeled_' + col] = df_Final_Chosen_ETFs_Std[col].apply(lambda x: setlabels(x)) #,thresholdMin = -.125,thresholdMax = .125
    frqtab = pd.crosstab(index=df_Final_Chosen_ETFs_Std['Labeled_' + col],columns=col,margins=True)#
    frq = pd.concat([frq,frqtab.loc[:,col]],axis=1)


frq = frq.T
frq['Up_Perc'] = frq['Up']/frq['All']*100
frq['Dw_Perc'] = frq['Down']/frq['All']*100

print("CLOSE ETF and CLOSE NDXT are correlated")
#print(X_Cols)
frq.loc[X_Cols[1:8:2],:]
## Save the File Final File so it can be used later in the Training/Testing Phase (when in a local PC)
filenameOut = './PreProc_AllETFs_Chosen.csv'
df_Final_Chosen_ETFs_Std.to_csv(path_or_buf = filenameOut,index_label = "index")
df_Final_Chosen_ETFs_Std.loc[:,['date','close']+ X_Cols[1:8:2]].set_index('date').plot(subplots=True, legend=True, figsize=(20,10)) #ax=axes[0,0],

df_Final_Chosen_ETFs_Std.loc[:,['date','close']+ X_Cols[2:9:2]].set_index('date').plot(subplots=True, legend=True, figsize=(20,10))


###### Identify variables/feature with higher chance of UP 

print(frq.columns.values)
pyplot.bar(np.arange(frq.shape[0]),frq['Up_Perc']) #,figsize = (8,2.5
pyplot.xticks(np.arange(frq.shape[0]), frq.index.values, fontsize=6,rotation=45)
pyplot.axhline(y=50,linewidth=4, color='r')
pyplot.rcParams["figure.figsize"] = [20,10]
pyplot.xlabel('Feature', fontsize=18)
pyplot.ylabel('Up (%)', fontsize=16)
pyplot.tick_params(labelsize = 14, rotation  = 90)
pyplot.show()
### EXPLORATORY VISUALIZATION
## Function to Round up values
def myround(x, prec=1, base=.5):
    return round(base * round(float(x)/base),prec)

NumCols = df_Final_Chosen_ETFs_Std.columns
print("HISTOGRAMS OF INDEXES")
print()
tmp = [25,34,38,35,39,36,40,37,41]
for col in df_Final_Chosen_ETFs_Std.columns[tmp]: ## LOOKS INTO DATA COLUMNS OF THE SUBMATRIX
    print("Variable to be Plotted",col.upper())
    fig, ax = pyplot.subplots()    
    maxi = myround(df_Final_Chosen_ETFs_Std[col].max())
    mini = myround(df_Final_Chosen_ETFs_Std[col].min())
    if maxi- mini >= 7:
        bin = np.arange(mini-.5,maxi+.5,.5)
    else:
        bin = np.arange(mini-.5,maxi+.5,.25)
    
    if col.upper().find('DAILYCHANGE')>=0:
        ax.set_xlabel('Daily Return Change (unit)')
        pyplot.xticks(np.arange(-.25,.25,.05), fontsize=9,rotation=45)
        df_Final_Chosen_ETFs_Std[col].hist(bins = np.arange(-.25,.25,.05), figsize = (15,3))
    elif col.upper().find('STD')>=0:
        ax.set_xlabel('Standarized Value')
        df_Final_Chosen_ETFs_Std[col].hist(bins = bin, figsize = (15,3))
    else:
        ax.set_xlabel(col.upper())
        df_Final_Chosen_ETFs_Std[col].hist(bins = bin, figsize = (15,3))
        

    ax.set_ylabel('Number of Samples')
    pyplot.show()
################## DEFINE HELPER FUNCTIONS FOR EVALUATING MODELS AND PERFORMING CLASSIFICATIONS

def EvaluateROC(y_test,y_pred, plot = False):
        tprs = []
        std_auc=[]
        mean_fpr = np.linspace(0, 1, 100)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        #std_auc = np.std(aucs)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        if plot:
            pyplot.rcParams["figure.figsize"] = [7,7]
            pyplot.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC Score (AUC = %0.2f)' % (roc_auc))
            pyplot.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)
            pyplot.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
            pyplot.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
            pyplot.xlim([-0.05, 1.05])
            pyplot.ylim([-0.05, 1.05])
            pyplot.xlabel('False Positive Rate')
            pyplot.ylabel('True Positive Rate')
            pyplot.title('Receiver operating characteristic Direction Prediction')
            pyplot.legend(loc="lower right")
            pyplot.show()
            
        return mean_auc,roc_auc_score(y_test,y_pred)
    
def RunClassificationPredictions(X_train, y_train, X_test, y_test, classifier, Plot=False):
    #print("Do FIT no GridSearch")
    classifier.fit(X_train, y_train)
    #print("Do SCORE no GridSearch")
    score = clf.score(X_test, y_test)
    #print("Do predict no GridSearch")
    y_pred = clf.predict(X_test).ravel()  
    # ROC Evaluation
    #print("Evaluate ROC no GridSearch")
    auc_mean , roc_auc_score_mean = EvaluateROC(y_test,y_pred,plot = Plot)
    return y_pred,score,auc_mean,roc_auc_score_mean

def RunGridSearchClassification(X_train, y_train, X_test, y_test, classifier,TSCV,param, Plot=False):
    #print("Do GridSearch")
    gsearch = GridSearchCV(estimator=classifier, cv=TSCV,param_grid=param,n_jobs=-1)
    #print("Do FIT and GridSearch")
    gsearch.fit(X_train, y_train)
    #print("Do SCORE and GridSearch")
    score = gsearch.score(X_test, y_test)
    #print("Do predict and GridSearch")
    y_pred = gsearch.predict(X_test).ravel()  
   #print("Evaluate ROC for GridSearch")
    auc_mean , roc_auc_score_mean = EvaluateROC(y_test,y_pred,plot = Plot)
    return y_pred,score,auc_mean,roc_auc_score_mean,gsearch.best_params_


### INITIALIZE VARIABLES FOR TRAINING
numeric_columns = ['open','high','low','close','openint']
#date_columns = [1,8,13,18]
closing_columns = ['date','ticker','close','Labeled_delta_close_etf',
                   'vix_close','Labeled_delta_close_vix',
                   'vxn_close','Labeled_delta_close_vxn',
                   'ndxt_close','Labeled_delta_close_ndxt',
                   'dailychange_close_etf','dailychange_close_vix',
                   'dailychange_close_vxn','dailychange_close_ndxt', u'Std_close']

####### TARGET FEATURES FOR TRAINING
X_Cols = ['ticker',  
          'delta_close_etf','dailychange_close_etf',
          'delta_close_vix','dailychange_close_vix',
          'delta_close_vxn','dailychange_close_vxn',
          'delta_close_ndxt','dailychange_close_ndxt']

####### LABEL FEATURE FOR TRAINING
Y_Cols = ['Labeled_delta_close_etf']


####### INITIATE VARIABLES FOR CLASSIFIERS AND GRIDSEARCHCV
randomstate = 1975
names = [
         "SVM",
         "Decision Tree",
         "Random Forest",
         "Gaussian Naive Bayes",
         "Neural Net"
]

classifiers = [
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),     
    GaussianNB(),
    MLPClassifier()
]

parametersets = [
    {'kernel':('linear','rbf', 'sigmoid'),'random_state':[randomstate],'C':[0.0001,.025,1],'gamma':[0.0001,.025,1,10]},
    {'max_depth':[100,10, 5], 'max_features':('log2','sqrt'),'random_state':[randomstate]},
    {'n_estimators':[10],'random_state':[randomstate]},
    {None},
    {'activation':['relu','logistic'],'solver':( 'sgd', 'adam'), 'alpha':[0.0001, 1,10] }, 
]


#############################################################
## Variable Targets -
## m -> Number of Days in the Future for forecasting
## n1 - > Moving Average for Market Index (days)
## n2 - > Moving Average gor Volatility (days)
#############################################################

m = [5,10,20,90,270]    #,5,10,20,90,270
n = [5,10,20,90,270]     # ,270,5,10,20

#############################################################

### In case of using the data set from previous code
PreProcessed_data_ETF = df_Final_Chosen_ETFs_Std

###READ THE FILE(S) THIS MAY USED DURING DEVELOPMENT TIMES TO SAVE TIME RE-PROCESSING OF THE DATA EACH TIME
### In case of reading the file from previous stages, Open Pre-processed Files
#filename = './Dataset/1Day/ETFs/PreProc_AllETFs_Chosen_WO_NoChange.csv'
### PreProcessed_data_ETF = pd.read_csv(filename, parse_dates=['date','date_x','date_y'],index_col='index')

#########################################################################################################
#### RUN PREDICTIONS AND HYPERPARAMETER OPTIMIZATION USING GRIDSEARCHCV AND TIME SERIES CROSS-VALIDATION
#########################################################################################################
########## CALCULATE VOLATILITY, MOMENTUM FOR ETF AND SECTOR INDEXES
########## EACH FEATURE CALCULATED AVERAGING OVER THE PAST N DAYS FOR 5,10,20,90,270

##  0. For each combination of ML Algorithm/Classifier, m, n1 and n2 execute the following:
##    1. Subset the original full dataset by Ticker/ETF 
##    2. Shift by "m" days to the past the target label
##    3. Because the last "m" rows now show as NA, these are removed from the dataset
##    4. Encode Label feature (targe predicted variable)
##    5. Calculate two features(volatility and momentum using Moving Average) to each variable given n1, n2 (N2 is parameter for ETF, and N1 for all other three indexes)
##       5.1 n1 and n2 may be 5,10,20,90,270. This will make NAs mane of the first rows of the data set
##       5.2 First "d" rows are removed from the dataset where d = (max(n1,n2)+1) th date
##    6. Subset dataset columns to only include explanatory features and target label feature
##    7. Transform the data: Apply Log10 (to daily returns) and Standardize all explanatory features 
##    8. Break dataset into TRAIN and TESTING sets with a 70% ratio for Training
##    9. Create Time Series Cross-Validator Object
##    10. Train Model using Cross-Validation and Hyper-Parameter optimization using GridSearchCV 
##       3.1 Random forest and Gausian Naive Bayes don't pass by GridSearchCV. The former because it needs to be out-of-the-cox and the latter because it does not have parametes to be passed to GridSearchCV
##    11. Predict on Testign given current m, n1 and n2 values
##    12. Evaluate accuracy using "accuracy score" and "AUC_ROC" score
##    13. Record on dataframe results from training/testing exercise: ['Ticker','INDEX_N','ETF_N','Forecast','Classifier','Accuracy Score','AUC Trap. Score','AUC_ROC Score', 'TrainingTesting Time', 'Optimized Hyper Parameters']


## Run simulation for one set of m,n1,n2
timestart  = time.time()
print("Local current time :", timestart)
Train_Cols = ['mom_ETF','vol_ETF','mom_vix','vol_vix','mom_vxn','vol_vxn','mom_ndxt','vol_ndxt']
All_Scores = pd.DataFrame(columns = ['Ticker','INDEX_N','ETF_N','Forecast','Classifier','Accuracy Score','AUC Trapezoidal Score','AUC_ROC_Score','Simulation Duration in Secs','Final Estimator'])
# Defines # of folds for TS Cross-Validation
TS_splits = 3
# Defines if ROC charts will be plot or not
ShowPlots = False

# In a local machine this file allows to record results for each iteration
#filenameOut = './Dataset/1Day/ETFs/20180309 TrainingTestingResults.csv'
print('Expected Number of Iterations (13 Tickers each) = ', len(m)*len(n)*len(n)*4,'\n')
ind = 0
Cycletime2 = 0

for i in np.arange(4):#np.arange(5):# THIS IS A WORK AROUND FOR THE ML ALGORITHM TRAINNING CYCLE BELOW. I HAVE REMOVED THE NEURAL NET ALGORITHM BECAUSE EACH ITERATION TAKES TOO LONG ON THIS PLATFORM
    print("\n################\n################ ML Algorithm:",names[i],"\n################")
    for n1 in n: #FOR INDEX
        #All_Scores.to_csv(path_or_buf = filenameOut,index_label = "Iteration")      # Used for saving in local PC
        for n2 in n: #FOR ETF
            for m1 in m:
                Cycletime = time.time()
                # Below PRINT statement is useful ofr one-to-one analysis. I've removed to minimize size of the Kernel
                #print('Start Estimator for 13 Tickers = %s n1 = %d, n2 = %d, m = %d' % (names[i],n1,n2,m1),'Last Training took ',myround(Cycletime2, prec=2, base=.1),'seconds')
                ## Setting up the simulation for one only Ticker so runing time don't over run Kaglles
                for group, matrix in PreProcessed_data_ETF[X_Cols+Y_Cols].loc[PreProcessed_data_ETF['ticker']!='IyW'].groupby('ticker'): #.loc[PreProcessed_data_ETF['ticker']!='IoooM',
                    ind +=1
                    matrix = matrix.copy()

                    ## Shift Target feature for Forecasting
                    matrix[Y_Cols] = matrix[Y_Cols].shift(-m1)  
                    matrix  = matrix[:-m1]

                    # Encode Target feature
                    le = LabelEncoder()
                    matrix[Y_Cols] = le.fit_transform(matrix[Y_Cols].values.ravel())

                    ### Create Features for Momentum and Volatility for each variable
                    matrix['mom_ETF'] = matrix['delta_close_etf'].rolling(window = n2).mean()
                    matrix['vol_ETF'] = matrix['dailychange_close_etf'].rolling(window = n2).mean()
                    matrix['mom_vix'] = matrix['delta_close_vix'].rolling(window = n1).mean()
                    matrix['vol_vix'] = matrix['dailychange_close_vix'].rolling(window = n1).mean()
                    matrix['mom_vxn'] = matrix['delta_close_vxn'].rolling(window = n1).mean()
                    matrix['vol_vxn'] = matrix['dailychange_close_vxn'].rolling(window = n1).mean()
                    matrix['mom_ndxt'] = matrix['delta_close_ndxt'].rolling(window = n1).mean()
                    matrix['vol_ndxt'] = matrix['dailychange_close_ndxt'].rolling(window = n1).mean()

                    ### Extract Explanatory Features and Target Features from raw dataset, remove NA rows after moving average features creation
                    FirstGoodSample = max(n1,n2)+1
                    matrix = matrix[Train_Cols+Y_Cols]
                    matrix = matrix[FirstGoodSample:]
                    X_All = matrix.loc[FirstGoodSample:,Train_Cols].values
                    Y_All = matrix.loc[FirstGoodSample:,Y_Cols].values

                    ### Transform Explanatory Features: Apply Log10 to Daily Returns and Standardize all explanatory features
                    X_All[:,[1,3,5,7]] = np.log10(X_All[:,[1,3,5,7]]+1)
                    scaler = StandardScaler()
                    X_All = scaler.fit_transform(X_All)

                    ### SEPARATE TRAININIG AND TESTING SETS 70% training, 30% Testing
                    train = int(len(X_All)*.7)
                    X_train = X_All[:train]
                    y_train = Y_All[:train].ravel()
                    X_test = X_All[train:]
                    y_test = Y_All[train:].ravel()

                    #Create TS Cross-Validator
                    my_cv = TimeSeriesSplit(n_splits = TS_splits).split(X_train)

                    #print('Start Training and Testing for ETF: ',group)

                    # Training Process including Cross-Validation and Hyper-Parameter optimization using GridSearchCV
                    for name,clf,GS_Param in list(zip(names,classifiers,parametersets))[i:i+1]:
                        ## This FOR Loop is failing so a work around was created as the first FOR loop of the cycle above. It's been kept for others to see and with the hope to fix it on a later day
                        ## I have opened a forum question for it: https://discussions.udacity.com/t/iterating-multiple-estimators-with-gridsearchcv-valueerror-need-more-than-0-values-to-unpack/617716
                        ##http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

                        # Localtime1 will be used to time length of time for each training/testing iteration
                        localtime1 = time.time()

                        # Run Classification and Testing procedure; it returns predicted Y values for testing set, and 
                        #print(matrix['ticker'].unique())
                        if name in ['Random Forest','Gaussian Naive Bayes']:
                            y_pred, score, auc_mean, roc_auc_score_mean = RunClassificationPredictions(X_train, y_train, X_test, y_test, clf, Plot = ShowPlots)
                            estim = clf.get_params()
                        else:
                            y_pred, score, auc_mean, roc_auc_score_mean,estim = RunGridSearchClassification(X_train, y_train, X_test, y_test, clf,my_cv,GS_Param, Plot = ShowPlots)

                        SimTime = time.time() - localtime1

                        # Record Results in Output Sumamry dataframe
                        All_Scores.loc[len(All_Scores)+1] = [group,n1,n2,m1,name,score,auc_mean,roc_auc_score_mean,SimTime,estim]   #['Ticker','INDEX_N','ETF_N','Forecast','Classifier','Accuracy Score','AUC Trap. Score','AUC_ROC Score', 'TrainingTesting Time', 'Optimized Hyper Parameters']]
                Cycletime2 = time.time() - Cycletime
######
timeend  = time.time()
print("Local current time :", timeend)
print("Total duration in Secs :", timeend  - timestart)



## Select the best Forecast timeframe coming from Traning/Testing. Details on my GitHub repository (https://github.com/lambertopisani/Udacity_ML_Projects/tree/master/6_Capstone).
Forecast_df = All_Scores[All_Scores.Forecast == 90]

ggplot(Forecast_df,aes(x = 'Classifier', y='AUC_ROC_Score', fill = 'Classifier')) + geom_boxplot() + facet_grid('INDEX_N~ETF_N') + theme_bw()+\
theme(strip_background=element_rect(color='blue', fill='blue', size=2),strip_text = element_text(size=9,color="white", face="bold")) +\
theme(legend_position = "right")  +\
theme(axis_text_x = element_text(size=8,face="bold",angle = 90, hjust = 0.5, vjust=1)) +\
ggtitle('AUC_ROC for Index Momentum(days) vs\nETF Momentum(days)\n @ Forecast of 90 days\n')
#ggplot(aes(x = 'Classifier', y='AUC_ROC_Score', fill = 'Classifier' ),data = Forecast_df) + geom_bar(position = position_dodge(width = 0.9)) 
All_Scores_Melted = pd.melt(All_Scores, id_vars=['Ticker', 'INDEX_N', 'ETF_N', 'Forecast', 'Classifier', 'Final Estimator'], value_vars=['Accuracy Score', 'AUC Trapezoidal Score', 'AUC_ROC_Score','Simulation Duration in Secs'], var_name='Metric', value_name='value', col_level=None)
All_Scores_Melted

ggplot(All_Scores_Melted[All_Scores_Melted.Metric != 'Simulation Duration in Secs'],aes(x = 'Metric', y='value', fill ='Metric')) + geom_boxplot() + facet_grid('Forecast ~ Classifier') + theme_bw()+\
theme(strip_background=element_rect(color='blue', fill='blue', size=2),strip_text = element_text(size=9,color="white", face="bold")) +\
theme(legend_position = "right")  +\
theme(axis_text_x = element_text(size=8,face="bold",angle = 90, hjust = 0.5, vjust=1)) +\
ggtitle('ML ALgorithm vs Forecast Horizon')
ggplot(All_Scores_Melted[All_Scores_Melted.Metric != 'Simulation Duration in Secs'],aes(x = 'Metric', y='value', fill ='Metric')) + geom_boxplot() + theme_bw()+\
theme(strip_background=element_rect(color='blue', fill='blue', size=2),strip_text = element_text(size=9,color="white", face="bold")) +\
theme(legend_position = "right")  +\
theme(axis_text_x = element_text(size=8,face="bold",angle = 90, hjust = 0.5, vjust=1)) +\
ggtitle('Evaluation Metric Comparison by ML Algorithm')
ggplot(All_Scores_Melted[All_Scores_Melted.Metric != 'Simulation Duration in Secs'],aes(x = 'Metric', y='value', fill ='Metric')) + geom_boxplot() + facet_grid('. ~ Classifier') + theme_bw()+\
theme(strip_background=element_rect(color='blue', fill='blue', size=2),strip_text = element_text(size=9,color="white", face="bold")) +\
theme(legend_position = "right")  +\
theme(axis_text_x = element_text(size=8,face="bold",angle = 90, hjust = 0.5, vjust=1)) +\
ggtitle('Evaluation Metric Comparison by ML Algorithm')
## Save the File Final File so it can be used later in the Training/Testing Phase (when in a local PC)
filenameOut = './All_Scores.csv'
All_Scores.to_csv(path_or_buf = filenameOut,index_label = "index")