from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib
import warnings
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style("whitegrid")
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

"""
This module provides helper methods to carry out linear regression
on flight data found on https://www.kaggle.com/usdot/flight-delays.

These methods are specific to the flight dataset and is not meant to be 
generic functions for other datasets.
"""
def select_kbest_reg(data_frame, target, k):
    """
    Selecting K-Best features regression.  Performs F-Test 
    :param data_frame: A pandas dataFrame with the training data
    :param target: target variable name in DataFrame
    :param k: desired number of features from the data
    :returns feature_scores: scores for each feature in the data as 
    pandas DataFrame
    """
    feat_selector = SelectKBest(f_regression, k=k)
    _ = feat_selector.fit(data_frame.drop(target, axis=1), data_frame[target])
    
    feat_scores = pd.DataFrame()
    feat_scores["F Score"] = feat_selector.scores_
    feat_scores["P Value"] = feat_selector.pvalues_
    feat_scores["Attribute"] = data_frame.drop(target, axis=1).columns
    
    return feat_scores 

def LinearRegressionModelScore(data, featurelist, cond, target, testsize, bintime = True):
    df = EncodeDepartureTimeMonthDayOfWeek(data, cond,
              featurelist, -1, bintime)
    features = np.delete(df.columns.values, 0)
    features_train, features_test, target_train, target_test = train_test_split(df[features],
                df[target], 
                test_size = testsize)
    
    # fit a model
    lm = linear_model.LinearRegression()
    model = lm.fit(features_train, target_train)
    predictions = lm.predict(features_test)
    return model.score(features_test, target_test), len(df.index), len(features_test), predictions, target_test, lm

def PredictedVsActual(data, cond, features, target, testsize, bindata = True):
    score, dflen, testlen, predictions, target_test, lm = LinearRegressionModelScore(data, features, cond, target, testsize, bindata)
    df = pd.DataFrame()
    df['Actual Delay'] = target_test.values
    df['Predicted Delay'] = predictions 
    df['Diff In Delay'] = predictions - target_test.values 
    return df;

def PlotPredictedVsActual(df, ymax):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    ax1 = axes[0]
    ax2 = ax1.twinx()  # set up the 2nd axis

    ax1.set_ylim([0, ymax])
    ax2.set_ylim([0, ymax])

    df['Actual Delay'].hist(ax=ax1)
    df['Predicted Delay'].hist(ax=ax2, color='red', alpha = 0.5)
    sns.distplot(df['Diff In Delay'], ax = axes[1]);
    sns.boxplot(df['Diff In Delay'], ax = axes[2]);

def EncodeDepartureTimeMonthDayOfWeek(data, cond, featureList, sampleSize, binMonthAndTime=True):
    newFeatureList = list(featureList)
    df = data.loc[cond];
    #filterDF = df.groupby(groupattr, group_keys=False).apply(lambda x: x.sample(min(len(x), sampleSize)))
    if (sampleSize > 0):
        maxSize = len(df)
        if sampleSize > maxSize:
            filterDF = df
        else:
            filterDF = df.sample(n=sampleSize)
    else:
        filterDF = df
    if (binMonthAndTime == False):
        return filterDF[newFeatureList]
    one_hot = pd.get_dummies(filterDF['DEPARTURE_TIME_BIN'], prefix='DEPARTURE_TIME_HOUR')
    df = filterDF.join(one_hot)
    newFeatureList.extend(one_hot) 
    
    #one_hot = pd.get_dummies(df['PRCP_BIN'], prefix='PRCP_D')
    #df = df.join(one_hot)
    #newFeatureList.extend(one_hot) 
    
    #one_hot = pd.get_dummies(df['WDF2_BIN'], prefix='WDF2_D')
    #df = df.join(one_hot)
    #newFeatureList.extend(one_hot) 

    #one_hot = pd.get_dummies(df['TAVG_BIN'], prefix='TAVG_D')
    #df = df.join(one_hot)
    #newFeatureList.extend(one_hot) 
    
    one_hot = pd.get_dummies(df['MONTH'], prefix='MONTH')
    df = df.join(one_hot)
    newFeatureList.extend(one_hot) 

    one_hot = pd.get_dummies(df['DAY_OF_WEEK'], prefix='DAY_OF_WEEK')
    df = df.join(one_hot)
    newFeatureList.extend(one_hot) 

    if 'AIRLINE' in df.columns:
        one_hot = pd.get_dummies(df['AIRLINE'], prefix='AIRLINE')
        df = df.join(one_hot)
        newFeatureList.extend(one_hot) 
    
    return df[newFeatureList]
    
def GetKBestFeatureList(data, attribute, cond, featureList, sampleSize, k, filterlist = []):
    newdf = EncodeDepartureTimeMonthDayOfWeek(data, cond, featureList, sampleSize)
    if filterlist:
        newdf=newdf[filterlist];
    return select_kbest_reg(newdf, attribute, k)

def PlotFeatures(featureList):
    attribute_fscore = featureList[['Attribute','F Score']]
    df = attribute_fscore.set_index('Attribute')
    df = df.sort_values('F Score')
    ax  = df.plot.bar(figsize=(14, 6))

def GetFeatureListsDFList(data, attr, cond, sample_size, count, features, k, filterlist = []):
    first = 0
    oldsize = 0
    for x in range(0, count):
        df = GetKBestFeatureList(data, attr, cond,
                  features, sample_size, k, filterlist)
        if (first == 1):
            olddf = pd.merge(olddf, df, on='Attribute', suffixes=('_'+str(oldsize),'_'+str(oldsize+1)))
        else:
            olddf = df
        oldsize = oldsize+1
        first = 1
    return olddf

def PlotFeatureLists(data, count, figwidth= 15, figheight = 6, attribute='F Score'):
    df = data.set_index(['Attribute'])
    fig = plt.figure(figsize=(figwidth,figheight))
    ax = fig.add_subplot(111)
    p = 0
    c = ['red','green','blue','yellow','black','red','green','blue','yellow','black']
    for x in range(0, count):
        df[attribugte+'_'+ str(x+1)].plot(kind='bar', color=c[p], ax=ax, position=p, width=0.25)
        p = p+1
        
    ax.set_ylabel = ('Sample')
    plt.show()
    
def PlotFeatureList(data, figwidth= 15, figheight = 6, attribute='F Score'):
    df = data.set_index(['Attribute'])
    fig = plt.figure(figsize=(figwidth,figheight))
    df = df.sort_values(by=attribute, ascending=False)
    df[attribute].plot(kind='bar', width=0.5)
    plt.show()   

def AnalyzeSampleSize(data, attribute, attr, initcount, increment, condition, features, k, filterlist=[]):
    datasetSize = len(data.loc[condition]);
    sample_size_list=np.arange(initcount+increment,datasetSize,increment)
    a = pd.DataFrame();
    initialdf = GetFeatureListsDFList(data, attr, condition, initcount, 1, features, k, filterlist)
    a[str(initcount)] = initialdf[attribute]
    for sample_size in sample_size_list:
        b = GetFeatureListsDFList(data, attr, condition, sample_size, 1, features, k, filterlist)
        a[str(sample_size)] = b[attribute]
    
    aT= a.transpose()
    aT.columns = initialdf['Attribute'].values
    return aT

def PerformLinearRegression(data, features, conds, condslbl, target, testsize, bin_data = True):
    df = pd.DataFrame(columns=['Condition' , 'Score'])
    popDF = pd.DataFrame(columns=['# of Data' , '# of Test Data'])
    meanSquareDF = pd.DataFrame(columns=['Condition' , 'MSE'])
    for cond, lbl in zip(conds, condslbl):
        score, dflen, testlen, predictions, target_test, lm = LinearRegressionModelScore(data, features, cond, target, testsize, bin_data)
        df = df.append({'Condition': lbl, 'Score': score}, ignore_index=True)
        popDF = popDF.append({'# of Data': dflen, '# of Test Data': testlen}, ignore_index=True)
        meanSquareDF = meanSquareDF.append({'Condition': lbl, 'MSE': mean_squared_error(target_test, predictions)}, ignore_index=True)
        
    return df, popDF, meanSquareDF;

def PerformNumberOfLinearRegression(data, features, count, conds, condslbl, target, testsize, bin_data):
    for x in range(0, count):
        df, popDF, meanSquareDF = PerformLinearRegression(data, features, conds, condslbl, target, testsize, bin_data)
        if x > 0:
            olddf = pd.merge(olddf, df, on='Condition', suffixes=('_'+str(x),'_'+str(x+1)));
            oldfMSEDF = pd.merge(oldfMSEDF, meanSquareDF, on='Condition', suffixes=('_'+str(x),'_'+str(x+1)));
        else:
            olddf = df;
            oldfMSEDF = meanSquareDF;
    MSEaT = oldfMSEDF.transpose()
    MSEaT.columns = oldfMSEDF['Condition'].values
    MSEaT[1:]
    aT = olddf.transpose()
    aT.columns = olddf['Condition'].values
    aT[1:]
    overalldf = pd.DataFrame();
    overalldf['# of Data'] = popDF['# of Data']
    overalldf['# of Test Data'] = popDF['# of Test Data']
    overalldf['MSE Mean'] = MSEaT[1:].mean().values
    overalldf['MSE Std'] = MSEaT[1:].std().values
    overalldf['R-Squared Mean'] = aT[1:].mean().values
    overalldf['R-Squared Std'] = aT[1:].std().values
    overalldf['Condition'] = aT[1:].mean().index
    return overalldf;

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#Read data
flights = pd.read_csv('../input/atlantaairportdata/ATL_')
# Read weather information for atlanta international airport
weather = pd.read_csv('../input/airport-weather/Weather.csv')

# Merge in weather data
mergeddf = pd.merge(flights, weather, on=['MONTH', "DAY"])
# Clean up data.  Fill NA with 0.  Assume 0 delay in cases where information is not entered
mergeddf = mergeddf.fillna(0)

# Let's creat a category for Departure time based on hour (0 to 2400 hr clock)
mergeddf['DEPARTURE_TIME_BIN'] = pd.cut(mergeddf['DEPARTURE_TIME'], bins=np.arange(0,2400, 100), labels=np.arange(23))

# Display distrubtion data of various features in the dataset
mergeddf.describe()
attr="DEPARTURE_DELAY"
# list of features.  Note additional features will be added such as departure time and airlines when
# calling the rh.AnalyzeSampleSize method.  For more information look at regressionhelper.py
features = [attr,'DAY','TAXI_OUT','WHEELS_OFF','DIVERTED','CANCELLED', 
            'AWND', 'PGTM', 'PRCP',
           'PSUN', 'SNOW', 'SNWD', 'TAVG', 'TMAX', 'TMIN', 'TSUN', 'WDF2',
           'WDF5', 'WESD', 'WESF', 'WSF2', 'WSF5']

# Splice the data.  Let's just look at departure delays between 10 and 240 minutes.  
cond = (mergeddf[attr] > 10) & (mergeddf[attr] <= 240)

# Create an array starting from 1000 samples and increment by 1000 sample size
k = 10
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,6))
df = AnalyzeSampleSize(mergeddf, 'F Score', attr, 1000, 1000, cond, features, k)
df.plot.line(legend=False, ax=ax, style='.-', title='F Score at different sample sizes (k = 10)')
plt.show()
k = 10
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,5))
df = AnalyzeSampleSize(mergeddf, 'P Value', attr, 1000, 1000, cond, features, 5)
df.plot.line(legend=False, ax=ax, style='.-', title='P-Value at different sample sizes (k = 10)')
plt.show()

# Let's plot the P-Value again for the smallest 20 P-Value features
k = 10
featuresFilter = [attr, 'DEPARTURE_TIME_HOUR_0', 'PRCP', 'DEPARTURE_TIME_HOUR_1', 'WSF2',
       'WSF5', 'AIRLINE_DL', 'DEPARTURE_TIME_HOUR_8',
       'DEPARTURE_TIME_HOUR_2', 'DEPARTURE_TIME_HOUR_15', 'TMIN',
       'DEPARTURE_TIME_HOUR_9', 'AIRLINE_EV', 'TAVG',
       'DEPARTURE_TIME_HOUR_18', 'MONTH_12', 'DEPARTURE_TIME_HOUR_10',
       'TMAX', 'DAY', 'MONTH_3', 'DEPARTURE_TIME_HOUR_7', 'DAY_OF_WEEK_5',
       'DAY_OF_WEEK_2', 'AWND', 'WDF2', 'MONTH_11', 'AIRLINE_NK', 'WDF5',
       'MONTH_6', 'AIRLINE_F9', 'AIRLINE_OO']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,6))
df = AnalyzeSampleSize(mergeddf, 'P Value', attr, 1000, 1000, cond, features, k, featuresFilter)
df.plot.line(legend=False, ax=ax, style='.-', title='P-Value of smallest 20 P-Value features at different sample sizes (k = 10)')
ax.annotate('stabalizes after 15,000 sample size',
            xy=(15, 0), xycoords='data',
            xytext=(170, 60), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom')
plt.show()
sample_size = 23932
k = 10
df = GetFeatureListsDFList(mergeddf, attr, cond, sample_size, 1, features, k)
df.sort_values(['P Value'], ascending=[1]).head(30)
attr = 'DEPARTURE_DELAY'
cond = (mergeddf[attr] > 10) & (mergeddf[attr] <= 240)

PlotFeatureList(df, 20, 6, 'F Score')
target = 'DEPARTURE_DELAY'
testsize=0.33
features = [target,'DAY','SCHEDULED_DEPARTURE','TAXI_OUT','WHEELS_OFF','DIVERTED','CANCELLED','AWND', 'PGTM', 'PRCP',
       'PSUN', 'SNOW', 'SNWD', 'TAVG', 'TMAX', 'TMIN', 'TSUN', 'WDF2',
       'WDF5', 'WESD', 'WESF', 'WSF2', 'WSF5']
conds = [(mergeddf[target] <= 10),
         (mergeddf[target] > 10) & (mergeddf[target] <= 120),
         (mergeddf[target] > 10) & (mergeddf[target] <= 180),
         (mergeddf[target] > 10) & (mergeddf[target] <= 240),
         (mergeddf[target] > 120),
         (mergeddf[target] > 140),
         (mergeddf[target] > 180),
         (mergeddf[target] > 240)]
condslbl = [target + ' < 10',
           target + ' > 10 and ' + target + ' < 120',
           target + ' > 10 and ' + target + ' < 180',
           target + ' > 10 and ' + target + ' < 240',
           target + ' > 120',
           target + ' > 140',
           target + ' > 180',
           target + ' > 240'
           ]
NUMBER_OF_ITERATIONS = 20
df = PerformNumberOfLinearRegression(mergeddf, features, NUMBER_OF_ITERATIONS, conds, condslbl, target, testsize, True);
pd.options.display.float_format = '{:.6f}'.format
df.sort_values(['R-Squared Mean'], ascending=[0])
target = 'DEPARTURE_DELAY'
cond = (mergeddf[target] > 10) & (mergeddf[target] <= 240)
predictedf = PredictedVsActual(mergeddf, cond, features, target, testsize, True)
PlotPredictedVsActual(predictedf, 14000)
ax = predictedf[['Actual Delay', 'Predicted Delay']].head(50).plot(figsize=(15,6))
target = 'WEATHER_DELAY'
testsize=0.33
features = [target,'DAY','SCHEDULED_DEPARTURE','TAXI_OUT','WHEELS_OFF','DIVERTED','CANCELLED','AWND', 'PGTM', 'PRCP',
            'PSUN', 'SNOW', 'SNWD', 'TAVG', 'TMAX', 'TMIN', 'TSUN', 'WDF2',
       'WDF5', 'WESD', 'WESF', 'WSF2', 'WSF5']
conds = [(mergeddf[target] <= 10),
         (mergeddf[target] > 10) & (mergeddf[target] <= 120),
         (mergeddf[target] > 10) & (mergeddf[target] <= 140),
         (mergeddf[target] > 10) & (mergeddf[target] <= 240),
         (mergeddf[target] > 120),
         (mergeddf[target] > 140),
         (mergeddf[target] > 180),
         (mergeddf[target] > 240)        
        ]
condslbl = [target + ' <= 10',
           target + ' > 10 and ' + target + ' <= 120',
           target + ' > 10 and ' + target + ' <= 180',
           target + ' > 10 and ' + target + ' <= 240',
           target + ' > 120',
           target + ' > 140',
           target + ' > 180',
           target + ' > 240'            
           ]
NUMBER_OF_ITERATIONS = 20
df = PerformNumberOfLinearRegression(mergeddf, features, NUMBER_OF_ITERATIONS, conds, condslbl, target, testsize, True);
df.sort_values(['R-Squared Mean'], ascending=[0])
target = 'WEATHER_DELAY'
cond = (mergeddf[target] > 10) & (mergeddf[target] <= 240)
predictedf = PredictedVsActual(mergeddf, cond, features, target, .3, True)
PlotPredictedVsActual(predictedf, 800)
ax =predictedf[['Actual Delay', 'Predicted Delay']].head(50).plot(figsize=(15,6))
