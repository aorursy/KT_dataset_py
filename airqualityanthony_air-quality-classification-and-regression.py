'''Internet on required'''

## Install and import GAM modelling from pyGAM 

!pip install pygam

from pygam import LinearGAM, s, f

## read Rdata files

!pip install pyreadr

import pyreadr

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib.colors import ListedColormap

##Import train test split for generation of X_train, y_train, X_test, y_test split

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

## Import Linear model from sklearn

from sklearn.linear_model import LinearRegression

##Import KNN model from sklearn

from sklearn.neighbors import KNeighborsClassifier

## Import Random Forest from sklearn

from sklearn.ensemble import RandomForestClassifier

##Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import cross_val_score



''' Data Download Functions importAURN R to python - conversion of rpy2 pyAURN to pyreadr'''

from urllib.request import urlretrieve

from urllib.error import HTTPError

import warnings





def importAURN(site, years):

    site = site.upper()



    # If a single year is passed then convert to a list with a single value

    if type(years) is int:

        years = [years]



    downloaded_data = []

    df = pd.DataFrame()

    errors_raised = False



    for year in years:

        # Generate correct URL and download to a temporary file

        url = f"https://uk-air.defra.gov.uk/openair/R_data/{site}_{year}.RData"



        try:

            filename, headers = urlretrieve(url)



            result = pyreadr.read_r(filename)



            # done! let's see what we got

#             print(result.keys()) # let's check what objects we got

    

            df1 = result[f"{site}_{year}"] # extract the pandas data frame for object df1

        

            df = df.append(df1)

        

        except HTTPError:

            errors_raised = True

            continue

#     df.set_index('date')

    return df





def importMeta():

    url = "http://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData"

    

    filename, headers = urlretrieve(url)



    result = pyreadr.read_r(filename)

    

    meta = result['AURN_metadata']

    

    meta = meta.drop_duplicates(subset=['site_id'])

    

    return meta
# ''' For CSV based data file '''



# # ''' Data Load - from CSV '''

# # df = pd.read_csv("../input/leamington-aurn-air-quality-data/LEAR.csv") 

# # df.head()

# df.columns



# '''Data Clean'''

# ## drop unecessary variables

# # df.drop(['Unnamed: 0'], inplace = True, axis = 1)

# df.drop(['date'], inplace = True, axis = 1)

# # df.drop(['latitude'], inplace = True, axis = 1)

# # df.drop(['longitude'], inplace = True, axis = 1)

# # df.drop(['site.type'], inplace = True, axis = 1)

# df.drop(['site'], inplace = True, axis = 1)

# df.drop(['code'], inplace = True, axis = 1)

# df.head()



# print(df.shape)

# ## Calculate amount of NA data in each column

# print((df.isna().sum())/len(df) * 100)



# ## clean data set dropping the >50 NA columns

# df.drop(['NV10'], inplace = True, axis = 1)

# df.drop(['V10'], inplace = True, axis = 1)

# df.drop(['NV2.5'], inplace = True, axis = 1)

# df.drop(['V2.5'], inplace = True, axis = 1)

# df.drop(['AT10'], inplace = True, axis = 1)

# df.drop(['AP10'], inplace = True, axis = 1)

# df.drop(['AT25'], inplace = True, axis = 1)

# df.drop(['AP25'], inplace = True, axis = 1)

# df.drop(['RAWPM25'], inplace = True, axis = 1)



# ## drop remaining NA rows

# df_clean = df.dropna()



# ## add classification label (0 = NO2 below 40ug/m3, 1 = NO2 above 40ug/m3)

# df_clean['label'] = np.where(df_clean['NO2'] >= 40, 1, 0)
''' Data Load - from importAURN function (live AURN air quality data)'''

##import AURN (site code, year (single value or list))

df = importAURN("LEAR", 2019)

## Import meta data about all AURN stations operating in the UK. 

meta = importMeta()
yearsaurn = list(range(2015,2020))
meta.columns
# remove stations that have been retired. 

meta_clean = meta[meta['date_ended'].isna()]



#create empty site list

sites = []

# populate site list with currently active AURN sites

for i in meta_clean['site_id']:

    sites.append(i)

#create empty dictionary   

aurndict = {}



# for each station run importAURN and filled against dictionary key for station. - takes a while

for x in sites:

    aurndict["{0}".format(x)]= importAURN(x, yearsaurn)
#example of the AURN station dataframe held in a dictionary value based on station cod

aurndict['ABD7']
# in each aurn dataframe set the date as the index

for i in aurndict.keys():

    aurndict[i].set_index('date', inplace = True)
#create dictionary for column lengths

column_lengths = {}

for x in aurndict.keys():

    column_lengths["{0}".format(x)] = len(aurndict[x].columns)

    

# AURN station with minimum pollutant types

min(column_lengths, key=column_lengths.get)
# AURN stations without NO2 measurements

no2 = []

for i in aurndict.keys():

    if "NO2" not in aurndict[i].columns:

        no2.append(i)

        

no2



#remove stations without no2 measurements

for x in no2: 

    del aurndict[x]
# Merge all AURN station dataframes into single dataframe

merged_df = pd.concat(aurndict.values(), axis = 0, join='outer', ignore_index=False)
'''Data Clean'''

merged_df.drop(['site'], inplace = True, axis = 1)

merged_df.drop(['code'], inplace = True, axis = 1)
## print shape of dataframe

print(merged_df.shape)

## Calculate amount of NA data in each column

print((merged_df.isna().sum())/len(merged_df) * 100)
((merged_df.isna().sum())/len(merged_df) * 100) > 70

merged_df.columns
columns_low_na = ['temp','wd','ws','NO2','NO','NOXasNO2','NV10','NV2.5','O3','PM10','PM2.5','V10','V2.5']
## select dataframe columns dropping the >70 NA columns

df_lowna = merged_df[columns_low_na]
print((df_lowna.isna().sum())/len(df_lowna) * 100)
## drop remaining NA rows

df_clean = df_lowna.dropna()
df_clean.shape
df_clean.head()
''' Linear Model '''



## Dataset for linear regression

## features read from columns

features_linear = df_clean.columns

## read features

features_linear

## drop the no2 feature to avoid overfit to target variable, drop nox feature as its also 1:1 relationship to no2.

features_linear = ['temp', 'wd', 'ws','NV10', 'NV2.5', 'O3','PM10', 'PM2.5', 'V10', 'V2.5','NO']

features = []



## loop through adding each variable in order to the model

for i in features_linear:

    features.append(i)

    X_linear = df_clean[features]

    y_linear = df_clean['NO2']



    X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear, y_linear, test_size=0.20, random_state=42)



    lm_model = LinearRegression()



    lm_model.fit(X_train_linear,y_train_linear)



    y_pred_linear = lm_model.predict(X_test_linear)



    print("Accuracy for",features, len(features),":",lm_model.score(X_test_linear, y_test_linear))



for idx, col_name in enumerate(X_train_linear.columns):

    print("The coefficient for {} is {}".format(col_name, lm_model.coef_[idx]))
'''Classification Models Below'''

## add classification label (0 = NO2 below 40ug/m3, 1 = NO2 above 40ug/m3)

df_clean['label'] = np.where(df_clean['NO2'] >= 40, 1, 0)
''' Random Forest '''



features = df_clean.columns

features_rf = ['temp', 'wd', 'ws','NV10', 'NV2.5', 'O3','PM10', 'PM2.5', 'V10', 'V2.5']





y_rf = df_clean['label']

X_rf = df_clean[features_rf]



normalized_X_rf = preprocessing.normalize(X_rf)

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(normalized_X_rf, y_rf, test_size=0.20, random_state=42)



rf = RandomForestClassifier(n_estimators = 1000, n_jobs=-1, random_state=1)



rf.fit(X_train_rf, y_train_rf)



pred_rf = rf.predict(X_test_rf)



print(' random forest Val accuracy : ', accuracy_score(pred_rf, y_test_rf))

print(classification_report(y_test_rf, pred_rf, target_names = ['Not Exceedance', 'Exceedance']))

## features name list for classification

features_knn = df_clean.columns

print(features_knn)
'''KNN Model'''

## features name list for classification

features_knn = df_clean.columns

## Standard list of features from AURN output.

features_knn = ['temp', 'wd', 'ws','NV10', 'NV2.5', 'O3','PM10', 'PM2.5', 'V10', 'V2.5']

features = []

## loop through variables running the model then adding the next variable and rerunning the model. 

for i in features_knn:

    features.append(i)

    ## set X and y. X = data without label, y = label

    X_knn = df_clean[features]

    y_knn = df_clean['label']



    ## normalise the features to scale  0 to 1

    normalized_X = preprocessing.normalize(X_knn)

    # split into train and test data

    X_train, X_test, y_train, y_test = train_test_split(normalized_X, y_knn, test_size=0.20, random_state=42)

    # KNN model creation and fit

    knn = KNeighborsClassifier(n_neighbors=14)

    knn.fit(X_train,y_train)

    y_pred_knn = knn.predict(X_test)



    # Model Accuracy, how often is the classifier correct?

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred_knn))

    print(classification_report(y_test, y_pred_knn, target_names = ['Not Exceedance', 'Exceedance']))
'''Optimising K: Long run time on CPU'''



## creating list of K for KNN

k_list = list(range(1,50,2))

## creating list of cv scores

cv_scores = []



## perform 10-fold cross validation

for k in k_list:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())

    

MSE = [1 - x for x in cv_scores]



plt.figure()

plt.figure(figsize=(15,10))

plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')

plt.xlabel('Number of Neighbors K', fontsize=15)

plt.ylabel('Misclassification Error', fontsize=15)

sns.set_style("whitegrid")

plt.plot(k_list, MSE)



plt.show()



## Print best k

best_k = k_list[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d." % best_k)
df_clean.columns
'''GAM Model: Long run time on CPU'''



## features minus no2 target variable

features_gam = features_linear

features_two = []



for i in features_gam:

    features_two.append(i)

    X_linear_gam = df_clean[features].to_numpy()

    y_linear_gam = df_clean['NO2'].to_numpy()



    X_train_linear_gam, X_test_linear_gam, y_train_linear_gam, y_test_linear_gam = train_test_split(X_linear_gam, y_linear_gam, test_size=0.20, random_state=42)



    gam = LinearGAM().gridsearch(X_train_linear_gam, y_train_linear_gam)



    print(len(features),gam.statistics_['pseudo_r2'])

    print(gam.statistics_['GCV'])

   