

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from pdpbox import pdp, get_dataset, info_plots

from eli5.sklearn import PermutationImportance

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report



import string

from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import FeatureUnion

from collections import Counter

import os

print(os.listdir("../input"))

import warnings

import eli5

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
df_kick = pd.read_csv("../input/ks-projects-201801.csv",parse_dates = ["launched", "deadline"])

df_kick.shape
df_kick.columns
df_kick.info()
df_kick.tail()
print(df_kick.shape)

df_kick = df_kick.dropna()

print(df_kick.shape)

# projects = projects[projects["currency"] == "USD"]

# projects = projects[projects["state"].isin(["failed", "successful"])]

# projects = projects.drop(["backers", "ID", "currency", "country", "pledged", "usd pledged", "usd_pledged_real", "usd_goal_real"], axis = 1)
def syllable_count(word):

    word = word.lower()

    vowels = "aeiouy"

    count = 0

    if word[0] in vowels:

        count += 1

    for index in range(1, len(word)):

        if word[index] in vowels and word[index - 1] not in vowels:

            count += 1

    if word.endswith("e"):

        count -= 1

    if count == 0:

        count += 1

    return count

## feature engineering

df_kick["syllable_count"]   = df_kick["name"].apply(lambda x: syllable_count(x))

df_kick["launched_month"]   = df_kick["launched"].dt.month

df_kick["launched_week"]    = df_kick["launched"].dt.week

df_kick["launched_day"]     = df_kick["launched"].dt.weekday

df_kick['launched_quarter'] = df_kick['launched'].dt.quarter

df_kick['launched_year']    = df_kick['launched'].dt.year

df_kick["is_weekend"]       = df_kick["launched_day"].apply(lambda x: 1 if x > 4 else 0)

df_kick["num_words"]        = df_kick["name"].apply(lambda x: len(x.split()))

df_kick["num_chars"]        = df_kick["name"].apply(lambda x: len(x.replace(" ","")))

df_kick["duration"]         = df_kick["deadline"] - df_kick["launched"]

df_kick["duration"]         = df_kick["duration"].apply(lambda x: int(str(x).split()[0]))

df_kick["state"]            = df_kick["state"].apply(lambda x: 1 if x=="successful" else 0)



#length of name

df_kick['name_len'] = df_kick.name.str.len()



# presence of !

df_kick['name_exclaim'] = (df_kick.name.str[-1] == '!').astype(int)



# presence of !

df_kick['name_question'] = (df_kick.name.str[-1] == '?').astype(int)



# number of words in the name

df_kick['name_words'] = df_kick.name.apply(lambda x: len(str(x).split(' ')))



# if name is uppercase

df_kick['name_is_upper'] = df_kick.name.str.isupper().astype(float)



#additional features from goal, pledge and backers columns

df_kick.loc[:,'goal_reached'] = df_kick['pledged'] / df_kick['goal'] # Pledged amount as a percentage of goal.

#The above field will be used to compute another metric

# In backers column, impute 0 with 1 to prevent undefined division.

df_kick.loc[df_kick['backers'] == 0, 'backers'] = 1 

df_kick.loc[:,'pledge_per_backer'] = df_kick['pledged'] / df_kick['backers'] 

# Pledged amount per backer.

#will create percentile buckets for the goal amount in a category

df_kick['goal_cat_perc'] =  df_kick.groupby(['category'])['goal'].transform(

                     lambda x: pd.qcut(x, [0, .35, .70, 1.0], labels =[1,2,3]))



#will create percentile buckets for the duration in a category

df_kick['duration_cat_perc'] =  df_kick.groupby(['category'])['duration'].transform(

                     lambda x: pd.qcut(x, [0, .35, .70, 1.0], labels =False, duplicates='drop'))
#creating a metric to see number of competitors for a given project in a given quarter

#number of participants in a given category, that launched in the same year and quarter and in the same goal bucket

ks_particpants_qtr=df_kick.groupby(['category','launched_year','launched_quarter','goal_cat_perc']).count()

ks_particpants_qtr=ks_particpants_qtr[['name']]

#since the above table has all group by columns created as index, converting them into columns

ks_particpants_qtr.reset_index(inplace=True)



#creating a metric to see number of competitors for a given project in a given month

#number of participants in a given category, that launched in the same year and month and in the same goal bucket

ks_particpants_mth=df_kick.groupby(['category','launched_year','launched_month','goal_cat_perc']).count()

ks_particpants_mth=ks_particpants_mth[['name']]

#since the above table has all group by columns created as index, converting them into columns

ks_particpants_mth.reset_index(inplace=True)



#creating a metric to see number of competitors for a given project in a given week

#number of participants in a given category, that launched in the same year and week and in the same goal bucket

ks_particpants_wk=df_kick.groupby(['category','launched_year','launched_week','goal_cat_perc']).count()

ks_particpants_wk=ks_particpants_wk[['name']]

#since the above table has all group by columns created as index, converting them into columns

ks_particpants_wk.reset_index(inplace=True)



#renaming columns of the derived table

colmns_qtr=['category', 'launched_year', 'launched_quarter', 'goal_cat_perc', 'participants_qtr']

ks_particpants_qtr.columns=colmns_qtr



colmns_mth=['category', 'launched_year', 'launched_month', 'goal_cat_perc', 'participants_mth']

ks_particpants_mth.columns=colmns_mth



colmns_wk=['category', 'launched_year', 'launched_week', 'goal_cat_perc', 'participants_wk']

ks_particpants_wk.columns=colmns_wk

#creating 2 metrics to get average pledge per backer for a category in a year according to the goal bucket it lies in and the success rate ie average pledged to goal ratio for the category and goal bucket in this year

#using pledge_per_backer (computed earlier) and averaging it by category in a launch year

ks_ppb_goal=pd.DataFrame(df_kick.groupby(['category','launched_year','goal_cat_perc'])['pledge_per_backer','goal_reached'].mean())

#since the above table has all group by columns created as index, converting them into columns

ks_ppb_goal.reset_index(inplace=True)

#renaming column

ks_ppb_goal.columns= ['category','launched_year','goal_cat_perc','avg_ppb_goal','avg_success_rate_goal']



#creating a metric: the success rate ie average pledged to goal ratio for the category in this year

ks_ppb_duration=pd.DataFrame(df_kick.groupby(['category','launched_year','duration_cat_perc'])['goal_reached'].mean())

#since the above table has all group by columns created as index, converting them into columns

ks_ppb_duration.reset_index(inplace=True)

#renaming column

ks_ppb_duration.columns= ['category','launched_year','duration_cat_perc','avg_success_rate_duration']
#creating 2 metrics to get average pledge per backer for a category in a year according to the goal bucket it lies in and the success rate ie average pledged to goal ratio for the category and goal bucket in this year

#using pledge_per_backer (computed earlier) and averaging it by category in a launch year

ks_ppb_goal=pd.DataFrame(df_kick.groupby(['category','launched_year','goal_cat_perc'])['pledge_per_backer','goal_reached'].mean())

#since the above table has all group by columns created as index, converting them into columns

ks_ppb_goal.reset_index(inplace=True)

#renaming column

ks_ppb_goal.columns= ['category','launched_year','goal_cat_perc','avg_ppb_goal','avg_success_rate_goal']



#creating a metric: the success rate ie average pledged to goal ratio for the category in this year

ks_ppb_duration=pd.DataFrame(df_kick.groupby(['category','launched_year','duration_cat_perc'])['goal_reached'].mean())

#since the above table has all group by columns created as index, converting them into columns

ks_ppb_duration.reset_index(inplace=True)

#renaming column

ks_ppb_duration.columns= ['category','launched_year','duration_cat_perc','avg_success_rate_duration']
#merging the particpants column into the base table

df_kick = pd.merge(df_kick, ks_ppb_goal, on = ['category', 'launched_year','goal_cat_perc'], how = 'left')

df_kick = pd.merge(df_kick, ks_ppb_duration, on = ['category', 'launched_year','duration_cat_perc'], how = 'left')
#merging the particpants column into the base table

df_kick = pd.merge(df_kick, ks_particpants_qtr, on = ['category', 'launched_year', 'launched_quarter','goal_cat_perc'], how = 'left')

df_kick = pd.merge(df_kick, ks_particpants_mth, on = ['category', 'launched_year', 'launched_month','goal_cat_perc'], how = 'left')

df_kick = pd.merge(df_kick, ks_particpants_wk, on = ['category', 'launched_year', 'launched_week','goal_cat_perc'], how = 'left')
#creating 2 metrics: mean and median goal amount

median_goal_cat=pd.DataFrame(df_kick.groupby(['category','launched_year','duration_cat_perc'])['goal'].median())

#since the above table has all group by columns created as index, converting them into columns

median_goal_cat.reset_index(inplace=True)

#renaming column

median_goal_cat.columns= ['category','launched_year','duration_cat_perc','median_goal_year']



mean_goal_cat=pd.DataFrame(df_kick.groupby(['category','launched_year','duration_cat_perc'])['goal'].mean())

#since the above table has all group by columns created as index, converting them into columns

mean_goal_cat.reset_index(inplace=True)

#renaming column

mean_goal_cat.columns= ['category','launched_year','duration_cat_perc','mean_goal_year']
#merging the particpants column into the base table

df_kick = pd.merge(df_kick, median_goal_cat, on = ['category', 'launched_year','duration_cat_perc'], how = 'left')

df_kick = pd.merge(df_kick, mean_goal_cat, on = ['category', 'launched_year','duration_cat_perc'], how = 'left')
print(df_kick.shape)

df_kick[:3]
# replacing all 'N,0"' values in the country column with 'NZERO' to avoid discrepancies while one hot encoding

df_kick = df_kick.replace({'country': 'N,0"'}, {'country': 'NZERO'}, regex=True)
list(df_kick)
df_kick.columns
#selecting the needed fields only

#this will lead to the final features list



#creating a list of columns to be dropped

drop_columns= ['ID','name','launched','deadline','usd pledged','usd_pledged_real','pledge_per_backer','goal_reached']

#dropping columns above

kick = df_kick.copy()

df_kick.drop(drop_columns, axis=1, inplace=True)
#these functions will be used on the textual column entries to remove '&','-' or white spaces

def replace_ampersand(val):

    if isinstance(val, str):

        return(val.replace('&', 'and'))

    else:

        return(val)



def replace_hyphen(val):

    if isinstance(val, str):

        return(val.replace('-', '_'))

    else:

        return(val)    

    

def remove_extraspace(val):

        if isinstance(val, str):

            return(val.strip())

        else:

            return(val) 



def replace_space(val):

        if isinstance(val, str):

            return(val.replace(' ', '_'))

        else:

            return(val)
#apply those functions to all cat columns

#this will remove special characters from the character columns.

#Since these fields will be one-hot encoded, the column names so derived should be compatible with the requied format

df_kick['category'] = df_kick['category'].apply(remove_extraspace)

df_kick['category'] = df_kick['category'].apply(replace_ampersand)

df_kick['category'] = df_kick['category'].apply(replace_hyphen)

df_kick['category'] = df_kick['category'].apply(replace_space)



df_kick['main_category'] = df_kick['main_category'].apply(remove_extraspace)

df_kick['main_category'] = df_kick['main_category'].apply(replace_ampersand)

df_kick['main_category'] = df_kick['main_category'].apply(replace_hyphen)

df_kick['main_category'] = df_kick['main_category'].apply(replace_space)
#missing value treatment

# Check for nulls.

df_kick.isnull().sum()
#creating a backup copy of the dataset

df_kick_copy= df_kick.copy()



df_kick_copy[:5]
for c in df_kick.columns:

    #this gives us the list of columns and the respective data types

    col_type = df_kick[c].dtype

    #looking through all categorical columns in the list above

    if col_type == 'object' :

        a=df_kick[c].unique()

        keys= range(a.shape[0])

        #initiating a dictionary

        diction={}

        for idx,val in enumerate(a):

        #looping through to create the dictionary with mappings

            diction[idx] = a[idx]

        #the above step maps integers to the values in the column

        # hence inverting the key-value pairs

        diction = {v: k for k, v in diction.items()}

        print(diction)

        # creating a dictionary for mapping the values to integers

        df_kick_copy[c] = [diction[item] for item in df_kick_copy[c]] 

        # converting data type to 'category'

        df_kick_copy[c] = df_kick_copy[c].astype('category')
# One-Hot encoding to convert categorical columns to numeric

print('start one-hot encoding')



df_kick_ip = pd.get_dummies(df_kick, prefix = [ 'category', 'main_category', 'currency','country'],

                             columns = [ 'category', 'main_category', 'currency','country'])

    

#this will have created 1-0 flag columns (like a sparse matrix)    

print('ADS dummy columns made')
df_kick_ip.columns
#creating 2 arrays: features and response



#features will have all independent variables

features=list(df_kick_ip)

features.remove('state')

#response has the target variable

response= ['state']
#creating a backup copy of the input dataset

df_kick_ip_copy= df_kick_ip.copy()
df_kick_ip_copy.shape
# normalize the data attributes

df_kick_ip_scaled_ftrs = pd.DataFrame(preprocessing.normalize(df_kick_ip[features]))

df_kick_ip_scaled_ftrs.columns=list(df_kick_ip[features])
df_kick_ip_scaled_ftrs[:3]

#kick_projects_ip[features].shape
#creating test and train dependent and independent variables

#Split the data into test and train (30-70: random sampling)

#will be using the scaled dataset to split 

train_ind, test_ind, train_dep, test_dep = train_test_split(df_kick_ip_scaled_ftrs, df_kick_ip[response], test_size=0.3, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(train_ind, train_dep)



acc_knn = round(knn.score(test_ind, test_dep) * 100, 2)

acc_knn
import math
features_count = train_ind.shape[1]



parameters_rf = {'n_estimators':[50], 'max_depth':[20], 'max_features': 

                     [math.floor(np.sqrt(features_count)), math.floor(features_count/3)]}



def random_forest_classifier(features, target):

    """

    To train the random forest classifier with features and target data

    :param features:

    :param target:

    :return: trained random forest classifier

    """

    clf = RandomForestClassifier(n_estimators=50,criterion='gini' ,max_depth=20, max_features=2)

    clf.fit(features, target)

    return clf
trained_model_RF= random_forest_classifier(train_ind[features], train_dep[response])
# Predict the on the train_data

test_ind["Pred_state_RF"] = trained_model_RF.predict(test_ind[features])



# Predict the on the train_data

train_ind["Pred_state_RF"] = trained_model_RF.predict(train_ind[features])



# Predict the on the train_data

df_kick_ip["Pred_state_RF"] = trained_model_RF.predict(df_kick_ip_scaled_ftrs)
# Train and Test Accuracy

print ("Train Accuracy :: ", accuracy_score(train_dep[response], trained_model_RF.predict(train_ind[features])))

print ("Test Accuracy  :: ", accuracy_score(test_dep[response], trained_model_RF.predict(test_ind[features])))

print ("Complete Accuracy  :: ", accuracy_score(df_kick_ip[response], trained_model_RF.predict(df_kick_ip_scaled_ftrs)))

print (" Confusion matrix of complete data is", confusion_matrix(df_kick_ip[response],df_kick_ip["Pred_state_RF"]))
## Feature importances

ftr_imp_rf=zip(features,trained_model_RF.feature_importances_)

for values in ftr_imp_rf:

    print(values)
feature_imp_RF=pd.DataFrame(list(zip(features,trained_model_RF.feature_importances_)))

column_names_RF= ['features','RF_imp']

feature_imp_RF.columns= column_names_RF
feature_imp_RF= feature_imp_RF.sort_values('RF_imp',ascending=False)

feature_imp_RF[:15]
df_kick[df_kick['state']=='successful']['backers'].value_counts()
df_kick['pledged_log'] = np.log(df_kick['usd_pledged_real'] + 1)

df_kick['goal_log'] = np.log(df_kick['usd_goal_real'] + 1)

df_kick['backers_log'] = np.log(df_kick['backers'] + 1)

sns.distplot(df_kick['backers_log'],kde=False)
sns.distplot(df_kick['goal_log'],kde=False)
sns.distplot(df_kick['pledged_log'],kde=False)
df_kick[df_kick['usd pledged']!=df_kick['usd_pledged_real']]

#         , 'usd_goal_real'']]
df_kick.head()
features
kick.columns
kick.state.value_counts()
kick['pledged_log'] = np.log(kick['usd_pledged_real'] + 1)

kick['goal_log'] = np.log(kick['usd_goal_real'] + 1)

kick['backers_log'] = np.log(kick['backers'] + 1)

sns.distplot(kick['backers_log'],kde=False)
sns.distplot(kick['pledged_log'],kde=False)
%matplotlib inline
plt.figure(figsize=(20,10))

failed = kick[kick['state']==0][['pledged_log','backers_log']]

success = kick[kick['state']==1][['pledged_log','backers_log']]



plt.scatter( failed['backers_log'],failed['pledged_log'], color='r', label='failed',alpha=0.3)

plt.scatter( success['backers_log'],success['pledged_log'], color='g', label='successful',alpha=0.3)



plt.xlabel('pledged_log')

plt.ylabel('backers_log')

plt.legend()

plt.show()