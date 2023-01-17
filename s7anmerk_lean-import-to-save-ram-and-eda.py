# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 



%matplotlib inline

sns.set()



#import pydot

from IPython.display import Image



import os

print(os.listdir("../input"))



# ignore warnings

import warnings

warnings.filterwarnings("ignore")



# To release RAM import gc collector

import gc

gc.enable()
# Reduce memory function was taken from the kaggle following kernel:

# https://www.kaggle.com/ashishpatel26/lightgbm-gbdt-dart-baysian-ridge-reg-lb-3-61

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
# Read CSV-Fikes



test_data = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv' )

# Print the head of the data 

print(test_data.head())

print('\n\n')

print(sample_submission.head(10))
# delete test data and the submission file for now

del test_data

del sample_submission

gc.collect
# Declaring a dictionary with column name and column type, 

# so that we can save memory while reading the csv-file

# the dictionary is based on the result from the memory reduce function



dtypesDict_tr = {

'id'                            :         'int32',

'target'                        :         'float16',

'severe_toxicity'               :         'float16',

'obscene'                       :         'float16',

'identity_attack'               :         'float16',

'insult'                        :         'float16',

'threat'                        :         'float16',

'asian'                         :         'float16',

'atheist'                       :         'float16',

'bisexual'                      :         'float16',

'black'                         :         'float16',

'buddhist'                      :         'float16',

'christian'                     :         'float16',

'female'                        :         'float16',

'heterosexual'                  :         'float16',

'hindu'                         :         'float16',

'homosexual_gay_or_lesbian'     :         'float16',

'intellectual_or_learning_disability':    'float16',

'jewish'                        :         'float16',

'latino'                        :         'float16',

'male'                          :         'float16',

'muslim'                        :         'float16',

'other_disability'              :         'float16',

'other_gender'                  :         'float16',

'other_race_or_ethnicity'       :         'float16',

'other_religion'                :         'float16',

'other_sexual_orientation'      :         'float16',

'physical_disability'           :         'float16',

'psychiatric_or_mental_illness' :         'float16',

'transgender'                   :         'float16',

'white'                         :         'float16',

'publication_id'                :         'int8',

'parent_id'                     :         'float32',

'article_id'                    :         'int32',

'funny'                         :         'int8',

'wow'                           :         'int8',

'sad'                           :         'int8',

'likes'                         :         'int16',

'disagree'                      :         'int16',

'sexual_explicit'               :         'float16',

'identity_annotator_count'      :         'int16',

'toxicity_annotator_count'      :         'int16'

}
# Read file to CSV  

train_data = pd.read_csv('../input/train.csv',dtype=dtypesDict_tr,parse_dates=['created_date'])  # nrows=10000000

train_data['created_date'] = pd.to_datetime(train_data['created_date']).values.astype('datetime64[M]')



gc.collect()
# Use the methos to well import the data

# reduce_mem_usage(train_data)

train_data.info()
# Look at the top of the dataset

train_data.head(3)
# Inspect the statistical summary of the dataset

train_data.describe()
pd.options.display.max_colwidth=300

# Print the most severe comments (with target value greater than 0.8) 

# to get a feeling for the data but also on the data structure

for n, v in enumerate(train_data.loc[train_data.target>0.8, 'comment_text']):

    print(n, ': ', v)

    if n == 10:

        break
pd.options.display.max_colwidth=300

# Print the non-toxic comments 

for n, v in enumerate(train_data.loc[train_data.target==0.0, 'comment_text']):

    print(n, ': ', v)

    if n == 10:

        break
# Plot the number of comments over time in the training dataset.

cnt_srs = train_data['created_date'].dt.date.value_counts()

cnt_srs = cnt_srs.sort_index()

plt.figure(figsize=(14,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')

plt.xticks(rotation='vertical')

plt.xlabel('Creation Date', fontsize=12)

plt.ylabel('Number of comments', fontsize=12)

plt.title("Number of comments over time in the train set")

plt.show()
# Plot the number of toxic comments over time in the training dataset.

cnt_srs = train_data.loc[train_data['target']>=0.5,'created_date'].dt.date.value_counts()

cnt_srs = cnt_srs.sort_index()

plt.figure(figsize=(14,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')

plt.xticks(rotation='vertical')

plt.xlabel('Creation Date', fontsize=12)

plt.ylabel('Number of comments', fontsize=12)

plt.title("Number of comments over time in train set")

plt.show()
# Plot the average toxicity over time.



toxicity_gb_time = train_data[train_data['target']>0.5]['target'].groupby(train_data['created_date']).count()/train_data['target'].groupby(train_data['created_date']).count()

#print(toxicity_gb_time)

toxicity_gb_time = toxicity_gb_time.fillna(0)



toxicity_gb_time = toxicity_gb_time.sort_index()



plt.figure(figsize=(14,6))

#sns.lineplot(x=toxicity_gb_time.index, y=toxicity_gb_time.values, label='target')

plt.plot(toxicity_gb_time.index, toxicity_gb_time.values, marker='o', linestyle='-', linewidth=2, markersize=0)



plt.xticks(rotation=45)

plt.xlabel('Creation Date', fontsize=12)

plt.ylabel('Ratio of Toxic Comments', fontsize=12)

plt.title("Ratio of Toxic Comments over Time")

plt.show()
#Plot distribution of the target variable

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.hist(train_data.target, bins=2)

ax1.set_title('Distribution of the target variable')

ax1.set_ylabel('count')



sns.distplot(train_data.target, ax=ax2)

#ax2.hist(train_data.target)

ax2.set_title('Distribution of the target variabl')

ax2.set_ylabel('count')



plt.show()
# Analyse the partition of toxicity classification against the full dataset

# When we interpret the accuracy we need to consider the fact that the number

# of toxic vs. non-toxic comments is not equally balanced



columns = ['target','severe_toxicity', 'obscene',

       'identity_attack', 'insult', 'threat']

# Create series with the number of comments that only contain toxicity values greater than 0.5

toxic_crit = train_data.loc[:, columns]

toxic_crit = toxic_crit>0.5

toxic_crit = toxic_crit.sum()

toxic_crit = toxic_crit.sort_values(ascending = False)

gc.collect()
# Plot the distribution of the comments with toxicity values greater than 0.5

plt.figure(figsize=(14,6))

sns.barplot(toxic_crit.index, toxic_crit.values, alpha=0.8, color='green')

plt.xticks(rotation=45)

plt.xlabel('Criterias', fontsize=12)

plt.ylabel('Number of occurences', fontsize=12)

plt.title("Toxicity annotation greater 0.5")

plt.show()
# Plot the distribution of the toxicity and identity annotator count

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

sns.boxplot(train_data.toxicity_annotator_count, ax=ax1)

ax1.set_title('Distribution of the toxicity_annotator_count')

ax1.set_ylabel('count')

sns.boxplot(train_data.identity_annotator_count, ax=ax2)

#ax2.hist(train_data.identity_annotator_count)

ax2.set_title('Distribution of the identity_annotator_count')

ax2.set_ylabel('count')

plt.show()
import matplotlib.gridspec as gridspec



# Create 3x2 sub plots

gs = gridspec.GridSpec(3, 2)



fig = plt.figure(figsize=(14, 18))

ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0

ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1

ax3 = fig.add_subplot(gs[1, 0]) # row 1, col 0

ax4 = fig.add_subplot(gs[1, 1]) # row 1, col 1

ax5 = fig.add_subplot(gs[2, 0]) # row 2, col 0



sns.distplot(train_data['severe_toxicity'],kde=False, hist=True, bins=30, label='severe_toxicity', ax=ax1)

ax1.set_title('Dist. of the severe_toxicity')

sns.distplot(train_data['obscene'],kde=False, hist=True, bins=30, label='obscene', ax=ax2)

ax2.set_title('Dist. of the obscene')

sns.distplot(train_data['identity_attack'],kde=False, hist=True, bins=30, label='identity_attack', ax=ax3)

ax3.set_title('Dist. of the identity_attack')

sns.distplot(train_data['insult'],kde=False, hist=True, bins=30, label='insult', ax=ax4)

ax4.set_title('Dist. of the insult')

sns.distplot(train_data['threat'],kde=False, hist=True, bins=30, label='threat', ax=ax5)

ax5.set_title('Dist. of the threat')



plt.show()
id_attr = ['male','female','transgender','other_gender','heterosexual','homosexual_gay_or_lesbian','bisexual', 'other_sexual_orientation', 'christian','jewish','muslim','hindu','buddhist',

   'atheist','other_religion','black','white', 'asian','latino','other_race_or_ethnicity','physical_disability','intellectual_or_learning_disability','psychiatric_or_mental_illness','other_disability']
# Analyse the partition of toxicity classification against the full dataset

# When we interpret the accuracy we need to consider the fact that the number

# of toxic vs. non-toxic comments is not equally balanced



other_attr = train_data.loc[:, id_attr]

other_attr = other_attr>0.5

other_attr = other_attr.sum()

other_attr = other_attr.sort_values(ascending = False)
#cnt_srs = test_df['first_active_month'].dt.date.value_counts()

#cnt_srs = cnt_srs.sort_index()



# Plot the distribution of the comments with identity values greater than 0.5

plt.figure(figsize=(14,6))

sns.barplot(other_attr.index, other_attr.values, alpha=0.8, color='green')

plt.xticks(rotation='vertical')

plt.xlabel('Criterias', fontsize=12)

plt.ylabel('Number of occurences', fontsize=12)

plt.title("Distribution of other attribute annotation greater 0.5")

plt.show()
# How many entries (comments) have identity features that are all 0 or NaN?

person_cat = train_data

person_cat_nan = person_cat[(np.isnan(person_cat['asian']) | (person_cat['asian'] == 0.0))

                            & (np.isnan(person_cat['atheist']) | (person_cat['atheist'] == 0.0)) 

                            & (np.isnan(person_cat['bisexual']) | (person_cat['bisexual'] == 0.0)) 

                            & (np.isnan(person_cat['black']) | (person_cat['black'] == 0.0)) 

                            & (np.isnan(person_cat['buddhist']) | (person_cat['buddhist'] == 0.0)) 

                            & (np.isnan(person_cat['christian']) | (person_cat['christian'] == 0.0)) 

                            & (np.isnan(person_cat['female']) | (person_cat['female'] == 0.0)) 

                            & (np.isnan(person_cat['heterosexual']) | (person_cat['heterosexual'] == 0.0)) 

                            & (np.isnan(person_cat['hindu']) | (person_cat['hindu'] == 0.0)) 

                            & (np.isnan(person_cat['homosexual_gay_or_lesbian']) | (person_cat['homosexual_gay_or_lesbian'] == 0.0)) 

                            & (np.isnan(person_cat['intellectual_or_learning_disability']) | (person_cat['intellectual_or_learning_disability'] == 0.0)) 

                            & (np.isnan(person_cat['jewish']) | (person_cat['jewish'] == 0.0)) 

                            & (np.isnan(person_cat['latino']) | (person_cat['latino'] == 0.0)) 

                            & (np.isnan(person_cat['male']) | (person_cat['male'] == 0.0)) 

                            & (np.isnan(person_cat['muslim']) | (person_cat['muslim'] == 0.0)) 

                            & (np.isnan(person_cat['other_disability']) | (person_cat['other_disability'] == 0.0)) 

                            & (np.isnan(person_cat['other_gender']) | (person_cat['other_gender'] == 0.0)) 

                            & (np.isnan(person_cat['other_race_or_ethnicity']) | (person_cat['other_race_or_ethnicity'] == 0.0)) 

                            & (np.isnan(person_cat['other_religion']) | (person_cat['other_religion'] == 0.0)) 

                            & (np.isnan(person_cat['other_sexual_orientation']) | (person_cat['other_sexual_orientation'] == 0.0)) 

                            & (np.isnan(person_cat['physical_disability']) | (person_cat['physical_disability'] == 0.0))

                            & (np.isnan(person_cat['psychiatric_or_mental_illness']) | (person_cat['psychiatric_or_mental_illness'] == 0.0)) 

                            & (np.isnan(person_cat['transgender']) | (person_cat['transgender'] == 0.0)) 

                            & (np.isnan(person_cat['white']) | (person_cat['white'] == 0.0))]



print("Apparently only " + str((person_cat_nan.shape[0] / train_data.shape[0])*100) + " % of the comments do not have a value greater than 0 in any of the identity columns")
# How many of the comments that do not have a value greater than 0 in any of the identity columns,

# have a value greater than 0.5 for toxicity? 

print("" + str((len(person_cat_nan[person_cat_nan['target']>0.5]['target'])/person_cat_nan.shape[0])*100) + " % of the comments that do not have a value greater than 0 in any of the identity columns, are toxic.")
# How many of the comments that do not have a value greater than 0 in any of the identity columns,

# have a value greater than 0.5 for identity_attack? 

len(person_cat_nan[person_cat_nan['identity_attack']>0.5]['identity_attack'])



print("Only " + str((len(person_cat_nan[person_cat_nan['identity_attack']>0.5]['identity_attack'])/person_cat_nan.shape[0])*100) + " % of the comments that do not have a value greater than 0 in any of the identity columns,\nhave a value greater than 0.5 for identity_attack.")
del person_cat_nan

gc.collect()
person_cat = person_cat[((person_cat['asian'] > 0.0))# & person_cat[person_cat.asian.notnull()])

                        | ((person_cat['atheist'] > 0.0))

                        | (person_cat['bisexual'] > 0.0) 

                        | (person_cat['black'] > 0.0)

                        | (person_cat['buddhist'] > 0.0)

                        | (person_cat['christian'] > 0.0)

                        | (person_cat['female'] > 0.0)

                        | (person_cat['heterosexual'] > 0.0)

                        | (person_cat['hindu'] > 0.0)

                        | (person_cat['homosexual_gay_or_lesbian'] > 0.0)

                        | (person_cat['intellectual_or_learning_disability'] > 0.0)

                        | (person_cat['jewish'] > 0.0)

                        | (person_cat['latino'] > 0.0)

                        | (person_cat['male'] > 0.0)

                        | (person_cat['muslim'] > 0.0)

                        | (person_cat['other_disability'] > 0.0)

                        | (person_cat['other_gender'] > 0.0)

                        | (person_cat['other_race_or_ethnicity'] > 0.0)

                        | (person_cat['other_religion'] > 0.0)

                        | (person_cat['other_sexual_orientation'] > 0.0)

                        | (person_cat['physical_disability'] > 0.0)

                        | (person_cat['psychiatric_or_mental_illness'] > 0.0)

                        | (person_cat['transgender'] > 0.0)

                        | (person_cat['white'] > 0.0)]



person_cat.shape

print("" + str((person_cat.shape[0] / train_data.shape[0])*100) + " % of the comments have a value greater than 0 in at least one of the identity columns")
print("" + str(((len(person_cat[person_cat['target']>0.5]['target']) / person_cat.shape[0])*100)) + " % of the comments that have a value greater than 0 in any of the identity columns, are toxic.")
# Explanation

# categories = ['target']+list(train_data)[slice(8,32)]

# SUM(x*y)/COUNT(identity_col > 0)

# categories.iloc[:, 1:] multiply all the identity columns  

# categories.iloc[:, 0]  with the target column

# categories.iloc[:, 1:].multiply(categories.iloc[:, 0], axis="index").sum()  create the sum

# categories.iloc[:, 1:][categories.iloc[:, 1:]>0].count()    devide by all the number of the identity values that are bigger than 0



# Percent of toxic comments related to different identities, 

# using target and popolation amount of each identity as weights:



categories = train_data.loc[:, ['target']+list(train_data)[slice(8,32)]].dropna() # take the column target and all the categorizing columns

# categories.iloc[:, 0] --> target column

# categories.iloc[:, 1] --> all identity columns

weighted_toxic = categories.iloc[:, 1:].multiply(categories.iloc[:, 0], axis="index").sum()/categories.iloc[:, 1:][categories.iloc[:, 1:]>0].count()

weighted_toxic = weighted_toxic.sort_values(ascending=False)

plt.figure(figsize=(10,8))

sns.set(font_scale=1)

ax = sns.barplot(x = weighted_toxic.values, y = weighted_toxic.index, saturation=0.5, alpha=0.99)

plt.ylabel('Categories')

plt.xlabel('Weighted Toxic')

plt.show()
# Percent of identity_attack related to different identities, 

# using target and popolation amount of each identity as weights:



categories = train_data.loc[:, ['identity_attack']+list(train_data)[slice(8,32)]] #.dropna() # take the column target and all the categorizing columns

# categories.iloc[:, 0] --> identity_attack column

# categories.iloc[:, 1] --> all identity columns

weighted_identity_attack = categories.iloc[:, 1:].multiply(categories.iloc[:, 0], axis="index").sum()/categories.iloc[:, 1:][categories.iloc[:, 1:]>0].count()

weighted_identity_attack = weighted_identity_attack.sort_values(ascending=False)

plt.figure(figsize=(10,8))

sns.set(font_scale=1)

ax = sns.barplot(x = weighted_identity_attack.values, y = weighted_identity_attack.index, saturation=0.5, alpha=0.99)

plt.ylabel('categories')

plt.xlabel('weighted identity_attack')

plt.show()
#Let's see if there are correlations between the identity columns in train_data and the target:



corrs = np.abs((person_cat.loc[:, ['target']+list(person_cat)[slice(8,32)]]).corr())

ordered_cols = (corrs).sum().sort_values().index

np.fill_diagonal(corrs.values, 0)

plt.figure(figsize=[8,8])

plt.imshow(corrs.loc[ordered_cols, ordered_cols], cmap='plasma', vmin=0, vmax=1)

plt.colorbar(shrink=0.7)

plt.xticks(range(corrs.shape[0]), list(ordered_cols), size=16, rotation=90)

plt.yticks(range(corrs.shape[0]), list(ordered_cols), size=16)

plt.title('Heat map of coefficients of correlation between identity categories and target', fontsize=17)

plt.show()

# Distribution of race and ethnicity



# Create 3x2 sub plots

import matplotlib.gridspec as gridspec



gs = gridspec.GridSpec(3, 2)



fig = plt.figure(figsize=(14, 18))

ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0

ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1

ax3 = fig.add_subplot(gs[1, 0]) # row 1, col 0

ax4 = fig.add_subplot(gs[1, 1]) # row 1, col 1

ax5 = fig.add_subplot(gs[2, 0]) # row 2, col 0

ax6 = fig.add_subplot(gs[2, 1]) # row 2, col 1



sns.distplot(person_cat['asian'],kde=False, hist=True, bins=30, label='asian', ax=ax1)

ax1.set_title('Dist. of the asian')

sns.distplot(person_cat['black'],kde=False, hist=True, bins=30, label='black', ax=ax2)

ax2.set_title('Dist. of the black')

sns.distplot(person_cat['jewish'],kde=False, hist=True, bins=30, label='jewish', ax=ax3)

ax3.set_title('Dist. of the jewish')

sns.distplot(person_cat['latino'],kde=False, hist=True, bins=30, label='latino', ax=ax4)

ax4.set_title('Dist. of the atino')

sns.distplot(person_cat['other_race_or_ethnicity'],kde=False, hist=True, bins=30, label='other_race_or_ethnicity', ax=ax5)

ax5.set_title('Dist. of the other_race_or_ethnicity')

sns.distplot(person_cat['other_race_or_ethnicity'],kde=False, hist=True, bins=30, label='other_race_or_ethnicity', ax=ax6)

ax6.set_title('Dist. of the other_race_or_ethnicity')



plt.show()
# Plot ethnicity features over time.



# Extract month and year from created_date and aggregate



# create dataframe with all the identity columns, target and creation date

withdate = train_data.loc[:, ['created_date', 'target']+list(train_data)[slice(8,32)]].dropna() 

# weight the identity scores, dividing each value by the sum of the whole column

raceweighted = withdate.iloc[:, 2:]/withdate.iloc[:, 2:].sum()  

# Multipy the raceweight columns witht the target

race_target_weighted = raceweighted.multiply(withdate.iloc[:, 1], axis="index")

# Create created_date column

race_target_weighted['created_date'] = pd.to_datetime(withdate['created_date']).values.astype('datetime64[M]')

# Group by the creation date

weighted_demo = race_target_weighted.groupby(['created_date']).sum().sort_index()
plt.figure(figsize=(14,6))

#sns.lineplot(x=toxicity_gb_time.index, y=toxicity_gb_time.values, label='target')

plt.plot(weighted_demo.index, weighted_demo['white'], marker='o', linestyle='-', linewidth=2, markersize=0)

plt.plot(weighted_demo.index, weighted_demo['asian'], marker='o', linestyle='-', linewidth=2, markersize=0)

plt.plot(weighted_demo.index, weighted_demo['black'], marker='o', linestyle='-', linewidth=2, markersize=0)

plt.plot(weighted_demo.index, weighted_demo['jewish'], marker='o', linestyle='-', linewidth=2, markersize=0)

plt.plot(weighted_demo.index, weighted_demo['latino'], marker='o', linestyle='-', linewidth=2, markersize=0)

plt.plot(weighted_demo.index, weighted_demo['other_race_or_ethnicity'], marker='o', linestyle='-', linewidth=2, markersize=0)



plt.xticks(rotation=45)

plt.xlabel('Creation Date', fontsize=12)

#plt.ylabel('Ratio of Toxic Comments', fontsize=12)

plt.title("Time Series Toxicity & Race")

plt.legend()

plt.show()
del toxic_crit, fig,ax1, ax2, other_attr, person_cat

del weighted_identity_attack

del weighted_toxic

del race_target_weighted

del raceweighted

gc.collect
# Print out all the variables in use

# %whos
train_ids = train_data[['created_date','id', 'publication_id', 'parent_id', 'article_id', 'target']]

train_ids.head(10)
# Check missing values

# Only parent_id seems to have missing values

train_ids.isnull().sum()
# HOW MANY BAD COMMENTS DIVIDED BY ALL COMMENTS PER PUBLICATION_ID (RATIO)

plt.figure(figsize=(14,6))

# Group by publication_id and count the number of comments with toxcicty (over 0.5) per publication_id group.

y = train_data[train_data['target']>0.5]['target'].groupby(train_data['publication_id']).count()/train_data['target'].groupby(train_data['publication_id']).count()

# Plot the values in descending order 

sns.barplot(y.index, y.sort_values(ascending =False), color='green')

plt.ylabel('Number of Toxic Comments', fontsize=12)

plt.title("Distribution of Toxic Comments over publication_id")
# Convert the data type of 'created date' column

#train_data['created_date'] = pd.to_datetime(train_data['created_date']).values.astype('datetime64[M]')

#train_data['created_date'].head()
# Plot the distribution of the ID-s over time 

plt.figure(figsize=(16, 6))

sns.lineplot(x='created_date', y='id', label='IDs', data=train_data)

plt.title('Distribution of IDs over years')
train_eda = train_data.loc[:, ["target", "sad", "wow", "funny", "likes", "disagree"]]

train_eda[train_eda['target']>0].head()
train_eda.describe()
train_eda.isnull().sum()
colormap = plt.cm.RdBu

plt.figure(figsize=(12,12))

sns.heatmap(train_eda.corr(), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True)

plt.title('Pair-wise correlation')
eda_wow = train_eda[train_eda["wow"] >= 1]

eda_wow.drop(["sad", "disagree", "funny", "likes"], axis=1).describe()
eda_funny = train_eda[train_eda["funny"] >= 1]

eda_funny.drop(["sad", "wow", "disagree", "likes"], axis=1).describe()
eda_sad = train_eda[train_eda["sad"] >= 1]

eda_sad.drop(["disagree", "wow", "funny", "likes"], axis=1).describe()
eda_likes = train_eda[train_eda["likes"] >= 1]

eda_likes.drop(["sad", "wow", "funny", "disagree"], axis=1).describe()
eda_disagree = train_eda[train_eda["disagree"] >= 1]

eda_disagree.drop(["sad", "wow", "funny", "likes"], axis=1).describe()
del columns, corrs, eda_likes, eda_sad, eda_wow, eda_funny

del eda_disagree

del train_eda

gc.collect
!pip install nltk

import nltk

import re



# Import the English language class

from spacy.lang.en import English



nltk.download('punkt')

nltk.download('stopwords')



# Import Counter

from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

stop_words = stopwords.words('english')





!pip install gensim

from nltk.corpus import stopwords  

import gensim 

from gensim.utils import simple_preprocess 

from gensim.parsing.preprocessing import STOPWORDS 

from nltk.stem import WordNetLemmatizer, SnowballStemmer 

from nltk.tokenize import word_tokenize
# Tokenize the comment text of toxic comments

tok_comments = [word_tokenize(com) for com in train_data[train_data['target']>0.5]['comment_text']]
# Remove stopwords

tokens = [[w for w in s if (w not in stop_words) & (len(w)>2)] for s in tok_comments]
from nltk.probability import FreqDist
#plot the most frequent words

tokens = np.array([np.array(s) for s in tokens])

fdist = FreqDist(np.concatenate(tokens))

fdist.plot(30,cumulative=False)

plt.show()
#plot the most frequent word pairs

from nltk import bigrams, ngrams

bigrams_tokens = bigrams(np.concatenate(tokens))

fdist_bigrams = FreqDist(list(bigrams_tokens))

fdist_bigrams.plot(30,cumulative=False)

plt.show()
def print_top_20_words(comment_text):

  

  # delete the numbers from the string

  #comment_text = re.sub(r'\d+', '', comment_text)



  # Create the nlp object

  nlp = English()



  # Process a text

  doc = nlp(comment_text)



  # Print the document text

  # print(doc.text)



  # Tokenize the article: tokens

  tokens = word_tokenize(doc.text)



  # Convert the tokens into lowercase: lower_tokens

  lower_tokens = [t.lower() for t in tokens]



  # Remove stopwords

  

  lower_tokens = [word for word in lower_tokens if word not in stopwords.words('english')]

  lower_tokens = [word.lower() for word in lower_tokens if word.isalpha()]





  # Create a Counter with the lowercase tokens: bow_simple

  bow_simple = Counter(lower_tokens)

  

  # Print the 20 most common tokens

  # print(bow_simple.most_common(10))

  d = dict()

  for elem in bow_simple.most_common(20):

    print(elem[1], elem[0])

    d[elem[0]] = elem[1]

    

  return d
downsampled_train_data = pd.read_csv('../input/train.csv',dtype=dtypesDict_tr,parse_dates=['created_date'], nrows=200000)
# Turn the whole comment_text column into list of strings (text)

comment_text = downsampled_train_data[downsampled_train_data['target'] >0.5]['comment_text'].to_string()



# List of top 20 most used words in comments with toxicity (target) higher than 0.7

# Work in chunks because nlp cannot process more than 1 Million characters

first_chunk = print_top_20_words(comment_text[:(len(comment_text)//4)])  # // devision returns a natural number without the rest

print('\n\n')

second_chunk = print_top_20_words(comment_text[(len(comment_text)//4+1):(2*len(comment_text)//4)])

print('\n\n')

third_chunk = print_top_20_words(comment_text[(2*len(comment_text)//4+1):(3*len(comment_text)//4)])

print('\n\n')

fourth_chunk = print_top_20_words(comment_text[(3*len(comment_text)//4+1):])
# Build an average out of the 2 dictionaries



newd1 = {}

for key in first_chunk.keys():

    for key2 in second_chunk.keys():

        if key in second_chunk.keys():

            newd1[key] = int(first_chunk.get(key)) + int(second_chunk.get(key))

        else:

            newd1[key] = first_chunk.get(key)

        if key2 not in first_chunk.keys():

            newd1[key2] = second_chunk.get(key2)



newd2 = {}   

for key in third_chunk.keys():

    for key2 in fourth_chunk.keys():

        if key in fourth_chunk.keys():

            newd2[key] = int(third_chunk.get(key)) + int(fourth_chunk.get(key))

        else:

            newd2[key] = third_chunk.get(key)

        if key2 not in third_chunk.keys():

            newd2[key2] = fourth_chunk.get(key2)



newd = {}

for key in newd1.keys():

    for key2 in newd2.keys():

        if key in newd2.keys():

            newd[key] = int(newd1.get(key)) + int(newd2.get(key))

        else:

            newd[key] = newd1.get(key)

        if key2 not in newd1.keys():

            newd[key2] = newd2.get(key2)





# Sort dictionary numerically descending

print("Top 20 most used words in comments with toxicity (target) higher than 0.5:")

i = 0

for key, value in sorted(newd.items(), key=lambda item: item[1], reverse=True):

    if i < 20:

        print("%s: %s" % (key, value))

    i = i + 1
# List of top 20 most used words in comments with identity attack higher than 0.5

comment_text = downsampled_train_data[downsampled_train_data['identity_attack'] >0.5]['comment_text'].to_string()

print("Top 20 most used words in comments with identity attack higher than 0.5:")

dict_identity_attack = print_top_20_words(comment_text)
from wordcloud import WordCloud ,STOPWORDS

from PIL import Image
def toxicwordcloud(dict1, subset=train_data[train_data.target>0.5], title = "Words Frequented"):

    stopword=set(STOPWORDS)

    wc= WordCloud(background_color="black",max_words=4000,stopwords=stopword)

    wc.generate(" ".join(list(dict1.keys())))

    plt.figure(figsize=(8,8))

    plt.xticks([])

    plt.yticks([])

    plt.axis('off')

    plt.title(title, fontsize=20)

    plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98)
toxicwordcloud(newd)
toxicwordcloud(dict_identity_attack)
del downsampled_train_data

gc.collect()
words = train_data["comment_text"].apply(lambda x: len(x) - len(''.join(x.split())) + 1)



train_data['words'] = words

words_toxic = train_data.loc[(train_data['words']<200)&(train_data['target'] >0.7)]['words']

words_identity_attack = train_data.loc[(train_data['words']<200)&(train_data['target'] >0.7)&(train_data['identity_attack'] >0.7)]['words']

words_threat = train_data.loc[(train_data['words']<200)&(train_data['target'] >0.7)&(train_data['threat'] >0.7)]['words']

words_nontoxic = train_data.loc[(train_data['words']<200)&(train_data['target'] <0.3)]['words']



sns.set()

plt.figure(figsize=(12,6))

plt.title("Comment Length (words)")

sns.distplot(words_toxic,kde=True,hist=False, bins=120, label='toxic')

sns.distplot(words_identity_attack,kde=True,hist=False, bins=120, label='identity_attack')

sns.distplot(words_threat,kde=True,hist=False, bins=120, label='threat')

sns.distplot(words_nontoxic,kde=True,hist=False, bins=120, label='nontoxic')

plt.legend(); plt.show()
# Create new features

train_data['total_length'] = train_data['comment_text'].apply(len)

train_data['capitals'] = train_data['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))

train_data['num_exclamation_marks'] = train_data['comment_text'].apply(lambda comment: comment.count('!'))

train_data['num_question_marks'] = train_data['comment_text'].apply(lambda comment: comment.count('?'))

train_data['num_symbols'] = train_data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))

train_data['num_words'] = train_data['comment_text'].apply(lambda comment: len(comment.split()))

train_data['num_smilies'] = train_data['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
# Correlations between new features and the toxicity features

features = ('total_length', 'capitals', 'num_exclamation_marks','num_question_marks', 'num_symbols', 'num_words', 'num_smilies')

columns = ('target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat')

rows = [{c:train_data[f].corr(train_data[c]) for c in columns} for f in features]

train_correlations = pd.DataFrame(rows, index=features)
train_correlations