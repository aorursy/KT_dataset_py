import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

import re 



# for pretty printing pandas dataframes

from IPython.display import display, HTML

# matplotlib setting

%matplotlib inline

plt.style.use('seaborn-notebook')
dir = '../input/'

data = pd.DataFrame()

for f in glob.glob((dir+'*.csv')):

    data = pd.concat([data, pd.read_csv(f)])



# load the structure of the data

data.info()
# samples from the data

data.head()
duplicated_cols = ['nearest_five_percent', 'compressed_image_url', 'category_url', 'category_slug']

less_important_cols = ['igg_image_url', 'url']



data_clean = data.drop(duplicated_cols+less_important_cols, axis = 1)
small_variance_cols = ['card_type', 'friend_contributors', 'friend_team_members', 'source_url']

for col in small_variance_cols:

    print('no of elements in', col, 'is:', len(np.unique(data[col])))

    

# drop the columns

data_clean = data_clean.drop(small_variance_cols, axis = 1)

data_clean.head()
# extract the number in balance and collected_percentage

for col in ['balance', 'collected_percentage']:

    data_clean[col] = pd.to_numeric(data_clean[col].str.replace(r"\D",''))



# from Juan Corporan's Analyzing a Project's Success notebook. To transform the time left.

def get_daysleft(time):

    if "hour" in time:

        return float(re.sub(r"\D", "", time))/24

    elif "day" in time:

        return float(re.sub(r"\D", "", time))

    else:

        return 0.0



data_clean.amt_time_left = data_clean.amt_time_left.apply(get_daysleft)



# transform in_forever_funding into indicator variable

def split_foreverfunding(x):

    if "true" in str(x).lower():

        return True

    elif "false" in str(x).lower() :

        return False

    else:

        return np.nan

    

data_clean.in_forever_funding = data_clean.in_forever_funding.apply(split_foreverfunding) 



# transform partner_name into indicator variable

def have_partner(x):

    if "null" in str(x).lower():

        return False

    else:

        return True



data_clean['partner_name'] = data_clean.partner_name.apply(have_partner)



# lower the string in category_name

data_clean.category_name = data_clean.category_name.str.lower()

data_clean['category_name'] = data_clean['category_name'].astype('category')



data_clean.head(5)  
# assume the most updated status is the record with the max number in balance

idx = data_clean.groupby(['id'])['balance'].transform(max) == data_clean['balance']

# now we can safely remove the duplicated rows since they contains the most updated records even they are duplicated. 

data_clean = data_clean[idx].drop_duplicates()
# focus on the projects that are closed, i.e., projects with no time left.

data_clean = data_clean[data_clean.amt_time_left <= 0]

# then we can drop the variable amt_time_left

data_clean = data_clean.drop(['amt_time_left'], axis = 1)
# We can estimate the goal of the projects using the collected_percentage and balance

data_clean['goal'] = data_clean['balance']*100/data_clean['collected_percentage']

# collected_percentage values can be zero

data_clean.loc[data_clean['goal'] == np.inf, 'goal'] = np.nan



# define success for funding

data_clean['success'] = data_clean['collected_percentage'].apply(lambda x: np.where(x >= 100,True,False))



data_clean.head()
fig, ax = plt.subplots(figsize=(6, 4))

ax = sns.barplot(y="goal", x="partner_name", hue="success", data=data_clean)
IQR = data_clean['goal'].quantile(0.75) - data_clean['goal'].quantile(0.25)

lower_b = data_clean['goal'].quantile(0.25) - 1.5*IQR

higher_b = data_clean['goal'].quantile(0.75) + 1.5*IQR

non_outlier = (data_clean.goal > lower_b) & (data_clean.goal < higher_b)



ax = sns.distplot(data_clean[non_outlier].goal, bins = 9)
fig, ax = plt.subplots(figsize=(6, 4))

ax = sns.barplot(y="cached_collected_pledges_count", x="partner_name", hue="success", data=data_clean)
from sklearn.model_selection import train_test_split



# get the subgroup of the data to predict the success of funding

subdata = data_clean[['cached_collected_pledges_count', 'currency_code', 'category_name', 

               'in_forever_funding', 'partner_name', 'goal', 'success']].dropna()

X = subdata.iloc[:, : len(subdata.columns) -1]

y = subdata['success']

one_hot_X = pd.get_dummies(X)



# split the data

X_train, X_test, y_train, y_test = train_test_split(one_hot_X, y, test_size=0.25, random_state=0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score



# create the clf

clf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators= 150, oob_score = True)



# fit and predict

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_naive_pred = np.random.randint(2, size=len(y_pred))

print('The F1 score of the clf is:', f1_score(y_test, y_pred),

      'which is better than random guess', f1_score(y_naive_pred, y_pred))
# The first 5 important features

importances = clf.feature_importances_

indices = np.argsort(importances)[-5:]



# The relative importance

top_5 = {one_hot_X.columns[x]: importances[x] for x in indices}

print(top_5)