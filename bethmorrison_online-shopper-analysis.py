import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import copy

import random

import itertools

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

%matplotlib inline
shoppers = pd.read_csv("../input/online-shoppers-intention/online_shoppers_intention.csv")
type(shoppers)
shoppers.head()
shoppers.shape ## 12330 rows, 18 columns
shoppers.describe()
shoppers.columns ## just the column names
shoppers.isnull().sum()
shoppers = shoppers.dropna(axis = 0)
shoppers.isnull().sum() ## no remaining null values
shoppers.shape ## should be 14 rows less
shoppers.info()
cat_cols = ['Month', 'OperatingSystems', 'Browser', 'Region',

           'TrafficType', 'VisitorType', 'Weekend', 'Revenue']

for col in cat_cols:

    shoppers[col] = shoppers[col].astype('category')
shoppers.dtypes 
for col in cat_cols:

    print(shoppers[col].unique())
sum(shoppers['VisitorType'] == 'Other') ## 85 have this other type
len(shoppers.loc[shoppers['ProductRelated_Duration'] == -1, 'ProductRelated_Duration'])
len(shoppers.loc[shoppers['Informational_Duration'] == -1, 'Informational_Duration'])
len(shoppers.loc[shoppers['Administrative_Duration'] == -1, 'Administrative_Duration'])
## okay so looks like the -1 are all in the same row so let's drop those rows

shoppers = shoppers.loc[shoppers['Administrative_Duration'] != -1, ]
## how many shoppers spent zero amount of time on the website - these ones maybe are

## rounded down because they spent so little time on the site (<0)

sum(shoppers[['ProductRelated_Duration',

              'Informational_Duration',

              'Administrative_Duration']].sum(axis = 1) == 0)
plt.figure(figsize = (16,8))

sns.countplot(x = "Region", data = shoppers, 

              order = shoppers['Region'].value_counts().index)

plt.title("Site Visitors by Region", fontsize = 16)

plt.xlabel("Region", fontsize = 16)

plt.ylabel("Number of Visitors", fontsize = 16)
shoppers.columns
dur = shoppers[['Administrative_Duration', 

          'Informational_Duration', 

          'ProductRelated_Duration']].sum(axis = 1)
site_duration = pd.DataFrame({"VisitorType": shoppers['VisitorType'],

                           "TotalDuration": dur

                           })
plt.figure(figsize = (16,8))

sns.boxplot('VisitorType', 'TotalDuration', data = site_duration)

plt.title("Time on Site by Visitor Type", fontsize = 16)

plt.xlabel("Visitor Type", fontsize = 16)

plt.ylabel("Total Time on Site (seconds)", fontsize = 16)

plt.ylim(10, 10000)
ret_duration = list(site_duration.loc[

    site_duration['VisitorType'] == 'Returning_Visitor', 'TotalDuration'])
new_duration = list(site_duration.loc[

    site_duration['VisitorType'] == 'New_Visitor', 'TotalDuration'])
import copy

import random
## write a permutation function for mean



def perm_mean(group_1, group_2, p): ## two lists and a numeric value for the number of permutations

    """Returns the p-value for a permutation test of difference in means between

    two groups"""

    

    ## observed difference in means

    obs_mean = np.abs(np.average(group_1) - np.average(group_2))

    

    ## pool the observations into a single list

    pooled_groups = list(group_1 + group_2)

    

    ## make a copy that can be randomly shuffled for the permutations

    pooled_copy = copy.copy(pooled_groups)

    

    ## a space to save permutation output

    perm_means = []

    

    ## permutations

    for i in range(0, p):

        ## randomly shuffle the pooled observations

        random.shuffle(pooled_copy)

        

        ## calculate differences in mean for each permutation

        perm_means.append(

            np.abs(np.average(

                pooled_copy[0:len(group_1)]) - np.average(pooled_copy[len(group_1):])))



    ## calculate the p-value as proportion of the permuted means that had a larger

    ## difference in means than the observed difference in means

    p_value = sum(perm_means >= obs_mean)/p

    

    return p_value

perm_mean(ret_duration, new_duration, 1000)
shoppers.columns
duration = list(shoppers['Administrative_Duration']) + list(

    shoppers['ProductRelated_Duration']) + list(

    shoppers['Informational_Duration'])
import itertools
duration_type = list(

    itertools.repeat('Administrative', len(shoppers['Administrative_Duration']))) + list(

    itertools.repeat('ProductRelated', len(shoppers['ProductRelated_Duration']))) + list(

    itertools.repeat('Informational', len(shoppers['Informational_Duration'])))
visitor_type = list(shoppers['VisitorType'])*3
duration_info = pd.DataFrame({"Visitor Type": visitor_type,

                           "Duration Type": duration_type,

                           "Duration": duration})
duration_info
plt.figure(figsize = (16,8))

sns.boxplot('Visitor Type', 'Duration', data = duration_info, hue = 'Duration Type')

plt.title("Duration on Each Page Type by Visitor Type", fontsize = 16)

plt.xlabel("Visitor Type", fontsize = 16)

plt.ylabel("Duration on Page (seconds)", fontsize = 16)

plt.show()
plt.figure(figsize = (16,8))

sns.boxplot('Visitor Type', 'Duration', data = duration_info, hue = 'Duration Type')

plt.title("Duration on Each Page Type by Visitor Type", fontsize = 16)

plt.xlabel("Visitor Type", fontsize = 16)

plt.ylabel("Duration on Page (seconds)", fontsize = 16)

plt.ylim((-100, 3000))

plt.show()
duration_info ## ok something got fucked up
shoppers
duration_info['Total Duration'] = list(site_duration['TotalDuration'])*3
duration_info
## percentages

duration_info['Duration Percent'] = duration_info['Duration']/duration_info['Total Duration']
duration_info
min(duration_info['Duration Percent'])
duration_info = duration_info.dropna()
plt.figure(figsize = (16,8))

sns.boxplot('Visitor Type', 'Duration Percent', data = duration_info, hue = 'Duration Type',

              palette = "colorblind")

plt.title("Duration on Each Page Type by Visitor Type", fontsize = 16)

plt.xlabel("Visitor Type", fontsize = 16)

plt.ylabel("Duration on Page (seconds)", fontsize = 16)

plt.show()
percent_ret = round(shoppers['VisitorType'].value_counts()['Returning_Visitor']/len(shoppers['VisitorType'])*100, 2)

percent_new = round(shoppers['VisitorType'].value_counts()['New_Visitor']/len(shoppers['VisitorType'])*100, 2)



print(f"{percent_ret} % visitors were returning")

print(f"{percent_new} % visitors were new")
ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 

                  'Nov', 'Dec']
## this bit might not be the cleanest and I wish I could figure out a better way

## dataframe that counts the entries in each group (visitor type & month)

month_info = shoppers.groupby(['VisitorType', 'Month']).count()
## only need one of those columns

month_info = pd.DataFrame(month_info.iloc[:, 1])
## turns the index into columns

month_info.reset_index(inplace = True)
## change column names

month_info.columns = ['Visitor Type', 'Month', 'Num. Visitors']
month_info['Num. Visitors'].fillna(value = 0, inplace = True)
month_info['Month']
month_info['Visitor Type'] = month_info['Visitor Type'].astype(str)
month_info['Month'] = month_info['Month'].astype(str)
to_add = pd.DataFrame([], columns = month_info.columns)
to_add['Visitor Type'] = ['New_Visitor', 'New_Visitor', 

                          'Other', 'Other', 

                          'Returning_Visitor', 'Returning_Visitor']
to_add['Month'] = ['Jan', 'Apr']*3
to_add['Num. Visitors'] = [0]*6
to_add
month_info = month_info.append(to_add)
month_info['Month'] = pd.Categorical(month_info['Month'], categories = ordered_months,

                                    ordered = True)
month_info
plt.figure(figsize=(12, 8))

sns.lineplot(x = 'Month', y = "Num. Visitors", hue = "Visitor Type", data = month_info, 

            hue_order = ["Returning_Visitor", "New_Visitor", "Other"], sizes=(2.5, 2.5))

plt.title("Number of Monthly Visitors by Visitor Type", fontsize=16)

plt.xlabel("Month", fontsize=16)

plt.xticks(rotation=45)

plt.ylabel("Number of Visitors", fontsize=16)

plt.show()
shoppers.columns
## closeness to special days in may? 

np.average(shoppers.loc[shoppers['Month'] == 'May', 'SpecialDay'])
np.average(shoppers.loc[shoppers['Month'] == 'Mar', 'SpecialDay'])
np.average(shoppers.loc[shoppers['Month'] == 'Nov', 'SpecialDay'])
np.average(shoppers.loc[shoppers['Month'] == 'Dec', 'SpecialDay'])
shoppers['SpecialDay'].unique()
shoppers.loc[shoppers['SpecialDay'] > 0, 'Month'].unique()
plt.figure(figsize=(12, 8))

sns.countplot(x = 'Revenue', data = shoppers)

plt.title("How many visitors generated revenue overall?", fontsize=16)

plt.xlabel("Revenue Generated?", fontsize=16)

plt.xticks(rotation=45)

plt.ylabel("Number of Visitors", fontsize=16)

plt.show()
print(round(sum(shoppers['Revenue'])/len(shoppers['Revenue']), 2)*100, "% of visitors generate revenue")
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics
shoppers.columns
shoppers.dtypes
cat_vars = list(shoppers.select_dtypes('category').columns)
num_vars = list(shoppers.select_dtypes('float').columns)
cat_vars = cat_vars[:-1] ## drop the revenue label
shoppers_dummies = pd.get_dummies(shoppers, columns = cat_vars)
shoppers_dummies.head()
shoppers_dummies.columns ## no revenue column because we already dropped it so no need to dop it now
X = shoppers_dummies ## independent variables
y = shoppers['Revenue'] ## dependent variable
y = y.astype(int) ## transform boolean to 0s and 1s
## use the stratify argument here because of the uneven distribution of true and false values

## in the target variable (revenue)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                test_size = 0.3, random_state = 42,

                                                stratify = y)
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn import svm

from sklearn import tree

from sklearn.metrics import classification_report

from sklearn import svm

from xgboost import XGBClassifier

from itertools import compress
# define models to test:

base_models = [("clf", LogisticRegression(random_state=42)),

               ("clf", DecisionTreeClassifier(random_state=42)),

               ("clf", svm.SVC(random_state=42))]
svm.SVC().get_params().keys()
check_params_lr = {'clf__C': [0.1, 1, 10, 100],

                  'pca__n_components':[2, 3, 4, 5, 6]}

check_params_dt = {'pca__n_components':[2, 3, 4, 5, 6],

               'clf__criterion':['gini', 'entropy'],

               'clf__min_samples_split': [2,3,4],

               'clf__max_depth': np.arange(3,15)}

check_params_svc = {'pca__n_components': [2, 3, 4, 5, 6],

                   'clf__C': [0.1, 1, 10, 100]}



check_params = [check_params_lr, check_params_dt, check_params_svc]
def model_fit(model, params, X_train, y_train, X_test, y_test):

    

    pipe = Pipeline([('sc1', StandardScaler()),

                     ('pca', PCA()),

                    model])

    

    gs = GridSearchCV(estimator = pipe,

                     param_grid = params,

                     scoring = 'accuracy',

                     cv = 5)

    

    gs.fit(X_train, y_train)

    

    # evaluate the model on the test set

    y_true, y_pred = y_test, gs.predict(X_test)



    # get classification report for the gs model

    print(classification_report(y_true, y_pred))

    
for mod, param in zip(base_models, check_params):

    model_fit(mod, param, X_train, y_train, X_test, y_test)