# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
suicide = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

suicide.head(10)
suicide.info()
suicide[' gdp_for_year ($) ']=suicide[' gdp_for_year ($) '].apply(lambda val: val.replace(',', ''))

suicide[' gdp_for_year ($) '] = pd.to_numeric(suicide[' gdp_for_year ($) '])

suicide[' gdp_for_year ($) ']
suicide.describe()
suicide['country'].nunique()
suicide['country'].unique()
suicide['generation'].value_counts()
from sklearn.model_selection import train_test_split

train, test = train_test_split(suicide, test_size=0.2, random_state = 1)
test.count()[0]/suicide.count()[0]*100
train.describe()
train.head(10)
import seaborn as sns

sns.countplot('sex',data=test)
sns.countplot('age',data=test)
test['generation'].value_counts()/len(test)
suicide['generation'].value_counts()/len(suicide)
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = 1)

for train_index, test_index in split.split(suicide, suicide['generation']):

    strat_train = suicide.loc[train_index]

    strat_test = suicide.loc[test_index]
strat_test['generation'].value_counts()/len(strat_test)
#Create a function to compare the proportions of different set

def generation_proportions(test_set):

    return test_set['generation'].value_counts()/len(test_set)

compare_props = pd.DataFrame({

                'Overall': generation_proportions(suicide),

                'Random': generation_proportions(test),

                'Stratified': generation_proportions(strat_test)}

                            )

compare_props['%err random'] = 100*(compare_props['Random'] - compare_props['Overall'])/ compare_props['Overall']

compare_props['%err stratified'] = 100*(compare_props['Stratified'] - compare_props['Overall'])/ compare_props['Overall']
compare_props
suicide = strat_train.copy()
corr_mat = suicide.corr()['suicides/100k pop'].sort_values(ascending=False)

corr_mat
sns.heatmap(suicide.corr(),annot=True)
sns.pairplot(suicide)
suicide.head(10)
import matplotlib.pyplot as plt

suicide['age'].replace({'5-14 years':'05-14','15-24 years':'15-24','25-34 years':'25-34','35-54 years':'35-54','55-74 years':'55-74','75+ years':'75+'},inplace=True)

sns.set_style('whitegrid')

sns.catplot('age','suicides/100k pop',kind='bar',data=suicide.sort_values(by='age'), hue ='sex',palette='coolwarm')

plt.xlim(0,5.5)
sns.distplot(suicide['suicides/100k pop'])
countries = suicide.groupby('country').mean().sort_values(by='suicides/100k pop',ascending=False)['suicides/100k pop']

countries.head(10)
import geopandas as gpd

import geoplot as gplt

#Create a variable holding the map of the world

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

world.plot()
#Merge the map data and our suicide data

merge = world.set_index('name').join(countries,how='inner')

merge.head()
merge.describe()
fig, (ax1,ax2) = plt.subplots(1,2, sharey=True,figsize=(20,20))

ax1.set_title('Suicide Rate')

ax2.set_title('Population')

merge.plot(column='suicides/100k pop',cmap='Reds',ax=ax1)

merge.plot(column='pop_est',cmap='Reds',ax=ax2)
merge[['suicides/100k pop','pop_est']].corr()
sns.distplot(suicide['year'])
#Seperate the feature we wants to predict from the training data

#Drop suicides_no and suicides/100k pop they are dependent variables. We could easily predict the rate of suicides by using suicides_no and population.  

suicide = strat_train.drop(['suicides_no','country-year'],axis=1)

suicide_labels = strat_train['suicides/100k pop']
suicide.info()
from sklearn.preprocessing import LabelEncoder



cat_attribs = suicide[[column for column in suicide.columns if suicide[column].dtype == 'object']]



le = LabelEncoder()



suicide_cat = cat_attribs.apply(lambda col: le.fit_transform(col))



suicide_cat.head(10)

suicide_cat_dummies = pd.get_dummies(suicide, columns=cat_attribs.columns, drop_first=True )

suicide_cat_dummies
1 - suicide['HDI for year'].count()/len(suicide)
sns.distplot(suicide['HDI for year'].dropna())
suicide.describe()['HDI for year']
median = suicide['HDI for year'].median()

filled_HDI = suicide['HDI for year'].fillna(median)

filled_HDI.describe()
sns.distplot(filled_HDI)
from sklearn.preprocessing import Imputer

 

imputer = Imputer(strategy='median')



num_attribs = suicide[suicide.columns[suicide.dtypes != 'object']]



#Since imputer just applies to numerical columns, we drop categorical columns

suicide_num = imputer.fit_transform(num_attribs)

suicide_num = pd.DataFrame(suicide_num,columns=suicide.columns[suicide.dtypes != 'object'])

suicide_num['HDI for year'].describe()



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaled_suicide_num = scaler.fit_transform(suicide_num)



scaled_suicide_num = pd.DataFrame(scaled_suicide_num,columns=suicide.columns[suicide.dtypes != 'object'])



scaled_suicide_num
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



num_pipeline = Pipeline([

    ('imputer', Imputer(strategy='median')),

    ('scaler', StandardScaler(with_mean=False))

])



cat_pipeline = Pipeline([

    ('encoder', OneHotEncoder(handle_unknown='ignore')),

    ('scaler', StandardScaler(with_mean=False))

])



full_pipeline = ColumnTransformer([

    ('num_pipeline', num_pipeline, list(num_attribs.columns)),

    ('cat_pipeline', cat_pipeline, list(cat_attribs.columns)),

])



suicide_prepared = full_pipeline.fit_transform(suicide)

type(suicide_prepared)
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(suicide_prepared,suicide_labels)

lr_predictions = lr.predict(suicide_prepared)

lr_predictions





#Find the mean difference between the predictions and the real values 

(lr_predictions-list(suicide_labels)).mean()
from sklearn.metrics import mean_squared_error



lrmse = np.sqrt(mean_squared_error(suicide_labels,lr_predictions))

lrmse
from sklearn.tree import DecisionTreeRegressor



dr = DecisionTreeRegressor(random_state=0)

dr.fit(suicide_prepared,suicide_labels)

dr_predictions = dr.predict(suicide_prepared)



drmse = np.sqrt(mean_squared_error(suicide_labels,dr_predictions))

drmse
from sklearn.model_selection import cross_val_score



scores_1 = cross_val_score(dr, suicide_prepared, suicide_labels, scoring = "neg_mean_squared_error", cv = 10)

tree_scores = np.sqrt(-scores_1)

tree_scores.mean()
scores_2 = cross_val_score(lr, suicide_prepared, suicide_labels, scoring = "neg_mean_squared_error", cv = 10)

lr_scores = np.sqrt(-scores_2)

lr_scores.mean()
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=30, random_state=42)

forest_reg.fit(suicide_prepared, suicide_labels)



scores_3 = cross_val_score(forest_reg, suicide_prepared, suicide_labels, scoring = "neg_mean_squared_error", cv = 10)

rf_scores = np.sqrt(-scores_3)

rf_scores.mean()
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8, 10]},

    

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4, 6,8 ]},

  ]



forest_reg = RandomForestRegressor(random_state=42)

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error', return_train_score=True,

                          error_score=np.nan)

grid_search.fit(suicide_prepared, suicide_labels) 
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
cvres_df = pd.DataFrame(cvres)

cvres_df["mean_score"] = cvres_df['mean_test_score'].apply(lambda x:np.sqrt(-x) )

cvres_df[["mean_score","params"]].sort_values(by='mean_score')
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_distribs = {

        'n_estimators': randint(low=1, high=200),

        'max_features': randint(low=1, high=10),

    }



forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,

                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

rnd_search.fit(suicide_prepared, suicide_labels)
rnd_search.best_params_

rnd_results = rnd_search.cv_results_

rnd_results = pd.DataFrame(rnd_results)

rnd_results['mean_score'] = rnd_results['mean_test_score'].apply(lambda x: np.sqrt(-x))

rnd_results[["mean_score","params"]].sort_values(by='mean_score')