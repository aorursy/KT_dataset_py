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

os.chdir('/kaggle/input/')

# Any results you write to the current directory are saved as output.
df = pd.read_csv('housing.csv')

df.columns
df['ocean_proximity'].value_counts()
df.info() # For values and the data type and number of the instances 
df.head()
df.describe()
import seaborn as sns

import matplotlib.pyplot as plt

plt.rc('figure',figsize = (12,14))


sns.pairplot(df)
%matplotlib inline

df.hist(bins = 50 , figsize = (20,15))

plt.show()
# Create a dataset



sns.boxplot(data =df)

def split_train_test(data,test_ratio):

    

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    

    return data.iloc[shuffled_indices[:test_set_size]],data.iloc[shuffled_indices[test_set_size:]]

import hashlib



def test_set_check(identifier,test_ratio,hash):

    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio



def split_train_test_by_id( data, test_ratio , id_column , hash = hashlib.md5):

    ids = data[id_column]

    in_test_set = ids.apply(lambda id_: test_set_check(id, test_ratio , hash))

    return data.loc[~in_test_set], data.iloc[in_test_set]



help(df.pipe)
# make stable key - identifier

#train_set,test_set =  df.assign(id = df['longitude'] *1000 + df['latitude']).pipe(split_train_test_by_id,test_ratio = 0.2,id_column = id)
# 3rd method is sklearn.modelselection

from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(df, test_size = 0.2 , random_state = 42)
# most important feature is the medain income 

plt.figure(figsize = (20,7))



plt.hist( train_set.median_income,bins = 50)

plt.xticks(np.arange(0,15,0.5))

plt.show()
0.5/1.5,15/1.5

df['income_cat'] = np.ceil(df['median_income']/1.5)

print(df.income_cat.unique())

df.income_cat.value_counts()/len(df) # % distribution
df.income_cat.where(df['income_cat']< 5, 5.0 , inplace = True)# pandas function where is reverse

df.income_cat.value_counts()/len(df)
try:

    df.drop('ioncome_cat',axis = 1,inplace=True)

except:

    print('It is already draoped')

    

print(df.income_cat.unique())


from sklearn.model_selection import StratifiedShuffleSplit



help(StratifiedShuffleSplit)
# instance of the StratifiedSufflesplit

split = StratifiedShuffleSplit(n_splits= 1 , train_size= 0.8,random_state= 42)



split.get_n_splits(df,df['income_cat'])

for train_index , test_index in split.split(df,df['income_cat']):

    strat_train_set = df.loc[train_index]

    strat_test_set = df.loc[test_index]




strat_test_set


strat_train_set
pd.merge(df['income_cat'].value_counts()/len(df),

         strat_train_set['income_cat'].value_counts()/len(strat_train_set),

         left_index=True,

         right_index=True,

         how='outer',indicator='index')
for set in (strat_train_set,strat_test_set):

    try:

        set.drop(['income_cat'],axis =  1 , inplace = True)

    except:

        print('it is already deleted')

    finally:

        print(strat_train_set.columns)
housing = strat_train_set.copy()


housing.plot(kind = 'scatter',x = 'longitude',y = 'latitude',figsize = (18,8))



housing.plot(kind = 'scatter',x = 'longitude',y = 'latitude',figsize = (18,8),alpha = 0.1)

fig = plt.figure()

housing.plot(kind = 'scatter',x = 'longitude',y='latitude', alpha = 0.4,

        s  = housing['population']/100, label = 'population',

        c = 'median_house_value',

        cmap = plt.get_cmap('jet'),colorbar = True,figsize = (23,10))

plt.ylabel('latitude')

plt.xlabel('longitude')

plt.title('Population size coloured with median house value ',fontdict={'size':32,'color':'blue'})

plt.show()
housing.corr()['median_house_value'].sort_values(ascending = False)
housing_wn_attributes = housing.assign(

    rooms_per_household = housing['total_rooms']/ housing['households'],

    bedrooms_per_room  = housing['total_bedrooms'] / housing['total_rooms'],

    population_per_household = housing['population'] / housing['households'])

pd.merge(housing_wn_attributes.corr()['median_house_value'],

        housing.corr()['median_house_value'],how = 'outer',left_index=True,right_index= True,

        suffixes = ['_new','_old']

       ).sort_values('median_house_value_old',ascending = False)
strat_train_set.columns
housing = strat_train_set.drop('median_house_value',axis = 1)

housing_labels = strat_train_set['median_house_value'].copy()
housing.info()
from sklearn.impute import SimpleImputer



help(SimpleImputer)
imputer  = SimpleImputer(strategy='median')



housing_num = housing.drop('ocean_proximity', axis =1 )



imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)



housing_tr = pd.DataFrame(X,columns= housing_num.columns)

housing_tr
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

housing_cat = housing['ocean_proximity']

housing_cat_encoded =  encoder.fit_transform(housing_cat)

encoder.classes_


housing_cat_encoded
from sklearn.preprocessing import OneHotEncoder



ohc = OneHotEncoder()



housing_cat= ohc.fit_transform(housing_cat_encoded.reshape(-1,1))



housing_cat
housing_cat.toarray()




from sklearn.preprocessing import LabelBinarizer



encoder = LabelBinarizer()

housing_cat_1hot = encoder.fit_transform(housing_cat)

housing_cat_1hot
# Custom Transformaer

from sklearn.base import BaseEstimator,TransformerMixin



rooms_ix,badroom_ix,population_ix, household_ix = 3,4,5,6



class CombineAttributesAdder(BaseEstimator,TransformerMixin):

    

    def __init__(self,add_badrooms_per_room = True):

        self.add_badrooms_per_room = add_badrooms_per_room

        

    def fit(self,X,y=None):

        return self



    def transform(self,X,y=None):

        

        population_per_household = X[:,population_ix]/X[:,household_ix]

        rooms_per_population = X[:,rooms_ix]/X[:,population_ix]

        

        if self.add_badrooms_per_room:

            badrooms_per_room = X[:,badroom_ix]/X[:,rooms_ix]

            return np.c_[X,population_per_household,rooms_per_population,badrooms_per_room]

        else:

            return np.c_[X,population_per_household,rooms_per_population]

    

attr_adder = CombineAttributesAdder(add_badrooms_per_room=False)

housing_extra_attribute = attr_adder.transform(housing.values)

    

housing_extra_attribute
from sklearn.pipeline import Pipeline,FeatureUnion

from sklearn.preprocessing import StandardScaler



class DataBaseSelector(BaseEstimator ,TransformerMixin ):

    

    def __init__(self,attribute):

        self.attribute = attribute

        

    def fit(self,X,y=None):

        return self

    

    def transform(self,X):

        return X[:,self.attribute].values
num_attribs = list(housing_num)

num_attribs
cat_attribs = ['ocean_proximity']



num_pipeline = Pipeline([

    ('imputer',SimpleImputer(strategy = 'median')),

    ('attributes_adder',CombineAttributesAdder()),

    ('standard_scalaer',StandardScaler()),

     ])



# cat_pipeline = Pipeline([

    

#     ('selector',DataBaseSelector(cat_attribs)),

#     ('label_binar',LabelBinarizer()),



# ])



# full_pipeline = FeatureUnion(

#     transformer_list=[

#     ('num_attributes',num_pipeline),

#     ('cat_attributes',cat_pipeline),

#     ])



from sklearn.compose import ColumnTransformer



full_pipeline = ColumnTransformer(

    [

        ('num',num_pipeline,num_attribs),

        ('cat',OneHotEncoder(),cat_attribs)

    ]



)
housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared.shape
housing_labels
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared,housing_labels)

some_data = housing.iloc[:5]

some_labels = housing_labels[:5]

some_data_prepared = full_pipeline.transform(some_data)

lin_reg.predict(some_data_prepared)
some_labels
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error





tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared,housing_labels)



housing_predictions = tree_reg.predict(some_data_prepared)

tree_mse = mean_squared_error(housing_predictions,some_labels)

tree_mse
from sklearn.model_selection import cross_val_score



score = cross_val_score(

    tree_reg,

    housing_prepared,

    housing_labels,

    scoring = 'neg_mean_squared_error',

    cv = 10,

    )



scores  = np.sqrt(-score)



def display_scores(scores):

    print('scores :' ,scores)

    print('Mean :',scores.mean())

    print('Standard Deviation',scores.std())
display_scores(scores)
import sklearn

sklearn.metrics.SCORERS.keys()
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared,housing_labels)





housing_predictions =  forest_reg.predict(some_data_prepared)

forest_mse = mean_squared_error(housing_predictions,some_labels)

np.sqrt(forest_mse)
scores = np.sqrt(-cross_val_score(

    forest_reg,

    housing_prepared,

    housing_labels,

    scoring = 'neg_mean_squared_error',

    cv = 10

))



display_scores(scores)
from sklearn.model_selection import GridSearchCV



param_grid = [

    

    {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},

    {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]}

]



forest_reg = RandomForestRegressor()



grid_search = GridSearchCV(

    

    forest_reg,

    param_grid= param_grid,

    cv = 5,

    scoring= 'neg_mean_squared_error'

    )



grid_search.fit(housing_prepared,housing_labels)
grid_search.best_params_
grid_search.best_estimator_
grid_search.cv_results_
grid_search.cv_results_.keys()
results = grid_search.cv_results_

for mean_scores , params in  zip(results['mean_test_score'],results['params']):

    print(np.sqrt(-mean_scores),' ',params)
from sklearn.model_selection import RandomizedSearchCV



help(np.random)
randomSearchCv = RandomizedSearchCV(

    forest_reg,

    

    param_distributions= {'n_estimators':np.arange(3,30),'max_features':np.arange(2,8)},

    cv = 5,

#    n_jobs = -1,

 #   n_iter= 30

    )#{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
random_search = randomSearchCv.fit(housing_prepared,housing_labels)
random_search.best_params_
feature_importance = grid_search.best_estimator_.feature_importances_

feature_importance
extra_attribs = ['rooms_per_hhold','pop_per_hhold','bedrooms_per_room']
encoder.classes_
cat_one_hot_attribs = list(encoder.classes_)

attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importance,attributes),reverse= True)
final_model = grid_search.best_estimator_

final_model
from sklearn.externals import joblib



#joblib.dump(final_model,'final_model')
strat_test_set.columns
X_test = strat_test_set.drop('median_house_value',axis = 1 )

y_test = strat_test_set['median_house_value'].copy()


X_test_prepared = full_pipeline.transform(X_test)

predictions = final_model.predict(X_test_prepared)

predictions
test_mse = mean_squared_error(predictions,y_test)

np.sqrt(test_mse)
final_model2 = random_search.best_estimator_

final_model2
predictions = final_model2.predict(X_test_prepared)



np.sqrt(mean_squared_error(predictions,y_test))