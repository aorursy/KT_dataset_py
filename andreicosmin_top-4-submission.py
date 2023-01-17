from IPython.display import Image

Image(url = 'https://scx2.b-cdn.net/gfx/news/2016/mountingtens.jpg')
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import scipy

%matplotlib inline
train_labels = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/train_labels.csv')

test = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/test_values.csv')

train = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/train_values.csv')
train_labels.head()
train_labels.info()
train.head()
train.info()
res = train_labels.building_id.equals(train.building_id)

print("Statement 'all building IDs match is'", res)
train['damage_grade'] = train_labels.damage_grade
train.describe()
fig = plt.subplots(figsize = (9,5))

sns.countplot(train.damage_grade)

plt.show()
pd.value_counts(train.damage_grade)
train.iloc[:,[i for i in range(15, 26)]].head()
train.iloc[:,[i for i in range(28, 39)]].head()
train.has_secondary_use.mean()
total = 0

for i in range(29, 39):

    col = train.columns[i]

    total+=train[col].mean()

print(f'The sum of means of the secondary_use columns is {total}')
plt.hist(train.age)

plt.show()
plt.hist(train.age,range=(0,175), bins = 15)

plt.show()
sns.barplot('damage_grade', 'age', data = train)
sns.barplot('damage_grade', 'land_surface_condition', data= train)
train.drop('building_id', inplace = True, axis = 1) #this column isn't needed
#14- 24 are columns containing superstructure info

superstructure_cols = []

for i in range(14, 25):

    superstructure_cols.append(train.columns[i])
corr = train[superstructure_cols+['damage_grade']].corr()
sns.heatmap(corr)
sns.barplot('damage_grade', 'has_superstructure_adobe_mud', data=train,)
#pearsonr correlation implies normal distribution

#The p-value roughly indicates the probability of an uncorrelated system producing datasets that have a

#Pearson correlation at least as extreme as the one computed from these datasets

scipy.stats.pearsonr(train.damage_grade, train.has_superstructure_mud_mortar_stone)
scipy.stats.pearsonr(train.damage_grade, train.has_superstructure_cement_mortar_brick)
superstructure_cols = []

for i in range(14, 25):

    superstructure_cols.append(train.columns[i])
secondary_use = []

for i in range(27, 37):

    secondary_use.append(train.columns[i])
corr = train[secondary_use +['damage_grade']].corr()
sns.heatmap(corr)
additional_num_data = []

for i in range(7):

    additional_num_data.append(train.columns[i])

additional_num_data.append(train.columns[26])
corr = train[additional_num_data+['damage_grade']].corr()
sns.heatmap(corr)
train.dtypes.value_counts()
print('Object data types:\n')

#we'll use the function later, without wanting to print anything

def get_obj(train, p = False):

    obj_types = []

    for column in train.columns:

        if train[column].dtype == 'object': 

            if p: print(column)

            obj_types.append(column)

    return obj_types

obj_types = get_obj(train, True)
def transform_to_int(train, obj_types):

    #Assign dictionaries with current values and replacements for each column

    d_lsc = {'n':0, 'o':1, 't':2}

    d_ft = {'h':0, 'i':1, 'r':2, 'u':3, 'w':4}

    d_rt = {'n':0, 'q':1, 'x':2}

    d_gft = {'f':0, 'm':1, 'v':2, 'x':3, 'z':4}

    d_oft = {'j':0, 'q':1, 's':2, 'x':3}

    d_pos = {'j':0, 'o':1, 's':2, 't':3}

    d_pc = {'a':0, 'c':1, 'd':2, 'f':3, 'm':4, 'n':5, 'o':6, 'q':7, 's':8, 'u':9}

    d_los = {'a':0, 'r':1, 'v':2, 'w':3}

    #Each positional index in replacements corresponds to the column in obj_types

    replacements = [d_lsc, d_ft, d_rt, d_gft, d_oft, d_pos, d_pc, d_los]



    #Replace using lambda Series.map(lambda)

    for i,col in enumerate(obj_types):

        train[col] = train[col].map(lambda a: replacements[i][a]).astype('int64')

transform_to_int(train, obj_types)
train.dtypes.value_counts()
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
#separate column to be predicted from the rest

y = train.pop('damage_grade') 

x = train.copy()
x_train, x_test, y_train, y_test = train_test_split(x, y)



rcf = RandomForestClassifier()

model = rcf.fit(x_train, y_train)



model.score(x_test, y_test)
y_pred = model.predict(x_test)
f1_score(y_test, y_pred,average='micro')
def get_conf_matrix(y_test, y_pred):    

    data = confusion_matrix(y_test, y_pred) #get confusion matrix

    cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test)) #build the confusion matrix as a dataframe table

    cm.index.name = 'Observed'

    cm.columns.name = 'Predicted'

    plt.figure(figsize = (10,7))

    sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 12}) #plot a heatmap

    plt.title("Confusion Matrix")

    plt.show()

get_conf_matrix(y_test, y_pred)
importance = pd.DataFrame({"Feature":list(x), "Importance": rcf.feature_importances_}) # build a dataframe with features and their importance

importance = importance.sort_values(by="Importance", ascending=False) #sort by importance

importance
#import the fitting methods to try

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier





classifiers = [

    KNeighborsClassifier(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier()]



def model_and_test(X, y, classifiers):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

    for model in classifiers:

        this_model = model.__class__.__name__ #get the name of the classifier

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        

        print(f'{this_model} f1 score:')

        score = f1_score(y_test, y_pred,average='micro')

        print(f'{score:.4f}')

        print('\n')
model_and_test(x, y, classifiers)
boxplot_cols=["geo_level_3_id","geo_level_2_id","geo_level_1_id","age", "area_percentage", "height_percentage"]

q=1

plt.figure(figsize=(20,20))

for j in boxplot_cols:

    plt.subplot(3,3,q)

    ax=sns.boxplot(train[j].dropna())

    plt.xlabel(j)

    q+=1

plt.show()
def remove_outliers(df, col_cutoff = 0.01, z_score = 3.5): #define a function to get rid of all outliers of the most important columns

    important_cols = importance[importance.Importance>col_cutoff]['Feature'].tolist() #get all columns with importance > 0.01.  

    df_new = df.copy() #init the new df

    for col in important_cols: df_new = df_new[np.abs(scipy.stats.zscore(df_new[col]))<z_score] #removing all rows where a z-score is >3

    return df_new
df = pd.concat([x, y], axis = 1)
df_new = remove_outliers(df)
y = df_new.pop('damage_grade')

x=df_new
sns.countplot(y)
import xgboost as xgb
#Bring up original data

def get_original():

    df = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/train_values.csv')

    df.drop('building_id', axis =1, inplace=True)

    obj_types = get_obj(df)

    transform_to_int(df, obj_types)

    df['damage_grade'] = train_labels.damage_grade



    return df

df = get_original()



# a function that will later be used to divide dataframe into x(independent variables) and y(dependent variable)

def get_xy(df):

    y = df.pop('damage_grade')

    x= df

    return x, y
y = df.damage_grade

x = df.drop('damage_grade', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
pd.value_counts(y_test) #to confirm that the original proportion of damage grades is preserved
def test_model(model, removing = False, col_cutoff = 0.01, z_score = 3.5):

    df_train = pd.concat([x_train, y_train], axis = 1) #combine them together, so outliers are simultaneously removed from both

    if removing: df_train = remove_outliers(df_train, col_cutoff, z_score) 

    x, y =get_xy(df_train)

    model.fit(x, y)



    y_pred = model.predict(x_test)

    print(f1_score(y_test, y_pred, average='micro'))

test_model(xgb.XGBRFClassifier())
models = [xgb.XGBRFClassifier(), xgb.XGBClassifier()]

for model in models:

    print(model.__class__.__name__, 'score:',end =' ')

    test_model(model, True)
xgbc = xgb.XGBClassifier()
'''

xgbc = xgb.XGBClassifier() #init xgbc 

for a in [0.01, 0.02, 0.05]:

    for b in [2.5, 3, 3.5]:

        print('removing outliers on columns with importance >,',a,'on z scores >',b,'. Score =', end=' ')

        test_model(xgbc, True, a, b) '''
'''for b in [2.5, 3, 3.5]:

    print('removing outliers on columns with importance > 0.1,','on z scores >',b,'. Score =', end=' ')

    test_model(xgbc, True, 0.1, b)'''
print('No outlier removal score:', end = ' ')

test_model(xgbc, False)
x, y = get_xy(df)
df_train = pd.concat([x, y], axis = 1) #combine them together, so outliers are simultaneously removed from both x and y

df_train = remove_outliers(df_train, 0.1, 3)

x, y =get_xy(df_train)
xgbc = xgb.XGBRFClassifier()
parameters = {'max_depth' : [5, 10, 20, 40]} #first looking for an optimal max_depth
from sklearn.model_selection import GridSearchCV

#grid search cv tries all the parameters individually using cross validation, default set to 5 folds

grid_search = GridSearchCV(xgbc, parameters, scoring="f1_micro", n_jobs=-1, verbose=3)

# grid_result = grid_search.fit(x, label_encoded_y)
def plot_score(grid_result, parameters, name):    

    means = grid_result.cv_results_['mean_test_score'] #get the means of the scores from the 5 folds

    stds = grid_result.cv_results_['std_test_score'] #standard error of scores for plotting error bars



    # plot scores vs parameter

    plt.errorbar(parameters[name], means, yerr=stds)

    pyplot.xlabel(name)

    pyplot.ylabel('f1 score')

#plot_score(grid_result,parameters, 'max_depth')
Image(url = 'https://imgur.com/16Lya4M.png')
xgbc = xgb.XGBRFClassifier(max_depth = 20)
n_estimators = [50, 100, 150, 200]

param2 = {'n_estimators':n_estimators}
grid_search_estimators = GridSearchCV(xgbc, param2, scoring="f1_micro", n_jobs=-1, verbose=3)

# grid_result_estimators = grid_search.fit(x, label_encoded_y)
# plot_score(grid_result_estimators,param2, 'n_estimators')
Image(url = 'https://i.imgur.com/FwgVGgk.png')
xgbc = xgb.XGBClassifier(max_depth = 20, n_estimators = 150)
params={

 "learning_rate"    : [0.1, 0.2, 0.3] ,

 "min_child_weight" : [ 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.05, 0.1, 0.2 , 0.3],

 "colsample_bylevel" :[0.2, 0.5, 0.8, 1.0],

 "colsample_bynode": [0.2, 0.5, 0.8, 1.0],

 "subsample": [0.2, 0.5, 0.8, 1.0],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

    }
from sklearn.model_selection import RandomizedSearchCV

rand_search = RandomizedSearchCV(xgbc,param_distributions=params ,n_iter=10,scoring='f1_micro',n_jobs=-1, verbose = 3)
# rand_res = rand_search.fit(x, y)
# best_params = rand_res.best_params_
best_params = {'subsample': 0.8,

 'min_child_weight': 5,

 'learning_rate': 0.1,

 'gamma': 0.05,

 'colsample_bytree': 0.3,

 'colsample_bynode': 0.8,

 'colsample_bylevel': 0.8}
xgbc = xgb.XGBClassifier( min_child_weight= 5, learning_rate= 0.1, gamma= 0.05, subsample= 0.8,colsample_bytree= 0.3, colsample_bynode= 0.8,

 colsample_bylevel= 0.8, max_depth = 20, n_estimators = 150)



# xgbc.fit(x, y) #final model
def submit_model(model, file_name): #I defined a function because I was submitting multiple models

    test = pd.read_csv('../input/richters-predictor-modeling-earthquake-damage/test_values.csv') #get the test csv into a dataframe

    submission_ids = test.pop('building_id') #get the building ids

    transform_to_int(test, get_obj(test)) #transform obj_types to int to predict damage grades

    submission_predictions = model.predict(test) #predict

    subbmission = pd.DataFrame({'building_id':submission_ids, 'damage_grade':submission_predictions}) #save buildings_ids and predicted damage grades to a data frame

    subbmission.to_csv(file_name, index = False) #save as a csv file
# submit_model(xgbc, 'submission_xgb4.csv')

#0.7477 score