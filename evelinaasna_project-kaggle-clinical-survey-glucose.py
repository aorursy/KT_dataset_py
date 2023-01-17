# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # plotting
from math import * # sqrt() etc
import seaborn as sns
# with %matplotlib inline you turn on the immediate display.
# %matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_dictionary_loc = '../input/CAB_data_dictionary.xlsx'
data_dic = pd.read_excel(data_dictionary_loc, dtype = object)
data_dic['File Content Description'] #well, how to import the correct column width? can be viewed using other programs
data_dic
data_u_pradesh = pd.read_csv('../input/CAB_09_UP.csv', low_memory = False) 
#needed to specify low_memory because columns (14, 43 had mixed types)
data_u_pradesh.head()
data = data_u_pradesh[(data_u_pradesh['age_code']=='Y')&(data_u_pradesh['age']>=18)]
len(data)
data = data.replace([-1, '-1'], np.nan)
cols_under5 = ['illness_type', 'illness_duration', 'treatment_type']
cols_under3 = ['first_breast_feeding', 'is_cur_breast_feeding', 'day_or_month_for_breast_feeding_', 'day_or_month_for_breast_feeding', 'water_month', 'ani_milk_month', 'semisolid_month_or_day', 'solid_month', 'vegetables_month_or_day']
data = data.drop(cols_under5, axis = 1)
data = data.drop(cols_under3, axis = 1)
data = data.drop(['state_code', 'psu_id', 'ahs_house_unit', 'house_hold_no', 'record_code_iodine_reason', 'sl_no', 'usual_residance', 'usual_residance_reason', 'identification_code', 'v54'], axis = 1)
data = data.drop('rural_urban', axis = 1)
display(np.unique(data['age_code']))
data = data.drop('age_code', axis = 1)
plt.hist(data.age.dropna(), bins = 50)
plt.title('Age')
plt.show
pd.value_counts(data['record_code_iodine'])
data = data.drop(['weight_measured', 'length_height_measured', 'length_height_code'], axis = 1)
data = data.rename(index=str, columns={"weight_in_kg": "weight", "length_height_cm": "height"})
plt.boxplot(data['weight'].dropna())
plt.title('Weight with outliers')
plt.show
plt.boxplot(data['height'].dropna())
plt.title('Height with outliers')
plt.show
# exclude any measurements where difference from median is larger than 3 standard deviations
def remove_outliers(data, feature):
    stdev = sqrt(np.var(data[feature].dropna()))
    median = np.median(data[feature].dropna())
    print("number of discarded measurements")
    display(len(data[[feature]].where(abs(data[feature] - median)>(3*stdev)).dropna()))
# keep original values if difference from mean is less than 3 standard deviations. NA otherwise
    return data[[feature]].where(abs(data[feature] - median)<(3*stdev), other = np.nan)
data['height'] = remove_outliers(data, 'height')
print('number of discarded measurements')
display(len(data[data['weight']<20]))
data['weight'] = data['weight'].where(data['weight']>20, other=np.nan)
plt.boxplot(data['weight'].dropna())
plt.title('Weight without outliers')
plt.show
plt.boxplot(data['height'].dropna())
plt.title('Height without outliers')
plt.show
data['bmi'] = data['weight']/(data['height']/100)**2
plt.hist(data['weight'].dropna(), bins = 50)
plt.title('Weight without outliers')
plt.show()
plt.hist(data['height'].dropna(), bins = 50)
plt.title('Height without outliers')
plt.show()
plt.hist(data['bmi'].dropna(), bins = 50)
plt.title('BMI')
plt.show()
# distribution of measurement differences
#plt.hist((data['bp_systolic'] - data['bp_systolic_2_reading']).dropna(), bins = 50)
#plt.hist((data['pulse_rate'] - data['pulse_rate_2_reading']).dropna(), bins = 50)
#plt.hist((data['bp_diastolic'] - data['bp_diastolic_2reading']).dropna(), bins = 50)
# for features where two measurements were taken, exclude any where difference between measurements is larger than 3 standard deviations
def remove_outliers_difference(data, col1, col2):
    stdev = sqrt((data[col1] - data[col2]).var())
# how many measurements were excluded
    print('number of discarded measurements')
    display(len(data[[col1, col2]].where(abs(data[col1] - data[col2])>(3*stdev)).dropna()))
# keep original values if difference of two measurements is less than 3 standard deviations. NA otherwise
    return data[[col1, col2]].where(abs(data[col1] - data[col2])<(3*stdev), other = np.nan)
data[['bp_systolic', 'bp_systolic_2_reading']] = remove_outliers_difference(data, 'bp_systolic', 'bp_systolic_2_reading')
data[['bp_diastolic', 'bp_diastolic_2reading']] = remove_outliers_difference(data, 'bp_diastolic', 'bp_diastolic_2reading')
data[['pulse_rate', 'pulse_rate_2_reading']] = remove_outliers_difference(data, 'pulse_rate', 'pulse_rate_2_reading')
# aggregate two reading by finding mean
def aggregate_readings(data, col1, col2):
    data[col1] = data.apply(lambda row: sum([row[col1], row[col2]])/2, axis = 1)
    data = data.drop(col2, axis = 1)
    return data
data = aggregate_readings(data, 'bp_systolic', 'bp_systolic_2_reading')
data = aggregate_readings(data, 'bp_diastolic', 'bp_diastolic_2reading')
data = aggregate_readings(data, 'pulse_rate', 'pulse_rate_2_reading')
# retain original values where resting blood pressure lower than beating. NA otherwise 
data[['bp_diastolic', 'bp_systolic']] = data[['bp_diastolic', 'bp_systolic']].where(data.bp_diastolic < data.bp_systolic, other = np.nan)
data = data.drop(['haemoglobin_test', 'haemoglobin'], axis = 1)
plt.hist(data.haemoglobin_level[~np.isnan(data.haemoglobin_level)], bins=50)
plt.title('Blood haemoglobin')
plt.show
data = data.drop(['diabetes_test', 'fasting_blood_glucose'], axis = 1)
data = data.rename(index = str, columns = {'fasting_blood_glucose_mg_dl' : 'glucose'})
plt.hist(data.glucose[~np.isnan(data.glucose)], bins=50)
plt.title('Blood sugar')
plt.show
plt.boxplot(data.glucose[~np.isnan(data.glucose)])
plt.title('Blood sugar')
plt.show
data['glucose'] = remove_outliers(data,'glucose')
cols_women = ['marital_status', 'gauna_perfor_not_perfor', 'duration_pregnanacy']
data['marital_status'] = data['marital_status'].where(~(data['marital_status']==8.0), other = np.nan)
# input errors have to be dealt with
plt.boxplot(data['duration_pregnanacy'].dropna())
plt.show
corr=data.corr()[['haemoglobin_level', 'pulse_rate', 'bp_diastolic', 'bp_systolic', 'glucose']]
corr.where(abs(corr)>0.1)
data_correlated = data.drop(['district_code', 'stratum', 'test_salt_iodine', 'record_code_iodine', 'date_of_birth', 'month_of_birth', 'duration_pregnanacy'], axis = 1)
corr = data_correlated.corr()[['haemoglobin_level', 'pulse_rate', 'bp_diastolic', 'bp_systolic', 'glucose']]
corr.where(abs(corr)>0.1)
dummieable =['district_code', 'stratum', 'record_code_iodine', 'sex', 'marital_status', 'gauna_perfor_not_perfor']
dummiedata = [data]
for dum in dummieable: 
    dummiedata.append(pd.get_dummies(data[dum], prefix = dum))
dummied_data = pd.concat(dummiedata, axis = 1)
print("Number of features", len(dummied_data.columns))
dummied_data.columns
dummied_data = dummied_data.drop(dummieable, axis =1)
print("Number of features", len(dummied_data.columns))
dummied_data.columns
rename_dict = {'marital_status_1.0': 'never_married', 'marital_status_2.0': 'married_no_gauna',
               'marital_status_3.0': 'married_and_gauna',
       'marital_status_4.0': 'remarried', 'marital_status_5.0': 'widow', 'marital_status_6.0': 'divorced',
       'marital_status_7.0': 'separated', 'gauna_perfor_not_perfor_1.0': 'pregnant',
       'gauna_perfor_not_perfor_2.0': 'lactating', 'gauna_perfor_not_perfor_3.0': 'non_pregnant_non_lactating',
        'sex_1': 'male', 'sex_2': 'female'}
dummied_data = dummied_data.rename(rename_dict, axis = 'columns').drop(['female'], axis = 1)
dummied_data.columns[70:]
ft_without_district = [x for x in dummied_data.columns if not x.startswith('district')]
ft_without_district.remove('glucose')
print("These", len(ft_without_district),"features are going to be compared to diabetes: \n")
print(ft_without_district)
dummied_data['diabetes'] = dummied_data['glucose'].apply(lambda x: 1 if x >= 100 else 0)
data['diabetes'] = data['glucose'].apply(lambda x: 1 if x >= 100 else 0)
dummied_data.diabetes.value_counts()
# make pairplots of each feature and blood glucose
import seaborn as sns
sns.pairplot(data.dropna(), x_vars = data.columns.drop(['diabetes', 'glucose'])[0:4] , y_vars = ['glucose'])
sns.pairplot(data.dropna(), x_vars = data.columns.drop(['diabetes', 'glucose'])[4:8] , y_vars = ['glucose'])
sns.pairplot(data.dropna(), x_vars = data.columns.drop(['diabetes', 'glucose'])[8:12] , y_vars = ['glucose'])
sns.pairplot(data.dropna(), x_vars = data.columns.drop(['diabetes', 'glucose'])[12:16] , y_vars = ['glucose'])
sns.pairplot(data.dropna(), x_vars = data.columns.drop(['diabetes', 'glucose'])[16:20] , y_vars = ['glucose'])
# getting relative frequency of high blood sugar
def diabetes_relative_freq(feature):
    subset = data.groupby(feature)
    high = pd.Series()
    for i in np.unique(data[feature]):
        high = high.append(pd.Series((subset['diabetes'].value_counts()[i]/sum(subset['diabetes'].value_counts()[i])).loc[1]))
    high.index = np.arange(1,len(subset)+1)  
    plt.bar(np.arange(1, len(subset)+1), high)
    plt.ylabel("Realtive freq of high blood sugar")
    plt.title(feature)
    plt.show
plt.rcParams['figure.figsize'] = [35, 15]
diabetes_relative_freq("district_code")
plt.rcParams['figure.figsize'] = [6, 6]
diabetes_relative_freq("stratum")
plt.rcParams['figure.figsize'] = [6, 6]
diabetes_relative_freq('sex')
ft_with_district = [x for x in dummied_data.columns if x.startswith('district')]

fig = plt.figure(figsize = (20, 50))

for counter, d in enumerate(ft_with_district): 
    df= dummied_data.loc[dummied_data[d] == 1].glucose.dropna()
    axes= fig.add_subplot(14, 5, 1+ counter)
    axes.hist(df, density = True, bins = 20)
    axes.set_title(d + ' ' +str(df.shape[0]))
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
plt.show()
var = []
for dist in ft_with_district:
    df= dummied_data.loc[dummied_data[dist] == 1].glucose.dropna()
    var.append(np.var(df.value_counts().sort_index()))
print("The lowest glucose distribution variance: ", np.min(var))
print("The highest glucose distribution variance ", np.max(var))

ft_district_droppable = []
for d in ft_with_district: 
    df= dummied_data.loc[dummied_data[d] == 1].glucose.dropna()
    if(np.var(df.value_counts().sort_index()))< 2000: 
        ft_district_droppable.append(d)
print("The following districts will be discarded: ", ft_district_droppable)
print(dummied_data.shape[1], "features before")
dummied_data.drop(ft_district_droppable, axis = 1, inplace = True)
print(dummied_data.shape[1], "features left")
#transform survey date into year and month
def parse(string):
    return int(string[6:])*10000 + int(string[3:5])*100 + int(string[:2])
dummied_data['year_month_day_survey'] = dummied_data.date_survey.apply(parse)
display(dummied_data[['date_survey', 'year_month_day_survey']].head(10))

ft_numeric = ['year_month_day_survey','test_salt_iodine', 'age', 'date_of_birth', 'month_of_birth', 'year_of_birth', 'weight', 
              'height', 'haemoglobin_level', 'bp_systolic', 'bp_diastolic', 'duration_pregnanacy',
              'bmi', 'pulse_rate']

dummied_data.drop('date_survey', axis = 1, inplace = True);
dummied_data['female'] = dummied_data['male'].apply(lambda x: 1 if x == 0 else 0)
ft_cat_no_distr = [x for x in dummied_data.columns if x not in ft_numeric + ft_with_district + ['glucose'] + ['diabetes']]
fig = plt.figure(figsize = (20, 50))
std_dict = {}

for counter, d in enumerate(ft_cat_no_distr): 
    df= dummied_data.loc[dummied_data[d] == 1].glucose.dropna()
    std_dict[d] = (np.std(df.value_counts().sort_index()/np.nanmean(dummied_data.glucose)))
    axes= fig.add_subplot(14, 5, 1+ counter)
    axes.hist(df, density = True, bins = 20)
    axes.set_title(d + ' ' +str(df.shape[0]))
    
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
plt.show()
std_dict
dummied_data['stratum_1_2'] = dummied_data['stratum_1'] + dummied_data.stratum_2
print(np.nanstd(dummied_data.stratum_1_2.value_counts().sort_index())/np.nanmean(dummied_data.pulse_rate))
dummied_data.drop(['stratum_1', 'stratum_2'], axis = 1, inplace = True)
ft_with_district = [x for x in dummied_data.columns if x.startswith('district')]
std_dict = {}
for f in ft_with_district: 
    df= dummied_data.loc[dummied_data[f] == 1].pulse_rate.dropna()
    std_dict[f] = (np.std(df.value_counts().sort_index()/np.nanmean(dummied_data.glucose)))
dummied_data['district_signi'] = dummied_data[ft_with_district].sum(axis = 0)
std_dict['district_signi'] = (np.std(df.value_counts().sort_index()/np.nanmean(dummied_data.glucose)))
std_dict
stds = sorted(list(std_dict.values()))
std_treshold = stds[np.round(int(len(stds)*3/4))]
for f in ft_with_district: 
    if std_dict[f] < std_treshold: 
        std_dict.pop(f)
dummied_data.drop([x for x in dummied_data[ft_with_district].columns if x not in std_dict.keys()], axis = 1, inplace= True)
print(dummied_data.shape[1], "features left")
#only apply on numerical features, as categorical ones have too many zeros and might be discarded

dummied_data_numeric = dummied_data[ft_numeric]
#centralize data
dummied_data_numeric_cent = (dummied_data_numeric-dummied_data_numeric.mean())
#normalize by mean to get relative information of the feature
d = (dummied_data_numeric_cent.apply(np.nanstd, axis= 0))/dummied_data_numeric.mean()
print("This is the normalized standard deviation: ")
display(d)
ft_numeric_selected = d.where(d > 0.11)
ft_numeric_selected = ft_numeric_selected.index[np.where(ft_numeric_selected > 0)].tolist()
print("The following features will be kept: ", )
display(ft_numeric_selected)     
print("The following features will be discarded: ")
ft_numeric_discarded = [x for x in ft_numeric if x not in ft_numeric_selected]
display(ft_numeric_discarded)
dummied_data['age_orig'] = dummied_data['age']
dummied_data = dummied_data.drop(ft_numeric, axis = 1)
dummied_data = pd.concat([dummied_data, dummied_data_numeric_cent[ft_numeric_selected]], axis = 1)
print(dummied_data.shape[1], "features left")
dummied_data.columns
dummied_data.drop('district_signi', axis = 1, inplace = True)
cols_women = ['never_married', 'married_no_gauna', 'married_and_gauna', 'remarried', 'widow', 'divorced', 'separated', 'pregnant', 'lactating', 'non_pregnant_non_lactating', 'duration_pregnanacy']
data_over45 = dummied_data.where(dummied_data['age_orig']>=45)
data_over45.drop('age_orig', axis = 1, inplace = True)

men_over45 = data_over45.where(dummied_data.male == 1).drop(cols_women, axis = 1).dropna()
women_over45 = data_over45.where(dummied_data.male == 0).dropna()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(men_over45.drop(['diabetes', 'glucose'], axis = 1).fillna(0),
                                                    men_over45.diabetes)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score as acc

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc_param_grid = {'n_estimators': np.arange(160, 170, 1), 
                  'criterion': ['entropy', 'gini']}
rfc_grid_search = GridSearchCV(rfc, rfc_param_grid, cv=20, return_train_score=True)
rfc_grid_search.fit(X_train, y_train)
rfc_grid_search.best_params_
print("Accuracy achieved by Random Forest with parameters above: ", 
     acc(y_test, rfc_grid_search.best_estimator_.predict(X_test)))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(women_over45.drop(['diabetes', 'glucose'], axis = 1).fillna(0),
                                                    women_over45.diabetes)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score as acc

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc_param_grid = {'n_estimators': np.arange(160, 170, 1), 
                  'criterion': ['entropy', 'gini']}
rfc_grid_search = GridSearchCV(rfc, rfc_param_grid, cv=20, return_train_score=True)
rfc_grid_search.fit(X_train, y_train)
rfc_grid_search.best_params_
print("Accuracy achieved by Random Forest with parameters above: ", 
     acc(y_test, rfc_grid_search.best_estimator_.predict(X_test)))
data_pregnant = dummied_data.where(dummied_data.pregnant == 1).dropna(how = "all")
# centralized age is used for modeling
data_pregnant.drop(['age_orig'], axis = 1, inplace = True)
#data_pregnant['diabetes'] = data_pregnant.glucose.apply(lambda x : 1 if x > 92 else 0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_pregnant.drop(['diabetes', 'glucose'], axis = 1).fillna(0),
                                                    data_pregnant.diabetes)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score as acc

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc_param_grid = {'n_estimators': np.arange(160, 170, 1), 
                  'criterion': ['entropy', 'gini']}
rfc_grid_search = GridSearchCV(rfc, rfc_param_grid, cv=20, return_train_score=True)
rfc_grid_search.fit(X_train, y_train)
rfc_grid_search.best_params_
print("Accuracy achieved by Random Forest with parameters above: ", 
     acc(y_test, rfc_grid_search.best_estimator_.predict(X_test)))
data_pregnant = dummied_data.where(dummied_data.pregnant == 1).dropna(how = "all")
# centralized age is used for modeling
data_pregnant.drop(['age_orig'], axis = 1, inplace = True)
data_pregnant['diabetes'] = data_pregnant.glucose.apply(lambda x : 1 if x > 92 else 0)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score as acc

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc_param_grid = {'n_estimators': np.arange(160, 170, 1), 
                  'criterion': ['entropy', 'gini']}
rfc_grid_search = GridSearchCV(rfc, rfc_param_grid, cv=20, return_train_score=True)
rfc_grid_search.fit(X_train, y_train)
rfc_grid_search.best_params_
print("Accuracy achieved by Random Forest with parameters above: ", 
     acc(y_test, rfc_grid_search.best_estimator_.predict(X_test)))