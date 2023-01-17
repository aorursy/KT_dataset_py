# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # plotting
from math import * # sqrt() etc
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
plt.hist(data.fasting_blood_glucose_mg_dl[~np.isnan(data.fasting_blood_glucose_mg_dl)], bins=50)
plt.title('Blood sugar')
plt.show
plt.boxplot(data.fasting_blood_glucose_mg_dl[~np.isnan(data.fasting_blood_glucose_mg_dl)])
plt.title('Blood sugar')
plt.show
data['fasting_blood_glucose_mg_dl'] = remove_outliers(data,'fasting_blood_glucose_mg_dl')
cols_women = ['marital_status', 'gauna_perfor_not_perfor', 'duration_pregnanacy']
data['marital_status'] = data['marital_status'].where(~(data['marital_status']==8.0), other = np.nan)
# input errors have to be dealt with
plt.boxplot(data['duration_pregnanacy'].dropna())
plt.show
corr=data.corr()[['haemoglobin_level', 'pulse_rate', 'bp_diastolic', 'bp_systolic', 'fasting_blood_glucose_mg_dl']]
corr.where(abs(corr)>0.1)
data_correlated = data.drop(['district_code', 'stratum', 'test_salt_iodine', 'record_code_iodine', 'date_of_birth', 'month_of_birth', 'duration_pregnanacy'], axis = 1)
corr = data_correlated.corr()[['haemoglobin_level', 'pulse_rate', 'bp_diastolic', 'bp_systolic', 'fasting_blood_glucose_mg_dl']]
corr.where(abs(corr)>0.1)
print(data.shape)
data.columns

data.height.value_counts().head()

weird_heights = data.height.value_counts().index[:3].tolist()
data_filter_helper = data.isin(weird_heights)
weird_heights_data = data.loc[data_filter_helper.height]
print(weird_heights_data.shape)

fig = plt.figure(figsize = (10, 30))

for counter, column in enumerate(weird_heights_data.columns): 
    axes= fig.add_subplot(7, 3, 1+ counter)
    axes.bar(weird_heights_data[column].value_counts().index, weird_heights_data[column].value_counts().values)
    axes.set_title(column)  
plt.subplots_adjust(wspace = 0.5)
plt.show()
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
dummied_data = dummied_data.rename(rename_dict, axis = 'columns')
dummied_data.columns[70:]
ft_with_district = [x for x in dummied_data.columns if x.startswith('district')]

fig = plt.figure(figsize = (20, 50))

for counter, d in enumerate(ft_with_district): 
    df= dummied_data.loc[dummied_data[d] == 1].pulse_rate.dropna()
    axes= fig.add_subplot(14, 5, 1+ counter)
    axes.hist(df, density = True, bins = 20)
    axes.set_title(d + ' ' +str(df.shape[0]))
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
plt.show()

df1= dummied_data.loc[dummied_data[ft_with_district[0]] == 1].pulse_rate.dropna()
df2= dummied_data.loc[dummied_data[ft_with_district[25]] == 1].pulse_rate.dropna()
print("A normally distributed pulse_rate has variance", np.var(df1.value_counts().sort_index()))
print("A non normally distributed pulse_rate has variance", np.var(df2.value_counts().sort_index()))

ft_district_droppable = []
for d in ft_with_district: 
    df= dummied_data.loc[dummied_data[d] == 1].pulse_rate.dropna()
    if(np.var(df.value_counts().sort_index()))< 1000: 
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
              'height', 'haemoglobin_level', 'bp_systolic', 'bp_diastolic', 'fasting_blood_glucose_mg_dl', 'duration_pregnanacy',
              'bmi']
dummied_data.drop('date_survey', axis = 1, inplace = True);
ft_cat_no_distr = [x for x in dummied_data.columns if x not in ft_numeric + ft_with_district + ['pulse_rate']]
fig = plt.figure(figsize = (20, 50))
std_dict = {}

for counter, d in enumerate(ft_cat_no_distr): 
    df= dummied_data.loc[dummied_data[d] == 1].pulse_rate.dropna()
    std_dict[d] = (np.std(df.value_counts().sort_index()/np.nanmean(dummied_data.pulse_rate)))
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
    std_dict[f] = (np.std(df.value_counts().sort_index()/np.nanmean(dummied_data.pulse_rate)))
dummied_data['district_signi'] = dummied_data[ft_with_district].sum(axis = 0)
std_dict['district_signi'] = (np.std(df.value_counts().sort_index()/np.nanmean(dummied_data.pulse_rate)))
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
ft_numeric_selected = d.where(d > 0.15)
ft_numeric_selected = ft_numeric_selected.index[np.where(ft_numeric_selected > 0)].tolist()
print("The following features will be kept: ", )
display(ft_numeric_selected)     
print("The following features will be discarded: ")
ft_numeric_discarded = [x for x in ft_numeric if x not in ft_numeric_selected]
display(ft_numeric_discarded)
dummied_data = dummied_data.drop(ft_numeric, axis = 1)
dummied_data_cent = pd.concat([dummied_data, dummied_data_numeric_cent[ft_numeric_selected]], axis = 1)
dummied_data = pd.concat([dummied_data, dummied_data_numeric[ft_numeric_selected]], axis = 1)
print(dummied_data.shape[1], "features left")
ft_female = ['married_no_gauna', 'never_married', 'married_and_gauna', 'remarried', 
             'widow', 'pregnant', 'lactating', 'non_pregnant_non_lactating', 'duration_pregnanacy']
men_data = dummied_data_cent.loc[dummied_data['male'] == 1].drop(ft_female + ['male', 'female'], axis = 1)
fem_data = dummied_data_cent.loc[dummied_data['female']== 1].drop(['male', 'female'], axis = 1)
print(men_data.shape[1], "features for men")
print(fem_data.shape[1], "features for women")
fem_data['age_orig'] = dummied_data.loc[dummied_data.female == 1].age
men_data['age_orig'] = dummied_data.loc[dummied_data.male == 1].age
fig = plt.figure(figsize = (10, 5))
axes = fig.add_subplot(2, 1, 1)
axes.hist(men_data.pulse_rate.dropna(), density = True, bins = 20);
axes.set_title('mens pulse rate, mean ' + str(np.nanmean(men_data.pulse_rate)))
axes = fig.add_subplot(2, 1, 2)
axes.hist(fem_data.dropna(), density = True, bins = 20);
axes.set_title('womans pulse rate, mean ' + str(np.nanmean(fem_data.pulse_rate)))
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
plt.show()
fem_data_postmeno = fem_data.loc[fem_data['age_orig'] >= 50]
plt.hist(fem_data_postmeno.pulse_rate.dropna(), density = True, bins = 20)
plt.title('pulse rate of elder women, mean ' + str(np.nanmean(fem_data_postmeno.pulse_rate)))
fem_data_postmeno.head()
#drop all women that did not give a pulse rate
print(fem_data_postmeno.shape)
fem_data_postmeno = fem_data_postmeno.where(fem_data_postmeno.pulse_rate.notna() == True).dropna(how = 'all')
print(fem_data_postmeno.shape)
def pulse_dange(x): 
    return (x >= 76) * 1
fem_data_postmeno['pulse_rate_dangerous'] = fem_data_postmeno.pulse_rate.apply(pulse_dange)
display(fem_data_postmeno[['pulse_rate', 'pulse_rate_dangerous']].head())
fem_data_postmeno_train = fem_data_postmeno.drop(['pulse_rate', 'pulse_rate_dangerous'], axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(fem_data_postmeno_train.fillna(0),
                                                    fem_data_postmeno.pulse_rate_dangerous)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score as acc
fem_data_postmeno_train.fillna(0).head()
#this will take ca 14min
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc_param_grid = {'n_estimators': np.arange(160, 170, 5), 
                  'criterion': ['entropy', 'gini']}
rfc_grid_search = GridSearchCV(rfc, rfc_param_grid, cv=20, return_train_score=True)
rfc_grid_search.fit(X_train, y_train)
rfc_grid_search.best_params_
print("Accuracy achieved by Random Forest with parameters above: ", 
     acc(y_test, rfc_grid_search.best_estimator_.predict(X_test)))
splits = [x for x in list(rfc_grid_search.cv_results_.keys()) if x.endswith('test_score') and x.startswith('split')]
best_rfc_scores = {}
for counter, x in enumerate(splits): 
    best_rfc_scores[counter]= (rfc_grid_search.cv_results_[x][1])
plt.scatter(best_rfc_scores.keys(), best_rfc_scores.values())
plt.title('variance in performance dependent on split');
imps = rfc_grid_search.best_estimator_.feature_importances_
important_features = [idx for idx in range(len(imps))if imps[idx]>0]
print("There are ", len(important_features), "features used for Random Forest: ")
plt.xticks(rotation='vertical')
plt.bar(fem_data_postmeno_train.columns[important_features], imps[important_features])
men_data_older = men_data.loc[men_data['age_orig'] >= 50]
plt.hist(men_data_older.pulse_rate.dropna(), density = True, bins = 20)
plt.title('pulse rate of elder men, mean ' + str(np.nanmean(men_data_older.pulse_rate)))
plt.show()
#drop all men that did not give a pulse rate
print(men_data_older.shape)
men_data_older = men_data_older.where(men_data_older.pulse_rate.notna() == True).dropna(how = 'all')
print(men_data_older.shape)
men_data_older['pulse_rate_dangerous'] = men_data_older.pulse_rate.apply(pulse_dange)
men_data_older_train = men_data_older.drop(['pulse_rate', 'pulse_rate_dangerous'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(men_data_older_train.fillna(0),
                                                    men_data_older.pulse_rate_dangerous)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
rfc_param_grid = {'n_estimators':[160, 170, 180, 200], 
                  'criterion': ['entropy', 'gini']}
rfc_grid_search = GridSearchCV(rfc, rfc_param_grid, cv=20)
rfc_grid_search.fit(X_train, y_train)
display(rfc_grid_search.best_params_)
print("Accuracy achieved by Random Forest: ", 
      acc(y_test, rfc_grid_search.best_estimator_.predict(X_test)))
imps = rfc_grid_search.best_estimator_.feature_importances_
important_features_men = [idx for idx in range(len(imps))if imps[idx]>0]
print("There are ", len(important_features_men), "features used for Random Forest for men: ")
#fig = plt.figure(figsize = (10, 5))
plt.xticks(rotation='vertical')
plt.bar(men_data_older_train.columns[important_features_men], imps[important_features_men])