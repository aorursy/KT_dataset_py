# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt # plotting
from math import * # sqrt() etc
# with %matplotlib inline you turn on the immediate display.
# %matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings("ignore")
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
data['diabetes'] = data['glucose'].apply(lambda x: 1 if x >= 100 else 0)
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
# get an impression of how many values are there again
print("About how many instances are we talking?\n")
print(data.height.value_counts().head())
weird_heights = data.height.value_counts().index[:3].tolist()
data_filter_helper = data.isin(weird_heights)
weird_heights_data = data.loc[data_filter_helper.height]
print("This affects ", weird_heights_data.shape[0], "instances. ")
print("That is ", weird_heights_data.shape[0]*100/data.shape[0], "% of our data in total. ")
weird_heights_data.describe()
fig = plt.figure(figsize = (10, 30))

for counter, column in enumerate(weird_heights_data.columns): 
    axes= fig.add_subplot(7, 4, 1+ counter)
    axes.bar(weird_heights_data[column].value_counts().index, weird_heights_data[column].value_counts().values)
    axes.set_title(column)  
plt.subplots_adjust(wspace = 0.5)
plt.show()
dummieable =['district_code', 'stratum', 'record_code_iodine', 'sex', 'marital_status', 'gauna_perfor_not_perfor']
dummiedata = [data]
for dum in dummieable: 
    dummiedata.append(pd.get_dummies(data[dum], prefix = dum))
dummied_data = pd.concat(dummiedata, axis = 1)
print("Number of features now: ", len(dummied_data.columns))
dummied_data.columns
dummied_data = dummied_data.drop(dummieable, axis =1)
print("Number of features after making categorical numeric: ", len(dummied_data.columns))
rename_dict = {'marital_status_1.0': 'never_married', 'marital_status_2.0': 'married_no_gauna',
               'marital_status_3.0': 'married_and_gauna',
       'marital_status_4.0': 'remarried', 'marital_status_5.0': 'widow', 'marital_status_6.0': 'divorced',
       'marital_status_7.0': 'separated', 'gauna_perfor_not_perfor_1.0': 'pregnant',
       'gauna_perfor_not_perfor_2.0': 'lactating', 'gauna_perfor_not_perfor_3.0': 'non_pregnant_non_lactating',
        'sex_1': 'male', 'sex_2': 'female'}
dummied_data = dummied_data.rename(rename_dict, axis = 'columns')
def parse(string):
    return int(string[6:])*10000 + int(string[3:5])*100 + int(string[:2])
dummied_data['year_month_day_survey'] = dummied_data.date_survey.apply(parse)
display(dummied_data[['date_survey', 'year_month_day_survey']].head(10)) #show how encoding looks like
dummied_data.drop('date_survey', axis = 1, inplace = True); #remove the original encoding
ft_numeric = ['year_month_day_survey','test_salt_iodine', 'age', 'date_of_birth', 'month_of_birth', 'year_of_birth', 'weight', 
              'height', 'haemoglobin_level', 'bp_systolic', 'bp_diastolic', 'glucose', 'duration_pregnanacy',
              'bmi', 'pulse_rate']
print("before: ")
display(dummied_data[ft_numeric].head())
#scale data to unit variance
cols = ["std_"+ x for x in  dummied_data[ft_numeric].columns]
dummied_data_numeric_std = pd.DataFrame(StandardScaler(with_mean = True).fit_transform(dummied_data[ft_numeric]), 
                                    columns = cols, index = dummied_data.index)
print("after scaling and centralizing: ")
dummied_data_numeric = dummied_data[ft_numeric]
dummied_data_numeric.head()
dummied_data_numeric_std.hist(figsize = (20, 20));
dummied_data_std = pd.concat([dummied_data.drop(ft_numeric, axis = 1), dummied_data_numeric_std], axis = 1)
dummied_data_std.head()
print("Missing values in both dummied data sets: ")
for c in dummied_data.columns: 
    nan_count = sum(dummied_data[c].isna())
    if(nan_count > 0):
        print(c, nan_count)
def drop_null_targets(data, target): 
     return data[data[target].notnull()]

data_anemia = drop_null_targets(dummied_data, 'haemoglobin_level')
data_anemia_std = drop_null_targets(dummied_data_std, 'std_haemoglobin_level')
data_glucose = drop_null_targets(dummied_data, 'glucose')
data_glucose_std = drop_null_targets(dummied_data_std, 'std_glucose')
data_heart = drop_null_targets(dummied_data, 'pulse_rate')
data_heart_std = drop_null_targets(dummied_data_std, 'std_pulse_rate')
#for control
print(dummied_data.shape)
print(dummied_data_std.shape)

anemia_relevant = ['stratum_0', 'stratum_1', 'stratum_2',"test_salt_iodine",'record_code_iodine_1', 'record_code_iodine_2', 
                   'record_code_iodine_3',"age","weight","height","haemoglobin_level","bp_systolic",
                   "bp_diastolic","pulse_rate","glucose","duration_pregnanacy","bmi", 'male', 'female']
temp = [x for x in anemia_relevant if not np.isin(x, ft_numeric)]
temp2 = [x for x in anemia_relevant if np.isin(x, ft_numeric)]
anemia_relevant_std = temp + ["std_" + x for x in temp2]
data_anemia_red = data_anemia[anemia_relevant]
data_anemia_std_red = data_anemia_std[anemia_relevant_std]

display(data_anemia_red.head())
display(data_anemia_red.describe(include = 'all'))
display(data_anemia_std_red.head())
display(data_anemia_std_red.describe(include = 'all'))
data_anemia_red['anemia'] = np.where(((data_anemia_red['male'] == 1) & (data_anemia_red['haemoglobin_level'] < 13.0)) |
                                    ((data_anemia_red['female'] == 1) & (data_anemia_red['haemoglobin_level'] < 12.0)), 1, 0)
data_anemia_std_red['anemia'] = data_anemia_red.anemia
display(data_anemia_std_red.head()) #no need to find the borders here.
data_anemia_red.head()
data_anemia_red = data_anemia_red.drop('haemoglobin_level', axis = 1)
data_anemia_std_red = data_anemia_std_red.drop('std_haemoglobin_level', axis = 1)
def impute_data(data):
    imputer = SimpleImputer()#fill up with mean
    data_i= pd.DataFrame(imputer.fit_transform(data), columns = data.columns, index = data.index)
    return data_i
    
data_anemia_red = impute_data(data_anemia_red)
data_anemia_std_red = impute_data(data_anemia_std_red)
display(data_anemia_red.head())
data_anemia_std_red.head()
def create_sampled_train_test_split(data, label, test_size, under = True):
    X_train,X_test,y_train,y_test = train_test_split(data, data[label],test_size=test_size)
    r = False
    if(under): 
        class_count = np.amin(X_train.groupby(label)[label].count().values)
    else: 
        class_count = np.amax(X_train.groupby(label)[label].count().values)
        r = True
    
    print("Classes in extracted train data before undersampling")
    display(X_train.groupby(label)[label].count())
    
    negative_cases = X_train[X_train[label] == 0].sample(n = class_count, replace = r)
    positive_sample = X_train[X_train[label] == 1].sample(n=class_count, replace = r)
    X_train_balanced = pd.concat([negative_cases, positive_sample])
    X_train_balanced.sort_index
    if(under):
        print("Classes after undersampling")
    else: 
        print("Classes after oversampling")
    display(X_train_balanced.groupby(label)[label].count())
    
    y_train = X_train_balanced.pop(label)
    y_test = X_test.pop(label)
    
    return X_train_balanced, X_test, y_train, y_test

#X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_red, label = 'anemia', test_size = 0.2)
#Now train a model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    print("Evaluation of", model)
    rf = model.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    confusion_matrix_result = confusion_matrix(y_test.values, y_pred)
    print("Confusion matrix:\n%s" % confusion_matrix_result)
    print(classification_report(y_test, y_pred))
    print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))
    
#evaluate_model(RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0), X_train, y_train, X_test, y_test)
data_glucose.diabetes.value_counts() #see how imbalanced the data is
def plot_ft_against_pred_label(data, ft, pred_label):     
    print("These", len(ft),"features are going to be compared to diabetes: \n")
    print(ft)
    #plot pairwise in rows of 4
    rest = len(ft)%4
    for l in np.arange(4, len(ft)- rest, 4): 
        sns.pairplot(data.dropna(), x_vars = ft[(l-4):l] , y_vars = [pred_label])
    sns.pairplot(data.dropna(), x_vars = ft[len(ft)-4: len(ft)] , y_vars = [pred_label])
    
#look at undummied data
ft_without_district = [x for x in data.columns if not x.startswith('district')]
ft_without_district.remove('diabetes')
ft_without_district.remove('glucose')
ft_without_district
plot_ft_against_pred_label(data, ft_without_district, 'glucose')
# getting relative frequency of high blood sugar
def diabetes_relative_freq(data, feature):
    subset = data.groupby(feature)
    high = pd.Series()
    for i in np.unique(data[feature]):
        high = high.append(pd.Series((subset['diabetes'].value_counts()[i]/sum(subset['diabetes'].value_counts()[i])).loc[1]))
    high.index = np.arange(1,len(subset)+1)  
    plt.bar(np.arange(1, len(subset)+1), high)
    plt.ylabel("Realtive freq of high blood sugar")
    plt.title(feature)
    plt.show()
plt.rcParams['figure.figsize'] = [35, 15]
diabetes_relative_freq(data, "district_code")
plt.rcParams['figure.figsize'] = [6, 6]
diabetes_relative_freq(data, "stratum")
plt.rcParams['figure.figsize'] = [6, 6]
diabetes_relative_freq(data, 'sex')
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
    df= data_glucose.loc[data_glucose[dist] == 1].glucose.dropna()
    var.append(np.var(df.value_counts().sort_index()))
print("The lowest glucose distribution variance: ", np.min(var))
print("The highest glucose distribution variance ", np.max(var))

ft_district_droppable = []
for d in ft_with_district: 
    df= data_glucose.loc[dummied_data[d] == 1].glucose.dropna()
    if(np.var(df.value_counts().sort_index()))< 2000: 
        ft_district_droppable.append(d)
print("The following districts will be discarded: ", ft_district_droppable)
print(data_glucose.shape[1], "features before in glucose not null data")
print(data_glucose_std.shape[1], "features before in glucose not null std data")
data_glucose.drop(ft_district_droppable, axis = 1, inplace = True)#district related features are named the same
data_glucose_std.drop(ft_district_droppable, axis = 1, inplace = True)
print(data_glucose.shape[1], "features left")
print(data_glucose_std.shape[1], "features left in glucose not null std data")
def plot_cat_against_pred_label(dummied_data, pred_label_list):#the binary feature must be first
    ft_cat_no_distr = [x for x in dummied_data.columns if x not in ft_numeric + ft_with_district + pred_label_list]
    fig = plt.figure(figsize = (20, 50))
    std_dict = {}

    for counter, d in enumerate(ft_cat_no_distr): 
        df= dummied_data.loc[dummied_data[d] == 1][pred_label_list[0]].dropna()
        #append the standard deviation among the pred label relative to the mean of the pred label
        std_dict[d] = (np.std(df.value_counts().sort_index()/np.nanmean(dummied_data[pred_label_list[0]])))
        axes= fig.add_subplot(14, 5, 1+ counter)
        axes.hist(df, density = True, bins = 20)
        axes.set_title(d + ' ' +str(df.shape[0]))
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
    plt.show()
    return std_dict

cat_std_dict = plot_cat_against_pred_label(data_glucose, ['glucose', 'diabetes'])
def plot_bar_dict(d):
    plt.figure(figsize = (14, 14))
    plt.xticks(rotation = 90)
    plt.bar(d.keys(), [d[y] for y in d.keys() ])
    plt.show()
plot_bar_dict(cat_std_dict)
data_glucose['stratum_1_2'] = data_glucose['stratum_1'] + data_glucose.stratum_2
data_glucose_std['stratum_1_2'] = data_glucose_std['stratum_1'] + data_glucose_std.stratum_2
print(np.nanstd(data_glucose.stratum_1_2.value_counts().sort_index())/np.nanmean(data_glucose.glucose))

def plot_distr_against_pred_label(dummied_data, pred_label_list):#the binary feature must be first
    ft_with_district = [x for x in dummied_data.columns if x.startswith('district')]
    fig = plt.figure(figsize = (20, 50))
    std_dict = {}

    for counter, d in enumerate([x for x in ft_with_district if x not in pred_label_list]): 
        df= dummied_data.loc[dummied_data[d] == 1][pred_label_list[0]].dropna()
        std_dict[d] = (np.std(df.value_counts().sort_index()/np.nanmean(dummied_data[pred_label_list[0]])))
        axes= fig.add_subplot(14, 5, 1+ counter)
        axes.hist(df, density = True, bins = 20)
        axes.set_title(d + ' ' +str(df.shape[0]))
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
    plt.show()
    return std_dict

distr_std_dict = plot_distr_against_pred_label(data_glucose, ['glucose', 'diabetes'])
plot_bar_dict(distr_std_dict)
def filter_by_std_prop(data, std_dict, factor = 3/4):
    stds = sorted(list(std_dict.values()))#sort ascending
    ft = list(std_dict.keys())
    std_treshold = stds[np.round(int(len(stds)*factor))]#1-factor of all keys will be kept
    for f in ft: 
        if std_dict[f] < std_treshold: 
            std_dict.pop(f)
    data.drop([x for x in data[ft].columns if x not in std_dict.keys()], axis = 1, inplace= True)
    print(data.shape[1], "features left")
#district_signi didnt make much sense here anyway, it wouldve been discarded
filter_by_std_prop(data_glucose, distr_std_dict)

diff = [x for x in data_glucose_std.columns if not np.isin(x,data_glucose.columns)]
dropped_districts = [x for x in diff if x.startswith('district')]
data_glucose_std.drop(dropped_districts, axis = 1, inplace = True)
data_glucose_std.shape, data_glucose.shape
print("Standard deviations of normalized heart related data")
display(data_glucose_std[['std_'+ x for x in ft_numeric]].describe().loc['std'])
desc = data_glucose_std[['std_'+ x for x in ft_numeric]].describe()
rel_std = pd.DataFrame({'rel_std': np.divide(desc.loc['std'].values, np.subtract(desc.loc['max'].values,desc.loc['min'].values)),
                        'ft': desc.columns})
print("Standard deviations of normalized heart related data relative to min max of the feature")
display(rel_std)

ft_to_drop = list(rel_std.loc[rel_std.rel_std <0.11].ft)
print("The following features will be dropped because their relative standard variation is too low: ")
ft_to_drop
if np.isin('std_glucose', ft_to_drop):
    ft_to_drop.remove('std_glucose')
ft_nostd_to_drop = [x[4:] for x in ft_to_drop ]
ft_nostd_to_drop
#This is a different result than in the glucose notebook. Lets see the outcome, because this is more precise
data_glucose_std.drop(ft_to_drop, inplace = True, axis = 1)
data_glucose.drop(ft_nostd_to_drop, inplace = True, axis = 1)
print("Now there are", len(data_glucose.columns), "in unscaled data")
print("Now there are", len(data_glucose_std.columns), "in scaled and centered data")

cols_women = ['never_married', 'married_no_gauna', 'married_and_gauna', 'remarried', 'widow', 'divorced', 'separated', 
              'pregnant', 'lactating', 'non_pregnant_non_lactating', 'duration_pregnanacy']
cols_women_std = ['never_married', 'married_no_gauna', 'married_and_gauna', 'remarried', 'widow', 'divorced', 'separated', 
              'pregnant', 'lactating', 'non_pregnant_non_lactating', 'std_duration_pregnanacy']
cols_w = [x for x in cols_women if not np.isin(x,ft_nostd_to_drop)]
cols_w_std = [x for x in cols_women_std if not np.isin(x,ft_to_drop)]
data_glucose_over45 = data_glucose.where(data_glucose['age']>=45)
data_glucose_over45_std = data_glucose_std.loc[list(data_glucose_over45.index), :]
print(data_glucose_over45.shape)
print(data_glucose_over45_std.shape)
men_glucose_over45 = data_glucose_over45.where(data_glucose_over45.male == 1).drop(cols_w + ['male', 'female'], axis = 1).dropna()
women_glucose_over45 = data_glucose_over45.where(data_glucose_over45.male == 0).drop(['male', 'female'], axis = 1).dropna()
men_glucose_over45_std = data_glucose_over45_std.loc[data_glucose_over45_std.male == 1].drop(cols_w_std + ['male', 'female'], axis = 1).dropna()
women_glucose_over45_std = data_glucose_over45_std.loc[data_glucose_over45_std.male == 0].drop(['male', 'female'], axis = 1).dropna()
print(men_glucose_over45.shape,
women_glucose_over45.shape,
men_glucose_over45_std.shape,
women_glucose_over45_std.shape)
#For random forest, knn, decision tree standardization is not necessary
X_train, X_test, y_train, y_test = train_test_split(men_glucose_over45.drop(['diabetes', 'glucose'], axis = 1).fillna(0),
                                                    men_glucose_over45.diabetes)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
#predictions for men, initially optimized with gridsearch
#rfc = RandomForestClassifier()
#rfc_param_grid = {'n_estimators': np.arange(160, 170, 1), 
#                  'criterion': ['entropy', 'gini']}
#rfc_grid_search = GridSearchCV(rfc, rfc_param_grid, cv=20, return_train_score=True)
#rfc_grid_search.fit(X_train, y_train)
#print(rfc_grid_search.best_params_)
r_m = RandomForestClassifier(criterion = 'gini', n_estimators = 168).fit(X_train, y_train)
print("Accuracy achieved by Random Forest with criterion gini, 168 estimators: ", 
     accuracy_score(y_test, r_m.predict(X_test)))
from sklearn.tree import DecisionTreeClassifier
#predictions for men, initially optimized with gridsearch
#dtc = DecisionTreeClassifier()
#dtc_param_grid = {'min_samples_split': np.arange(2, 10, 1), 'criterion': ['entropy', 'gini'], "splitter":["best", "random"]}
#dtc_grid_search = GridSearchCV(dtc, dtc_param_grid, cv=20, return_train_score=True)
#dtc_grid_search.fit(X_train, y_train)
#print(dtc_grid_search.best_params_)
dt = DecisionTreeClassifier(criterion = 'gini', splitter = "random", min_samples_split = 7).fit(X_train, y_train)
print("Accuracy achieved by Decision Tree with criterion gini, 7 min leaves, random splitter: ", 
     accuracy_score(y_test, dt.predict(X_test)))
from sklearn.neighbors import KNeighborsClassifier
#predictions for men, initially optimized with gridsearch
#knn = KNeighborsClassifier()
#knn_param_grid = {'n_neighbors': np.arange(3,10,1)}
#knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=20, return_train_score=True)
#knn_grid_search.fit(X_train, y_train)
#print(knn_grid_search.best_params_)
kn = KNeighborsClassifier(n_neighbors = 8).fit(X_train, y_train)
print("Accuracy achieved by KNN with 8 neighbors: ", 
     accuracy_score(y_test, kn.predict(X_test)))
#Using standardized data for SVM
X_train, X_test, y_train, y_test = train_test_split(men_glucose_over45_std.drop(['diabetes', 'std_glucose'], axis = 1).fillna(0),
                                                    men_glucose_over45_std.diabetes)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.svm import SVC
#predictions for men, initially optimized with gridsearch
#svc = SVC()
#svc_param_grid = {'kernel': ["linear", "rbf", "poly", "sigmoid"]}
#svc_grid_search = GridSearchCV(svc, svc_param_grid, cv=20, return_train_score=True)
#svc_grid_search.fit(X_train, y_train)
#print(svc_grid_search.best_params_)
sv = SVC(kernel = 'rbf').fit(X_train, y_train)
print("Accuracy achieved by SVM with rbf kernel: ", 
     accuracy_score(y_test, sv.predict(X_test)))
X_train, X_test, y_train, y_test = train_test_split(women_glucose_over45.drop(['diabetes', 'glucose'], axis = 1).fillna(0),
                                                    women_glucose_over45.diabetes)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
#first for upper limit 100mg/dl
data_glucose_pregnant = data_glucose.loc[data_glucose.pregnant == 1].drop(['male', 'female'], axis = 1)
data_glucose_pregnant = impute_data(data_glucose_pregnant)
data_glucose_pregnant_std = data_glucose_std.loc[data_glucose_std.pregnant == 1].drop(['male', 'female'], axis = 1)

# I will try imputing instead of dropping all nan rows
data_glucose_pregnant_std = impute_data(data_glucose_pregnant_std)
# centralized age is used for modeling
data_glucose_pregnant_std.diabetes.value_counts()
X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_glucose_pregnant_std.drop(['std_glucose'], axis = 1), label = 'diabetes', test_size = 0.2, under = False)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
#data_glucose_pregnant_std.columns
#X_train, X_test, y_train, y_test = train_test_split(data_glucose_pregnant_std.drop(['diabetes', 'std_glucose'], axis = 1),
                                                    #data_glucose_pregnant_std.diabetes)

rfc = RandomForestClassifier(criterion = 'gini', n_estimators = 162)#was optimized with GridSearch below
#rfc_param_grid = {'n_estimators': np.arange(162, 180, 3), 
#                  'criterion': ['entropy', 'gini']}#
#rfc_grid_search = GridSearchCV(rfc, rfc_param_grid, cv=20, return_train_score=True)
#rfc_grid_search.fit(X_train, y_train)
#print(rfc_grid_search.best_params_)
evaluate_model(rfc, X_train, y_train, X_test, y_test)
#now for upper limit 92mg/dl
data_glucose_pregnant.drop(['diabetes'], axis = 1, inplace = True)
data_glucose_pregnant_std.drop(['diabetes'], axis = 1, inplace = True)
data_glucose_pregnant['diabetes'] = data_glucose_pregnant.glucose.apply(lambda x : 1 if x > 92 else 0)
data_glucose_pregnant_std['diabetes'] = data_glucose_pregnant.diabetes
data_glucose_pregnant_std.diabetes.value_counts()
X_train, X_test, y_train, y_test = train_test_split(data_glucose_pregnant_std.drop(['diabetes', 'std_glucose'], axis = 1),
                                                    data_glucose_pregnant_std.diabetes)

#rfc = RandomForestClassifier()
#rfc_grid_search = GridSearchCV(rfc, rfc_param_grid, cv=20, return_train_score=True)
#rfc_grid_search.fit(X_train, y_train)
#print(rfc_grid_search.best_params_)
rfc = RandomForestClassifier(criterion = 'gini', n_estimators = 171)
evaluate_model(rfc, X_train, y_train, X_test, y_test)
imps = rfc.feature_importances_
important_features = [idx for idx in range(len(imps))if imps[idx]>0]
print("There are ", len(important_features), "features used for Random Forest for pregnant women: ")
plt.xticks(rotation='vertical')
plt.bar(data_glucose_pregnant_std.columns[important_features], imps[important_features])
plt.show()
distr_std_dict = plot_distr_against_pred_label(data_heart, ['pulse_rate'])
filter_by_std_prop(data_heart, distr_std_dict)
diff = [x for x in data_heart_std.columns if not np.isin(x,data_heart.columns)]
dropped_districts = [x for x in diff if x.startswith('district')]
data_heart_std.drop(dropped_districts, axis = 1, inplace = True)
data_heart_std.shape, data_heart.shape
cat_std_dict = plot_cat_against_pred_label(data_heart, ['pulse_rate'])
cat_std_dict
data_heart['stratum_1_2'] = data_heart['stratum_1'] + data_heart.stratum_2
data_heart_std['stratum_1_2'] = data_heart_std['stratum_1'] + data_heart_std.stratum_2
print(np.nanstd(data_heart.stratum_1_2.value_counts().sort_index())/np.nanmean(data_heart.pulse_rate))
#first we need a deviation standard dict of numerical features relative to their min-max-value
ft_numeric_std = ['std_'+x for x in ft_numeric]
ft_numeric_std_heart = ft_numeric_std
ft_numeric_std_heart.remove('std_pulse_rate')
desc = data_heart_std[ft_numeric_std_heart].describe()
desc
std_dict = {}
for i, c in enumerate(desc.columns):
    std_dict[c] = desc.iloc[2, i]/(desc.iloc[7, i]- desc.iloc[3, i])#2->std, 3->min, 7->max
plot_bar_dict(std_dict)
filter_by_std_prop(data_heart_std, std_dict, factor = 0.3)
dropped = [x for x in ft_numeric_std_heart if not np.isin(x, data_heart_std.columns)]
print("The following features have been discarded:")
dropped
data_heart.drop([x[4:] for x in dropped], axis = 1, inplace = True)
data_heart.shape, data_heart_std.shape
cols_w = [x for x in cols_w if np.isin(x, data_heart.columns)]
cols_w_std = [x for x in cols_w_std if np.isin(x, data_heart_std.columns)]#in case a numeric feature was kept
#cols_w, cols_w_std
men_data_heart = data_heart.loc[data_heart['male'] == 1].drop(cols_w + ['male', 'female'], axis = 1)
fem_data_heart = data_heart.loc[data_heart['female']== 1].drop(['male', 'female'], axis = 1)
men_data_heart_std = data_heart_std.loc[data_heart_std['male'] == 1].drop(cols_w_std + ['male', 'female'], axis = 1)
fem_data_heart_std = data_heart_std.loc[data_heart_std['female']== 1].drop(['male', 'female'], axis = 1)
print(men_data_heart.shape[1], "features for men")
print(fem_data_heart.shape[1], "features for women")
print(men_data_heart_std.shape[1], "features for men std")
print(fem_data_heart_std.shape[1], "features for women std")
men_data_heart = impute_data(men_data_heart)
fem_data_heart = impute_data(fem_data_heart)
men_data_heart_std = impute_data(men_data_heart_std)
fem_data_heart_std= impute_data(men_data_heart_std)
print("mean pulse rate of women", fem_data_heart.pulse_rate.mean())
print("mean pulse rate of men", men_data_heart.pulse_rate.mean())
fem_data_heart_postmeno = fem_data_heart.loc[fem_data_heart['age'] >= 50]
men_data_heart_older = men_data_heart.loc[men_data_heart['age'] >= 50]

fig = plt.figure(figsize = (15, 5))
#plot pulse rate of postmeno women
axes = fig.add_subplot(1, 2, 1)
axes.hist(fem_data_heart_postmeno.pulse_rate.dropna(), density = True, bins = 20)
axes.axvline(76, color = 'red', label = 'critical pulse rate')
axes.set_title('pulse rate of elder women, mean ' + str(np.nanmean(fem_data_heart_postmeno.pulse_rate)))
#plot pulse rate of older men
axes = fig.add_subplot(1, 2, 2)
axes.hist(men_data_heart_older.pulse_rate.dropna(), density = True, bins = 20)
axes.axvline(76, color = 'red', label = 'critical pulse rate')
axes.set_title('pulse rate of elder men, mean ' + str(np.nanmean(men_data_heart_older.pulse_rate)))
axes.legend()
plt.show()
def pulse_dange(x): 
    return (x >= 76) * 1
fem_data_heart_postmeno['pulse_rate_dangerous'] = fem_data_heart_postmeno.pulse_rate.apply(pulse_dange)
men_data_heart_older['pulse_rate_dangerous'] = men_data_heart_older.pulse_rate.apply(pulse_dange)
men_count= men_data_heart_older.pulse_rate_dangerous.sum()
fem_count = fem_data_heart_postmeno.pulse_rate_dangerous.sum()
print(men_count, "older men are affected by increased heart attack risk, that is",men_count/men_data_heart_older.shape[0], "%" )
print(fem_count, "older women are affected by increased heart attack risk, that is",fem_count/fem_data_heart_postmeno.shape[0], "%" )
men_data_heart_older.shape, fem_data_heart_postmeno.shape