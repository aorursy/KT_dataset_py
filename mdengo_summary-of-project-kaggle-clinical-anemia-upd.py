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

corr_dummied_data = dummied_data.corr()
corr_dummied_data_std = dummied_data_std.corr()
m = (corr_dummied_data.mask(np.eye(len(corr_dummied_data), dtype=bool)).abs() > 0.4).any()
# keeping the feature if it has an above 0.4 correlation with at least one other feature

corr_dd = corr_dummied_data.loc[m, m]
corr_matrix_plot = corr_dd
title = 'Pairwise Correlations'

f, ax = plt.subplots(figsize=(10, 8))
        
# Diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with a color bar
sns.heatmap(corr_matrix_plot, cmap=cmap, center=0, linewidths=.25, cbar_kws={"shrink": 0.6})

# Set the ylabels 
ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
ax.set_yticklabels(list(corr_matrix_plot.index), size = int(160 / corr_matrix_plot.shape[0]));

# Set the xlabels 
ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
ax.set_xticklabels(list(corr_matrix_plot.columns), size = int(160 / corr_matrix_plot.shape[1]));
plt.title(title, size = 14)
plt.savefig('correlations.png')
plt.savefig('correlations.pdf')
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
data_anemia_red_corr = data_anemia_red.corr()
data_anemia_std_red_corr = data_anemia_std_red.corr()
display(data_anemia_std_red_corr)
display(data_anemia_red_corr)
data_anemia_women = data_anemia_red.where(data_anemia_red.male == 0)
data_anemia_men = data_anemia_red.where(data_anemia_red.male == 1)
fig = plt.figure(figsize = (15, 5))
#plot anemia status of all women
axes = fig.add_subplot(1, 2, 1)
axes.fill_between(x = [0, 12], y1= [0.28, 0.28], color = 'lightcoral' )
axes.fill_between(x = [12, 18], y1= [0.28, 0.28], color = 'lightgreen' )
axes.hist(data_anemia_women.haemoglobin_level.dropna(), density = True, bins = 20)
axes.axvline(12, color = 'red', label = 'critical haemoglobin level')
axes.set_xlabel('haemoglobin [g/dl]')
axes.set_ylabel('freq')
axes.set_title('haemoglobin of women, mean ' + str(np.nanmean(data_anemia_women.haemoglobin_level)))
#anemia of all men
axes = fig.add_subplot(1, 2, 2)
axes.fill_between(x = [0, 13], y1= [0.28, 0.28], color = 'lightcoral' )
axes.fill_between(x = [13, 18], y1= [0.28, 0.28], color = 'lightgreen' )
axes.hist(data_anemia_men.haemoglobin_level.dropna(), density = True, bins = 20)
axes.axvline(13, color = 'red', label = 'critical haemoglobin level')
axes.set_xlabel('haemoglobin [g/dl]')
axes.set_ylabel('freq')
axes.set_title('haemoglobin of men, mean ' + str(np.nanmean(data_anemia_men.haemoglobin_level)))
axes.legend()
plt.savefig("anemia_men_women.pdf")
plt.show()
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
    
    print("Classes before under- or oversampling")
    display(X_train.groupby(label)[label].count())
    
    negative_cases = X_train[X_train[label] == 0].sample(n = class_count, replace = r)
    positive_sample = X_train[X_train[label] == 1].sample(n=class_count, replace = r)
    X_train_balanced = pd.concat([negative_cases, positive_sample])
    X_train_balanced.sort_index
    print("Classes after under- or oversampling")
    display(X_train_balanced.groupby(label)[label].count())
    
    y_train = X_train_balanced.pop(label)
    y_test = X_test.pop(label)
    
    return X_train_balanced, X_test, y_train, y_test

X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_red, label = 'anemia', test_size = 0.2)
#Now train a model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    print("Evaluation of", model)
    rf = model.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    confusion_matrix_result = confusion_matrix(y_test.values, y_pred)
    print("Confusion matrix:\n%s" % confusion_matrix_result)
    print(classification_report(y_test, y_pred))
    print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))
    
evaluate_model(RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0), X_train, y_train, X_test, y_test)
#repeat the same on standardized data set
X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_std_red, label = 'anemia', test_size = 0.2)
evaluate_model(RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0), X_train, y_train, X_test, y_test)
#try oversampling
X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_std_red, label = 'anemia', test_size = 0.2, under = False)
evaluate_model(RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0), X_train, y_train, X_test, y_test)
from sklearn.feature_selection import SelectKBest, f_classif
select_k_best_classifier = SelectKBest(score_func=f_classif, k=13)

select_k_best_classifier.fit_transform(data_anemia_red, data_anemia_red.anemia)
mask = select_k_best_classifier.get_support()
relevant_columns = data_anemia_red.columns[mask]
display(relevant_columns)

select_k_best_classifier.fit_transform(data_anemia_std_red, data_anemia_std_red.anemia)
mask = select_k_best_classifier.get_support()
relevant_columns_std = data_anemia_std_red.columns[mask]
display(relevant_columns_std)
data_anemia_new = data_anemia_red[relevant_columns]
data_anemia_std_new = data_anemia_std_red[relevant_columns_std]
display(data_anemia_new.head())
display(data_anemia_std_new.head())
X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_new, label = 'anemia', test_size = 0.2)
evaluate_model(RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0), X_train, y_train, X_test, y_test)
#repeat the same on standardized data set
X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_std_new, label = 'anemia', test_size = 0.2)
evaluate_model(RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0), X_train, y_train, X_test, y_test)
#try oversampling
X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_new, label = 'anemia', test_size = 0.2, under = False)
evaluate_model(RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0), X_train, y_train, X_test, y_test)
#try oversampling
X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_std_new, label = 'anemia', test_size = 0.2, under = False)
evaluate_model(RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0), X_train, y_train, X_test, y_test)
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_new, label = 'anemia', test_size = 0.2, under = False)
evaluate_model(KNeighborsClassifier(n_neighbors=3), X_train, y_train, X_test, y_test)
from sklearn.model_selection import cross_val_score

# filtering just the odd numbers from 1 to 50
neighbors = list(filter(lambda x: x % 2 != 0, list(range(1,50))))
cv_scores = [] # cross-validation scores

for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
errors = [1 - x for x in cv_scores]
optimal_n = neighbors[errors.index(min(errors))]
print("The optimal number of neighbors is %d" % optimal_n)

# plot misclassification errors for each n
plt.plot(neighbors, errors)
plt.xlabel('Number of Neighbors')
plt.ylabel('Misclassification Error')
plt.show()
X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_new, label = 'anemia', test_size = 0.2, under = False)
evaluate_model(KNeighborsClassifier(n_neighbors=optimal_n), X_train, y_train, X_test, y_test)
# let's try on the standardized dataset as well
X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_std_new, label = 'anemia', test_size = 0.2, under = False)
evaluate_model(KNeighborsClassifier(n_neighbors=optimal_n), X_train, y_train, X_test, y_test)
# and on the undersampled datasets
X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_new, label = 'anemia', test_size = 0.2)
evaluate_model(KNeighborsClassifier(n_neighbors=optimal_n), X_train, y_train, X_test, y_test)
# standardized dataset with undersampling
X_train, X_test, y_train, y_test = create_sampled_train_test_split(data_anemia_std_new, label = 'anemia', test_size = 0.2)
evaluate_model(KNeighborsClassifier(n_neighbors=optimal_n), X_train, y_train, X_test, y_test)