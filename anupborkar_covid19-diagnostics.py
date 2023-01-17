"""
Created on Sun May  3 18:20:04 2020
@author: Anup Borkar
"""
%reset -f
# 1.1 For data manipulations
import numpy as np
import pandas as pd
# 1.2 For plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0}) #just to suppress warning for max plots of 20
#import matplotlib
#import matplotlib as mpl     # For creating colormaps
import seaborn as sns
# 1.3 For data processing
from sklearn.preprocessing import StandardScaler
# 1.4 OS related
import os

# 1.5 Go to folder containing data file
#os.chdir("../input/")
#os.listdir()            # List all files in the folder

# 1.6 Read file and while reading file,
df = pd.read_csv("../input/uncover/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv")
df.head()
#counting the number of NAs by each Row
num_of_NA = (df.isnull().sum(axis=1))
#num_of_NA
df['patient_age_quantile'] = pd.to_numeric(df['patient_age_quantile'])
df = df.fillna(np.nan)
df['urine_leukocytes']=df['urine_leukocytes'].str.replace("<","") #Cleaning the column as I found some Special Character
if (pd.to_numeric(df['urine_leukocytes'].values.any())<1000):
    df['urine_leukocytes']=df['urine_leukocytes'].replace({999}, inplace=True) 
# let me get our output (positive (1) or negative(0)),
# transforming our target column in a binary format
df.loc[:, 'sars_cov_2_exam_result_flag'] = df['sars_cov_2_exam_result']
df['sars_cov_2_exam_result_flag'].replace({"positive": "1", "negative": "0"}, inplace=True)
df['sars_cov_2_exam_result_flag'].unique()
# removing patient id (it is not necessary)
# we MUST remove this column 'cause one patient doesn't has impact in others.
# so, predictors cannot try to use this information
df=df.drop(['patient_id','patient_addmited_to_regular_ward_1_yes_0_no',
       'patient_addmited_to_semi_intensive_unit_1_yes_0_no',
       'patient_addmited_to_intensive_care_unit_1_yes_0_no','sars_cov_2_exam_result'], axis=1)
df
df.replace(False, -1, inplace=True) #Replace all False values with -1
df.replace(True, +1, inplace=True) #Replace all True vaules with 1
# these columns have categorical data. Transforming them in numerical ones and putting in some order
columns = ['respiratory_syncytial_virus', 'influenza_a', 'influenza_b','parainfluenza_1', 'coronavirusnl63',
           'rhinovirus_enterovirus', 'coronavirus_hku1', 'parainfluenza_3','chlamydophila_pneumoniae', 'adenovirus',
           'parainfluenza_4','coronavirus229e', 'coronavirusoc43', 'inf_a_h1n1_2009','bordetella_pertussis', 'metapneumovirus',
           'parainfluenza_2', 'influenza_b_rapid_test','influenza_a_rapid_test', 'strepto_a', 'urine_esterase', 'urine_aspect',
           'urine_ph', 'urine_hemoglobin', 'urine_bile_pigments','urine_ketone_bodies', 'urine_nitrite','urine_urobilinogen', 
           'urine_protein','urine_leukocytes', 'urine_crystals','urine_hyaline_cylinders', 'urine_granular_cylinders', 
           'urine_yeasts','urine_color']
#Replacing default values available into Numerical datatype
for i in columns:
    df[i].replace({"not_detected":0,          "detected":1,        "Não Realizado": 0,
                   "negative":0,              "positive":1,        "not_done":-1,
                   "absent":0,                "present":1,         "clear":0,
                   "lightly_cloudy":1,        "cloudy":2,          "altered_coloring":10,
                   "Ausentes":0,              "Urato Amorfo --+":1,"Oxalato de Cálcio -++":2,
                   "Oxalato de Cálcio +++":3, "Urato Amorfo +++":3,"light_yellow":1,
                   "yellow":2,                "citrus_yellow":3,   "orange":4, "nan":"",
                   "normal":1
                  }, inplace=True)

df['urine_color'].unique() # Just to check whether changes affected or not
df.head()
#Let's start plotting some graphs
sns.distplot(df.patient_age_quantile)
sns.despine()               # Plot with and without it
#not using urine_yeasts, urine_granular_cylinders, urine_hyaline_cylinders,parainfluenza_2,urine_nitrite from columns
#array as these variable don't have values for drawing
cols =    ['respiratory_syncytial_virus', 'influenza_a', 'influenza_b','parainfluenza_1', 'coronavirusnl63',
           'rhinovirus_enterovirus', 'coronavirus_hku1', 'parainfluenza_3','chlamydophila_pneumoniae', 'adenovirus',
           'parainfluenza_4','coronavirus229e', 'coronavirusoc43', 'inf_a_h1n1_2009','bordetella_pertussis', 'metapneumovirus',
           'influenza_b_rapid_test','influenza_a_rapid_test', 'strepto_a', 'urine_esterase', 'urine_aspect',
           'urine_ph', 'urine_hemoglobin', 'urine_bile_pigments','urine_ketone_bodies', 'urine_urobilinogen', 
           'urine_protein','urine_leukocytes', 'urine_crystals','urine_color']

fig = plt.figure(figsize = (20,30))
for i in range(len(cols)):
    plt.subplot(10,3,i+1)
    sns.kdeplot(df[cols[i]],  bw=2) #cumulative=True,

#Relationship between Patient Age Quantile and Urine Colour light_yellow=1,yellow=2,citrus_yellow=3,orange=4
sns.boxplot(x = 'urine_color',                    # Discrete 
            y = 'patient_age_quantile',           # Continuous
            data = df
            )
#Relationship between Patient Age Quantile and Urine Colour with 'Notch'
sns.boxplot(x = 'urine_color',                          # Discrete
            y = 'patient_age_quantile',                 # Continuous
            data = df,
            notch = True              
            )
df.describe() #describing summary about the dataset
sns.jointplot(df.phosphor,
              df.patient_age_quantile,
              kind = "hex"
              )
#Segragating Columns into Categories like Corona Related, Urine Related and Test Related for plotting
cols_corona = [ 'respiratory_syncytial_virus', 'coronavirusnl63',
            'coronavirus_hku1', 'coronavirus229e', 'coronavirusoc43']
cols_urine=['urine_esterase', 'urine_aspect','urine_ph', 'urine_hemoglobin', 'urine_bile_pigments',
            'urine_ketone_bodies', 'urine_urobilinogen', 'urine_protein','urine_leukocytes', 
            'urine_crystals','urine_color']
cols_test = ['influenza_b_rapid_test','influenza_a_rapid_test', 'strepto_a']
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
fig = plt.figure(figsize = (25,30))
for i in range(len(cols_corona)):
    plt.subplot(10,5,i+1)
    sns.distplot(
    df[cols_corona[i]], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1});#.set(xlabel='Legend ', ylabel='Value');
    sns.despine()   
    #sns.kdeplot(df[cols[i]],  bw=2) #cumulative=True,

fig = plt.figure(figsize = (25,30))
for i in range(len(cols_urine)):
    plt.subplot(10,5,i+1)
    sns.countplot(df[cols_urine[i]])
    
#Lets draw Count Plot for Test Related Colums
fig, ax = plt.subplots(1, 3, figsize=(20, 10))
for variable, subplot in zip(cols_test, ax.flatten()):
    sns.countplot(df[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for variable, subplot in zip(cols_urine, ax.flatten()):
    sns.countplot(df[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
sorted_nb = df.groupby(['patient_age_quantile'])['coronavirus229e'].median().sort_values()
sns.boxplot(x=df['coronavirus_hku1'], y=df['patient_age_quantile'], order=list(sorted_nb.index))
#columns = list(df.head(0))
#columns
#df.select_dtypes(include=['category']).dtypes
sns.catplot(x = 'urine_color',
            y = 'patient_age_quantile',
            row = 'urine_color',
            col = 'sars_cov_2_exam_result_flag',
            estimator = np.mean ,
            kind = 'box',
            data =df)

#sns.jointplot(x = 'sars_cov_2_exam_result_flag', y = 'patient_age_quantile',data = df, notch=True)
#Heatmap of Urine Related Tests
df_cols=df[cols_urine]
corr = df_cols.corr()
g = sns.heatmap(corr, vmax=.3, center=0,
square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
sns.despine()
g.figure.set_size_inches(15,10)
plt.show()