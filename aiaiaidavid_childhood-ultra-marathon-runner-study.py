# Import libraries

import numpy as np

import pandas as pd



import seaborn as sns

sns.set_color_codes("pastel")

sns.set(style="whitegrid")



import matplotlib.pyplot as plt

# Pretty display for notebooks

%matplotlib inline



# Set precision to two decimals

pd.set_option("display.precision", 2)



import warnings

warnings.filterwarnings("ignore")
!ls ../input
# Upload data file into dataframe df

path = "../input/childhood-ultramarathon-runner-study-jan-2020/"

fileName = "Childhood Ultramarathon Runner Study v0.92.csv"

df = pd.read_csv(path + fileName, sep=',')

df.info()
df
df.info(verbose=True)
df.describe()
df.describe(include=['object', 'bool'])
df_demographics = df[['GENDER', 'CURRENT_AGE', 'CURRENT_HEIGHT', 'CURRENT_WEIGHT',

                     'HEIGHT_AT_25', 'WEIGHT_AT_25', 'BIRTH_PLACE', 

                     'ETHNIC_ORIGIN', 'MARITAL_STATUS', 'NBR_BIO_CHILDREN',

                     'HIGHEST_EDUCATION', 'WORKING_STATUS', 

                     'AVG_WORK_HOURS_PW_LAST_12_MONTHS', 'OCCUPATION',

                     'RETIRE_YEAR']]

df_demographics.info()
# Calculate Body Max Index (BMI) and add columns

df_demographics['CURRENT_BMI'] = df_demographics['CURRENT_WEIGHT'] / ((df_demographics['CURRENT_HEIGHT']/100)**2)

df_demographics['BMI_AT_25'] = df_demographics['WEIGHT_AT_25'] / (df_demographics['HEIGHT_AT_25']/100)**2
df_demographics.describe(include='all')
# Gender and age stats

f, axes = plt.subplots(1, 2, figsize=(12, 4))

g = df_demographics.GENDER.value_counts()

sns.barplot(g.index, g.values, ax=axes[0])

sns.distplot(df_demographics.CURRENT_AGE, kde=False, bins=12, ax=axes[1])

g
# Height, weight and BMI histograms

f, axes = plt.subplots(2, 3, figsize=(18, 8))

sns.distplot(df_demographics.CURRENT_HEIGHT, bins=12, kde=False, ax=axes[0,0])

sns.distplot(df_demographics.CURRENT_WEIGHT, bins=12, kde=False, ax=axes[0,1])

sns.distplot(df_demographics.CURRENT_BMI, bins=12, kde=False, ax=axes[0,2])

sns.distplot(df_demographics.HEIGHT_AT_25, bins=12, kde=False, ax=axes[1,0])

sns.distplot(df_demographics.WEIGHT_AT_25, bins=12, kde=False, ax=axes[1,1])

sns.distplot(df_demographics.BMI_AT_25, bins=12, kde=False, ax=axes[1,2])
df_demographics[['CURRENT_HEIGHT', 'CURRENT_WEIGHT', 'HEIGHT_AT_25', 'WEIGHT_AT_25', 'CURRENT_BMI', 'BMI_AT_25']].describe()
# Birth place stats

f, axes = plt.subplots(1, 1, figsize=(6, 4))

bp = df_demographics.BIRTH_PLACE.value_counts()

sns.barplot(bp.values, bp.index,  orient='h')

bp
# Ethnic origin stats

f, axes = plt.subplots(1, 1, figsize=(6, 4))

eo = df_demographics.ETHNIC_ORIGIN.value_counts()

sns.barplot(eo.values, eo.index,  orient='h')

eo
# Education stats

f, axes = plt.subplots(1, 1, figsize=(8, 6))

ed = df_demographics.HIGHEST_EDUCATION.value_counts()

sns.barplot(ed.values, ed.index,  orient='h')

ed
# Occupation stats

f, axes = plt.subplots(1, 1, figsize=(6, 8))

oc = df_demographics.OCCUPATION.value_counts()

sns.barplot(oc.values, oc.index,  orient='h')

# sns.barplot(oc.values, oc.index,  orient='h', color='grey')

# f.savefig("fig1.pdf", orientation="portrait", bbox_inches='tight')

oc
# Work status stats

f, axes = plt.subplots(1, 1, figsize=(6, 4))

ws = df_demographics.WORKING_STATUS.value_counts()

sns.barplot(ws.values, ws.index, orient='h')

ws
# Marital status stats

f, axes = plt.subplots(1 ,1, figsize=(6, 4))

ms = df_demographics.MARITAL_STATUS.value_counts()

sns.barplot(ms.values, ms.index, orient='h')

ms
# Biological children

f, axes = plt.subplots(1 ,1, figsize=(6, 4))

ch = df_demographics.NBR_BIO_CHILDREN.value_counts()

sns.barplot(ch.values, ch.index, orient='h')

ch
df_health_habits = df[['SMOKED_100_IN_LIFE', 'AGE_1ST_SMOKE', 'CURRENTLY_SMOKING', 'YEARS_SINCE_STOP_SMOKING', '12_DRINKS_IN_ANY_ONE_YEAR',

                     '12_DRINKS_IN_ENTIRE_LIFE', 'DRINK_DAYS_PW_LAST_12_MONTHS', 'HEAVY_DRINKING_DAYS_LAST_12_MONTHS',

                     'DIET']]
df_health_habits.info()
df_health_habits.describe(include='all')
# Smoking stats

f, axes = plt.subplots(1, 3, figsize=(18, 4))

s1 = df_health_habits.SMOKED_100_IN_LIFE.value_counts()

sns.barplot(s1.index, s1.values, ax=axes[0])

s2 = df_health_habits.CURRENTLY_SMOKING.value_counts()

sns.barplot(s2.index, s2.values, ax=axes[1])

sns.distplot(df_health_habits.AGE_1ST_SMOKE, kde=False, bins=7, ax=axes[2])

print(s1)

print()

print(s2)
# Drinking stats

f, axes = plt.subplots(1, 2, figsize=(12, 4))

dr1 = df_health_habits['12_DRINKS_IN_ENTIRE_LIFE'].value_counts()

sns.barplot(dr1.index, dr1.values, ax=axes[0])

dr2 = df_health_habits['12_DRINKS_IN_ANY_ONE_YEAR'].value_counts()

sns.barplot(dr2.index, dr2.values, ax=axes[1])

print(dr1)

print()

print(dr2)
# Some more drinking stats

f, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.distplot(df_health_habits.DRINK_DAYS_PW_LAST_12_MONTHS, bins=12, kde=False, ax=axes[0])

sns.distplot(df_health_habits.HEAVY_DRINKING_DAYS_LAST_12_MONTHS, bins=12, kde=False, ax=axes[1])
# Diet stats

f, axes = plt.subplots(1 ,1, figsize=(6, 4))

di = df_health_habits.DIET.value_counts()

sns.barplot(di.values, di.index, orient='h')

di
df_childhood_UM_history = df[['AGE_1ST_UM', 'DISTANCE_1ST_UM', 'NBR_UM_BEFORE_19', 'NBR_UM_50KM_BEFORE_19', 'NBR_UM_80KM_BEFORE_19', 'NBR_UM_100KM_BEFORE_19', 'NBR_UM_160KM_BEFORE_19',

                              'NBR_UM_OVER_160KM_BEFORE_19', 'UM_TRAIN_TIMES_PW_BEFORE_19', 'UM_TRAIN_HOURS_PW_BEFORE_19', 'UM_TRAIN_DISTANCE_PW_BEFORE_19', 'UM_TRAIN_%_ASPHALT_BEFORE_19',

                              'UM_TRAIN_%_TRAIL_BEFORE_19', 'UM_TRAIN_%_TRACK_BEFORE_19', 'UM_TRAIN_%_TREADMILL_BEFORE_19', 'UM_TRAIN_%_OTHER_BEFORE_19', 'SUFFERED_UM_INJURIES_UNDER_19',

                              'UM_INJURY_KNEE_PAIN', 'UM_INJURY_ANKLE_TENDINOPATHY', 'UM_INJURY_ACHILLES_TENDINOPATHY', 'UM_INJURY_HIP_FLEXOR_STRAIN', 'UM_INJURY_ILIOTIBIAL_BAND_ISSUE',

                              'UM_INJURY_FOOT_STRESS_FRACTURE', 'UM_INJURY_TIBIA_STRESS_FRACTURE', 'UM_INJURY_CALF_STRAIN', 'UM_INJURY_ANKLE_SPRAIN', 'UM_INJURY_HAMSTRING_STRAIN',

                              'UM_INJURY_PLANTAR_FASCIITIS', 'UM_INJURY_MEDIAL_TIBIAL_STRESS_SYNDROME', 'UM_INJURY_COMPARTMENT_SYNDROME', 'UM_INJURY_MORTON_NEUROMA', 'UM_INJURY_BURSITIS',

                              'UM_INJURY_MISSED_TRAINING_BEFORE_19', 'UM_TRAINING_AFFECTED_ACADEMIC_PERF_BEFORE_19', 'UM_TRAINING_AFFECTED_OTHER_SPORTS_PERF_UNDER_19']]
df_childhood_UM_history.info()
df_childhood_UM_history.describe()
# 1st marathon stats

f, axes = plt.subplots(2, 2, figsize=(12,8))

sns.distplot(df_childhood_UM_history.AGE_1ST_UM, kde=False, bins=11, ax=axes[0,0])

sns.distplot(df_childhood_UM_history.NBR_UM_BEFORE_19, kde=False, bins=30, ax=axes[0, 1])

sns.distplot(df_childhood_UM_history.DISTANCE_1ST_UM, kde=False, bins=60, ax=axes[1, 0])

sns.distplot(df_childhood_UM_history.UM_TRAIN_DISTANCE_PW_BEFORE_19, kde=False, bins=12, ax=axes[1, 1])
print('Age  Freq')

df_childhood_UM_history.AGE_1ST_UM.value_counts().sort_index(ascending=False)
print('NbrUM  Freq')

df_childhood_UM_history.NBR_UM_BEFORE_19.value_counts().sort_index(ascending=False)
print('DistUM  Freq')

df_childhood_UM_history.DISTANCE_1ST_UM.value_counts().sort_index(ascending=False)
print('TrainKm  Freq')

df_childhood_UM_history.UM_TRAIN_DISTANCE_PW_BEFORE_19.value_counts().sort_index(ascending=False)
# Check for nulls

df_childhood_UM_history.isnull().any()
# Fill NaN in UM_INJURY columns with zero (meaning not present)

df_childhood_UM_history.fillna(value=0, inplace=True)

df_childhood_UM_history.isnull().any()
correlation = df_childhood_UM_history.corr() 

plt.figure(figsize=(24, 14))

heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
df_reasons_for_running_UM = df[['REASONS_UM_MY_IDEA', 'REASONS_UM_RUN_WITH_PARENTS', 'REASONS_UM_RUN_WITH_FRIENDS', 'REASONS_UM_RUN_WITH_FAMILY_OTHER',

                                'REASONS_UM_PARENTS ENCOURAGED_ME', 'REASONS_UM_HOW_FAR_CAN_I_GO', 'REASONS_UM_COACH_ENCOURAGED_ME', 'REASONS_UM_FOR_THE_CHALLENGE']]



df_reasons_for_running_UM.head()
df_reasons_for_running_UM.fillna(value=0, inplace=True)

df_reasons_for_running_UM.head()
run_reasons = df_reasons_for_running_UM.sum()

df_run_reasons = pd.DataFrame(data=run_reasons, index=df_reasons_for_running_UM.columns, columns=['NBR_RESPONSES'])

df_run_reasons
# Plot a pie chart with % values

from pylab import rcParams

rcParams['figure.figsize'] = 8,8

fig, ax = plt.subplots()

ax.pie(df_run_reasons.NBR_RESPONSES, labels=df_run_reasons.index, autopct='%1.1f%%', startangle=45, pctdistance=0.7, labeldistance=1.05, shadow=True )

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
# Injuries associated to UM activity suffered before the age of 19 

f, axes = plt.subplots(1, 1, figsize=(6, 4))

d = df_childhood_UM_history.SUFFERED_UM_INJURIES_UNDER_19.value_counts()

sns.barplot(d.index, d.values)

d
# Ultra-Marathon related injuries

df_UM_injuries = df_childhood_UM_history[['UM_INJURY_KNEE_PAIN', 'UM_INJURY_ANKLE_TENDINOPATHY', 'UM_INJURY_ACHILLES_TENDINOPATHY',  'UM_INJURY_HIP_FLEXOR_STRAIN',

                                          'UM_INJURY_ILIOTIBIAL_BAND_ISSUE', 'UM_INJURY_FOOT_STRESS_FRACTURE', 'UM_INJURY_TIBIA_STRESS_FRACTURE', 'UM_INJURY_CALF_STRAIN',

                                          'UM_INJURY_ANKLE_SPRAIN', 'UM_INJURY_HAMSTRING_STRAIN', 'UM_INJURY_PLANTAR_FASCIITIS', 'UM_INJURY_MEDIAL_TIBIAL_STRESS_SYNDROME',

                                          'UM_INJURY_COMPARTMENT_SYNDROME', 'UM_INJURY_MORTON_NEUROMA', 'UM_INJURY_BURSITIS']]
df_injuries = df_UM_injuries.sum()

df_injuries = pd.DataFrame(data=df_injuries, index=df_UM_injuries.columns, columns=['NBR_CASES'])

df_injuries.sort_values(by='NBR_CASES', ascending=False)
df_injuries['INDEX'] = df_injuries.index.str.replace("UM_INJURY_", "")

df_injuries['INDEX'] = df_injuries['INDEX'].str.replace("_", " ")

#df_injuries['INDEX'] = df_injuries['INDEX'].str.lower()

df_injuries.set_index('INDEX', inplace=True)
# Plot a pie chart with % values

rcParams['figure.figsize'] = 10,10

fig1, ax1 = plt.subplots()



# theme = plt.get_cmap('Greys')

# ax1.set_prop_cycle("color", [theme(1. * (i+2) / 30) for i in range(15)])

explode = [0.1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0]



ax1.pie(df_injuries.NBR_CASES, labels=df_injuries.index, autopct='%1.1f%%', startangle=45, pctdistance=0.7, labeldistance=1.05, shadow=True, explode=explode )

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
# Total count of injuries reported (some participants reported more than one injury)

df_injuries.sum()
# Other sports

df_other_sports = df[['OTHER_SP_Baseball', 'OTHER_SP_Basketball', 'OTHER_SP_Cross-Country Running', 'OTHER_SP_Golf', 'OTHER_SP_Tennis', 'OTHER_SP_Soccer', 'OTHER_SP_Football', 

                      'OTHER_SP_Track_and_Field', 'OTHER_SP_Swimming', 'OTHER_SP_Skiing', 'OTHER_SP_Volleyball', 'OTHER_SP_Others']]

          

#df_other_sports
# Number of participants that practised other sports (whether for competition or recreational purposes)

dos = df_other_sports.sum(axis=1)

f, axes = plt.subplots(1 ,1, figsize=(6, 4))

sns.distplot(dos.values, kde=False, bins=8, axlabel='NBR_OF_SPORTS')

print('Sports', 'Children')

dos.value_counts().sort_index()
df_sports = df_other_sports.sum()

df_sports = pd.DataFrame(data=df_sports, index=df_other_sports.columns, columns=['NBR_RESPONSES'])

df_sports = df_sports.sort_values(by='NBR_RESPONSES', ascending=False).head(20)

df_sports
df_sports['INDEX'] = df_sports.index.str.replace("OTHER_SP_", "")

df_sports['INDEX'] = df_sports['INDEX'].str.replace("_", " ")

df_sports.set_index('INDEX', inplace=True)
df_sports
# Plot a pie chart with % values

rcParams['figure.figsize'] = 10,10

fig2, ax2 = plt.subplots()



explode = [0.1,0,0,0,0,0,0,0,0,0,0,0] 

# theme = plt.get_cmap('Greys')

# ax2.set_prop_cycle("color", [theme(1. * (i+2) / 24) for i in range(12)])



ax2.pie(df_sports.NBR_RESPONSES, labels=df_sports.index, autopct='%1.1f%%', startangle=20, pctdistance=0.7, labeldistance=1.05, shadow=True, explode=explode)

ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



# plt.savefig("fig2.pdf", orientation="landscape", bbox_inches='tight')

plt.show()

df_effect_performance = df[['UM_INJURY_MISSED_TRAINING_BEFORE_19', 'UM_TRAINING_AFFECTED_ACADEMIC_PERF_BEFORE_19', 'UM_TRAINING_AFFECTED_OTHER_SPORTS_PERF_UNDER_19']]

df_effect_performance.head()
# Q. Did you miss training due to UM related injury?

f, axes = plt.subplots(1, 1, figsize=(6, 4))

d = df_effect_performance.UM_INJURY_MISSED_TRAINING_BEFORE_19.value_counts()

axes.set(xlabel='DID YOU MISS UM TRAINING DUE TO INJURY BEFORE 19?')

sns.barplot(d.index, d.values)

d
# Q. Did UM training affect your academic performance?

f, axes = plt.subplots(1, 1, figsize=(6, 4))

d = df_effect_performance.UM_TRAINING_AFFECTED_ACADEMIC_PERF_BEFORE_19.value_counts()

axes.set(xlabel='DID UM TRAINING AFFECT YOUR ACADEMIC PERFROMANCE BEFORE 19?')

sns.barplot(d.index, d.values)

d
# Q. Did UM training affect your performance in other sports?

f, axes = plt.subplots(1, 1, figsize=(6, 4))

d = df_effect_performance.UM_TRAINING_AFFECTED_OTHER_SPORTS_PERF_UNDER_19.value_counts()

axes.set(xlabel='DID UM TRAINING AFFECT OTHER SPORTS PERFORMANCE BEFORE 19?')

sns.barplot(d.index, d.values)

d
df_adult_exercise_history = df[['RUNNING_AFTER_19', 'COMPETITIONS_5KM_AFTER_19', 'COMPETITIONS_10KM_AFTER_19', 'COMPETITIONS_22KM_AFTER_19', 'COMPETITIONS_44KM_AFTER_19', 

                                'COMPETITIONS_50KM_AFTER_19', 'COMPETITIONS_70KM_AFTER_19', 'COMPETITIONS_100KM_AFTER_19', 'COMPETITIONS_160KM_AFTER_19', 'COMPETITIONS_OVER_160KM_AFTER_19',

                                'YEAR_MOST_RECENT_MARATHON', 'DISTANCE_MOST_RECENT_MARATHON', 'INTEND_TO_RUN_ANOTHER_UM', 'RUN_TIMES_PW_LAST_12_MONTHS', 'RUN_HOURS_PW_LAST_12_MONTHS',

                                'RUN_DISTANCE_PW_LAST_12_MONTHS', 'RUN_%_ASPHALT_LAST_12_MONTHS', 'RUN_%_TRAIL_LAST_12_MONTHS', 'RUN_%_TRACK_LAST_12_MONTHS', 'RUN_%_TREADMILL_LAST_12_MONTHS',

                                'RUN_%_OTHER_LAST_12_MONTHS']]
# Q. Did you continue running after the age of 19?

f, axes = plt.subplots(1, 1, figsize=(6, 4))

d = df_adult_exercise_history.RUNNING_AFTER_19.value_counts()

axes.set(xlabel='RUNNING AFTER 19?')

sns.barplot(d.index, d.values)

d
# Q. Do you intend to run another UM after 19?

f, axes = plt.subplots(1, 1, figsize=(6, 4))

d = df_adult_exercise_history.INTEND_TO_RUN_ANOTHER_UM.value_counts()

axes.set(xlabel='INTEND TO RUN ANOTHER UM?')

sns.barplot(d.index, d.values)

d
# More running stats after the age of 19

# Categories

categ = ['To 50km', 'To 100km', 'To 160km', 'Over 160km']

freq = df_adult_exercise_history.DISTANCE_MOST_RECENT_MARATHON.value_counts().sort_index()

f, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].set(xlabel='DISTANCE MOST RECENT MARATHON')

sns.barplot(categ, freq.values, ax=axes[0],  )

sns.distplot(df_adult_exercise_history.RUN_DISTANCE_PW_LAST_12_MONTHS, kde=False, bins=10, ax=axes[1])

freq
# Reasons for stopping running

df_stop_running = df[['REASON_STOP_RUNNING_INJURY', 'REASON_STOP_RUNNING_WORK_STUDY',

                     'REASON_STOP_RUNNING_OTHER_SPORT', 'REASON_STOP_RUNNING_FAMILY',

                     'REASON_STOP_RUNNING_BORED']]

df_stop_running.head()
df_stop_running.fillna(value=0, inplace=True)

df_stop_running.head()
df_stop_reasons = df_stop_running.sum()

df_stop_reasons = pd.DataFrame(data=df_stop_reasons, index=df_stop_running.columns, columns=['NBR_RESPONSES'])

df_stop_reasons
# Make labels nice

df_stop_reasons['INDEX'] = df_stop_reasons.index.str.replace("REASON_STOP_RUNNING_", "")

df_stop_reasons['INDEX'] = df_stop_reasons['INDEX'].str.replace("_", " ")

df_stop_reasons.set_index('INDEX', inplace=True)
# Plot a pie chart with % values

rcParams['figure.figsize'] = 10,10

fig3, ax3 = plt.subplots()



# theme = plt.get_cmap('Greys')

# ax3.set_prop_cycle("color", [theme(1. * (i+2) / 12) for i in range(5)])

explode = [0.1, 0,0,0,0]



ax3.pie(df_stop_reasons.NBR_RESPONSES, labels=df_stop_reasons.index, autopct='%1.1f%%', startangle=45, pctdistance=0.7, labeldistance=1.05, shadow=True, explode=explode )

ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



# plt.savefig("fig4.pdf", orientation="landscape", bbox_inches='tight')

plt.show()

df_adult_medical_history = df[['CORONARY_ARTERY_DISEASE', 'HEART_ATTACK', 'ANGINA_PECTORIS', 'ARRHYTMIA', 'CONGESTIVE_HEART_FAILUTE', 'HYPERTENSION', 'TRANSIENT_ISCHEMIC_ATTACK',

                               'THROPMOBOPHLEBITIS', 'PERIPHERAL_VASCULAR_DISEASE', 'VARICOSE_VAINS','PULMONARY_EMBOLISM', 'RHEUMATIC_FEVER', 'EXERCISE_INDUCE_ASTHMA', 'OTHER_ASTHMA',

                               'EMPHYSEMA', 'CHRONIC_BRONCHITIS', 'LUNG_CANCER', 'PROSTATE_CANCER', 'BREAST_CANCER', 'COLON_CANCER', 'BLADDER_CANCER', 'MELANOMA', 'NON_HODGKIN_LYMPHOMA',

                               'KIDNEY_CANCER', 'THYROID_CANCER', 'PANCREATIC_CANCER', 'LEUKEMIA', 'OSTEOPOROSIS', 'CHRONIC_LOW_BACK_PAIN', 'CHRONIC_NECK_PAINS', 'KNEE_OSTEOARTHRITIS', 

                               'HIP_OSTEOARTHRITIS', 'OTHER_JOINTS_OSTEOARTHRITIS', 'ANTERIOR_CRUCIATE_LIGAMENT', 'KNEE_MENISCUS_INJURY', 'RHEUMATOID_ARTHRITIS', 'FIBROMYALGYA_SYNDROME',

                               'EXERCISE_RELATED_STRESS_FRACTURE', 'HIP_STRAIN', 'TORN_ATFL', 'TRAUMA_FRACTURE', 'TOMMY_JOHN_SURGERY', 'ANXIETY', 'DEPRESSION', 'MULTIPLE_SCLEROSIS',

                               'PARKINSON_DISEASE', 'EPILEPSY', 'POLIO_WITH_RECURRENT_WEAKNESS', 'ALZHEIMER_DISEASE', 'OCD', 'STOMACH_ULCER', 'DUODENAL_ULCER', 'COLITIS', 'GASTRIC_ACID_REFLUX',

                               'DIVERTICULITIS', 'GALL_BLADDER_DISEASE', 'LIVER_DISEASES', 'HEMORRHOIDS', 'INTERSTITIAL_CYSTITIS', 'DIABETES', 'CATARACTS', 'GLAUCOMA', 'MIGRAINES', 

                               'SIGNIFICANT_VISION_PROBLEM', 'SIGNIFICANT_HEARING_PROBLEM', 'SLEEPING_DIFFICULTIES', 'HYPERCHOLESTEROLEMIA', 'KYDNEY_DISORDERS', 'PROSTATE_PROBLEMS',

                               'THYROID_PROBLEMS', 'INCONTINENCE', 'ALLERGIES', 'OBESITY', 'ANOREXIA', 'ALCOHOL_DRUG_ABUSE', 'ANEMIA', 'PITUITARY_ADENOMA']]
df_medical_conditions = df_adult_medical_history.count()

df_medical_stats = df_medical_conditions.sort_values(ascending=False).head(50)

df_medical_stats = df_medical_stats[df_medical_stats.values>0]  # Only plot elements with freq > 0

df_medical_stats
# Plot a pie chart with % values

rcParams['figure.figsize'] = 16,16

fig4, ax4 = plt.subplots()

ax4.pie(df_medical_stats.values, labels=df_medical_stats.index, autopct='%1.1f%%', startangle=45, pctdistance=0.7, labeldistance=1.05, shadow=True)

ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
df_adult_medical_history_by_gender = df[['GENDER', 'EXERCISE_RELATED_STRESS_FRACTURE', 'HEMORRHOIDS', 'ANXIETY', 'DEPRESSION', 'CHRONIC_LOW_BACK_PAIN', 'KNEE_MENISCUS_INJURY', 'ARRHYTMIA',

                                         'HYPERTENSION', 'ALLERGIES', 'SLEEPING_DIFFICULTIES', 'GASTRIC_ACID_REFLUX', 'EXERCISE_INDUCE_ASTHMA', 'CHRONIC_NECK_PAINS', 'MIGRAINES', 'OBESITY',

                                         'ANTERIOR_CRUCIATE_LIGAMENT', 'KYDNEY_DISORDERS', 'THYROID_PROBLEMS', 'STOMACH_ULCER', 'COLITIS', 'ANOREXIA', 'OTHER_JOINTS_OSTEOARTHRITIS',

                                         'HYPERCHOLESTEROLEMIA', 'RHEUMATOID_ARTHRITIS', 'LEUKEMIA', 'PROSTATE_CANCER', 'PROSTATE_PROBLEMS', 'ANEMIA', 'TORN_ATFL', 'TRAUMA_FRACTURE',

                                         'TOMMY_JOHN_SURGERY', 'PITUITARY_ADENOMA', 'VARICOSE_VAINS', 'MULTIPLE_SCLEROSIS', 'LIVER_DISEASES', 'ALCOHOL_DRUG_ABUSE', 'HEART_ATTACK',

                                         'PROSTATE_PROBLEMS', 'GLAUCOMA', 'RHEUMATIC_FEVER', 'INTERSTITIAL_CYSTITIS', 'MELANOMA', 'OCD', 'PULMONARY_EMBOLISM']]
df_mh_bg = df_adult_medical_history_by_gender.groupby(by='GENDER').count()

dfm = df_mh_bg.transpose()

dfm.T
rcParams['figure.figsize'] = 9,9

fig5, ax5 = plt.subplots()

ax5.set_title('Main conditions reported by FEMALE participants after 19', fontsize=16)

ax5.pie(dfm.Female[dfm.Female>0], labels=dfm[dfm.Female>0].index, autopct='%1.1f%%', startangle=10, pctdistance=0.7, labeldistance=1.05, shadow=True),

ax5.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
rcParams['figure.figsize'] = 9,9

fig6, ax6 = plt.subplots()

ax6.set_title('Main conditions reported by MALE participants after 19', fontsize=16)

ax6.pie(dfm.Male[dfm.Male>0], labels=dfm[dfm.Male>0].index, autopct='%1.1f%%', startangle=50, pctdistance=0.7, labeldistance=1.05, shadow=True),

ax6.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
# Same analysis by gender for injuries before 19

df_injuries_before_19_by_gender = df[['GENDER', 'UM_INJURY_KNEE_PAIN', 'UM_INJURY_ANKLE_TENDINOPATHY', 'UM_INJURY_ACHILLES_TENDINOPATHY', 'UM_INJURY_HIP_FLEXOR_STRAIN',

                                      'UM_INJURY_ILIOTIBIAL_BAND_ISSUE', 'UM_INJURY_FOOT_STRESS_FRACTURE',  'UM_INJURY_TIBIA_STRESS_FRACTURE', 'UM_INJURY_CALF_STRAIN',

                                      'UM_INJURY_ANKLE_SPRAIN', 'UM_INJURY_HAMSTRING_STRAIN', 'UM_INJURY_PLANTAR_FASCIITIS', 'UM_INJURY_MEDIAL_TIBIAL_STRESS_SYNDROME', 

                                      'UM_INJURY_COMPARTMENT_SYNDROME', 'UM_INJURY_MORTON_NEUROMA', 'UM_INJURY_BURSITIS']]
df_1nj_19_bg = df_injuries_before_19_by_gender.groupby(by='GENDER').count()

df_i19 = df_1nj_19_bg.transpose()

df_i19['INDEX'] = df_i19.index.str.replace("UM_INJURY_", "")

df_i19['INDEX'] = df_i19['INDEX'].str.replace("_", " ")

df_i19.set_index('INDEX', inplace=True)

df_i19.T
rcParams['figure.figsize'] = 9,9

fig7, ax7 = plt.subplots()

ax7.set_title('Main injuries reported by FEMALE participants before 19', fontsize=16)

ax7.pie(df_i19.Female[df_i19.Female>0], labels=df_i19[df_i19.Female>0].index, autopct='%1.1f%%', startangle=30, pctdistance=0.7, labeldistance=1.05, shadow=True)

ax7.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
# Plot a pie chart with % values

rcParams['figure.figsize'] = 9,9

fig10, ax10 = plt.subplots()

ax10.set_title('Main injuries reported by MALE participants before 19', fontsize=16)

# theme = plt.get_cmap('Greys')

# ax10.set_prop_cycle("color", [theme(1. * (i+2) / 30) for i in range(15)])

explode = [0.1, 0,0,0,0,0,0,0,0,0,0,0,0,0]



ax10.pie(df_i19.Male[df_i19.Male>0], labels=df_i19[df_i19.Male>0].index, autopct='%1.1f%%', startangle=30, pctdistance=0.7, labeldistance=1.05, shadow=True, explode=explode)

ax10.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



# plt.savefig("fig3.pdf", orientation="landscape", bbox_inches='tight')

plt.show()

df_age_and_injuries = df[['AGE_1ST_UM', 'UM_INJURY_KNEE_PAIN', 'UM_INJURY_ANKLE_TENDINOPATHY', 'UM_INJURY_ACHILLES_TENDINOPATHY',

                          'UM_INJURY_HIP_FLEXOR_STRAIN', 'UM_INJURY_ILIOTIBIAL_BAND_ISSUE', 'UM_INJURY_FOOT_STRESS_FRACTURE', 

                          'UM_INJURY_TIBIA_STRESS_FRACTURE', 'UM_INJURY_CALF_STRAIN', 'UM_INJURY_ANKLE_SPRAIN',

                          'UM_INJURY_HAMSTRING_STRAIN', 'UM_INJURY_PLANTAR_FASCIITIS', 'UM_INJURY_MEDIAL_TIBIAL_STRESS_SYNDROME',

                          'UM_INJURY_COMPARTMENT_SYNDROME', 'UM_INJURY_MORTON_NEUROMA', 'UM_INJURY_BURSITIS',

                          'EXERCISE_RELATED_STRESS_FRACTURE', 'HEMORRHOIDS', 'ANXIETY', 'DEPRESSION', 'CHRONIC_LOW_BACK_PAIN',

                          'KNEE_MENISCUS_INJURY', 'ARRHYTMIA', 'HYPERTENSION', 'ALLERGIES', 'SLEEPING_DIFFICULTIES', 

                          'GASTRIC_ACID_REFLUX', 'EXERCISE_INDUCE_ASTHMA', 'CHRONIC_NECK_PAINS', 'MIGRAINES', 'OBESITY' ]]
df_age_and_injuries_stats = df_age_and_injuries.groupby('AGE_1ST_UM').count()

df_age_and_injuries_stats.T
df_nbr_UM_and_injuries = df[['NBR_UM_BEFORE_19', 'UM_INJURY_KNEE_PAIN', 'UM_INJURY_ANKLE_TENDINOPATHY', 'UM_INJURY_ACHILLES_TENDINOPATHY',

                          'UM_INJURY_HIP_FLEXOR_STRAIN', 'UM_INJURY_ILIOTIBIAL_BAND_ISSUE', 'UM_INJURY_FOOT_STRESS_FRACTURE', 

                          'UM_INJURY_TIBIA_STRESS_FRACTURE', 'UM_INJURY_CALF_STRAIN', 'UM_INJURY_ANKLE_SPRAIN',

                          'UM_INJURY_HAMSTRING_STRAIN', 'UM_INJURY_PLANTAR_FASCIITIS', 'UM_INJURY_MEDIAL_TIBIAL_STRESS_SYNDROME',

                          'UM_INJURY_COMPARTMENT_SYNDROME', 'UM_INJURY_MORTON_NEUROMA', 'UM_INJURY_BURSITIS',

                          'EXERCISE_RELATED_STRESS_FRACTURE', 'HEMORRHOIDS', 'ANXIETY', 'DEPRESSION', 'CHRONIC_LOW_BACK_PAIN',

                          'KNEE_MENISCUS_INJURY', 'ARRHYTMIA', 'HYPERTENSION', 'ALLERGIES', 'SLEEPING_DIFFICULTIES', 

                          'GASTRIC_ACID_REFLUX', 'EXERCISE_INDUCE_ASTHMA', 'CHRONIC_NECK_PAINS', 'MIGRAINES', 'OBESITY']]
df_nbr_UM_and_injuries = df_nbr_UM_and_injuries.groupby('NBR_UM_BEFORE_19').count()

df_nbr_UM_and_injuries.T
df_women = df[['AGE_1ST_PERIOD', 'STOPPED_HAVING_PERIODS', 'EVER_BEEN_PREGNANT', 'NBR_PREGNANCIES', 'NBR_BIRTHS', 'AGE_1ST_PREGNANCY', 'AGE_MOST_RECENT_PREGNANCY']]

df_women.head()
# Medical history stats for women

f, axes = plt.subplots(1, 3, figsize=(18, 4))

sns.distplot(df_women.AGE_1ST_PERIOD, kde=False, bins=6, ax=axes[0])

sns.distplot(df_women.AGE_1ST_PREGNANCY, kde=False, bins=6, ax=axes[1])

sns.distplot(df_women.AGE_MOST_RECENT_PREGNANCY, kde=False, bins=6, ax=axes[2])
df_women[['AGE_1ST_PERIOD', 'AGE_1ST_PREGNANCY', 'AGE_MOST_RECENT_PREGNANCY']].describe()
# Medical history stats for women (pregnancy)

f, axes = plt.subplots(1, 4, figsize=(24, 4))

w0 = df_women.EVER_BEEN_PREGNANT.value_counts()

w1 = df_women.NBR_PREGNANCIES.value_counts()

w2 = df_women.NBR_BIRTHS.value_counts()

w3 = df_women.STOPPED_HAVING_PERIODS.value_counts()

axes[0].set(xlabel='EVER BEEN PREGNANT?')

axes[1].set(xlabel='NUMBER OF PREGNANCIES')

axes[2].set(xlabel='NUMBER OF BIRTHS')

axes[3].set(xlabel='STOPPED HAVING PERIODS?')

sns.barplot(w0.index, w0.values, ax=axes[0])

sns.barplot(w1.index, w1.values, ax=axes[1])

sns.barplot(w2.index, w2.values, ax=axes[2])

sns.barplot(w3.index, w3.values, ax=axes[3])

print(w0)

print()

print(w1)

print()

print(w2)

print()

print(w3)
df_sub_assess = df[['CONSIDER_YOURSELF_HEALTHY', 'UM_RUNNING_BEFORE_19_EFFECT_IN_YOUR_HEALTH', 'RECOMMEND_CHILDREN_TO_RUN_UM']]

df_sub_assess
# Self-assessment key stats

f, axes = plt.subplots(1, 3, figsize=(18, 4))

sa0 = df_sub_assess.CONSIDER_YOURSELF_HEALTHY.value_counts()

sa1 = df_sub_assess.UM_RUNNING_BEFORE_19_EFFECT_IN_YOUR_HEALTH.value_counts()

sa2 = df_sub_assess.RECOMMEND_CHILDREN_TO_RUN_UM.value_counts()

axes[0].set(xlabel='CONSIDER YOURSELF HEALTHY?')

axes[1].set(xlabel='EFFECT IN YOUR HEALTH OF UM RUNNING BEFORE 19?')

axes[2].set(xlabel='RECOMMEND CHILDREN TO RUN UM?')

sns.barplot(sa0.index, sa0.values, ax=axes[0])

sns.barplot(sa1.index, sa1.values, ax=axes[1])

sns.barplot(sa2.index, sa2.values, ax=axes[2])

print(sa0)

print()

print(sa1)

print()

print(sa2)
# Additional pie charts 



fig, ax = plt.subplots(1,2, figsize=(16, 8))



# theme = plt.get_cmap('Greys')

# ax[0].set_prop_cycle("color", [theme(1. * (i+2) / 10) for i in range(5)])

# ax[1].set_prop_cycle("color", [theme(1. * (i+2) / 10) for i in range(5)])



ax[0].set_title('Effect of ultra running during childhood ', fontsize=16)

ax[1].set_title('Would you recommend ultra running during childhood?', fontsize=16)



ax[0].pie(sa1.values, labels=sa1.index, autopct='%1.1f%%', startangle=45, pctdistance=0.7, labeldistance=1.05, shadow=True)

ax[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[1].pie(sa2.values, labels=sa2.index, autopct='%1.1f%%', startangle=45, pctdistance=0.7, labeldistance=1.05, shadow=True)

ax[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



# plt.savefig("fig5.pdf")

plt.show()

# Exercise addiction score

df_addict_score = df[['SPORT_IS_MOST_IMPORTANT_THING_IN_MY_LIFE', 'CONFLICTS_ARISEN_WITH_FAMILY_TOO_MUCH_SPORT', 'USE_EXERCISE_TO_CHANGE_MOOD',

                      'INCREASED_AMOUNT_OF_SPORT_OVERTIME', 'FEEL_MOODY_WHEN_MISS_TRAINING', 'CANT_CUT_DOWN_SPORT_AMOUNT']]



df_addict_score
dict_response = {'Strongly disagree' : 1,

                'Disagree' : 2,

                'Neither agree nor disagree' : 3,

                'Agree' : 4,

                'Strongly agree' : 5

                }



# Function that returns the value from the dictionary

def response_value(st):

  return dict_response[st]
for x in dict_response:

  df_addict_score.replace(to_replace=x, value=response_value(x), inplace=True)
df_addict_score
s = df_addict_score.sum(axis=1)

df_addict_score['SPORT_ADDICTION_SCORE'] = s

df_addict_score
# Sports addiction score distribution plot

f, axes = plt.subplots(1, 1, figsize=(12, 6))

sns.distplot(df_addict_score.SPORT_ADDICTION_SCORE, kde=False, bins=(0,13,24,30))