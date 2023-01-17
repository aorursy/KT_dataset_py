import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
cardio_df = pd.read_csv('../input/cardiovascular-disease-dataset/cardio_train.csv',sep=';')

cardio_df.head()
cardio_df.drop(columns=['id'], inplace=True)

cardio_df.columns = ['age(day)', 'gender', 'height(cm)', 'weight(kg)', 'systolic', 'diastolic', 

                  'cholesterol', 'glucose', 'smoker', 'alcohol', 'physical activity', 'CVD']
cardio_df['age(years)'] = cardio_df['age(day)'].apply(lambda x: math.floor(x / 365))
# Import relevant libraries for visualization

import matplotlib.pyplot as plt

import seaborn as sns
#How many people in this dataset have CVD?

cardio_df['CVD'].value_counts().plot.pie(figsize=(5, 5))

plt.title("Number of people with CVD vs not CVD")

plt.show()
smoker_group = cardio_df.groupby(['smoker'], as_index=False)

count = smoker_group.count()['CVD']

num_smokers = smoker_group.sum()['CVD']

percentage_smokers_with_cvd = num_smokers / count * 100

plt.bar(x=[0,1], height=percentage_smokers_with_cvd, align='center', tick_label=["Non-Smoker", "Smoker"])

plt.title("Percentage of Non-smokers vs Smokers with CVD")

plt.show()
smoker_group.mean()[['systolic', 'diastolic']]
cholesterol_group = cardio_df.groupby(['cholesterol'], as_index=False)

cholesterol_count = cholesterol_group.count()['CVD']

cholesterol_sum = cholesterol_group.sum()['CVD']

cholesterol_percentage = cholesterol_sum / cholesterol_count * 100



cholesterol_percentage



plt.bar(x=[0,1,2], height=cholesterol_percentage, align='center', tick_label=["1", "2", "3"])

plt.title("Percentage of Cholesterol level with CVD")

plt.ylabel("% of people with CVD")

plt.xlabel("Cholesterol Level")

plt.show()
gender_group = cardio_df.groupby(['gender'], as_index=False)

gender_count = gender_group.count()['CVD']

gender_sum = gender_group.sum()['CVD']

gender_percentage = gender_sum / gender_count * 100



# 0 = 1 (women)

# 1 = 2 (men)

plt.bar(x=[0,1], height=gender_percentage, yerr=gender_percentage.std(), align='center', tick_label=["Women", "Men"])

plt.title("Gender vs CVD")

plt.ylabel("% of people with CVD")

plt.xlabel("Gender")

plt.show()
alcohol_group = cardio_df.groupby(['alcohol'], as_index=False)

alcohol_count = alcohol_group.count()['CVD']

alcohol_sum = alcohol_group.sum()['CVD']

alcohol_percentage = alcohol_sum / alcohol_count * 100



plt.bar(x=[0,1], height=alcohol_percentage, yerr=alcohol_percentage.std(), align='center', tick_label=["Non-Drinkers", "Drinkers"])

plt.title("Effect of Alcohol on CVD")

plt.ylabel("% of people with CVD")

plt.xlabel("Drinking Status")

plt.show()
glucose_group = cardio_df.groupby(['glucose'], as_index=False)

glucose_count = glucose_group.count()['CVD']

glucose_sum = glucose_group.sum()['CVD']

glucose_percentage = glucose_sum / glucose_count * 100



plt.bar(x=[0,1,2], height=glucose_percentage, yerr=glucose_percentage.std(), align='center', tick_label=["1", "2", "3"])

plt.title("Effect of Glucose levels on CVD")

plt.ylabel("% of people with CVD")

plt.xlabel("Glucose Levels")

plt.show()
physical_group = cardio_df.groupby(['physical activity'], as_index=False)

physical_count = physical_group.count()['CVD']

physical_sum = physical_group.sum()['CVD']

physical_percentage = physical_sum / physical_count * 100



plt.bar(x=[0,1], height=physical_percentage, yerr=physical_percentage.std(), align='center', tick_label=["non-active", "active"])

plt.title("Effect of Physical Activity on CVD")

plt.ylabel("% of people with CVD")

plt.xlabel("Activity Status")

plt.show()
def get_high_SBP(sbp):

    if sbp > 140:

        return 1

    else:

        return 0

    

def get_high_DBP(dbp):

    if dbp > 90:

        return 1

    else:

        return 0



cardio_df['high SBP'] = cardio_df['systolic'].apply(get_high_SBP)

cardio_df['high DBP'] = cardio_df['diastolic'].apply(get_high_DBP)



high_sbp_group = cardio_df.groupby(['high SBP'], as_index=False)

high_sbp_count = high_sbp_group.count()['CVD']

high_sbp_sum = high_sbp_group.sum()['CVD']

high_sbp_percentage = high_sbp_sum / high_sbp_count * 100



high_dbp_group = cardio_df.groupby(['high DBP'], as_index=False)

high_dbp_count = high_dbp_group.count()['CVD']

high_dbp_sum = high_dbp_group.sum()['CVD']

high_dbp_percentage = high_dbp_sum / high_dbp_count * 100



fig = plt.figure(figsize=(6,10))

ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)

ax1.bar(x=[0,1], height=high_sbp_percentage, yerr=high_sbp_percentage.std(), align='center', tick_label=["not high", "high"])

ax1.set_title('High SBP and CVD')

ax1.set_ylabel("% of people with CVD")

ax1.set_xlabel("SBP status")



ax2.bar(x=[0,1], height=high_dbp_percentage, yerr=high_dbp_percentage.std(), align='center', tick_label=["not high", "high"])

ax2.set_title('High DBP and CVD')

ax2.set_ylabel("% of people with CVD")

ax2.set_xlabel("DBP status")



fig.show()
cardio_df['bmi'] = cardio_df['weight(kg)'] / (cardio_df['height(cm)'] / 100)**2
def get_bmi_groups(bmi):

    if bmi >= 16 and bmi <18.5:

        return "Underweight"

    elif bmi >= 18.5 and bmi < 25 :

        return "Normal weight"

    elif bmi >= 25 and bmi < 30:

        return "Overweight"

    elif bmi >= 30 and bmi < 35:

        return "Obese Class I (Moderately obese)"

    elif bmi >= 35 and bmi < 40:

        return "Obese Class II (Severely obese)"

    elif bmi >= 40 and bmi < 45:

        return "Obese Class III (Very severely obese)"

    elif bmi >= 45 and bmi < 50:

        return "Obese Class IV (Morbidly Obese)"

    elif bmi >= 50 and bmi < 60:

        return "Obese Class V (Super Obese)"

    elif bmi >= 60:

        return "Obese Class VI (Hyper Obese)"

    

    

cardio_df["bmi_group"] = cardio_df['bmi'].apply(get_bmi_groups)

cardio_df["bmi_group"] = cardio_df["bmi_group"].astype('category')



# Lets visualize the results



bmi_group_groups = cardio_df.groupby(['bmi_group'], as_index=False)

bmi_group_count = bmi_group_groups.count()['CVD']

bmi_group_sum = bmi_group_groups.sum()['CVD']

bmi_group_percentage = bmi_group_sum / bmi_group_count * 100



"""

0 = Normal Weight

1 = Obese Class 1

2 = Obese Class 2

3 = Obese Class 3

4 = Obese Class 4

5 = Obese Class 5

6 = Obese Class 6

7 = Overweight

8 = Underweight

"""



plt.figure(figsize=(18,4))

plt.bar(x=range(0,9), height=bmi_group_percentage, yerr=bmi_group_percentage.std(), align='center',

       tick_label=["Normal Weight", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Overweight", "Underweight"])

plt.title("BMI Class vs CVD rate")

plt.xlabel("BMI Group")

plt.ylabel("% of people with CVD")

plt.show()
def predicted_men_waist(bmi, age):

    b0 = 22.61306

    b1BMI = 2.520738*(bmi)

    b2AGE = 0.1583812*(age)

    return b0 + b1BMI + b2AGE



def predicted_women_waist(bmi, age):

    c0 = 28.81919

    c1BMI = 2.218007*(bmi)

    age_35 = 0

    if age > 35:

        age_35 = 1

    

    c2IAGE35 = -3.688953 * age_35

    IAGE35 = -0.6570163 * age_35

            

    c3AGEi = 0.125975*(age)

    

    return (c0 + c1BMI + c2IAGE35 + IAGE35 + c3AGEi)



def man_or_woman_waist(row):

    if row['gender'] == 1:

        return predicted_women_waist(row['bmi'], row['age(years)'])

    else: 

        return predicted_men_waist(row['bmi'], row['age(years)'])



cardio_df['waist(cm)'] = cardio_df.apply(man_or_woman_waist, axis=1)
# Lets visualize the effect of waist circumference on CVD using cut offs from guidelines

# Cut off for men is > 103 cm

# Cut off for women is > 88 cm



def get_if_waist_over_cutoff(row):

    if row['gender'] == 1 and row['waist(cm)'] > 88:

        return 1

    elif row['gender'] == 2 and row['waist(cm)'] > 103: 

        return 1

    

    return 0



cardio_df['waist cut off'] = cardio_df.apply(get_if_waist_over_cutoff, axis=1)



# Visualizing the results 



waist_cut_off_group = cardio_df.groupby(['waist cut off'], as_index=False)

waist_cut_off_group_count = waist_cut_off_group.count()['CVD']

waist_cut_off_group_sum = waist_cut_off_group.sum()['CVD']

waist_cut_off_group_percentage = waist_cut_off_group_sum / waist_cut_off_group_count * 100



plt.bar(x=[0,1], height=waist_cut_off_group_percentage, yerr=waist_cut_off_group_percentage.std(), align='center', 

        tick_label=["Below Cut Off", "Over Cut Off"])

plt.title("Waist Cut Off vs CVD")

plt.ylabel("% of people with CVD")

plt.xlabel("Waist Cut Off or Not")

plt.show()
def mean_arterial_pressure(row):

    mean_ap = (row['systolic'] + 2*row['diastolic']) / 3

    return mean_ap



cardio_df['map'] = cardio_df.apply(mean_arterial_pressure, axis=1)
def get_pulse_pressure(row):

    pulse_pressure = (row['systolic'] - row['diastolic'])

    return pulse_pressure



cardio_df['pulse_pressure'] = cardio_df.apply(get_pulse_pressure, axis=1)
# Encode BMI group to catergoical values so our classifer can handle the data.

cardio_df['bmi_group'] = cardio_df['bmi_group'].astype('category')

cardio_df['bmi_group_cat'] = cardio_df['bmi_group'].cat.codes



cardio_df.columns
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split 

from sklearn.metrics import f1_score



rf_clf = RandomForestClassifier()



## Lets just apply all columns and see what happens

X = cardio_df[['gender', 'height(cm)', 'weight(kg)', 'systolic',

               'diastolic', 'cholesterol', 'glucose', 'smoker', 'alcohol',

               'physical activity', 'age(years)', 'high SBP', 'high DBP', 'bmi',

               'waist(cm)', 'waist cut off', 'map', 'pulse_pressure',

               'bmi_group_cat']]



y = cardio_df['CVD']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf_clf.fit(X_train, y_train)

predictions = rf_clf.predict(X_test)



f1 = f1_score(y_test, predictions) 



print(f1)
from sklearn.feature_selection import RFE



rfe = RFE(rf_clf, 3, step=1)

rfe.fit(X, y)



feature_rank = pd.DataFrame()

feature_rank['Features'] = ['gender', 'height(cm)', 'weight(kg)', 'systolic',

                           'diastolic', 'cholesterol', 'glucose', 'smoker', 'alcohol',

                           'physical activity', 'age(years)', 'high SBP', 'high DBP', 'bmi',

                           'waist(cm)', 'waist cut off', 'map', 'pulse_pressure',

                           'bmi_group_cat']

feature_rank['Ranking'] = rfe.ranking_

feature_rank
top_3_features_rf_clf = RandomForestClassifier()



X = cardio_df[['systolic', 'bmi', 'waist(cm)']]

y = cardio_df['CVD']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



top_3_features_rf_clf.fit(X_train, y_train)

predictions = top_3_features_rf_clf.predict(X_test)



f1_top_3 = f1_score(y_test, predictions) 



print(f1)

print(f1_top_3)
import xgboost as xgb

import catboost as cb

import lightgbm as lgb



# XGB model

data_dmatrix = xgb.DMatrix(data=X, label=y)

xgb_clf = xgb.XGBClassifier()

xgb_clf.fit(X_train, y_train)

xgb_predictions_train = xgb_clf.predict(X_train)



# CatBoost model

cb_clf = cb.CatBoostClassifier(silent=True)

cb_clf.fit(X_train, y_train)

cb_predictions_train = cb_clf.predict(X_train)



# LightGBM model

lg_clf = lgb.LGBMClassifier(silent=True)

lg_clf.fit(X_train, y_train)

lg_predictions_train = lg_clf.predict(X_train)



ensemble_df = X_train

ensemble_df['xgb'] = xgb_predictions_train

ensemble_df['cb'] = cb_predictions_train

ensemble_df['lg'] = lg_predictions_train



# Final ensemble model will be XGB

final_clf = xgb.XGBClassifier()

final_clf.fit(ensemble_df, y_train)
ensemble_df.head()
def ensemble_predict(X_test, xgb_clf, cb_clf, lg_clf, final_clf):

    xgb_predict = xgb_clf.predict(X_test)

    cb_predict = cb_clf.predict(X_test)

    lg_predict = lg_clf.predict(X_test)

    

    ensemble_df = X_test.copy(deep=True)

    ensemble_df['xgb'] = xgb_predict

    ensemble_df['cb'] = cb_predict

    ensemble_df['lg'] = lg_predict

    print(ensemble_df.columns)

    final_predictions = final_clf.predict(ensemble_df)

    

    return final_predictions





final_predictions = ensemble_predict(X_test, xgb_clf, cb_clf, lg_clf, final_clf)
ensemble_f1 = f1_score(y_test, final_predictions)

print(ensemble_f1)