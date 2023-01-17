#import libraries, including...

import os 

#data structuring, statistics, and other math we'll use
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

#visualization
%matplotlib inline
import seaborn as sns
sns.light_palette("purple", as_cmap=True)

#machine learning and statistics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats.stats import pearsonr
# Load our school explorer data and update the index column name to match our other datasets

school_data = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv', index_col = 'Location Code')
SHSAT_results = pd.read_csv('../input/20172018-shsat-admissions-test-offers/2017-2018_SHSAT_Admissions_Test_Offers_By_Sending_School.csv', index_col = 'Feeder School DBN')
school_safety = pd.read_csv('../input/ny-2010-2016-school-safety-report/2010-2016-school-safety-report.csv', index_col = 'DBN') #Open NYC dataset from Kaggle
school_demographics = pd.read_csv('../input/ny-school-demographics-and-accountability-snapshot/2006-2012-school-demographics-and-accountability-snapshot.csv', index_col = 'DBN') #Open NYC dataset from Kaggle
specialized_hs_locations = pd.read_csv('../input/nyc-specialized-high-school-locations/elite_eight_data.csv') #Source: Infocusp competition submission. Link: https://www.kaggle.com/infocusp/recommendations-to-passnyc-based-on-data-analysis/data
SHSAT_testing_locations = pd.read_csv('../input/shsat-testing-locations/SHSAT_testing_locations.csv') #Private dataset containing lat/

#Note - I used a private version of the specialized_hs_locations dataset because of a typo in the original, 
#but I give full credit to Infocusp

raw_datasets = [school_data, SHSAT_results, school_safety, specialized_hs_locations, school_demographics]
# INITIAL CLEAN-UP (aka preprocessing)

# Removing unecessary columns before cleaning and merging

SHSAT_results.drop(['Feeder School Name'], axis=1, inplace=True) 
school_safety = school_safety[['Major N', 'Oth N', 'NoCrim N', 'Prop N', 'Vio N']]
school_demographics = pd.DataFrame(school_demographics[['frl_percent','sped_percent']])

# fix values before running computations

# In our SHSAT results, replacing '0-5' with 0 in the 'Count' columns

SHSAT_results['# of Test-takers'] = SHSAT_results['Count of Testers'][(SHSAT_results['Count of Testers'].astype(str)) != "0-5"]
SHSAT_results['# of Offers'] = SHSAT_results['Count of Offers'][(SHSAT_results['Count of Offers'].astype(str)) != "0-5"]

SHSAT_results.fillna(value = 0, inplace = True)

school_safety = school_safety.loc[school_safety.index.dropna()] # remove rows with blank DBNs

# Convert dollars, percents, and numbers to the correct format; swap out non-numbers with 0s

dollar_columns = ['School Income Estimate']
percent_columns = ['Percentage of Black/Hispanic students','Percent Asian', 'Percent Black', 'Percent ELL', 
                   'Percent Hispanic', 'Percent Black / Hispanic','Percent White', 'Student Attendance Rate', 
                   'Percent of Students Chronically Absent', 'Rigorous Instruction %', 
                   'Collaborative Teachers %', 'Supportive Environment %','Effective School Leadership %', 
                   'Strong Family-Community Ties %', 'Trust %', 'sped_percent']

def fix_dollars(df):
    for cols in dollar_columns:
        if cols in df:
            df[cols] = df[cols].astype(np.object_).str.replace('$','').str.replace(',','').astype(float)

def fix_percents(df):
    for cols in percent_columns:
        if cols in df:
            df[cols] = (df[cols].astype(np.object_).str.replace('%','').astype(float) / 100)

for dfs in raw_datasets:
    fix_dollars(dfs)
    fix_percents(dfs)
# Making new columns with secondary datasets

school_safety['Average Total Crimes 2013-2016'] = school_safety[['Major N', 'Oth N', 'NoCrim N', 'Prop N', 'Vio N']].sum(axis=1)
school_safety = school_safety.groupby(['DBN']).mean()
school_safety = school_safety['Average Total Crimes 2013-2016']
SHSAT_results = SHSAT_results[['Count of Students in HS Admissions','# of Test-takers','# of Offers']]
school_demographics = (school_demographics.groupby('DBN').mean() / 100)
school_demographics.columns = ['Free Meals %', 'Special Ed %']
# MERGING DATASETS

# merge datasets together
school_data = school_data.join(SHSAT_results).join(school_safety).join(school_demographics)
school_data = school_data.dropna(subset=['Count of Students in HS Admissions'])
## Feature creation within our aggregated dataset ##

# Creating our dependent, or predicted, variable
school_data['% Taking SHSAT'] = (school_data['# of Test-takers'].astype(float) 
                                 / school_data['Count of Students in HS Admissions'].astype(float))

school_data['% Receiving Offers'] = (school_data['# of Offers'].astype(float) 
                                     / school_data['Count of Students in HS Admissions'].astype(float))
   
# Estimate the % of HS candidates receiving 4s for outreach. Our 2017-2018 SHSAT cohort were 6th graders in 2015-16.                            
school_data['% of 2017-18 SHSAT Takers Receiving 4s in 2016'] = ((school_data['Grade 6 Math 4s - All Students'].astype(float) 
                                                / school_data['Grade 6 Math - All Students Tested'].astype(float))
                                                + (school_data['Grade 6 ELA 4s - All Students'].astype(float) 
                                                / school_data['Grade 6 ELA - All Students Tested'].astype(float)) 
                                                / 2)

# Estimate the % of 5th graders receiving 4s for medium-term interventions, since this is the 2018-2019 class.
school_data['% of 2018-19 SHSAT Takers Receiving 4s in 2016'] =  ((school_data['Grade 5 Math 4s - All Students'].astype(float) 
                                                / school_data['Grade 5 Math - All Students Tested'].astype(float))
                                                + (school_data['Grade 5 ELA 4s - All Students'].astype(float) 
                                                / school_data['Grade 5 ELA - All Students Tested'].astype(float)) 
                                                / 2)

# Aggregate % of students with 4s for each grade level to see how strong of a predictor each is
school_data['% of 7th Graders Receiving 4s'] =  ((school_data['Grade 7 Math 4s - All Students'].astype(float) 
                                                / school_data['Grade 7 Math - All Students Tested'].astype(float))
                                                + (school_data['Grade 7 ELA 4s - All Students'].astype(float) 
                                                / school_data['Grade 7 ELA - All Students Tested'].astype(float)) 
                                                / 2)

school_data['% of 8th Graders Receiving 4s'] =  ((school_data['Grade 8 Math 4s - All Students'].astype(float) 
                                                / school_data['Grade 8 Math - All Students Tested'].astype(float))
                                                + (school_data['Grade 8 ELA 4s - All Students'].astype(float) 
                                                / school_data['Grade 8 ELA - All Students Tested'].astype(float)) 
                                                / 2)

school_data['% of 3rd Graders Receiving 4s'] =  ((school_data['Grade 3 Math 4s - All Students'].astype(float) 
                                                / school_data['Grade 3 Math - All Students tested'].astype(float))
                                                + (school_data['Grade 3 ELA 4s - All Students'].astype(float) 
                                                / school_data['Grade 3 ELA - All Students Tested'].astype(float)) 
                                                / 2)

school_data['% of 4th Graders Receiving 4s'] =  ((school_data['Grade 4 Math 4s - All Students'].astype(float) 
                                                / school_data['Grade 4 Math - All Students Tested'].astype(float))
                                                + (school_data['Grade 4 ELA 4s - All Students'].astype(float) 
                                                / school_data['Grade 4 ELA - All Students Tested'].astype(float)) 
                                                / 2)

# Average the CC scores accross schools for longer-term interventions.
school_data['Average CC Scores'] = (school_data['Average ELA Proficiency'] + 
                                    school_data['Average Math Proficiency'] / 2)
# calculate distance between schools and nearest testing center and specialized high school

# taken from https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas and edited

def haversine(longitude1, latitude1, target_dataset):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    distances = []
    lon1, lat1 = map(radians, [longitude1, latitude1])
    for i in range(len(target_dataset)):
    # convert decimal degrees to radians 
        lon2, lat2 = map(radians, [target_dataset.loc[i,"Long"], target_dataset.loc[i,"Lat"]])
        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        km = 6367 * c
        distances.append(km)
    return min(distances)

# these loops iterate separately over our schools to find the nearest SHS and testing locations, respectively
for index, row in school_data.iterrows():
    school_data.loc[index, 'KM to nearest SHS'] = haversine(row['Longitude'], row['Latitude'], specialized_hs_locations)

for index, row in school_data.iterrows():
    school_data.loc[index, 'KM to nearest Testing Location'] = haversine(row['Longitude'], row['Latitude'], SHSAT_testing_locations)
# Create subsets for segmenting students, predictors, target, and school ID for easy review

demographics = ['Percent Black / Hispanic', 'Economic Need Index', 'Percent ELL', 
                'Community School?', 'Special Ed %']

model_predictors = ['KM to nearest SHS','KM to nearest Testing Location', 'Average Total Crimes 2013-2016',
                   'Economic Need Index','Rigorous Instruction %', 'Collaborative Teachers %', 
                    'Supportive Environment %', 'Effective School Leadership %', 
                    'Strong Family-Community Ties %', 'Trust %', 'Special Ed %', 'Percent of Students Chronically Absent'] 
                    #Some data we expect to add value as a predictor and as a means of segmenting schools.

model_target = ['% Taking SHSAT']

# Start with a predictor aimed at the short-term to identify the highest-impact schools ('quick wins').
# I call these variables 'short_term' because they'll be easy to act on right away.

short_term_targets = school_data[model_predictors]
short_term_targets = short_term_targets.join(school_data['% of 2017-18 SHSAT Takers Receiving 4s in 2016'])
short_term_targets = short_term_targets.fillna(short_term_targets.mean())

Rand_Forest = RandomForestRegressor(min_samples_leaf=10, n_estimators=100, n_jobs=1, random_state=0)
# parameter tuning
RF_params = {"max_depth": [3,5,6,None],
              "max_features": [0.33,0.67,1.0],
              "min_samples_leaf": [4,9,16]}
RF_Gridsearch = GridSearchCV(Rand_Forest, RF_params, cv=3, n_jobs=1)
RF_Gridsearch.fit(short_term_targets, school_data[model_target].values.ravel())
Rand_Forest = Rand_Forest.set_params(**RF_Gridsearch.best_params_)
Rand_Forest.fit(short_term_targets, school_data[model_target])

# delete variables which are not used or almost unused to keep the model on the simpler side
short_term_targets = short_term_targets.loc[:, Rand_Forest.feature_importances_>0.01]
Rand_Forest.fit(short_term_targets, school_data[model_target])

# Save the model's predictions as a new variable
short_term_prediction = school_data[model_target]
short_term_prediction['Prediction'] = Rand_Forest.predict(short_term_targets)
# Visualize predictor strength

short_term_weights = pd.Series(index=short_term_targets.columns, data=Rand_Forest.feature_importances_).sort_values(ascending=True)
short_term_weights.plot(kind='barh', figsize=(10,15), color="purple");
short_term_prediction['Underperformance Gap'] = (short_term_prediction["Prediction"] - 
                                                 short_term_prediction['% Taking SHSAT'])

short_term_prediction = short_term_prediction.sort_values('Underperformance Gap', ascending=False)
short_term_prediction['Underperformance in # of Eight Graders'] = (short_term_prediction['Underperformance Gap'].astype(float)
                                                                   * school_data['Count of Students in HS Admissions'].astype(float))
short_term_prediction = short_term_prediction.join(school_data[['School Name', 'Address (Full)']])

short_term_prediction.head()
# Prioritize schools

# Students scoring 4s on Common Core exams were excluded as a weighing variable because they could favor schools based on which grade levels they serve.

prioritization_metrics = pd.DataFrame((school_data['Economic Need Index'] + school_data['Percent ELL'] 
                          + school_data['Percent Black / Hispanic'].astype(float) / 3) 
                                      * school_data['Count of Students in HS Admissions'].astype(float))

short_term_prediction["Prioritization Score"] = prioritization_metrics
short_term_prediction = short_term_prediction.join(school_data[demographics])
# Estimate the % of 7th graders receiving 4s for longer-term interventions.
test_scores_by_grade = ['% of 3rd Graders Receiving 4s','% of 4th Graders Receiving 4s', 
                        '% of 2018-19 SHSAT Takers Receiving 4s in 2016', '% of 2017-18 SHSAT Takers Receiving 4s in 2016',
                       '% of 7th Graders Receiving 4s', '% of 8th Graders Receiving 4s']

correlations = pd.DataFrame()

DV = school_data['% Taking SHSAT']

for items in test_scores_by_grade:
    grade_check = school_data[items] >= 0
    #check that the school has students of that grade
    grade_test_scores = school_data[items]
    correlations[items] = pearsonr(DV[grade_check],grade_test_scores[grade_check])
    
correlations = correlations.loc[0,:]
# Perform the same analysis using 7th grade scores to find schools worth targeting with tutoring programs

targets_excl_grades = school_data[model_predictors]
targets_excl_grades = targets_excl_grades.fillna(targets_excl_grades.mean())

Rand_Forest_2 = RandomForestRegressor(min_samples_leaf=10, n_estimators=100, n_jobs=1, random_state=0)

RF_Gridsearch.fit(targets_excl_grades, school_data[model_target].values.ravel())
Rand_Forest_2 = Rand_Forest_2.set_params(**RF_Gridsearch.best_params_)
Rand_Forest_2.fit(targets_excl_grades, school_data[model_target])

# delete variables which are not used or almost unused to keep the model on the simpler side
targets_excl_grades = targets_excl_grades.loc[:, Rand_Forest_2.feature_importances_>0.015]
Rand_Forest_2.fit(targets_excl_grades, school_data[model_target])

# Save the model's predictions as a new variable
prediction_excl_grades = school_data[model_target]
prediction_excl_grades['Prediction'] = Rand_Forest_2.predict(targets_excl_grades)
excl_grades_weights = pd.Series(index=targets_excl_grades.columns, data=Rand_Forest_2.feature_importances_).sort_values(ascending=True)
excl_grades_weights.plot(kind='barh', figsize=(10,15), color="purple");
short_term_prediction.head(15)
short_term_prediction.to_csv("Chris_Deitrick_PASSNYC_School_Recommendations.csv")