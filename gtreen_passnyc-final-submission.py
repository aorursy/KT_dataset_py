# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

schools_data = pd.read_csv ('../input/nyc-augmented-schools-dataset/nyc_augmented_school_explorer.csv')
nyc_shs_admissions_data = pd.read_csv ('../input/ny-times-shs-acceptance-rates/nytdf.csv')

# Any results you write to the current directory are saved as output.

took_test_list=[]
offered_list  =[]
found_list    =[]
not_found_cnt = 0

for i in range(len(schools_data)):
    found = False
    took_test = 0
    offered   = 0
    for j in range(len(nyc_shs_admissions_data)):
        if (schools_data['Location Code'][i]==nyc_shs_admissions_data['DBN'][j]):
            found = True
            took_test = nyc_shs_admissions_data['NumSHSATTestTakers'][j]
            offered   = nyc_shs_admissions_data['NumSpecializedOffers'][j]
            if (took_test=='0-5'):
                took_test = 0
            if (offered=='0-5'):
                offered = 0
            break
    if (found==False):
        not_found_cnt += 1
        print ('Warning: could not find applications and offers data for school %s.' % schools_data['School Name'][i])
    took_test_list.append(took_test)
    offered_list.append(offered)
    found_list.append(found)

print ('Could not find application/offer data for %d schools.' % not_found_cnt)


took_test_list = np.array(took_test_list)
valid_took_test_list = took_test_list[found_list]
offered_list   = np.array(offered_list)
valid_offered_list = offered_list[found_list]

schools_data = schools_data[found_list]
schools_data = pd.DataFrame(data=np.array(schools_data),columns=schools_data.keys())

schools_data['Took test 2017'] = valid_took_test_list.astype('float')
schools_data['Offered 2017']   = valid_offered_list.astype('float')



common_data  = []
demographics_data  = []
grade7_data = []

# common data params from original School Explorer
common_data.append(schools_data['Economic Need Index'].astype('float'))
common_data.append(schools_data['Student Attendance Rate'].astype('float'))
common_data.append(schools_data['Percent of Students Chronically Absent'].astype('float'))
common_data.append(schools_data['Rigorous Instruction %'].astype('float'))
common_data.append(schools_data['Collaborative Teachers %'].astype('float'))
common_data.append(schools_data['Supportive Environment %'].astype('float'))
common_data.append(schools_data['Effective School Leadership %'].astype('float'))
common_data.append(schools_data['Strong Family-Community Ties %'].astype('float'))
common_data.append(schools_data['Trust %'].astype('float'))
common_data.append(schools_data['Average ELA Proficiency'].astype('float'))
common_data.append(schools_data['Average Math Proficiency'].astype('float'))
common_data.append(schools_data['Percent ELL'].astype('float'))
# params added from other sources
common_data.append(schools_data['Zip Density'].astype('float'))
common_data.append(schools_data['District Public Assistance %'].astype('float'))
common_data.append(schools_data['District U.S. Citizen %'].astype('float'))
common_data.append(schools_data['District Permanent Resident Alien %'].astype('float'))
common_data.append(schools_data['Nearby Auto Collisions'].astype('float'))
common_data.append(schools_data['Grade 6 Acceptance Rate'].astype('float'))


# demographics data params
demographics_data.append(schools_data['Percent Asian'].astype('float'))
demographics_data.append(schools_data['Percent Black'].astype('float'))
demographics_data.append(schools_data['Percent Hispanic'].astype('float'))
demographics_data.append(schools_data['Percent White'].astype('float'))
# params added from other sources
demographics_data.append(schools_data['District Asian %'].astype('float'))
demographics_data.append(schools_data['District Black %'].astype('float'))
demographics_data.append(schools_data['District Hispanic %'].astype('float'))
demographics_data.append(schools_data['District White %'].astype('float'))


# get the 2017 total count of ELA and Math-tested students
grade7_ela_count = schools_data['Grade 7 ELA - All Students Tested']
grade7_math_count = schools_data['Grade 7 Math - All Students Tested']

# only look at Grade 7 data for this model (since the SHSAT data is from 2017 and School Explorer is 2016)
grade7_data.append(schools_data['Grade 7 ELA 4s - Limited English Proficient'].astype('float')/grade7_ela_count)
grade7_data[-1].name = 'Grade 7 ELA 4s - Limited English Proficient'
grade7_data.append(schools_data['Grade 7 Math 4s - Limited English Proficient'].astype('float')/grade7_math_count)
grade7_data[-1].name = 'Grade 7 Math 4s - Limited English Proficient'
grade7_data.append(schools_data['Grade 7 ELA 4s - Economically Disadvantaged'].astype('float')/grade7_ela_count)
grade7_data[-1].name = 'Grade 7 ELA 4s - Economically Disadvantaged'
grade7_data.append(schools_data['Grade 7 Math 4s - Economically Disadvantaged'].astype('float')/grade7_math_count)
grade7_data[-1].name = 'Grade 7 Math 4s - Economically Disadvantaged'
grade7_data.append(schools_data['Grade 7 ELA 4s - Asian or Pacific Islander'].astype('float')/grade7_ela_count)
grade7_data[-1].name = 'Grade 7 ELA 4s - Asian or Pacific Islander'
grade7_data.append(schools_data['Grade 7 Math 4s - Asian or Pacific Islander'].astype('float')/grade7_math_count)
grade7_data[-1].name = 'Grade 7 Math 4s - Asian or Pacific Islander'
grade7_data.append(schools_data['Grade 7 ELA 4s - Black or African American'].astype('float')/grade7_ela_count)
grade7_data[-1].name = 'Grade 7 ELA 4s - Black or African American'
grade7_data.append(schools_data['Grade 7 Math 4s - Black or African American'].astype('float')/grade7_math_count)
grade7_data[-1].name = 'Grade 7 Math 4s - Black or African American'
grade7_data.append(schools_data['Grade 7 ELA 4s - Hispanic or Latino'].astype('float')/grade7_ela_count)
grade7_data[-1].name = 'Grade 7 ELA 4s - Hispanic or Latino'
grade7_data.append(schools_data['Grade 7 Math 4s - Hispanic or Latino'].astype('float')/grade7_math_count)
grade7_data[-1].name = 'Grade 7 Math 4s - Hispanic or Latino'
grade7_data.append(schools_data['Grade 7 ELA 4s - White'].astype('float')/grade7_ela_count)
grade7_data[-1].name = 'Grade 7 ELA 4s - White'
grade7_data.append(schools_data['Grade 7 Math 4s - White'].astype('float')/grade7_math_count)
grade7_data[-1].name = 'Grade 7 Math 4s - White'

# get the 2017 SHSAT test-taking and acceptance data

grade7_tested_count =np.max( np.array([grade7_ela_count, grade7_math_count]), axis=0)

Y_tested   = np.array(schools_data['Took test 2017'])/grade7_tested_count
Y_accepted = schools_data['Offered 2017']/grade7_tested_count

X = common_data + demographics_data + grade7_data
X_keys = []
for i in range(len(X)):
    X_keys.append(X[i].name)





total_students      = np.sum(schools_data['Grade 7 ELA - All Students Tested'])
asian_students_pct   = np.sum(schools_data['Grade 7 ELA - All Students Tested']*(schools_data['Percent Asian']/100.0))/total_students
black_students_pct   = np.sum(schools_data['Grade 7 ELA - All Students Tested']*(schools_data['Percent Black']/100.0))/total_students
hispanic_students_pct   = np.sum(schools_data['Grade 7 ELA - All Students Tested']*(schools_data['Percent Hispanic']/100.0))/total_students
white_students_pct   = np.sum(schools_data['Grade 7 ELA - All Students Tested']*(schools_data['Percent White']/100.0))/total_students

print ('Student population distribution:')
print ('    Asian: %f' % asian_students_pct)
print ('    Black: %f' % black_students_pct)
print ('    Hispanic: %f' % hispanic_students_pct)
print ('    White: %f' % white_students_pct)
print ('')

def print_highest_acceptance_rates (data, Y_a, N):
    y_a = np.copy(Y_a)
    for i in range (N):
        value = np.max(y_a)
        index = np.argmax(y_a)
        print ('%s, acceptance rate=%f' % (data['School Name'][index], value))
        print ('   - demographics percentages: %d Asian, %d Black, %d Hispanic, %d White' % (
                   data['Percent Asian'][index], data['Percent Black'][index], 
                   data['Percent Hispanic'][index], data['Percent White'][index]))
        y_a[index] = 0.0


print_highest_acceptance_rates (schools_data, Y_accepted, 10)
from sklearn import preprocessing
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor


def cross_validate (X, Y, n_folds=10, max_depth=10, max_features='auto', n_estimators=10):
    k_fold = KFold(len(Y), n_folds=n_folds, shuffle=True)#, random_state=0)
    model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=max_depth,
        #max_features='auto', max_leaf_nodes=None,
        max_features=max_features, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=n_estimators, n_jobs=1,
        oob_score=False, random_state=0, verbose=0, warm_start=False)
    cv_scores = cross_val_score(model, X, Y, cv=k_fold, n_jobs=1)
    print ('CV scores: %s' % str(cv_scores))
    print ('CV mean: %f, std. dev: %f' % (np.mean(cv_scores), np.std(cv_scores)))
    model.fit(X, Y)
    return model, np.mean(cv_scores), np.std(cv_scores)


# normalize the input features
X_scaled = preprocessing.scale (np.array(X).transpose())
Y_t      = np.array(Y_tested)
Y_a      = np.array(Y_accepted)

# perform k-fold cross-validation

depths   = [5, 10, 15, 20]
features = [5, 10, 15, 20]

best_cv=0
best_depth=0
best_features=0

for i in range(len(depths)):
    for j in range(len(features)):
        print ('Cross-validation trial with max depth=%d and max features=%d...' % (depths[i], features[j]))
        model, cv_mean, _ = cross_validate (X_scaled, Y_a, n_folds=5, max_depth=depths[i], max_features=features[j], n_estimators=100)
        if (cv_mean>best_cv):
            best_cv = cv_mean
            best_depth=depths[i]
            best_features=features[j]

print ('After k-fold cross-validation, the best max-depth setting is %d, the best max-features setting is %d' % (best_depth, best_features))
model, _, _  = cross_validate (X_scaled, Y_a, n_folds=5, max_depth=best_depth, max_features=best_features, n_estimators=100)


def print_feature_importances (model, keys):
    importances = model.feature_importances_
    for i in range(len(keys)):
        max_index = np.argmax(importances)
        print ('Key: %s, importance = %f' % (keys[max_index], importances[max_index]))
        importances[max_index] = 0.0
        
      
print_feature_importances (model, X_keys)

        


# get all of the adjustment factors

asian_mean = np.mean(schools_data['Percent Asian']/100.0)
asian_sdv = np.std(schools_data['Percent Asian']/100.0)
black_mean = np.mean(schools_data['Percent Black']/100.0)
black_sdv = np.std(schools_data['Percent Black']/100.0)
hispanic_mean = np.mean(schools_data['Percent Hispanic']/100.0)
hispanic_sdv = np.std(schools_data['Percent Hispanic']/100.0)
white_mean = np.mean(schools_data['Percent White']/100.0)
white_sdv = np.std(schools_data['Percent White']/100.0)

# asian_students_pct holds the percent of the overall middle school student body that is asian, etc.

asian_pop_adj_factor = (asian_students_pct-asian_mean)/asian_sdv
black_pop_adj_factor = (black_students_pct-black_mean)/black_sdv
hispanic_pop_adj_factor = (hispanic_students_pct-hispanic_mean)/hispanic_sdv
white_pop_adj_factor = (white_students_pct-white_mean)/white_sdv

asian_ela_mean = np.mean(grade7_data[4])
asian_ela_sdv = np.std(grade7_data[4])
black_ela_mean = np.mean(grade7_data[6])
black_ela_sdv = np.std(grade7_data[6])
hispanic_ela_mean = np.mean(grade7_data[8])
hispanic_ela_sdv = np.std(grade7_data[8])
white_ela_mean = np.mean(grade7_data[10])
white_ela_sdv = np.std(grade7_data[10])

asian_ela4s_adj_factor = (asian_students_pct-asian_ela_mean)/asian_ela_sdv
black_ela4s_adj_factor = (black_students_pct-black_ela_mean)/black_ela_sdv
hispanic_ela4s_adj_factor = (hispanic_students_pct-hispanic_ela_mean)/hispanic_ela_sdv
white_ela4s_adj_factor = (white_students_pct-white_ela_mean)/white_ela_sdv

asian_math_mean = np.mean(grade7_data[5])
asian_math_sdv = np.std(grade7_data[5])
black_math_mean = np.mean(grade7_data[7])
black_math_sdv = np.std(grade7_data[7])
hispanic_math_mean = np.mean(grade7_data[9])
hispanic_math_sdv = np.std(grade7_data[9])
white_math_mean = np.mean(grade7_data[11])
white_math_sdv = np.std(grade7_data[11])


asian_math4s_adj_factor = (asian_students_pct-asian_math_mean)/asian_math_sdv
black_math4s_adj_factor = (black_students_pct-black_math_mean)/black_math_sdv
hispanic_math4s_adj_factor = (hispanic_students_pct-hispanic_math_mean)/hispanic_math_sdv
white_math4s_adj_factor = (white_students_pct-white_math_mean)/white_math_sdv




# adjust demographics data to represent student population means
def modify_demographics_data (X, X_keys):
    X_out = np.copy(X)
    for i in range(len(X_keys)):
        key = X_keys[i]
        if (key=='Percent Asian'):
            X_out[:,i] = asian_pop_adj_factor
        elif (key=='Percent Black'):
            X_out[:,i] = black_pop_adj_factor
        elif (key=='Percent Hispanic'):
            X_out[:,i] = hispanic_pop_adj_factor
        elif (key=='Percent White'):
            X_out[:,i] = white_pop_adj_factor
        elif (key=='Grade 7 ELA 4s - Asian or Pacific Islander'):
            X_out[:,i] = asian_ela4s_adj_factor
        elif (key=='Grade 7 ELA 4s - Black or African American'):
            X_out[:,i] = black_ela4s_adj_factor
        elif (key=='Grade 7 ELA 4s - Hispanic or Latino'):
            X_out[:,i] = hispanic_ela4s_adj_factor
        elif (key=='Grade 7 ELA 4s - White'):
            X_out[:,i] = white_ela4s_adj_factor
        elif (key=='Grade 7 Math 4s - Asian or Pacific Islander'):
            X_out[:,i] = asian_math4s_adj_factor
        elif (key=='Grade 7 Math 4s - Black or African American'):
            X_out[:,i] = black_math4s_adj_factor
        elif (key=='Grade 7 Math 4s - Hispanic or Latino'):
            X_out[:,i] = hispanic_math4s_adj_factor
        elif (key=='Grade 7 Math 4s - White'):
            X_out[:,i] = white_math4s_adj_factor
        elif ('Asian' in key or 'Black' in key or 'Hispanic' in key or 'White' in key):
            X_out[:,i] = np.zeros(X.shape[0])
    return X_out



X_mod           = modify_demographics_data (X_scaled, X_keys)
# predict with as-is demographics data
pred            = model.predict(X_scaled)
# predict with modified demographics data
pred_mod        = model.predict(X_mod)



def list_N_model_results (Y, pred, names, N, Y_tag='actual', pred_tag='predicted'):
    residuals = pred-Y
    no_results = abs(N)
    for i in range(no_results):
       if (N>0):
           top = np.argmax(residuals)
       else:
           top = np.argmin(residuals)
       print ('%d. %s %s=%f, %s=%f' %((i+1), names[top], pred_tag, pred[top], Y_tag, Y[top]))
       residuals[top] = 0.0

def print_school_data (school_name, data):
    keys = data.keys()
    for i in range (len(data)):
        if (data['School Name'][i]==school_name):
            for j in range (len(keys)):
                print ('%s: %s' %(keys[j], str(data[keys[j]][i])))


# top-10 schools where demographic model value is lower than non-demographic model
print ('\nTop schools where the neutral model outperforms the demographics-aware model:')
list_N_model_results (pred_mod, pred, schools_data['School Name'], -20, Y_tag='neutral', pred_tag='demographics')



print_school_data('SUCCESS ACADEMY CHARTER SCHOOL - HARLEM 2',schools_data)