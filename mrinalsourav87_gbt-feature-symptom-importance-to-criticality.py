import numpy as np

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
data_source_path = "../input/covid19-patient-precondition-dataset/covid.csv"

covid = pd.read_csv( data_source_path, 

                    encoding = "ISO-8859-1", 

                    low_memory = False)



covid.info()



# Cleaning the data to keep only the rows containing 1, 2 values as 97 and 99 are essentialling missing data

covid = covid.loc[(covid.intubed == 1) | (covid.intubed == 2)]

covid = covid.loc[(covid.pneumonia == 1) | (covid.pneumonia == 2)]

covid = covid.loc[(covid.diabetes == 1) | (covid.diabetes == 2)]

covid = covid.loc[(covid.copd == 1) | (covid.copd == 2)]

covid = covid.loc[(covid.asthma == 1) | (covid.asthma == 2)]

covid = covid.loc[(covid.inmsupr == 1) | (covid.inmsupr == 2)]

covid = covid.loc[(covid.hypertension == 1) | (covid.hypertension == 2)]

covid = covid.loc[(covid.other_disease == 1) | (covid.other_disease == 2)]

covid = covid.loc[(covid.cardiovascular == 1) | (covid.cardiovascular == 2)]

covid = covid.loc[(covid.obesity == 1) | (covid.obesity == 2)]

covid = covid.loc[(covid.renal_chronic == 1) | (covid.renal_chronic == 2)]

covid = covid.loc[(covid.tobacco == 1) | (covid.tobacco == 2)]

covid = covid.loc[(covid.covid_res == 1) | (covid.covid_res == 2)]

covid = covid.loc[(covid.icu == 1) | (covid.icu == 2)]



# Modifying data to encode it into True/False instead of 1/0 

covid.sex = covid.sex.apply(lambda x: True if x == 1 else False)

covid.intubed = covid.intubed.apply(lambda x: True if x == 1 else False)

covid.pneumonia = covid.pneumonia.apply(lambda x: True if x == 1 else False)

covid.diabetes = covid.diabetes.apply(lambda x: True if x == 1 else False)

covid.copd = covid.copd.apply(lambda x: True if x == 1 else False)

covid.asthma = covid.asthma.apply(lambda x: True if x == 1 else False)

covid.inmsupr = covid.inmsupr.apply(lambda x: True if x == 1 else False)

covid.hypertension = covid.hypertension.apply(lambda x: True if x == 1 else False)

covid.other_disease = covid.other_disease.apply(lambda x: True if x == 1 else False)

covid.cardiovascular = covid.cardiovascular.apply(lambda x: True if x == 1 else False)

covid.obesity = covid.obesity.apply(lambda x: True if x == 1 else False)

covid.renal_chronic = covid.renal_chronic.apply(lambda x: True if x == 1 else False)

covid.tobacco = covid.tobacco.apply(lambda x: True if x == 1 else False)

covid.covid_res = covid.covid_res.apply(lambda x: True if x == 1 else False)

covid.icu = covid.icu.apply(lambda x: True if x == 1 else False)



#including pregnancy == True, False otherwise ( EVEN FOR UNKNOWN DATA )

covid.pregnancy = covid.pregnancy.apply(lambda x: True if x == 1 else False)
# filtering out records where covid results returned negative

covid = covid.loc[covid.covid_res==True]



# I create a new field called 'critical1', 

# which would be True for both where deaths have occurred OR where

# patients needed ICU. 

covid['critical1'] = (covid.date_died != '9999-99-99') | (covid.icu == True)



# I create another field called 'critical2', 

# which would be True for both where 'critical1' from above is True

# OR where patient required intubation identified by intubed is also True

covid['critical2'] = (covid.critical1 == True) | (covid.intubed == True) 



# We will fit two different classifiers to check if there is any change in symptom importance

# predicted by either. 



print(covid.shape)



# removing columns not needed  

covid.drop(columns = ['patient_type', 

                      'contact_other_covid', 

                      'covid_res', 

                      'entry_date', 

                      'date_symptoms', 

                      'date_died'], inplace=True)
# selecting data for GBT classifier

symptoms = covid[['sex',

                  'pneumonia',

                  'age',

                  'pregnancy',

                  'diabetes',

                  'copd',

                  'asthma',

                  'inmsupr',

                  'hypertension',

                  'other_disease',

                  'cardiovascular', 

                  'obesity',

                  'renal_chronic',

                  'tobacco']]

label1 = covid['critical1']

label2 = covid['critical2']
gbt1 = GradientBoostingClassifier(random_state=0)

gbt2 = GradientBoostingClassifier(random_state=0)

gbt1.fit(symptoms, label1)

gbt2.fit(symptoms, label2)
# Extracting feature importance, features here being the symptoms

importance1 = gbt1.feature_importances_

importance2 = gbt2.feature_importances_



def get_importance(list_symptoms, importance):

    sym_imp_map = [] 

    for sym, imp in zip(list_symptoms, importance):

        sym_imp_map.append((sym,imp))

    return sym_imp_map 



symptom_imp_map1 = get_importance(symptoms.columns, importance1)

symptom_imp_map2 = get_importance(symptoms.columns, importance2)



print('\nFor death or ICU criticality, we have: \n')

sorted(symptom_imp_map1, reverse=True, key=lambda x: x[1])
print('\n For death, ICU OR intubed patients, we have: \n')

sorted(symptom_imp_map2, reverse=True, key=lambda x: x[1])