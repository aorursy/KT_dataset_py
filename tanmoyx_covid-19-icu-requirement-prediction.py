# Import the necessary libraries



import numpy as np

import pandas as pd
# Display all columns and rows



pd.set_option('display.max_columns', None)
covid = pd.read_csv("../input/covid19-patient-precondition-dataset/covid.csv", encoding = "ISO-8859-1", low_memory = False)

covid.head()
covid.shape
covid.info()
covid.describe()
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
# Modifying data to get it converted to One Hot Encoded data



covid.sex = covid.sex.apply(lambda x: x if x == 1 else 0)

covid.intubed = covid.intubed.apply(lambda x: x if x == 1 else 0)

covid.pneumonia = covid.pneumonia.apply(lambda x: x if x == 1 else 0)

covid.diabetes = covid.diabetes.apply(lambda x: x if x == 1 else 0)

covid.copd = covid.copd.apply(lambda x: x if x == 1 else 0)

covid.asthma = covid.asthma.apply(lambda x: x if x == 1 else 0)

covid.inmsupr = covid.inmsupr.apply(lambda x: x if x == 1 else 0)

covid.hypertension = covid.hypertension.apply(lambda x: x if x == 1 else 0)

covid.other_disease = covid.other_disease.apply(lambda x: x if x == 1 else 0)

covid.cardiovascular = covid.cardiovascular.apply(lambda x: x if x == 1 else 0)

covid.obesity = covid.obesity.apply(lambda x: x if x == 1 else 0)

covid.renal_chronic = covid.renal_chronic.apply(lambda x: x if x == 1 else 0)

covid.tobacco = covid.tobacco.apply(lambda x: x if x == 1 else 0)

covid.covid_res = covid.covid_res.apply(lambda x: x if x == 1 else 0)

covid.icu = covid.icu.apply(lambda x: x if x == 1 else 0)
covid.shape
covid.describe()
# Removing Covid neagtive patient info



covid = covid.loc[(covid.covid_res == 1)]
# We will go ahead and drop a few columns which are intuitively not very useful to us, either because they have a huge number of empty data(97, 98, 99) or they are not just significant in predicting ICU requirement as we can not standardise them to a meaningful feature.



covid.drop(columns = ['patient_type', 'pregnancy', 'contact_other_covid', 'covid_res', 'entry_date', 'date_symptoms', 'date_died'], inplace=True)
covid.shape
covid.describe()