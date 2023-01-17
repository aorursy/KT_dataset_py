# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Treatment Prediciton.csv')

df.shape
# Null values are coded as ? replaceing then with Nan

df.replace('?',np.nan, inplace=True)
# Percent of null values present in the corresponding column



null = round(df.isnull().sum()/len(df)*100,2)

null[null > 0]
# As Medical Speciality is important fill null values as Unknown



df.medical_specialty.fillna('Unknown', inplace=True)
# 2 Columns where gender is invalid



df[df['gender']=='Unknown/Invalid'].iloc[:,:5]
# Dropping weight, payercode columns due to high null values 

# Dropping empty records of race

# Drop gender records that are invalid



df.drop(['weight','payer_code'],1, inplace=True)

df = df[df.race.notnull()]

df = df[df['gender']!='Unknown/Invalid']

df.shape
# Handling missing values in diagnosis

# If newer diagnosis is null, take the result of previous diagnosis as the value



df.diag_1 = np.where(df.diag_1.isnull(), df.diag_2, df.diag_1)



df.diag_2 = np.where(df.diag_2.isnull(), df.diag_1, df.diag_2)

df.diag_3 = np.where(df.diag_3.isnull(), df.diag_2, df.diag_3)
null = round(df.isnull().sum()/len(df)*100,2)

null[null > 0]



# Null Values are treated
# Converting Age 



def Ageimpute(col):

    if col=='[0-10)':

        return 5

    elif col=='[10-20)':

        return 15

    elif col=='[20-30)':

        return 25

    elif col=='[30-40)':

        return 35

    elif col=='[40-50)':

        return 45

    elif col=='[50-60)':

        return 55

    elif col=='[60-70)':

        return 65

    elif col=='[70-80)':

        return 75

    elif col=='[80-90)':

        return 85

    else:

        return 95

    

df.age = df.age.apply(Ageimpute)

df.age = df.age.astype(object)
print('No of Admission Type : ',df.admission_type_id.nunique())

print('No of Discharge Type : ',df.discharge_disposition_id.nunique())

print('No of Sources Type   : ',df.admission_source_id.nunique())
# Reducing Admission Type Id Categories



# Emergency (1,2,7)

# Elective (3)

# Newborn (4)

# Unknown (5,6,8)



def Admissiontype_impute(col):

    if ((col == 1) or (col == 2) or (col == 7)):

        return 'Emergency'

    elif col == 3:

        return 'Elective'

    elif col == 4:

        return 'Newborn'

    else:

        return 'Unknown'

    

df.admission_type_id = df.admission_type_id.apply(Admissiontype_impute)
# Reducing Discharge Disposition Id Category



# Home (1,6,7,8,13)

# Hospital (2,9,10,22,23,28)

# Discharged to special care facility (3,4,5,14,24,27)

# Expired or hospice (11,19,20)

# Outpatient (15,16,17)

# Null/Unknown (6,18,25)



def Discharge_impute(col):

    if ((col == 1) or (col == 6) or (col == 7) or (col == 8) or (col == 13)):

        return 'Home'

    elif ((col == 2) or (col == 9) or (col == 10) or (col == 22) or (col == 23) or (col == 28)):

        return 'Hospital'

    elif ((col == 3) or (col == 4) or (col == 5) or (col == 14) or (col==24) or (col == 27)):

        return 'SC_facility'

    elif ((col == 11) or (col == 19) or (col==20)):

        return 'Expired'

    elif ((col == 15) or (col == 16) or (col == 17) or (col == 12)):

        return 'Outpatient'

    else:

        return 'Unknown'



df.discharge_disposition_id = df.discharge_disposition_id.apply(Discharge_impute)
# Reducing Category Admission Source Id



# 1. Referral (1,2,3)

# 2. From another facility (4,5,6,10,22,25)

# 3. Emergency (7)

# 4. Law/Enforcement (8)

# 5. Null, Not mapped (9,17,20)

# 6. Birth (11,13,14) 



def Source_impute(col):

    if ((col == 1) or (col == 2) or (col == 3)):

        return 'Referrel'

    elif ((col == 2) or (col == 4) or (col == 5) or (col == 6) or (col == 10) or (col == 22) or (col == 25)):

        return 'Facility'

    elif col == 7:

        return 'Emergency'

    elif col == 8:

        return 'Law'

    elif ((col == 9) or (col == 17) or (col == 20)):

        return 'Unknown'

    else:

        return 'Birth'



df.admission_source_id = df.admission_source_id.apply(Source_impute)
# Dropping Expired and Hospice patients



df = df[df.discharge_disposition_id != 'Expired']
## Removing the characters from diagnosis to group them



import re

df.diag_1 = df.diag_1.apply(lambda x: re.sub('[VE]','',str(x)))

df.diag_2 = df.diag_2.apply(lambda x: re.sub('[VE]','',str(x)))

df.diag_3 = df.diag_3.apply(lambda x: re.sub('[VE]','',str(x)))



# Converting them back to float

df.diag_1 = df.diag_1.astype(float)

df.diag_2 = df.diag_2.astype(float)

df.diag_3 = df.diag_3.astype(float)
def diag_impute(col):

    if np.floor(col == 250):

        return 'Diabetes'

    

    elif ((col >= 390) and (col <= 459)) or (col == 785):

        return ' Circulatory'

    elif (col >=460 and col <=519) or (col == 786):

        return 'Respiratory'

    elif (col >= 520 and col <= 579) or (col == 787):

        return 'Digestive'

    elif (col>=800 and col<=999):

        return 'Injury'

    elif (col>=710 and col <=785):

        return 'Musculoskeletal'

    elif (col>=580 and col<=629) or col == 788:

        return 'Genitourinary'

    elif (col>=140 and col<=239):

        return 'Neoplasm'

    else :

        return 'Other'

    

df.diag_1 = df.diag_1.apply(diag_impute)

df.diag_2 = df.diag_2.apply(diag_impute)

df.diag_3 = df.diag_3.apply(diag_impute)
df = df.drop(['encounter_id','patient_nbr'],1)

df.reset_index(inplace=True, drop=True)

df.head().T
df.describe().T
fig, axes = plt.subplots(1,2)

fig.set_figwidth(16)



sns.countplot(x= df['race'], hue = df['Target'], ax = axes[0]).set_title('Race VS. Medication')

sns.countplot(df['gender'], hue = df['Target'], ax = axes[1]).set_title("Gender of Patient VS. Medication")



axes[0].set(xlabel = 'Race')

axes[1].set(xlabel = 'Gender')

            

plt.tight_layout;