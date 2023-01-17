import pandas as pd

import re

import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, balanced_accuracy_score
train_dataset = pd.read_csv('../input/train_LTFS.csv')

train_dataset.columns
preprocessed_dataset = train_dataset



# will do some experiment and check which feature is affecting the loan defaults.

#it seems like few branches have more loan default than others. we include this in final training

'''

preprocessed_dataset.groupby('branch_id')['loan_default'].value_counts()

preprocessed_dataset.groupby('Employment.Type')['loan_default'].value_counts()

preprocessed_dataset.groupby('PERFORM_CNS.SCORE.DESCRIPTION')['loan_default'].value_counts()

preprocessed_dataset.groupby('NO.OF_INQUIRIES')['loan_default'].value_counts()

preprocessed_dataset.groupby('credit_history_duration')['loan_default'].value_counts()

'''



#creating one new column 'age' from date of birth column and drop the previous column



age_calculation = []

#In pandas, one issue is there. year before 68 will be counted as 2068, 2067, 2068....To fix this issue, re is used here

for x in list(preprocessed_dataset['Date.of.Birth']):

    if int(re.findall('\d+-\d+-(\d+)', x)[0]) <= 68 and int(re.findall('\d+-\d+-(\d+)', x)[0]) >= 20:

        age_calculation.append(pd.to_datetime('today').year - pd.to_datetime(x).year + 100)

    else:

        age_calculation.append(pd.to_datetime('today').year - pd.to_datetime(x).year)

    

preprocessed_dataset['age'] = age_calculation

preprocessed_dataset = preprocessed_dataset.drop('Date.of.Birth', axis = 1)





#converting the credit history column data into year wise. This value '1yrs 11mo' will be converted to 2.

#similarily '0yrs 3mon' will be converted to 0. if month value is less than 6 then same year, else next year value  



credit_history_length = []

for duration in list(preprocessed_dataset['CREDIT.HISTORY.LENGTH']):

    month_value = float(re.findall('yrs\s(\d+)\w+', duration)[0])

    year_value = float(re.findall('(\d+)yrs', duration)[0])

    cal = year_value + (month_value/12) 

    credit_history_length.append(cal)

        

preprocessed_dataset['credit_history_length'] = credit_history_length

preprocessed_dataset = preprocessed_dataset.drop('CREDIT.HISTORY.LENGTH', axis = 1)







#simple way to check different unique value in particular column and it's total count

'''

a = np.array(preprocessed_dataset['age'])

a, b = np.unique(a, return_counts=True)

dict(zip(a, b))

'''





preprocessed_dataset.head()
preprocessed_dataset = preprocessed_dataset.drop('UniqueID', axis = 1)

preprocessed_dataset = preprocessed_dataset.drop('Current_pincode_ID', axis = 1)

preprocessed_dataset = preprocessed_dataset.drop('MobileNo_Avl_Flag', axis = 1)

preprocessed_dataset = preprocessed_dataset.drop('DisbursalDate', axis = 1)

preprocessed_dataset.head()
# for handling the AVERAGE.ACCT.AGE feature

average_account_age = []

for duration in list(preprocessed_dataset['AVERAGE.ACCT.AGE']):

    month_value = float(re.findall('yrs\s(\d+)\w+', duration)[0])

    year_value = float(re.findall('(\d+)yrs', duration)[0])

    cal = year_value + (month_value/12) 

    average_account_age.append(cal)

        

preprocessed_dataset['average_account_age'] = average_account_age

preprocessed_dataset = preprocessed_dataset.drop('AVERAGE.ACCT.AGE', axis = 1)

preprocessed_dataset.groupby('average_account_age')['loan_default'].value_counts()
preprocessed_dataset.groupby('PERFORM_CNS.SCORE.DESCRIPTION')['loan_default'].value_counts()

label_encoding = LabelEncoder()

preprocessed_dataset['PERFORM_CNS.SCORE.DESCRIPTION'] = label_encoding.fit_transform(preprocessed_dataset['PERFORM_CNS.SCORE.DESCRIPTION'])

preprocessed_dataset.groupby('PERFORM_CNS.SCORE.DESCRIPTION')['loan_default'].value_counts()
#dealing with missing data in Employment Type by directly deleting those rows

preprocessed_dataset['Employment.Type'].unique()

preprocessed_dataset = preprocessed_dataset.dropna(subset = ['Employment.Type'])





#preprocessed_dataset['Employment.Type'] = preprocessed_dataset['Employment.Type'].astype('category')

#preprocessed_dataset.columns[preprocessed_dataset.isnull().any()]





#handling the employee Type feature and converting it into one hot encoding

label_encoding = LabelEncoder()

preprocessed_dataset['Employment.Type'] = label_encoding.fit_transform(preprocessed_dataset['Employment.Type'])



one_hot_encoder = OneHotEncoder(categorical_features=[6])

one_hot_encoder_matrix = one_hot_encoder.fit_transform(preprocessed_dataset).toarray()

employment_type_dataframe = pd.DataFrame(one_hot_encoder_matrix, columns = ['Employment_Type_Salaried', 'Employment_Type_Self_Employed','disbursed_amount', 'asset_cost', 'ltv', 'branch_id', 'supplier_id',

       'manufacturer_id', 'State_ID', 'Employee_code_ID',

       'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag',

       'Passport_flag', 'PERFORM_CNS.SCORE', 'PERFORM_CNS.SCORE.DESCRIPTION',

       'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS',

       'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT',

       'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS',

       'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT',

       'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT', 'NEW.ACCTS.IN.LAST.SIX.MONTHS',

       'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', 'NO.OF_INQUIRIES',

       'loan_default', 'age', 'credit_history_length', 'average_account_age'])



print(preprocessed_dataset.shape)

print(employment_type_dataframe.shape)

print(preprocessed_dataset.head())

print(employment_type_dataframe.head())

preprocessed_dataset = employment_type_dataframe



#preprocessed_dataset.iloc[:,:-1].head()



#reordering the columns indices

preprocessed_dataset = preprocessed_dataset[['Employment_Type_Salaried', 'Employment_Type_Self_Employed',

       'disbursed_amount', 'asset_cost', 'ltv', 'branch_id', 'supplier_id',

       'manufacturer_id', 'State_ID', 'Employee_code_ID', 'Aadhar_flag',

       'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag',

       'PERFORM_CNS.SCORE', 'PERFORM_CNS.SCORE.DESCRIPTION', 'PRI.NO.OF.ACCTS',

       'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE',

       'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS',

       'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE',

       'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT',

       'SEC.INSTAL.AMT', 'NEW.ACCTS.IN.LAST.SIX.MONTHS',

       'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', 'NO.OF_INQUIRIES',

        'age', 'credit_history_length', 'average_account_age', 'loan_default']]



#preprocessed_dataset.iloc[:,:-1].head()



#normalization of data

min_max_scaler = MinMaxScaler()

preprocessed_dataset = min_max_scaler.fit_transform(preprocessed_dataset)

preprocessed_dataset[:, :-1]





X_Train, X_Test, Y_Train, Y_Test = train_test_split(preprocessed_dataset[:,:-1], preprocessed_dataset[:, -1],

                                                   test_size = 0.33, random_state = 42)



clf = RandomForestClassifier(n_estimators=75, criterion='gini', max_depth=12, verbose = 1, max_features='sqrt')

clf.fit(X_Train, Y_Train)
prediction_value = clf.predict(X_Test)

prediction_value
accuracy_score_ = accuracy_score(Y_Test, prediction_value, normalize=False)

balanced_accuracy_score_ = balanced_accuracy_score(Y_Test, prediction_value)

print(balanced_accuracy_score_)

print(accuracy_score_)