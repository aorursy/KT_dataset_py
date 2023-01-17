import pandas as pd
import numpy as np
import tqdm
have_corona = pd.read_excel('./have_corona/test value.xlsx')
dont_have_corona = pd.read_excel('./dont_have/test value.xlsx')
have_corona_users = pd.read_excel('./have_corona/user.xlsx')
dont_have_corona_users = pd.read_excel('./dont_have/user.xlsx')
mamad_csv = pd.read_csv('./data2/mtest.csv')
mamad_user_csv = pd.read_csv('./data2/user.csv')
have_corona_userids = list(set(have_corona.userid.values))
dont_have_corona_userids = list(set(dont_have_corona.userid.values))
# Drop repeated rows in "mamad_csv" dataset

def drop_dplicated_rows(mamad_csv):
    print('Before drop mamad_csv shape:', mamad_csv.shape)
    mamad_csv.drop_duplicates(subset=['test', 'testValue', 'date', 'userid', 'covid'],
                          keep='first', inplace=True)
    mamad_csv.reset_index(drop=True, inplace=True)
    print('After drop train shape:', mamad_csv.shape)

# drop_dplicated_rows(mamad_csv)
have_corona['covid'] = 1
drop_dplicated_rows(have_corona)
# this function will convert 'Not seen', etc .. strings to Zero
def convert_some_strings_to_zero(dataframe):
    dataframe.replace({'Not  seen': 0,
                       'Not Detected': 0,
                       'Loose' : 0,
                       'not seen' : 0,
                       'Not seen':0},
                      inplace=True)

convert_some_strings_to_zero(mamad_csv)
# convert_some_strings_to_zero(mamad_csv_joint)
def find_duplicate(user_id, test, value, data_frame):
    array = list(data_frame.loc[(data_frame['userid'] == user_id) 
                       & (data_frame['test'] == test) 
                       & (data_frame['testValue'] == str(value))]['testValue'].index)
    data_frame.drop(array[1:] ,inplace=True)
def remove_duplicated_tests_for_same_patient(user_ids, test_types, data_frame):
    for user_id in tqdm.tqdm_notebook(user_ids):
        for test_type in tqdm.tqdm_notebook(test_types):
            find_duplicate(user_id, test_type, 0,data_frame)
# remove all zero results for tests.
userids = list(set(mamad_csv.userid.values))
test_types = list(mamad_csv.test.unique())
remove_duplicated_tests_for_same_patient(userids, test_types, mamad_csv)
# mamad_csv2 is after removing Nan from data frame.
# commnet next line if you have for your dataframe.
mamad_csv.to_csv('./mamad_csv2.csv', index=False)


""" =======================================  RESUME FORM HERE  =========================================== """
# load data 
mamad_csv = pd.read_csv('./mamad_csv2.csv')
mamad_csv.head(1)
mamad_csv.drop(['covid'], axis=1, inplace=True)
mamad_csv.head(1)
mamad_csv_joint = pd.merge(mamad_csv, mamad_user_csv, on="userid", how='outer')
mamad_csv_joint.drop(['id'], axis=1, inplace=True)
mamad_csv_joint
def remove_question_mark(dataframe, char):
    index_array = list(dataframe[dataframe["test"].str.find(char) != -1].index)
    for row in index_array:
        dataframe.loc[row, 'test'] = dataframe.loc[row, 'test'].replace(char, '')
        
remove_question_mark(mamad_csv_joint, '?')
mamad_csv_joint[mamad_csv_joint["test"].str.find('?') != -1]
#  remove space before and after
mamad_csv_joint = mamad_csv_joint.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
# this one is outlier
print(mamad_csv_joint[mamad_csv_joint.test == "PCR   HSV"])
print(mamad_csv_joint[mamad_csv_joint.test == "pcr   hsv"])
print(mamad_csv_joint[mamad_csv_joint.test == "pcr hsv"])
# remove question mark rows
mamad_csv_joint.drop(list(mamad_csv_joint[mamad_csv_joint.test == ""].index), inplace=True)
test_types = list(mamad_csv_joint.test.unique())
mamad_csv_joint[mamad_csv_joint.test == ""]
#  lower case all test types
mamad_csv_joint['test'] = mamad_csv_joint['test'].str.lower()
#  lower case all testValues types
mamad_csv_joint['testValue'] = mamad_csv_joint['testValue'].str.lower()
mamad_csv_joint.head(15)

#   =======================  we have multiple white space between testValues let's remove them   =======================  
mamad_csv_joint['testValue'] = mamad_csv_joint['testValue'].apply(lambda x : " ".join(x.split()))
mamad_csv_joint['test'] = mamad_csv_joint['test'].apply(lambda x : " ".join(x.split()))
test_types = list(mamad_csv_joint.test.unique())
i = 0
total_rows = 0
indexes_to_drop = []
def count_test_per_patient(dataframe, test_type, i, total_rows):
    test_tested_on_patient_count= dataframe[dataframe['test'] == test_type].userid.nunique()
    if test_tested_on_patient_count < 90:
        i += 1
        total_rows += dataframe[dataframe['test'] == test_type].shape[0]
        indexes_to_drop.extend(list(dataframe[dataframe['test'] == test_type].index))
    return i, total_rows

for test_type in test_types:
        i, total_rows = count_test_per_patient(mamad_csv_joint, test_type, i, total_rows)
print("we have {} rows" .format(total_rows))
print(f'we have {i} test type with less than 90 repeat for each usesr')
print(f'at the end {len(indexes_to_drop)} rows must be deleted.')
        
# سطر هایی که ازمایشی با فراوانی کمتر از ۹۰ دارند حذف میشوند.
print(f'shape before remove is {mamad_csv_joint.shape}')
mamad_csv_joint.drop(indexes_to_drop, inplace=True)
print(f'shape after remove is {mamad_csv_joint.shape}')

# update test types
test_types = list(mamad_csv_joint.test.unique())
len(test_types)
mamad_csv_joint.reset_index(drop=True, inplace=True)
mamad_csv_joint.head(5)
# All the test Types with miss value
akbar = mamad_csv_joint.copy()
# change numeric values to numeric data type
mamad_csv_joint['testValue'] = mamad_csv_joint['testValue'].apply(lambda x : pd.to_numeric(x,errors='ignore'))
print(f'we have {mamad_csv_joint[mamad_csv_joint.testValue == 0]["test"].nunique()} test type with miss value')
all_testType_with_miss_value = mamad_csv_joint[mamad_csv_joint.testValue == 0]['test'].unique()
all_testType_with_miss_value
# all the rows with string type ( not number ) in testValue
akbar[pd.to_numeric(akbar.testValue, errors='coerce').isnull()]
#  =======================   let's see from all the test types. how many of them have string value too
mixed_dataType_tests_counts = len(set(akbar[pd.to_numeric(akbar.testValue, errors='coerce').isnull()]['test']))
akbar_testTypes_with_mixed_values = list(akbar[pd.to_numeric(akbar.testValue, errors='coerce').isnull()]['test'].unique())

print(f'we have {len(test_types)} test types. and we have {mixed_dataType_tests_counts} test type with mixed data type value')
# ========================== all the strings for testValue Column  =======================  


# print(f'we have {akbar[pd.to_numeric(akbar.testValue, errors="coerce").isnull()]["testValue"].nunique()} unique string ')
# print(akbar[pd.to_numeric(akbar.testValue, errors='coerce').isnull()]['testValue'].unique())

def check_values(df):
    print('let\'s see which testValues are werid')
    for testValue in df[pd.to_numeric(df.testValue, errors='coerce').isnull()]['testValue'].unique():
        if len(testValue) < 4 :
            print(testValue)

check_values(mamad_csv_joint)
#  =======================   let's remove thsi "." value row  =======================  
print(mamad_csv_joint.shape)
mamad_csv_joint.drop(list(mamad_csv_joint[mamad_csv_joint.testValue == "."].index), inplace=True)
mamad_csv_joint.drop(list(mamad_csv_joint[mamad_csv_joint.testValue == "+/-"].index), inplace=True)
print(mamad_csv_joint.shape)
#   =======================  merge all kind off positives in blood  =======================  
"""we have these for blood :
[ negative, trace, positive(+), positive(++), +, positive(+++), positive(++++), 2+, 3+, 4+ ]
"""
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood') & ( mamad_csv_joint.testValue == 'positive(+)' ), 'testValue'] = 'positive'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood') & ( mamad_csv_joint.testValue == 'positive(++)' ), 'testValue'] = 'positive'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood') & ( mamad_csv_joint.testValue == '+' ), 'testValue'] = 'positive'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood') & ( mamad_csv_joint.testValue == 'positive(+++)' ), 'testValue'] = 'positive'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood') & ( mamad_csv_joint.testValue == 'positive(++++)' ), 'testValue'] = 'positive'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood') & ( mamad_csv_joint.testValue == 'positive(++++)' ), 'testValue'] = 'positive'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood') & ( mamad_csv_joint.testValue == '2+' ), 'testValue'] = 'positive'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood') & ( mamad_csv_joint.testValue == '3+' ), 'testValue'] = 'positive'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood') & ( mamad_csv_joint.testValue == '4+' ), 'testValue'] = 'positive'


#   =======================  SAME APPROACH FOR OTHER TESTS Too.  =======================  






# ========================= cahnge  WBC  =============================

string_test_value_rows = mamad_csv_joint[pd.to_numeric(mamad_csv_joint.testValue, errors='coerce').isnull()]

print(string_test_value_rows[string_test_value_rows.test == 'wbc'].testValue.unique() )
# above line print out : ['many', '0-2', 'moderate', 'few', 'rare', '50-100', '-', '>100', '25-50']

# let's change test type to blood and urine
mamad_csv_joint.loc[(mamad_csv_joint.test == 'wbc') & ( mamad_csv_joint.testValue == '0-2' ), 'test'] = 'blood_wbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'wbc') & ( mamad_csv_joint.testValue == '15-20' ), 'test'] = 'blood_wbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'wbc') & ( mamad_csv_joint.testValue == '15-30' ), 'test'] = 'blood_wbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'wbc') & ( mamad_csv_joint.testValue == '25-50' ), 'test'] = 'blood_wbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'wbc') & ( mamad_csv_joint.testValue == '50-100' ), 'test'] = 'blood_wbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'wbc') & ( mamad_csv_joint.testValue == '>100' ), 'test'] = 'blood_wbc'
# let's change them to numeric
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood_wbc') & ( mamad_csv_joint.testValue == '0-2' ), 'testValue'] = 0
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood_wbc') & ( mamad_csv_joint.testValue == '15-20' ), 'testValue'] = 15
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood_wbc') & ( mamad_csv_joint.testValue == '15-30' ), 'testValue'] = 25
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood_wbc') & ( mamad_csv_joint.testValue == '25-50' ), 'testValue'] = 25
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood_wbc') & ( mamad_csv_joint.testValue == '50-100' ), 'testValue'] = 100
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood_wbc') & ( mamad_csv_joint.testValue == '>100' ), 'testValue'] = 100


# ===================================== NOW URINE PART ==============================
mamad_csv_joint.loc[(mamad_csv_joint.test == 'wbc') & ( mamad_csv_joint.testValue == 'few' ), 'test'] = 'urine_wbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'wbc') & ( mamad_csv_joint.testValue == 'moderate' ), 'test'] = 'urine_wbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'wbc') & ( mamad_csv_joint.testValue == 'many' ), 'test'] = 'urine_wbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'wbc') & ( mamad_csv_joint.testValue == 'rare' ), 'test'] = 'urine_wbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'wbc') & ( mamad_csv_joint.testValue == '-' ), 'test'] = 'urine_wbc'
# let's change them to numeric
mamad_csv_joint.loc[(mamad_csv_joint.test == 'urine_wbc') & ( mamad_csv_joint.testValue == 'few' ), 'testValue'] = 2
mamad_csv_joint.loc[(mamad_csv_joint.test == 'urine_wbc') & ( mamad_csv_joint.testValue == 'moderate' ), 'testValue'] = 5
mamad_csv_joint.loc[(mamad_csv_joint.test == 'urine_wbc') & ( mamad_csv_joint.testValue == 'many' ), 'testValue'] = 25
mamad_csv_joint.loc[(mamad_csv_joint.test == 'urine_wbc') & ( mamad_csv_joint.testValue == 'rare' ), 'testValue'] = 0
mamad_csv_joint.loc[(mamad_csv_joint.test == 'urine_wbc') & ( mamad_csv_joint.testValue == '-' ), 'testValue'] = 0
# ========================= cahnge  RBC ===========================

string_test_value_rows = mamad_csv_joint[pd.to_numeric(mamad_csv_joint.testValue, errors='coerce').isnull()]

print(string_test_value_rows[string_test_value_rows.test == 'rbc'].testValue.unique() )
# above line print out: ['25-50', 'few', '-', 'rare', 'moderate', '>100', '0-2', 'many','50-100', '15-20', '15-30']

# let's change test type to blood and urine
mamad_csv_joint.loc[(mamad_csv_joint.test == 'rbc') & ( mamad_csv_joint.testValue == '0-2' ), 'test'] = 'blood_rbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'rbc') & ( mamad_csv_joint.testValue == '15-20' ), 'test'] = 'blood_rbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'rbc') & ( mamad_csv_joint.testValue == '15-30' ), 'test'] = 'blood_rbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'rbc') & ( mamad_csv_joint.testValue == '25-50' ), 'test'] = 'blood_rbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'rbc') & ( mamad_csv_joint.testValue == '50-100' ), 'test'] = 'blood_rbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'rbc') & ( mamad_csv_joint.testValue == '>100' ), 'test'] = 'blood_rbc'
# let's change them to numeric
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood_rbc') & ( mamad_csv_joint.testValue == '0-2' ), 'testValue'] = 0
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood_rbc') & ( mamad_csv_joint.testValue == '15-20' ), 'testValue'] = 15
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood_rbc') & ( mamad_csv_joint.testValue == '15-30' ), 'testValue'] = 25
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood_rbc') & ( mamad_csv_joint.testValue == '25-50' ), 'testValue'] = 25
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood_rbc') & ( mamad_csv_joint.testValue == '50-100' ), 'testValue'] = 100
mamad_csv_joint.loc[(mamad_csv_joint.test == 'blood_rbc') & ( mamad_csv_joint.testValue == '>100' ), 'testValue'] = 100


# ===================================== NOW URINE PART ==============================
mamad_csv_joint.loc[(mamad_csv_joint.test == 'rbc') & ( mamad_csv_joint.testValue == 'few' ), 'test'] = 'urine_rbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'rbc') & ( mamad_csv_joint.testValue == 'moderate' ), 'test'] = 'urine_rbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'rbc') & ( mamad_csv_joint.testValue == 'many' ), 'test'] = 'urine_rbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'rbc') & ( mamad_csv_joint.testValue == 'rare' ), 'test'] = 'urine_rbc'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'rbc') & ( mamad_csv_joint.testValue == '-' ), 'test'] = 'urine_rbc'
# let's change them to numeric
mamad_csv_joint.loc[(mamad_csv_joint.test == 'urine_rbc') & ( mamad_csv_joint.testValue == 'few' ), 'testValue'] = 2
mamad_csv_joint.loc[(mamad_csv_joint.test == 'urine_rbc') & ( mamad_csv_joint.testValue == 'moderate' ), 'testValue'] = 5
mamad_csv_joint.loc[(mamad_csv_joint.test == 'urine_rbc') & ( mamad_csv_joint.testValue == 'many' ), 'testValue'] = 25
mamad_csv_joint.loc[(mamad_csv_joint.test == 'urine_rbc') & ( mamad_csv_joint.testValue == 'rare' ), 'testValue'] = 0
mamad_csv_joint.loc[(mamad_csv_joint.test == 'urine_rbc') & ( mamad_csv_joint.testValue == '-' ), 'testValue'] = 0
# ========================= cahnge ep-cell ======================
print(string_test_value_rows[string_test_value_rows.test == 'ep-cell'].testValue.unique() )
# above line will print out : ['rare' 'few' 'moderate' '-' 'many' '20-25']


# ========================= ep-cell ======================
mamad_csv_joint.loc[(mamad_csv_joint.test == 'ep-cell') & ( mamad_csv_joint.testValue == 'rare' ), 'testValue'] = 0
mamad_csv_joint.loc[(mamad_csv_joint.test == 'ep-cell') & ( mamad_csv_joint.testValue == 'few' ), 'testValue'] = 5
mamad_csv_joint.loc[(mamad_csv_joint.test == 'ep-cell') & ( mamad_csv_joint.testValue == 'moderate' ), 'testValue'] = 10
# maybe next line we should change to -1. so we don't change it. and change it in positive negative part.
# mamad_csv_joint.loc[(mamad_csv_joint.test == 'ep-cell') & ( mamad_csv_joint.testValue == '-' ), 'testValue'] = 0
mamad_csv_joint.loc[(mamad_csv_joint.test == 'ep-cell') & ( mamad_csv_joint.testValue == 'many' ), 'testValue'] = 50
mamad_csv_joint.loc[(mamad_csv_joint.test == 'ep-cell') & ( mamad_csv_joint.testValue == '20-25' ), 'testValue'] = 10
# ========================= cahnge glucose ===========================
print(string_test_value_rows[string_test_value_rows.test == 'glucose'].testValue.unique() )
# above line will print out : ['negative' 'positive(+)' '+' 'positive(++++)' 'trace' 'positive(++)']
mamad_csv_joint.loc[(mamad_csv_joint.test == 'glucose') & ( mamad_csv_joint.testValue == 'trace' ), 'testValue'] = 1
mamad_csv_joint.loc[(mamad_csv_joint.test == 'glucose') & ( mamad_csv_joint.testValue == 'negative' ), 'testValue'] = 'negative'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'glucose') & ( mamad_csv_joint.testValue == 'positive(+)' ), 'testValue'] = 'positive'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'glucose') & ( mamad_csv_joint.testValue == 'positive(++++)' ), 'testValue'] = 'positive'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'glucose') & ( mamad_csv_joint.testValue == 'positive(++)' ), 'testValue'] = 'positive'
mamad_csv_joint.loc[(mamad_csv_joint.test == 'glucose') & ( mamad_csv_joint.testValue == '+' ), 'testValue'] = 'positive'
# ========================= cahnge negative and positive to -1 , 1 ================================

mamad_csv_joint.loc[( mamad_csv_joint.testValue == 'positive' ), 'testValue'] = 1
mamad_csv_joint.loc[( mamad_csv_joint.testValue == 'positive(++)' ), 'testValue'] = 1
mamad_csv_joint.loc[( mamad_csv_joint.testValue == 'positive(+++)' ), 'testValue'] = 1
mamad_csv_joint.loc[( mamad_csv_joint.testValue == '2+' ), 'testValue'] = 1
mamad_csv_joint.loc[( mamad_csv_joint.testValue == '+' ), 'testValue'] = 1

mamad_csv_joint.loc[( mamad_csv_joint.testValue == '-------' ), 'testValue'] = -1
mamad_csv_joint.loc[( mamad_csv_joint.testValue == 'positive(+)' ), 'testValue'] = 1
mamad_csv_joint.loc[( mamad_csv_joint.testValue == 'negative' ), 'testValue'] = -1
mamad_csv_joint.loc[( mamad_csv_joint.testValue == '-' ), 'testValue'] = -1

# ======================= check if there still remain any testValue is kind of positive or negative ===================
# forexample positive++++ or positive4+ or negative- etc. 
# we removed many strings. Now Let's see what we have in testValue of string types.
string_test_value_rows2 = mamad_csv_joint[pd.to_numeric(mamad_csv_joint.testValue, errors='coerce').isnull()]
string_test_value_rows2.testValue.unique()
# ======================================== BUGABLE may conditions in if cause bug =========================================
#  =======================   Convert testValues which all the values are strings
# NOTE : WE CAN'T UNDERSTAND WHAT NUMBER IS EQUAL TO WHAT TEXT

one_hot_dataFrame = pd.DataFrame(columns=['testType', 'prevValue', 'newValue'])

def find_meghdar_ha(df, col, test_type): 
    df2 = df.copy()
    df2.drop(list(df2[df2.testValue == 0].index), inplace=True)
    a = list(pd.to_numeric(df2[df2.test == test_type]['testValue'], errors='coerce').isnull())
    if len(a) - sum(a) == 0 and len(a) != 0:
        strings_for_this_test = df2[df2[col] == test_type].testValue.value_counts().keys().tolist()
        change_string_values_to_number(df, strings_for_this_test, test_type)
        
def change_string_values_to_number(df, arrayOfStrings, testType):
    for idx, value in enumerate(arrayOfStrings):
        df.loc[(df.test == testType) & ( df.testValue == value ), 'testValue'] = idx + 1
        
def find_meghdar_ha2(df, col, test_type): 
    df2 = df.copy()
#     df2.drop(list(df2[df2.testValue == 0].index), inplace=True)
    a = list(pd.to_numeric(df2[df2.test == test_type]['testValue'], errors='coerce').isnull())
    if len(a) - sum(a) == 0 and len(a) != 0:
        strings_for_this_test = df2[df2[col] == test_type].testValue.value_counts().keys().tolist()
        change_string_values_to_number(df, strings_for_this_test, test_type)
        
def change_string_values_to_number2(df, arrayOfStrings, testType):
    for idx, value in enumerate(arrayOfStrings):
        df.loc[(df.test == testType) & ( df.testValue == value ), 'testValue'] = idx + 1
    
for test_type in test_types:
#     find_meghdar_ha(mamad_csv_joint, 'test', test_type)
    find_meghdar_ha2(mamad_csv_joint, 'test', test_type)    
# we removed many strings. Now Let's see what we have in testValue of string types.
string_test_value_rows2 = mamad_csv_joint[pd.to_numeric(mamad_csv_joint.testValue, errors='coerce').isnull()]
string_test_value_rows2.testValue.unique()
# some rows still have string value. What to do
# 1. first we can replace some special words with number

mamad_csv_joint.loc[( mamad_csv_joint.testValue == 'rare' ), 'testValue'] = 0
mamad_csv_joint.loc[( mamad_csv_joint.testValue == 'few' ), 'testValue'] = 5
mamad_csv_joint.loc[( mamad_csv_joint.testValue == 'moderate' ), 'testValue'] = 10
mamad_csv_joint.loc[( mamad_csv_joint.testValue == 'trace' ), 'testValue'] = 1
mamad_csv_joint.loc[( mamad_csv_joint.testValue == 'many' ), 'testValue'] = 50
mamad_csv_joint.loc[( mamad_csv_joint.testValue == '0-1' ), 'testValue'] = 0
mamad_csv_joint.loc[( mamad_csv_joint.testValue == '0-2' ), 'testValue'] = 1


# assume seen means positive
mamad_csv_joint.loc[( mamad_csv_joint.testValue == 'were seen' ), 'testValue'] = 1
mamad_csv_joint.loc[( mamad_csv_joint.testValue == 'present' ), 'testValue'] = 1
# we removed many strings. Now Let's see what we have in testValue of string types.
string_test_value_rows2 = mamad_csv_joint[pd.to_numeric(mamad_csv_joint.testValue, errors='coerce').isnull()]
string_test_value_rows2.testValue.unique()
a = list(pd.to_numeric(akbar[akbar.test == 'bacteria']['testValue'], errors='coerce').isnull())
print(sum(a))
print(len(a))
akbar[akbar.test == 'bacteria']['testValue']


