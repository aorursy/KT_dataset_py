# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("loading data takes about 1 minute....")



train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')



#sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')



print("loading successful!")
print(train_transaction.info(), "\n")

#print(train_transaction.describe(), "\n")

#print(train_transaction.head(), "\n")

print(train_transaction.shape, "\n")

print(train_transaction.columns, "\n")



print(test_transaction.shape, "\n")

print(test_transaction.columns, "\n")



print(train_transaction.isFraud, "\n")



print(train_transaction.isFraud.isnull().sum(), "\n")  # 0 missing values in target



print("percent of fraudulent train-transactions: ", len(train_transaction.loc[train_transaction.isFraud == 1])*100/len(train_transaction))
y_train = train_transaction["isFraud"]



# drop target column from train dataframe

train_transaction = train_transaction.drop(columns = ['isFraud'])



print(y_train.shape, "\n")

print(train_transaction.shape, "\n")
print(train_identity.shape, "\n")

print(train_identity.columns, "\n")

print(train_identity.head(), "\n")



print("\n\n")



print(test_identity.shape, "\n")

print(test_identity.columns, "\n")

print(test_identity.head(), "\n")
print("train_transaction.index: \n", train_transaction.index, "\n")

print("train_identity.index: \n", train_identity.index, "\n")



print(train_identity.id_01.value_counts(), "\n")    #  the id features seem to have many different values

print(train_identity.id_07.value_counts(), "\n")    #  the id features seem to have many different values

print(train_identity.DeviceType.value_counts(), "\n")  

print(train_identity.DeviceInfo.value_counts(), "\n")  
# sadly the id columns of test_identity are called id-01 instead of id_01, which is their name in the train dataframe.

# hence we must first rename all the 38 id columns in test_identity, before we can concat the dataframes.

# I simply used print(train_identity.columns) to get the list of correct column names, and now

# we will just assign them as the column names to the test_identity dataframe.



test_identity.columns = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08',

       'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',

       'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',

       'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',

       'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',

       'DeviceInfo']





print("before concatting: \n\n")

print("train_transaction.shape: ", train_transaction.shape, "\n")

print("test_transaction.shape: ", test_transaction.shape, "\n")

print("train_transaction.index: ", train_transaction.index, "\n")

print("test_transaction.index: ", test_transaction.index, "\n")



print("train_identity.shape: ", train_identity.shape, "\n")

print("test_identity.shape: ", test_identity.shape, "\n")

print("train_identity.index: ", train_identity.index, "\n")

print("test_identity.index: ", test_identity.index, "\n")







transaction_data = pd.concat([train_transaction, test_transaction])

identity_data = pd.concat([train_identity, test_identity])





print("after concatting: \n\n")

print("transaction_data.shape: ", transaction_data.shape, "\n")

print("transaction_data.index: ", transaction_data.index, "\n")

print("identity_data.shape: ", identity_data.shape, "\n")

print("identity_data.index: ", identity_data.index, "\n")
# lets generate some useful lists of columns

# we want a list of numerical features

# and a list of categorical features



c = (identity_data.dtypes == 'object')

n = (identity_data.dtypes != 'object')

cat_id_cols = list(c[c].index)

num_id_cols = list(n[n].index) 



print(cat_id_cols, "\n")

print("number categorical identity features: ", len(cat_id_cols), "\n\n")

print(num_id_cols, "\n")

print("number numerical identity features: ", len(num_id_cols))
# lets generate some useful lists of columns

# we want a list of numerical features

# and a list of categorical features



c = (transaction_data.dtypes == 'object')

n = (transaction_data.dtypes != 'object')

cat_trans_cols = list(c[c].index)

num_trans_cols = list(n[n].index) 



print(cat_trans_cols, "\n")

print("number categorical transaction features: ", len(cat_trans_cols), "\n\n")

print(num_trans_cols, "\n")

print("number numerical transaction features: ", len(num_trans_cols))
# we save the shapes in these variables before deleting the dataframes

shape_of_train_trans = train_transaction.shape

shape_of_train_id    = train_identity.shape



shape_of_test_trans  = test_transaction.shape

shape_of_test_id     = test_identity.shape



del train_transaction

del train_identity

del test_transaction

del test_identity



print("deletion successful!")
print(identity_data.id_12, "\n")

print(identity_data.id_15, "\n")

print(identity_data.id_16, "\n")
for i in cat_id_cols:

    print(identity_data[i].value_counts())

    print(i, "missing values: ", identity_data[i].isnull().sum())

    print(identity_data[i].isnull().sum()*100/ len(identity_data[i]), "\n")
#  categorical identity features:



#  id_12:   2 values       0  missing values

#  id_15:   3 values    8178  missing values

#  id_16:   2 values   31053  missing values

#  id_23:   3 values  275909  missing values 96%

#  id_27:   2 values  275909  missing values 96%

#  id_28:   2 value     8384  missing values 

#  id_29:   2 values    8384  missing values  

#  id_30:  75 values  137916  missing values 48%

#  id_31: 130 values:   9233  missing values  

#  id_33: 260 values: 142180  missing values 50%

#  id_34:   4 values: 136160  missing values 47%

#  id_35:   2 values:   8178  missing values

#  id_36:   2 values:   8178  missing values

#  id_37:   2 values:   8178  missing values

#  id_38:   2 values:   8178  missing values

#  DeviceType: 2 values 8399  missing values

#  DeviceInfo: 2799 values,  52417  missing values
low_missing_cat_id_cols = []      # lower than 15% missing values

medium_missing_cat_id_cols = []   # between 15% and 60% missing

many_missing_cat_id_cols = []     # more than 60% missing



for i in cat_id_cols:

    percentage = identity_data[i].isnull().sum() * 100 / len(identity_data[i])

    if percentage < 15:

        low_missing_cat_id_cols.append(i)

    elif percentage >= 15 and percentage < 60:

        medium_missing_cat_id_cols.append(i)

    else:

        many_missing_cat_id_cols.append(i)

        

print("cat_id_cols: \n\n")      

print("number low missing: ", len(low_missing_cat_id_cols), "\n")

print("number medium missing: ", len(medium_missing_cat_id_cols), "\n")

print("number many missing: ", len(many_missing_cat_id_cols), "\n")
for i in num_id_cols:

    print(identity_data[i].value_counts())

    print(i, "missing values: ", identity_data[i].isnull().sum()) 

    print(identity_data[i].isnull().sum()*100/len(identity_data[i]), "\n") # missing percent
#  numerical identity  features:



#  id_01:       77 values,       0  missing values  

#  id_02:   115655 values,    8292  missing values 

#  id_03:       24 values,  153335  missing values 54%

#  id_04:       15 values,  153335  missing values 54%

#  id_05:       93 values,   14525  missing values 

#  id_06:      101 values,   14525  missing values

#  id_07:       84 values,  275926  missing values 96%

#  id_08:       94 values,  275926  missing values 96%

#  id_09:       46 values,  136876  missing values 48%

#  id_10:       62 values,  136876  missing values 48%

#  id_11:      365 values,    8384  missing values

#  id_13:       54 values,   28534  missing values

#  id_14:       25 values,  134739  missing values 47%

#  id_17:      104 values,   10805  missing values

#  id_18:       18 values,  190152  missing values 66%

#  id_19:      522 values,   10916  missing values

#  id_20:      394 values,   11246  missing values

#  id_21:      490 values,  275922  missing values 96%

#  id_22:       25 values,  275909  missing values 96%

#  id_24:       12 values,  276653  missing values 97%

#  id_25:      341 values,  275969  missing values 96%

#  id_26:       95 values,  275930  missing values 96%

#  id_32:        4 values,  137883  missing values 48%
low_missing_num_id_cols = []      # lower than 15% missing values

medium_missing_num_id_cols = []   # between 15% and 60% missing

many_missing_num_id_cols = []     # more than 60% missing



for i in num_id_cols:

    percentage = identity_data[i].isnull().sum() * 100 / len(identity_data[i])

    if percentage < 15:

        low_missing_num_id_cols.append(i)

    elif percentage >= 15 and percentage < 60:

        medium_missing_num_id_cols.append(i)

    else:

        many_missing_num_id_cols.append(i)

        

print("num_id_cols: \n\n")        

print("number low missing: ", len(low_missing_num_id_cols), "\n")

print("number medium missing: ", len(medium_missing_num_id_cols), "\n")

print("number many missing: ", len(many_missing_num_id_cols), "\n")
for i in cat_trans_cols:

    print(transaction_data[i].value_counts())

    print(i, transaction_data[i].isnull().sum(), "missing values")

    print(i, transaction_data[i].isnull().sum()*100/len(transaction_data[i]), "\n")  # missing percent
#  categorical transaction features:



#  ProductCD:      5  values,      0 missing values

#  card4:          4  values,   4663 missing values

#  card6:          4  values,   4578 missing values 

#  P_emaildomain  59  values, 163648 missing values,15%  

#  R_emaildomain  60  values, 824070 missing values 75%  

#  M1:             2  values, 447739 missing values 41%

#  M2:             2  values, 447739 missing values 41% 

#  M3:             2  values, 447739 missing values 41%

#  M4:             3  values, 519189 missing values 47%  

#  M5:             2  values, 660114 missing values 60% 

#  M6:             2  values, 328299 missing values 30% 

#  M7:             2  values, 581283 missing values 53% 

#  M8:             2  values, 581256 missing values 53% 

#  M9:             2  values, 581256 missing values 53% 
low_missing_num_trans_cols = []      # lower than 15% missing values

medium_missing_num_trans_cols = []   # between 15% and 60% missing

many_missing_num_trans_cols = []     # more than 60% missing



for i in num_trans_cols:

    percentage = transaction_data[i].isnull().sum() * 100 / len(transaction_data[i])

    if percentage < 15:

        low_missing_num_trans_cols.append(i)

    elif percentage >= 15 and percentage < 60:

        medium_missing_num_trans_cols.append(i)

    else:

        many_missing_num_trans_cols.append(i)

        

print("num_trans_cols: \n\n")        

print("number low missing: ", len(low_missing_num_trans_cols), "\n")

print("number medium missing: ", len(medium_missing_num_trans_cols), "\n")

print("number many missing: ", len(many_missing_num_trans_cols), "\n")
low_missing_cat_trans_cols = []      # lower than 15% missing values

medium_missing_cat_trans_cols = []   # between 15% and 60% missing

many_missing_cat_trans_cols = []     # more than 60% missing



for i in cat_trans_cols:

    percentage = transaction_data[i].isnull().sum() * 100 / len(transaction_data[i])

    if percentage < 15:

        low_missing_cat_trans_cols.append(i)

    elif percentage >= 15 and percentage < 60:

        medium_missing_cat_trans_cols.append(i)

    else:

        many_missing_cat_trans_cols.append(i)

        

print("cat_trans_cols: \n\n")    

print("number low missing: ", len(low_missing_cat_trans_cols), "\n")

print("number medium missing: ", len(medium_missing_cat_trans_cols), "\n")

print("number many missing: ", len(many_missing_cat_trans_cols), "\n")
# Summary so far:



# we have 2 dataframes:   transaction_data and identity_data



####################################################################

# features:



# transaction_data:     14 categorical and 378 numerical features

# identity_data:        17 categorical and  23 numerical features

####################################################################

# missing values:



# cat_trans_cols:      4 low,    8 medium,    2 many 

# num_trans_cols:    176 low,   35 medium,  167 many



# cat_id_cols:        11 low,    4 medium,    2 many 

# num_id_cols:         9 low,    6 medium,    8 many

####################################################################
print("shape before dropping num_trans_cols: ", transaction_data.shape, "\n")        

transaction_data = transaction_data.drop(columns = many_missing_num_trans_cols)

print("shape after dropping num_trans_cols: ", transaction_data.shape, "\n\n")    





print("shape before dropping num_id_cols: ", identity_data.shape, "\n")        

identity_data = identity_data.drop(columns = many_missing_num_id_cols)

print("shape after dropping num_id_cols: ", identity_data.shape, "\n")





# because we dropped some numerical columns from the dataframe,

# we must create the list 'num_trans_cols' and

# 'num_id_cols' again such that the dropped cols are no longer in them

n = (transaction_data.dtypes != 'object')

num_trans_cols = list(n[n].index) 



n = (identity_data.dtypes != 'object')

num_id_cols = list(n[n].index) 
from sklearn.impute import SimpleImputer



print("index before imputation: ", transaction_data.index, "\n")

print("columns before imputation: ", transaction_data.columns, "\n")



print("starting imputation..... \n\n")

my_imputer = SimpleImputer(strategy = 'mean') 

my_imputer.fit(transaction_data[low_missing_num_trans_cols])



#print("values before imputing: ", train_transaction[low_missing_num_trans_cols], "\n")



transaction_data[low_missing_num_trans_cols] = my_imputer.transform(transaction_data[low_missing_num_trans_cols])



print("index after imputation: ", transaction_data.index, "\n")

print("columns after imputation: ", transaction_data.columns, "\n")
print("values after imputing: ", transaction_data[low_missing_num_trans_cols], "\n")



print("As we can see the imputation was successful! \n")
print("index before imputation: ", identity_data.index, "\n")

print("columns before imputation: ", identity_data.columns, "\n")





my_imputer = SimpleImputer(strategy = 'mean') 

my_imputer.fit(identity_data[low_missing_num_id_cols])



print("starting imputation....\n")

identity_data[low_missing_num_id_cols] = my_imputer.transform(identity_data[low_missing_num_id_cols])



print("index after imputation: ", identity_data.index, "\n")

print("columns after imputation: ", identity_data.columns, "\n")
print("index before imputation: ", transaction_data.index, "\n")

print("columns before imputation: ", transaction_data.columns, "\n")



print("values before imputing: ", transaction_data[medium_missing_num_trans_cols], "\n")



print("starting imputation.....\n\n")

my_imputer = SimpleImputer(strategy = 'median') 

my_imputer.fit(transaction_data[medium_missing_num_trans_cols])



transaction_data[medium_missing_num_trans_cols] = my_imputer.transform(transaction_data[medium_missing_num_trans_cols])



print("index after imputation: ", transaction_data.index, "\n")

print("columns after imputation: ", transaction_data.columns, "\n")
print("values after imputing: ", transaction_data[medium_missing_num_trans_cols], "\n")
print("index before imputation: ", identity_data.index, "\n")

print("columns before imputation: ", identity_data.columns, "\n")





my_imputer = SimpleImputer(strategy = 'median') 

my_imputer.fit(identity_data[medium_missing_num_id_cols])



print("values before imputing: ", identity_data[medium_missing_num_id_cols], "\n")



identity_data[medium_missing_num_id_cols] = my_imputer.transform(identity_data[medium_missing_num_id_cols])



print("index after imputation: ", identity_data.index, "\n")

print("columns after imputation: ", identity_data.columns, "\n")
print(transaction_data[num_trans_cols].isnull().sum().sum())
print(identity_data[num_id_cols].isnull().sum().sum())
print("transaction_data.memory_usage(): ", transaction_data.info(), "\n")  # 1.8 GB



print("identity_data.memory_usage(): ", identity_data.info(), "\n")        #  72 MB
object_counter = 0

int_counter = 0

float_counter = 0



not_detected = []



for i in transaction_data.columns:

        if transaction_data[i].dtype == 'object':

            object_counter += 1

        elif transaction_data[i].dtype == 'int':

            int_counter += 1

        elif transaction_data[i].dtype in ['float', 'float16', 'float32', 'float64']:

            float_counter += 1

        else:

            not_detected.append(i)

            

print("transaction_data has ", "\n")

print(object_counter, "object columns, \n")

print(int_counter, "int columns, \n")

print(float_counter, "float columns \n")



total = object_counter + int_counter  + float_counter



if total != len(transaction_data.columns):

    

    print("D DOUBLE DANGER: some columns have not been detected!!")

    print("these columns have not been detected: ", not_detected)

    for i in not_detected:

        print(identity_data[i].dtype, "\n")
object_counter = 0

int_counter = 0

float_counter = 0



not_detected = []



for i in identity_data.columns:

        if identity_data[i].dtype == 'object':

            object_counter += 1

        elif identity_data[i].dtype == 'int':

            int_counter += 1

        elif identity_data[i].dtype in ['float', 'float16', 'float32', 'float64']:

            float_counter += 1

        else:

            not_detected.append(i)

            

            

print("identity_data has ", "\n")

print(object_counter, "object columns, \n")

print(int_counter, "int columns, \n")

print(float_counter, "float columns \n")



total = object_counter + int_counter  + float_counter



if total != len(identity_data.columns):

    

    print("D DOUBLE DANGER: some columns have not been detected!!")

    print("these columns have not been detected: ", not_detected)    

    for i in not_detected:

        print(identity_data[i].dtype, "\n")
# the integer datatypes have the following ranges:



#   int8:  -128 to 127, range = 255  



#  int16:  -32,768 to 32,767, range = 65,535



#  int32:  -2,147,483,648 to 2,147,483,647, range = 4,294,967,295



#  int64:  -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807,

#           range = 18,446,744,073,709,551,615





#  By default all numerical columns in pandas are in int64 or float64.

#  This means that when we find a numerical integer column whose 

#  values do not exceed one of the ranges shown above, we can then

#  convert this datatype down to a smaller one. 
#  this function detects all the numerical columns,

#  that can be converted to a smaller datatype.



def detect_num_cols_to_shrink(list_of_num_cols, dataframe):

 

    convert_to_int8 = []

    convert_to_int16 = []

    convert_to_int32 = []

    

    #  sadly the datatype float8 does not exist

    convert_to_float16 = []

    convert_to_float32 = []

    

    for col in list_of_num_cols:

        

        if dataframe[col].dtype in ['int', 'int8', 'int32', 'int64']:

            describe_object = dataframe[col].describe()

            minimum = describe_object[3]

            maximum = describe_object[7]

            diff = abs(maximum - minimum)



            if diff < 255:

                convert_to_int8.append(col)

            elif diff < 65535:

                convert_to_int16.append(col)

            elif diff < 4294967295:

                convert_to_int32.append(col)   

                

        elif dataframe[col].dtype in ['float', 'float16', 'float32', 'float64']:

            describe_object = dataframe[col].describe()

            minimum = describe_object[3]

            maximum = describe_object[7]

            diff = abs(maximum - minimum)



            if diff < 65535:

                convert_to_float16.append(col)

            elif diff < 4294967295:

                convert_to_float32.append(col) 

        

    list_of_lists = []

    list_of_lists.append(convert_to_int8)

    list_of_lists.append(convert_to_int16)

    list_of_lists.append(convert_to_int32)

    list_of_lists.append(convert_to_float16)

    list_of_lists.append(convert_to_float32)

    

    return list_of_lists
num_cols_to_shrink_trans = detect_num_cols_to_shrink(num_trans_cols, transaction_data)



convert_to_int8 = num_cols_to_shrink_trans[0]

convert_to_int16 = num_cols_to_shrink_trans[1]

convert_to_int32 = num_cols_to_shrink_trans[2]



convert_to_float16 = num_cols_to_shrink_trans[3]

convert_to_float32 = num_cols_to_shrink_trans[4]



print("convert_to_int8 :", convert_to_int8, "\n")

print("convert_to_int16 :", convert_to_int16, "\n")

print("convert_to_int32 :", convert_to_int32, "\n")



print("convert_to_float16 :", convert_to_float16, "\n")

print("convert_to_float32 :", convert_to_float32, "\n")
print("starting with converting process....")



for col in convert_to_int16:

    transaction_data[col] = transaction_data[col].astype('int16') 

    

for col in convert_to_int32:

    transaction_data[col] = transaction_data[col].astype('int32') 



for col in convert_to_float16:

    transaction_data[col] = transaction_data[col].astype('float16')

    

for col in convert_to_float32:

    transaction_data[col] = transaction_data[col].astype('float32')

    

print("successfully converted!")
num_cols_to_shrink_id = detect_num_cols_to_shrink(num_id_cols, identity_data)



convert_to_int8 = num_cols_to_shrink_id[0]

convert_to_int16 = num_cols_to_shrink_id[1]

convert_to_int32 = num_cols_to_shrink_id[2]



convert_to_float16 = num_cols_to_shrink_id[3]

convert_to_float32 = num_cols_to_shrink_id[4]



print("convert_to_int8 :", convert_to_int8, "\n")

print("convert_to_int16 :", convert_to_int16, "\n")

print("convert_to_int32 :", convert_to_int32, "\n")



print("convert_to_float16 :", convert_to_float16, "\n")

print("convert_to_float32 :", convert_to_float32, "\n")
for col in convert_to_float16:

    identity_data[col] = identity_data[col].astype('float16')

    

for col in convert_to_float32:

    identity_data[col] = identity_data[col].astype('float32')

    

    

print("successfully converted!")
print("transaction_data.memory_usage(): ", transaction_data.info(), "\n")   # now uses 615 MB



print("identity_data.memory_usage(): ", identity_data.info(), "\n")         # now uses 48 MB
#errrrrorr
print("shape before dropping many_missing_cat_trans_cols: ", transaction_data.shape, "\n")        

transaction_data = transaction_data.drop(columns = many_missing_cat_trans_cols)

print("shape after dropping many_missing_cat_trans_cols: ", transaction_data.shape, "\n\n")    



print("shape before dropping many_missing_cat_id_cols: ", identity_data.shape, "\n")        

identity_data = identity_data.drop(columns = many_missing_cat_id_cols)

print("shape after dropping many_missing_cat_id_cols: ", identity_data.shape, "\n")





# because we dropped some categorical columns from the dataframe,

# we must create the list 'cat_trans_cols' and

# 'cat_id_cols' again such that the dropped cols are no longer in them

c = (transaction_data.dtypes == 'object')

cat_trans_cols = list(c[c].index) 



c = (identity_data.dtypes == 'object')

cat_id_cols = list(c[c].index) 
for col in cat_trans_cols:

    print(col, transaction_data[col].nunique(), "\n")
low_card_trans_cols = ["ProductCD", "card4", "card6", "M1", "M2", "M3", "M4", "M6", "M7", "M8", "M9"]

high_card_trans_cols = ["P_emaildomain"]



print("lists successfully created!")
for i in cat_trans_cols:

    most_frequent_value = transaction_data[i].mode()[0]

    print("For column: ", i, "the most frequent value is: ", most_frequent_value, "\n")

    transaction_data[i].fillna(most_frequent_value, inplace = True)
from sklearn.preprocessing import LabelEncoder

    

label_encoder = LabelEncoder()

print("transaction_data.shape before label-encoding: ", transaction_data.shape, "\n")



transaction_data[high_card_trans_cols] = label_encoder.fit_transform(transaction_data[high_card_trans_cols])



print("transaction_data.shape after label-encoding: ", transaction_data.shape, "\n")

print("transaction_data[high_card_trans_cols] after label_encoding: ",transaction_data[high_card_trans_cols], "\n")
for col in cat_id_cols:

    print(col, identity_data[col].nunique(), "\n")
low_card_id_cols =  ["id_12", "id_15", "id_16", "id_28", "id_29", "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType"]

high_card_id_cols = ["id_30", "id_31", "id_33", "DeviceInfo"]

    

print("lists successfully created!")
for i in cat_id_cols:

    most_frequent_value = identity_data[i].mode()[0]

    print("For column: ", i, "the most frequent value is: ", most_frequent_value, "\n")

    identity_data[i].fillna(most_frequent_value, inplace = True)
label_encoder = LabelEncoder()



print("identity_data.shape before label-encoding: ", identity_data.shape, "\n")



for col in high_card_id_cols:

    identity_data[col] = label_encoder.fit_transform(identity_data[col])



print("identity_data.shape after label-encoding: ", identity_data.shape, "\n")

print("identity_data[high_card_id_cols] after label_encoding: ",identity_data[high_card_id_cols], "\n")
print(transaction_data.info())
print(identity_data.info())
print("shape before encoding: ", transaction_data.shape, "\n")

print("columns to encode: ", low_card_trans_cols, "\n")

print("transaction_data.columns.to_list() before encoding: ", transaction_data.columns.to_list(), "\n")





# this line does the onehot encoding

low_card_trans_encoded = pd.get_dummies(transaction_data[low_card_trans_cols], dummy_na = False)

transaction_data.drop(columns = low_card_trans_cols, inplace = True)



print("shape after encoding: ", transaction_data.shape, "\n\n")

print("shape of new dataframe: ", low_card_trans_encoded.shape, "\n\n")

print("newly generated columns: ", low_card_trans_encoded.columns, "\n")

print("low_card_trans_encoded.info(): ", low_card_trans_encoded.info(),"\n")

print("transaction_data.columns.to_list() after encoding: ", transaction_data.columns.to_list(), "\n")
print("shape before encoding: ", identity_data.shape, "\n")

print("columns to encode: ", low_card_id_cols, "\n")



# this line does the onehot encoding

low_card_id_encoded = pd.get_dummies(identity_data[low_card_id_cols], dummy_na = False)

identity_data.drop(columns = low_card_id_cols, inplace = True)





print("shape after encoding: ", identity_data.shape, "\n\n")

print("shape of new dataframe: ", low_card_id_encoded.shape, "\n\n")

print("newly generated columns: ", low_card_id_encoded.columns, "\n")

print("low_card_id_encoded.info(): ", low_card_id_encoded.info())
print(transaction_data.isnull().sum().sum(), "\n")

print(low_card_trans_encoded.isnull().sum().sum())
print(identity_data.isnull().sum().sum(), "\n")

print(low_card_id_encoded.isnull().sum().sum())
print(transaction_data.info(), "\n")

print(low_card_trans_encoded.info())
print(identity_data.info(), "\n")

print(low_card_id_encoded.info())
print("transaction_data.shape before concatting: ", transaction_data.shape, "\n")

print("low_card_trans_encoded.shape before concatting: ", low_card_trans_encoded.shape, "\n")



transaction_concatted = pd.concat([transaction_data, low_card_trans_encoded], axis = 1)



print("transaction_concatted.shape after concatting: ", transaction_concatted.shape, "\n")

print("transaction_concatted.columns after concatting: ", transaction_concatted.columns, "\n")



#del low_card_trans_encoded

#del transaction_data



print(transaction_concatted.info())
print("identity_data.shape before concatting: ", identity_data.shape, "\n")

print("low_card_id_encoded.shape before concatting: ", low_card_id_encoded.shape, "\n")



identity_concatted = pd.concat([identity_data, low_card_id_encoded], axis = 1)



print("identity_concatted.shape after concatting: ", identity_concatted.shape, "\n")

print("identity_concatted.columns after concatting: ", identity_concatted.columns, "\n")



#del low_card_id_encoded

#del identity_data



print(identity_concatted.info())
print("transaction_concatted.shape before splitting up: ", transaction_concatted.shape, "\n")



# shape of train_transaction was (590540, 393), 

# shape of test_transaction  was (506691, 392)

train_transaction = transaction_concatted.iloc[0:590540]

test_transaction = transaction_concatted.iloc[590540:]



print("train_transaction.shape after splitting up: ", train_transaction.shape, "\n")

print("test_transaction.shape after splitting up: ", test_transaction.shape, "\n")
print("identity_concatted.shape before splitting up: ", identity_concatted.shape, "\n")



# shape of train_identity was  (144233, 40)

# shape of test_identity  was  (141907, 40)

train_identity = identity_concatted.iloc[0:144233]

test_identity = identity_concatted.iloc[144233:]



print("train_identity.shape after splitting up: ", train_identity.shape, "\n")

print("test_identity.shape after splitting up: ", test_identity.shape, "\n")
print("train_transaction.shape before concatting: ", train_transaction.shape, "\n")

print("train_identity.shape before concatting: ", train_identity.shape, "\n")



train_data  = pd.concat([train_transaction, train_identity], axis = 1)



print("train_data.shape: ", train_data.shape)
counter = 0



for i in train_data.columns:

    

    summ = train_data[i].isnull().sum()

    print(i, summ)

    if summ > 0:

        counter += 1

        

print("\n number of columns with missing values: ", counter)
print("test_transaction.shape before concatting: ", test_transaction.shape, "\n")

print("test_identity.shape before concatting: ", test_identity.shape, "\n")



test_data  = pd.concat([test_transaction, test_identity], axis = 1)



print("test_data.shape: ", test_data.shape)
counter = 0



for i in test_data.columns:

    

    summ = test_data[i].isnull().sum()

    print(i, summ)

    if summ > 0:

        counter += 1

        

print("\n number of columns with missing values: ", counter)
print(test_data["id_35_F"])
'''

######################

#####  Dask  #########

######################





import dask.dataframe as dd



transaction_final_dask = dd.from_pandas(transaction_final, npartitions = 3)





print(type(transaction_final_dask), "\n\n")

print(transaction_final_dask, "\n\n")

print(transaction_final_dask.shape, "\n")





identity_final_dask = dd.from_pandas(identity_final, npartitions = 3)



print(type(identity_final_dask), "\n\n")

print(identity_final_dask, "\n\n")

print(identity_final_dask.shape, "\n")



del transaction_final

del identity_final



print("deletion successful!")





concatted = dd.concat([transaction_final_dask,identity_final_dask], axis = 1) 



print(concatted.shape)





# now we convert back to pandas



# this line converts from dask dataframe to pandas dataframe

final_dataframe = concatted.compute()





print(type(final_dataframe), "\n")



print(final_dataframe.shape)

'''
#########################################################################################

##  calculate how many different values it takes, to get about 90% of all data per column

######################################################################################### 

    

''' 

categorical_cols = []



cat_train_data = train_data[categorical_cols]



#  Erstelle dataframe, in welchem der Wert durch den jeweiigen value_count ersetzt wurde

value_count_cat_train_data = cat_train_data.apply(lambda x: x.map(x.value_counts()))





for i in value_count_cat_train_data.columns:

    

    count = value_count_cat_train_data[value_count_cat_train_data[i] < 10].shape[0]

    print("number of values in column: ", i, "occurring less than 10 times: ", count, "\n")

    print("\n")





for i in categorical_cols:

    mode = train_data[i].mode()[0]

    print("For column: ", i, "the most frequent value is: ", mode, "\n")

    value_count_cat_train_data[i].replace(1, mode, inplace = True)







print("Das hier sind die value_counts der ersetzten Dataframes: ", "\n")

print(value_count_cat_train_data.DRG.value_counts(), "\n")   

print(value_count_cat_train_data.Hauptdiagnose.value_counts(), "\n")   

print(value_count_cat_train_data.Aufnahmediagnose.value_counts(), "\n") 





train_data["DRG"] = value_count_cat_train_data["DRG"]

train_data["Hauptdiagnose"] = value_count_cat_train_data["Hauptdiagnose"]

train_data["Aufnahmediagnose"] = value_count_cat_train_data["Aufnahmediagnose"]

'''

import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier







# A parameter grid for XGBoost

params = {

        'min_child_weight': [1,3,5],

        'subsample': [0.8, 0.9, 1.0],

        'colsample_bytree': [0.8, 0.9, 1.0],

        'max_depth': [5, 6, 7]

        }





#####################################################################



xgb = XGBClassifier(learning_rate = 0.02, 

                    n_estimators = 800, 

                    objective = 'binary:logistic',

                    silent = False, 

                    n_jobs = -1,

                    eval_metric='auc',

                    tree_method='gpu_hist')





#####################################################################

folds = 3

param_comb = 3



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1)



random_search = RandomizedSearchCV(xgb, 

                                   param_distributions = params, 

                                   n_iter = param_comb, 

                                   scoring = 'roc_auc', 

                                   n_jobs = -1, 

                                   cv = skf.split(train_data,y_train), 

                                   verbose = 3, 

                                   random_state = 1)





random_search.fit(train_data, y_train)



#####################################################################
print('\n All results:')

print(random_search.cv_results_)





print('\n Best estimator:')

print(random_search.best_estimator_)
'''

XGBClassifier(base_score=0.5, 

              booster='gbtree', 

              colsample_bylevel=1,

              colsample_bynode=1, 

              colsample_bytree=1.0, 

              eval_metric='auc',

              gamma=0, 

              gpu_id=0, 

              importance_type='gain',

              interaction_constraints='', 

              learning_rate=0.02, 

              max_delta_step=0,

              max_depth=6, 

              min_child_weight=1, 

              missing=nan,

              monotone_constraints='(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0...,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)',

              n_estimators=800, 

              n_jobs=-1, 

              num_parallel_tree=1, 

              random_state=0,

              reg_alpha=0, 

              reg_lambda=1, 

              scale_pos_weight=1, 

              silent=False,

              subsample=0.8, 

              tree_method='gpu_hist', 

              validate_parameters=1,

              verbosity=None)

'''              
import xgboost as xgb



clf = xgb.XGBClassifier(objective = 'binary:logistic',

                        colsample_bylevel=1,

                        colsample_bynode=1, 

                        colsample_bytree=1.0, 

                        eval_metric='auc',

                        n_estimators=800,           # possible improvement: use xgb.cv to find best parameters ?

                        n_jobs=-1,

                        max_depth=6,

                        min_child_weight=1, 

                        learning_rate=0.02,

                        subsample=0.8,

                        verbosity = 3,            # this prints out many information during the running process

                        tree_method='gpu_hist')   # this line enables the  GPU accelerator





#######################################################

# With these parameters:



#clf = xgb.XGBClassifier(objective = 'binary:logistic'

#                        colsample_bylevel=1,

#                        colsample_bynode=1, 

#                        colsample_bytree=1.0, 

#                        eval_metric='auc',

#                        n_estimators=800,           # possible improvement: use xgb.cv to find best parameters ?

#                        n_jobs=-1,

#                        max_depth=6,

#                        min_child_weight=1, 

#                        learning_rate=0.02,

#                        subsample=0.8,

#                        colsample_bytree=1.0,

#                        verbosity = 3,            # this prints out many information during the running process

#                        tree_method='gpu_hist')   # this line enables the  GPU accelerator





# - it takes 6 minutes with GPU accelerator

# - private score: , resulting in rank  /6351

# - public score:  , resulting in rank  /6381

#######################################################







print("starting training process..... \n") 

clf.fit(train_data, y_train)
# this part here was done before I tried out RandomizedSearchCV



'''

import xgboost as xgb



clf = xgb.XGBClassifier(objective = 'binary:logistic'

                        n_estimators=500,           # possible improvement: use xgb.cv to find best parameters ?

                        n_jobs=4,

                        max_depth=6,

                        learning_rate=0.05,

                        subsample=1.0,

                        colsample_bytree=1.0,

                        verbosity = 3,            # this prints out many information during the running process

                        tree_method='gpu_hist')   # this line enables the  GPU accelerator





#######################################################

# With these parameters:



#clf = xgb.XGBClassifier(n_estimators=500,           

#                        n_jobs=4,

#                        max_depth=9,

#                        learning_rate=0.05,

#                        subsample=0.9,

#                        colsample_bytree=0.9,

#                        verbosity = 3)            # this prints out many information during the running process

#                        tree_method='gpu_hist')   # this line enables the  GPU accelerator





# - it takes 25 minutes with CPU and no GPU accelerator

# - it takes 6 minutes with GPU accelerator

# - private score: 0.906720, resulting in rank  3738/6351

# - public score:  0.936289, resulting in rank  3964/6381

#######################################################







print("starting training process..... \n") 

clf.fit(train_data, y_train)

'''


sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')



sample_submission['isFraud'] = clf.predict_proba(test_data)[:,1]

sample_submission.to_csv('simple_xgboost.csv')



print("saving was successful!")
