import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

from sklearn.preprocessing import StandardScaler #column standardization

from sklearn.preprocessing import OneHotEncoder 

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from scipy.stats import chi2_contingency



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/network-intrusion-detection/Train_data.csv')
test = pd.read_csv('/kaggle/input/network-intrusion-detection/Test_data.csv')
# checking number of columns and type of each columnb

train.info()
# check the first 10 records of train dataset

train.head(10)
train['protocol_type'].value_counts()
train['flag'].value_counts()
pd.set_option('display.max_row', None)

train['service'].value_counts()
train['class'].value_counts()
pro_flg_serv = train.groupby(['protocol_type','service','class'])['class'].count()

pro_flg_serv
train.describe()
# https://stackoverflow.com/questions/28576540/how-can-i-normalize-the-data-in-a-range-of-columns-in-my-pandas-dataframe

# Normalize the data across numeric columns in dataset. Therefore, removing 4 object columns from list

cols_to_norm = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 

                'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 

                'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 

                'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 

                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 

                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 

                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

train[cols_to_norm] = StandardScaler().fit_transform(train[cols_to_norm])

test[cols_to_norm] = StandardScaler().fit_transform(test[cols_to_norm])
train.head()
# https://stackoverflow.com/questions/38913965/make-the-size-of-a-heatmap-bigger-with-seaborn

corr_df = train[cols_to_norm].corr(method='pearson')

fig, ax = plt.subplots(figsize=(12,12)) 

sns.heatmap(corr_df)
pd.set_option('display.max_column',None)

corr_df
# Since num_outbound_cmds and is_host_login is NAN value, we are dropping it from column and row.

corr_df.drop(index=['is_host_login','num_outbound_cmds'], columns=['is_host_login','num_outbound_cmds'], inplace=True)
corr_df.shape
# https://thispointer.com/python-pandas-how-to-add-rows-in-a-dataframe-using-dataframe-append-loc-iloc/

corr_col_df = pd.DataFrame(columns=["Column1","Column2","Corr_value"])

for i in corr_df.columns:

    for j in corr_df.index:

        if (i != j) and (corr_df[i][j] > 0.7):

            corr_col_df = corr_col_df.append({ "Column1" : i, "Column2" : j, "Corr_value" : corr_df[i][j] }, ignore_index=True)

            #print(i, "\t", j, "\t", corr_df[i][j])

            

corr_col_df
# records are repeating while is creating unnecessary complexity in analysis. Will try to remove duplicate records



ind_list = []

for i in range(len(corr_col_df)):

    for j in range(len(corr_col_df)):

        #print("j", j)

        #print("corr", uni_corr_col_df['Corr_value'][j])

        #print("columns", uni_corr_col_df['Column1'][i], uni_corr_col_df['Column2'][j])

        if ((i!=j) and (corr_col_df['Corr_value'][i] == corr_col_df['Corr_value'][j]) 

            and (corr_col_df['Column1'][i] == corr_col_df['Column2'][j]) 

            and (corr_col_df['Column2'][i] == corr_col_df['Column1'][j])):

            ind_list.append([i,j])



# Unique pair value from list - 

# https://www.geeksforgeeks.org/python-remove-duplicates-from-nested-list/

# https://stackoverflow.com/questions/47051854/remove-duplicates-based-on-the-content-of-two-columns-not-the-order

for i in ind_list:

    i.sort()

uni_ind_list = list(set(tuple(i) for i in ind_list)) 



# store unique records into dataframe

uni_corr_col_df = pd.DataFrame(columns=["Column1","Column2","Corr_value"])

for i in uni_ind_list:

    uni_corr_col_df = uni_corr_col_df.append(corr_col_df.iloc[i[0]], ignore_index=True)

    

uni_corr_col_df
# identifying columns to delete from dataframe

col_corr = set() # Set of all the names of deleted columns

for i in range(len(uni_corr_col_df)):

    if (uni_corr_col_df['Column1'][i] not in col_corr):

        colname = uni_corr_col_df['Column2'][i] # getting the name of column

        col_corr.add(colname)



col_corr = list(col_corr)

print(col_corr)



# dropping identified columns from train and test dataset

train.drop(col_corr, axis=1, inplace=True)

test.drop(col_corr, axis=1, inplace=True)
train.shape
test.shape
col_corr = set() # Set of all the names of deleted columns

for i in range(len(corr_df.columns)):

    for j in range(i):

        if (corr_df.iloc[i, j] >= 0.7) and (corr_df.columns[j] not in col_corr):

            colname = corr_df.columns[i] # getting the name of column

            col_corr.add(colname)

col_corr = list(col_corr)

print(col_corr)



# dropping identified columns from train and test dataset



#Uncomment below code if you are using Method 2.

#train.drop(col_corr, axis=1, inplace=True)

#test.drop(col_corr, axis=1, inplace=True)
train.shape
test.shape
#https://towardsdatascience.com/chi-squared-test-for-feature-selection-with-implementation-in-python-65b4ae7696db

# we need to pass data in cross tabular format to chi2_contingency. Therefore, using pd.crosstab

# we are assuming that significant value is 0.05

alpha = 0.05

stat, p, dof, expected = chi2_contingency(pd.crosstab(train['protocol_type'], train['class']))

print("p", p)



if p<=alpha:

    print("\nprotocol_type and class columns are dependent")

else:

    print("\nprotocol_type and class columns are independent")
stat, p, dof, expected = chi2_contingency(pd.crosstab(train['service'], train['class']))

print("p", p)



if p<=alpha:

    print("\nservice and class columns are dependent")

else:

    print("\nservice and class columns are independent")
stat, p, dof, expected = chi2_contingency(pd.crosstab(train['flag'], train['class']))

print("p", p)



if p<=alpha:

    print("\nflag and class columns are dependent")

else:

    print("\nflag and class columns are independent")
# convert dependent variable to number.

label_encoder = LabelEncoder()

train['class'] = label_encoder.fit_transform(train['class'])
y = train['class']

y.shape
X = train.drop('class', axis=1)

X.shape


X = pd.get_dummies(X)
X.head()
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression(random_state=45).fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics.confusion_matrix(y_test,y_pred)
metrics.f1_score(y_test, y_pred)


X['protocol_type'] = label_encoder.fit_transform(X['protocol_type'])

X['service'] = label_encoder.fit_transform(X['service'])

X['flag'] = label_encoder.fit_transform(X['flag'])



X['protocol_type'] = X['protocol_type'].astype('category')

X['service'] = X['service'].astype('category')

X['flag'] = X['flag'].astype('category')
X.head()
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression(random_state=45).fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics.confusion_matrix(y_test,y_pred)
metrics.f1_score(y_test, y_pred)
test.shape
test = pd.get_dummies(test)

test.shape
# Here we are checking if there is any column which is present in test dataset but not in train dataset. 

# If yes, then we will delete it from test dataset because model is not trained on those columns.

for i in list(test.columns):

    if i not in list(X.columns):

        print(i)

        test.drop(i, axis=1, inplace=True)
test.shape
# in order to fetch particular column index, we can use df.columns.get_loc()

# https://stackoverflow.com/questions/13021654/get-column-index-from-column-name-in-python-pandas

# to add a column on particular index loc in dataframe, we can use df.insert()

# https://stackoverflow.com/questions/18674064/how-do-i-insert-a-column-at-a-specific-column-index-in-pandas



# Here, we are identifying missing columns from test dataset.

for i in list(X.columns):

    if i not in list(test.columns):

        print(i)

        ind = X.columns.get_loc(i)

        test.insert(loc=ind, column=i,value=0)
test.shape
test.head()
model.predict(test)