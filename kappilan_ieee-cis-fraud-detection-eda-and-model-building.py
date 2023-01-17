# Importing the required packages

#! pip install -U imbalanced-learn

import numpy as np

import pandas as pd

from pandas.plotting import table

import seaborn as sns



from scipy.stats import zscore



#matplotlib Packages

import matplotlib.pyplot as matplot

from matplotlib import pyplot as plt 









#sklearn packages

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split 

from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.cluster import KMeans

from sklearn.svm import SVC

from sklearn import metrics



# SMOTE for up sampling

from imblearn.combine import SMOTETomek



# setting for better readability of the output

pd.set_option('display.max_columns', None) 

pd.set_option('display.max_rows', None)

pd.options.display.float_format = '{:.3f}'.format





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Creating a function to find the distribution of the dataset, to check distribution before and after imputation 

def distribution(Source):

        print("Columns that are int32,int64 = ",Source.select_dtypes(include=['int32','int64']).columns)

        print("Columns that are flaot32,float64 = ",Source.select_dtypes(include=['float64']).columns)

        print("Columns that are objects = ",Source.select_dtypes(include=['object']).columns)

        a = pd.Series(Source.select_dtypes(include=['int32','int64']).columns)

        leng = len(a)

        for j in range(0,len(a)):

            f, axes = matplot.subplots(1, 2, figsize=(10, 10))

            sns.boxplot(Source[a[j]].value_counts(), ax = axes[0])

            sns.distplot(Source[a[j]].value_counts(), ax = axes[1])

            matplot.subplots_adjust(top =  1.5, right = 10, left = 8, bottom = 1)



        a = pd.Series(Source.select_dtypes(include=['float64']).columns)

        leng = len(a)

        for j in range(0,len(a)):

            matplot.Text('Figure for float64')

            f, axes = matplot.subplots(1, 2, figsize=(10, 10))

            sns.boxplot(Source[a[j]].value_counts(), ax = axes[0])

            sns.distplot(Source[a[j]].value_counts(), ax = axes[1])

            matplot.subplots_adjust(top =  1.5, right = 10, left = 8, bottom = 1)



        a = pd.Series(Source.select_dtypes(include=['object']).columns)

        leng = len(a)

        for j in range(0,len(a)):

            matplot.subplots()

            sns.countplot(Source[a[j]])
# Function to check the Quartile 

def quartile_check(Source):

    a = pd.Series(Source.select_dtypes(include=['int32','int64','float64','float32']).columns)

    leng = len(a)

    for j in range(0,len(a)):

        print("Quantiles for {}".format(a[j]))

        print(Source[a[j]].quantile([.01, .025, .1, .25, .5, .75, .9, .975, .99]))

        print("-----------------------------------------------------------------")
# Function for univariate plots

def univariate_plots(Source):

        print("Columns that are int32,int64 = ",Source.select_dtypes(include=['int32','int64']).columns)

        print("Columns that are flaot32,float64 = ",Source.select_dtypes(include=['float64']).columns)

        print("Columns that are objects = ",Source.select_dtypes(include=['object']).columns)

        a = pd.Series(Source.select_dtypes(include=['int32','int64']).columns)

        leng = len(a)

        for j in range(0,len(a)):

            f, axes = plt.subplots(1, 2, figsize=(10, 10))

            sns.boxplot(Source[a[j]], ax = axes[0])

            sns.distplot(Source[a[j]], ax = axes[1])

            plt.subplots_adjust(top =  1.5, right = 10, left = 8, bottom = 1)



        a = pd.Series(Source.select_dtypes(include=['float64']).columns)

        leng = len(a)

        for j in range(0,len(a)):

            plt.Text('Figure for float64')

            f, axes = plt.subplots(1, 2, figsize=(10, 10))

            sns.boxplot(Source[a[j]], ax = axes[0])

            sns.distplot(Source[a[j]], ax = axes[1])

            plt.subplots_adjust(top =  1.5, right = 10, left = 8, bottom = 1)



        a = pd.Series(Source.select_dtypes(include=['object']).columns)

        leng = len(a)

        for j in range(0,len(a)):

            plt.subplots()

            sns.countplot(Source[a[j]])
# importing the data

#identity_df = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

#transaction_df = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")

#identity_test_df = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

#transaction_test_df = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")



identity_df = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

transaction_df = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")

identity_test_df = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

transaction_test_df = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")

transaction_df.head(10)
identity_df.head(10)
print(identity_df.shape)
print(transaction_df.shape)

# The total Number of rows in Transaction dataset does not match with the total number of identity dataset. 

# There is high chance we mightnot have identity for few of the transactions.

identity_df.dtypes

# The data types seems to be a mixer of float, int and object
transaction_df.dtypes
identity_df.isnull().sum()
transaction_df.isnull().sum()
identity_test_df.isnull().sum()
transaction_test_df.isnull().sum()
# Finding the distribution of all the features which seems to be a categorical variable also checking if any numerical value has typo which led to datatype object

for col in identity_df.columns:

    if identity_df[col].dtypes == "object":

        print("----------")

        print(col)

        print("----------")

        print(identity_df[col].value_counts(dropna=False))

        print("----------")



# There is no numerical column misclassified as object because of incoorect value
# Getting the uniques values of each of the categorical variables  

for col in identity_df.columns:

    if identity_df[col].dtypes == "object":

        print("----------")

        print(col)

        print("----------")

        print(identity_df[col].nunique())

        print("----------")
# Finding the distribution of all the features which seems to be a categorical variable. -- Transaction Dataset

for col in transaction_df.columns:

    if transaction_df[col].dtypes == "object":

        print("----------")

        print(col)

        print("----------")

        print(transaction_df[col].value_counts(dropna=False))

        print("----------")

        

# There is no numerical column misclassified as object because of incoorect value
# Getting the uniques values of each of the categorical variables  

for col in transaction_df.columns:

    if transaction_df[col].dtypes == "object":

        print("----------")

        print(col)

        print("----------")

        print(transaction_df[col].nunique())

        print("----------")
id_corr_mat = identity_df.corr()

tran_corr_mat = transaction_df.corr()
#id_corr_mat.reset_index(inplace=True)

#tran_corr_mat.reset_index(inplace=True)
#id_corr_mat.columns
#id_corr_mat
#id_corr_mat[(id_corr_mat['id_01']>0.95) | (id_corr_mat['id_01']<-0.95)]['index']
#for i in id_corr_mat.columns:

 #   if i=='index':

  #      continue

   # else:

    #    for i in id_corr_mat

     #   print("highly correlated columns for {}".format(i))

      #  print("-----------------------------")

       # id_corr_mat[(id_corr_mat[i]>0.95) | (id_corr_mat[i]<-0.95)]['index']

        #print("-----------------------------")

    #print(j)

    #break
#id_corr_mat.head()
# Checking for null values in the identity dataset. From the below observation we can see there are a huge null values. We have to handle these null values. 

identity_df.isnull().sum()
# Creating a Dataframe with column name and the number of null values, so that we can drop the columns which has more than 75% 

# of data as null. 

is_null = identity_df.isnull().sum()

is_null = is_null.to_frame()

is_null["col_name"]=is_null.index

is_null.columns = ["null_count","col_name"]

# Creating a Dataframe with column name and the number of null values, so that we can drop the columns which has more than 75% 

# of data as null. 

test_is_null = identity_test_df.isnull().sum()

test_is_null = test_is_null.to_frame()

test_is_null["col_name"]=test_is_null.index

test_is_null.columns = ["null_count","col_name"]

# Getting the list of columns that are not to be removed. This is used to get the distribution for these columns before and after imputation

identity_non_removed_columns = []

inc = 0

for i,col in is_null.values:

    

    if (i/144233) <= 0.75:

        identity_non_removed_columns.append(col)
# distribution of data before the imputation is made. 

#distribution(identity_df[identity_non_removed_columns])



# Th distribution shows there are outliers
# before the imputation is made. 

#quartile_check(identity_df[identity_non_removed_columns]),file=open("identity_quartile.csv","a")

# Dropping the columns that has more than 75% of data as null.

inc = 0

for i,col in is_null.values:

    

    if (i/144233) > 0.75:

        identity_df.drop(labels=col,axis=1,inplace=True)

        print("column dropped {} with {} null values".format(col,i))

        inc = inc + 1

print("total number of columns dropped {}".format(inc))

# all the columns dropped has 96% of missing data.
identity_test_df.shape
# Dropping the columns that has more than 75% of data as null.

inc = 0

for i,col in test_is_null.values:

    

    if (i/141907) > 0.75:

        identity_test_df.drop(labels=col,axis=1,inplace=True)

        print("column dropped {} with {} null values".format(col,i))

        inc = inc + 1

print("total number of columns dropped {}".format(inc))

# all the columns dropped has 96% of missing data.
identity_test_df.columns = identity_df.columns

# The column names in the train and test datasets have slight variation, keeping the same name will avoid confusion 
# repeating the same for transaction dataset

is_null = transaction_df.isnull().sum()

is_null = is_null.to_frame()

is_null["col_name"]=is_null.index

is_null.columns = ["null_count","col_name"]
# Getting the list of columns that are not to be removed

transaction_non_removed_columns = []

inc = 0

for i,col in is_null.values:

    

    if (i/590540) <= 0.75:

        transaction_non_removed_columns.append(col)
#quartile_check(transaction_df[transaction_non_removed_columns])
# distribution of data before the imputation is made. 

#distribution(transaction_df[transaction_non_removed_columns])

# The distribution shows there are presence of outliier is the data. Since the categorical variabes are masked and representedd as int, we are not treating those columns
# repeating the same for transaction dataset

is_null = transaction_df.isnull().sum()

is_null = is_null.to_frame()

is_null["col_name"]=is_null.index

is_null.columns = ["null_count","col_name"]
# Dropping the columns that has more than 75% of data as null.

inc = 0

removed_columns = []

for i,col in is_null.values:

    

    if (i/590540) > 0.75:

        transaction_df.drop(labels=col,axis=1,inplace=True)

        print("column dropped {} with {} null values".format(col,i))

        inc = inc + 1

        removed_columns.append(col)

print("total number of columns dropped {}".format(inc))
# Dropping the columns that has more than 75% of data as null.

inc = 0

removed_columns = []

for i,col in is_null.values:

    

    if (i/590540) > 0.75:

        transaction_test_df.drop(labels=col,axis=1,inplace=True)

        print("column dropped {} with {} null values".format(col,i))

        inc = inc + 1

        removed_columns.append(col)

print("total number of columns dropped {}".format(inc))
sample = pd.read_csv("/kaggle/input/ieee-fraud-detection/sample_submission.csv")
print(transaction_df.shape)

print(transaction_test_df.shape)
#impute using measures of central tendency. Let us impute with mean for



#creating a list with categorical variable( as described in the data definition)

cat = ["DeviceType","DeviceInfo","id_12","id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24","id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36","id_37","id_38"]



# Imputing in iteration 

for col in identity_df.columns:

    if identity_df[col].isnull().sum() > 0:

        if col in cat:

            identity_df[col].fillna(identity_df[col].mode()[0],inplace=True)

            print("column {} has been imputed with mode".format(col))

        else:

            

            identity_df[col].fillna(identity_df[col].median(),inplace=True)

            print("column {} has been imputed with mean".format(col))          

            
identity_test_df.columns = identity_df.columns
#impute using measures of central tendency. Let us impute with mean for



#creating a list with categorical variable( as described in the data definition)

cat = ["DeviceType","DeviceInfo","id_12","id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24","id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36","id_37","id_38"]



# Imputing in iteration 

for col in identity_test_df.columns:

    if identity_test_df[col].isnull().sum() > 0:

        if col in cat:

            identity_test_df[col].fillna(identity_test_df[col].mode()[0],inplace=True)

            print("column {} has been imputed with mode".format(col))

        else:

            

            identity_test_df[col].fillna(identity_test_df[col].median(),inplace=True)

            print("column {} has been imputed with mean".format(col))          

            
cat = ["ProductCD","card1","card2","card3","card4","card5","card6","addr1", "addr2","P_emaildomain", "R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9",]

for col in transaction_df.columns:

    if transaction_df[col].isnull().sum() > 0:

        if col in cat:

            #print(col)

            transaction_df[col].fillna(transaction_df[col].mode()[0],inplace=True)

            print("column {} has been imputed with mode".format(col))

        else:

            #print(col)

            transaction_df[col].fillna(transaction_df[col].median(),inplace=True)

            print("column {} has been imputed with mean".format(col))
cat = ["ProductCD","card1","card2","card3","card4","card5","card6","addr1", "addr2","P_emaildomain", "R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9",]

for col in transaction_test_df.columns:

    if transaction_test_df[col].isnull().sum() > 0:

        if col in cat:

            #print(col)

            transaction_test_df[col].fillna(transaction_test_df[col].mode()[0],inplace=True)

            print("column {} has been imputed with mode".format(col))

        else:

            #print(col)

            transaction_test_df[col].fillna(transaction_test_df[col].median(),inplace=True)

            print("column {} has been imputed with mean".format(col))
# distribution of data after the imputation is made. 

#distribution(identity_df[identity_non_removed_columns])



#Imputation does not seem to have a much impact to the distribution
#print(quartile_check(identity_df[identity_non_removed_columns]),file=open("identity_quartile.csv","a"))
#quartile_check(transaction_df[transaction_non_removed_columns])
# distribution of data before the imputation is made. 

#distribution(transaction_df[transaction_non_removed_columns])



# Imputation has no much impact to the distribution
# Distribution of Transaction Amout for Fraud and Legitimate transaction



plt.figure(figsize=(50,15))

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

f.suptitle('Amount per transaction by class')

bins = 50

ax1.hist(transaction_df[transaction_df["isFraud"]==1]["TransactionAmt"], bins = bins,histtype="stepfilled")

ax1.set_title('Fraud')

ax2.hist(transaction_df[transaction_df["isFraud"]==0]["TransactionAmt"], bins = bins,histtype="stepfilled")

ax2.set_title('Legit')

plt.xlabel('Amount ($)')

plt.ylabel('Number of Transactions')

plt.xlim((0, 33000))

plt.yscale('log')

plt.show();



# The distribution of fraud transaction is in the lower range and legitimate transaction hass higher values
# getting the number of unique values from the card columns to understand the column values.  
print("Unique value present in card1 {}".format(transaction_df["card1"].nunique()))

print("Unique value present in card2 {}".format(transaction_df["card2"].nunique()))

print("Unique value present in card3 {}".format(transaction_df["card3"].nunique()))

print("Unique value present in card4 {}".format(transaction_df["card4"].nunique()))

print("Unique value present in card5 {}".format(transaction_df["card5"].nunique()))

print("Unique value present in card6 {}".format(transaction_df["card6"].nunique()))
transaction_df[transaction_df["card4"]=="discover"]["card3"].value_counts()
transaction_df[transaction_df["card4"]=="discover"]["card4"].value_counts()

transaction_df[transaction_df["card4"]=="discover"]["card3"].value_counts()
c11= transaction_df.groupby(["card1"])["TransactionAmt"].max()

c12= transaction_df.groupby(["card1"])["TransactionAmt"].min()

c13= transaction_df.groupby(["card1"])["TransactionAmt"].mean()

c1 = pd.merge(c11,c12,on="card1")

c1 = pd.merge(c1,c13,on="card1")

c1 = c1.reset_index()

c1.columns = ["card1","Max","Min","Avg"]

c1.index = c1["card1"]

c1.drop("card1",axis=1,inplace=True)
c1.sort_values(by="Max",ascending=False).head(10).plot(kind="bar")

# Checking the distribution for top 10 category, majority of the transaction has occured in the 16075 category 
c21= transaction_df.groupby(["card2"])["TransactionAmt"].max()

c22= transaction_df.groupby(["card2"])["TransactionAmt"].min()

c23= transaction_df.groupby(["card2"])["TransactionAmt"].mean()

c2 = pd.merge(c21,c22,on="card2")

c2 = pd.merge(c2,c23,on="card2")

c2 = c2.reset_index()

c2.columns = ["card2","Max","Min","Avg"]

c2.index = c2["card2"]

c2.drop("card2",axis=1,inplace=True)
c2.sort_values(by="Max",ascending=False).head(10).plot(kind="bar")

# Checking the distribution for top 10 category, majority of the transaction has occured in the 514 category 
c31= transaction_df.groupby(["card3"])["TransactionAmt"].max()

c32= transaction_df.groupby(["card3"])["TransactionAmt"].min()

c33= transaction_df.groupby(["card3"])["TransactionAmt"].mean()

c3 = pd.merge(c31,c32,on="card3")

c3 = pd.merge(c3,c33,on="card3")

c3 = c3.reset_index()

c3.columns = ["card3","Max","Min","Avg"]

c3.index = c3["card3"]

c3.drop("card3",axis=1,inplace=True)
c3.sort_values(by="Max",ascending=False).head(10).plot(kind="bar")

# Checking the distribution for top 10 category, majority of the transaction has occured in the 150 category 
c51= transaction_df.groupby(["card5"])["TransactionAmt"].max()

c52= transaction_df.groupby(["card5"])["TransactionAmt"].min()

c53= transaction_df.groupby(["card5"])["TransactionAmt"].mean()

c5 = pd.merge(c51,c52,on="card5")

c5 = pd.merge(c5,c53,on="card5")

c5 = c5.reset_index()

c5.columns = ["card5","Max","Min","Avg"]

c5.index = c5["card5"]

c5.drop("card5",axis=1,inplace=True)
c5.sort_values(by="Max",ascending=False).head(10).plot(kind="bar")

# Checking the distribution for top 10 category, majority of the transaction has occured in the 102 category 
card_detail = transaction_df[["card1","card2","card3","card4","card5","card6"]]
IQR = transaction_df["TransactionAmt"].quantile(0.75) - transaction_df["TransactionAmt"].quantile(0.25)

upper_limit = transaction_df["TransactionAmt"].quantile(0.75) + (IQR * 1.5)

lower_limit = transaction_df["TransactionAmt"].quantile(0.25) - (IQR * 1.5)

print("Upper Limit :{}".format(upper_limit))

print("Lower Limit :{}".format(lower_limit))

#Any thing lower than the upper limit or lower than the lower limit will be treated as outlier, 

#We are not handling the outlier and just observing the distribution.
# Distribution excluding the outlier limit

sns.distplot(transaction_df[transaction_df["TransactionAmt"] < 247.000000]["TransactionAmt"])
# Positive Outlier

sns.distplot(transaction_df[transaction_df["TransactionAmt"] > 247.000000]["TransactionAmt"])
# Transaction Amount cannot be negative. There is no negative outlier 

transaction_df[transaction_df["TransactionAmt"] < -79].shape
#univariate_plots(transaction_df)
#univariate_plots(identity_df)
total = len(transaction_df)

total_amt = transaction_df.groupby(['isFraud'])['TransactionAmt'].sum().sum()



print(transaction_df["isFraud"].value_counts())

print(transaction_df.groupby(["isFraud"])["TransactionAmt"].sum().astype(int))



plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



transaction_df["isFraud"].value_counts().plot(kind="bar",ax=ax1,title="Over all number of Legit and Fraudelent Transaction")



for i in ax1.patches:

    # get_x pulls left or right; get_height pushes up or down

    ax1.text(i.get_x()-.03, i.get_height()+.5, 

            str(round(i.get_height()/total*100, 2))+'%', fontsize=15,

                color='dimgrey')





ax2 = plt.subplot(122)

transaction_df.groupby(["isFraud"])["TransactionAmt"].sum().plot(kind="bar",ax=ax2,title="Over all Legit and Fraudelent Transaction Amount")





for i in ax2.patches:

    # get_x pulls left or right; get_height pushes up or down

    ax2.text(i.get_x()-.03, i.get_height()+.5, 

            str(round(i.get_height()/total_amt*100, 2))+'%', fontsize=15,

                color='dimgrey')







plt.show()
transaction_df["Day"] = transaction_df["TransactionDT"]/86400
transaction_df["Day"] = transaction_df["Day"].astype(int)
transaction_df["Day of the week"] = ((transaction_df["Day"] - 1)%7)+ 1
transaction_df["Month"] = ((transaction_df["Day"]//30) % 30) + 1
transaction_df["Day of Month"] = transaction_df["Day"] % 30 
transaction_df["Day of Month"] = transaction_df["Day of Month"].replace(0,30)
transaction_df["Hour of the Day"] = (transaction_df["TransactionDT"] // 3600) % 24
transaction_test_df["Day"] = transaction_df["TransactionDT"]/86400
transaction_test_df["Day"] = transaction_df["Day"].astype(int)
transaction_test_df["Day of the week"] = ((transaction_df["Day"] - 1)%7)+ 1
transaction_test_df["Month"] = ((transaction_df["Day"]//30) % 30) + 1
transaction_test_df["Day of Month"] = transaction_df["Day"] % 30 
transaction_test_df["Day of Month"] = transaction_df["Day of Month"].replace(0,30)
transaction_test_df["Hour of the Day"] = (transaction_df["TransactionDT"] // 3600) % 24
#transaction_df.drop("Day",axis=1,inplace=True)

#transaction_test_df.drop("Day",axis=1,inplace=True)

#transaction_df.drop("Month",axis=1,inplace=True)

#transaction_test_df.drop("Month",axis=1,inplace=True)
#transaction_df.drop("TransactionDT",axis=1,inplace=True)

#transaction_test_df.drop("TransactionDT",axis=1,inplace=True)
transaction_df.dtypes
trans_amt_df1 = transaction_df.groupby(["Month","isFraud"])["TransactionAmt"].sum().astype(int).to_frame()

trans_amt_df1 = trans_amt_df1.reset_index()

tmp1 = trans_amt_df1.groupby(["Month"])["TransactionAmt"].sum().to_frame().reset_index()

trans_amt_df1 = pd.merge(trans_amt_df1,tmp1,on="Month")

trans_amt_df1["Percent Contribution"] = (trans_amt_df1["TransactionAmt_x"]/trans_amt_df1["TransactionAmt_y"])*100

trans_amt_df11= trans_amt_df1[trans_amt_df1["isFraud"]==1][["Month","Percent Contribution"]]

trans_amt_df11
transaction_df["P_emaildomain"].nunique()
# For better understanding grouping all the mail domain that has less tha 3% fraud recorded as others

transaction_df.loc[transaction_df.P_emaildomain.isin(transaction_df.P_emaildomain.value_counts()[(transaction_df[transaction_df["isFraud"]==1].P_emaildomain.value_counts())/transaction_df.P_emaildomain.value_counts() < 0.035].index), 'P_emaildomain'] = "Others"
transaction_test_df["P_emaildomain"].nunique()
# Grouping the emaildomain similar to train dataset

cond = transaction_test_df['P_emaildomain'].isin(transaction_df['P_emaildomain']) == False

transaction_test_df.loc[transaction_test_df[cond].index,'P_emaildomain'] = "Others"



print(transaction_df["P_emaildomain"].nunique())

print(transaction_test_df["P_emaildomain"].nunique())


for col in transaction_df.columns:

    listv = []

    if transaction_df[col].dtypes == "object":

        for var in transaction_df[col].unique():

            listv.append(var)

        per = pd.crosstab(transaction_df[col], transaction_df['isFraud'], normalize='index') * 100

        per = per.reset_index()

        per.rename(columns={0:'Legit', 1:'Fraud'}, inplace=True)





        plt.figure(figsize=(15,6))



        ax1 = plt.subplot(121)



        g1= sns.countplot(x=col, hue='isFraud', data=transaction_df,order=listv,palette="Set2")

        gt = g1.twinx()

        gt = sns.pointplot(x=col, y='Fraud', data=per, color='black', legend=False,order=listv)

        gt.set_ylabel("% of Fraud Transactions", fontsize=12)



        g1.set_title("Number of Legit and Fraudelent Transaction and Fraud contributio", fontsize=14)

        g1.set_ylabel("Count", fontsize=12)





# set individual bar lables using above list

        for i in ax1.patches:

    # get_x pulls left or right; get_height pushes up or down

            ax1.text(i.get_x()-.03, i.get_height()+.5, 

                    str(round(i.get_height(), 2)), fontsize=15,

                        color='dimgrey')

    



        plt.show()



# Observing the distribution of the categorical variables and checkin the % contribution of fraudulent transaction 

# by each of its class.



        

per = pd.crosstab(transaction_df['Month'], transaction_df['isFraud'], normalize='index') * 100

per = per.reset_index()

per.rename(columns={0:'Legit', 1:'Fraud'}, inplace=True)





plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



g1= sns.countplot(x='Month', hue='isFraud', data=transaction_df,palette="Set2")

gt = g1.twinx()

gt = sns.pointplot(x='Month', y='Fraud', data=per, color='red', legend=False)

gt.set_ylabel("% of Fraud Transactions", fontsize=12)



g1.set_title("Number of Legit and Fraudelent Transaction Every Month", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





# set individual bar lables using above list

for i in ax1.patches:

    # get_x pulls left or right; get_height pushes up or down

    ax1.text(i.get_x()-.03, i.get_height()+.5, 

            str(round(i.get_height(), 2)), fontsize=15,

                color='dimgrey')

    



plt.show()



p_cont1 = pd.crosstab(transaction_df['Day of the week'], transaction_df['isFraud'], normalize='index') * 100

p_cont1 = p_cont1.reset_index()

p_cont1.rename(columns={0:'Legit', 1:'Fraud'}, inplace=True)





listv = []

for var in transaction_df['Day of the week'].unique():

    listv.append(var)



plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



g1= sns.countplot(x='Day of the week', hue='isFraud', data=transaction_df,order=listv,palette="Set2")

gt = g1.twinx()

gt = sns.pointplot(x='Day of the week', y='Fraud', data=p_cont1, color='black', legend=False,order=listv)

gt.set_ylabel("% of Fraud Transactions", fontsize=12)



g1.set_title("Number of Legit and Fraudelent Transaction for Day of Week", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





 



plt.show()
per = pd.crosstab(transaction_df['Hour of the Day'], transaction_df['isFraud'], normalize='index') * 100

per = per.reset_index()

per.rename(columns={0:'Legit', 1:'Fraud'}, inplace=True)





listv = []

for var in transaction_df['Hour of the Day'].unique():

    listv.append(var)



plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



g1= sns.countplot(x='Hour of the Day', hue='isFraud', data=transaction_df,order=listv,palette="Set2")

gt = g1.twinx()

gt = sns.pointplot(x='Hour of the Day', y='Fraud', data=per, color='black', legend=False,order=listv)

gt.set_ylabel("% of Fraud Transactions", fontsize=12)



g1.set_title("Number of Legit and Fraudelent Transaction on Hour of Day basis", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





 



plt.show()



# Number of Fraudelent Transactions seems close enough every month
transaction_df["card_cat_type"] = transaction_df["card4"] + " " + transaction_df["card6"]
transaction_test_df["card_cat_type"] = transaction_test_df["card4"] + " " + transaction_test_df["card6"]
per = pd.crosstab(transaction_df['card_cat_type'], transaction_df['isFraud'], normalize='index') * 100

per = per.reset_index()

per.rename(columns={0:'Legit', 1:'Fraud'}, inplace=True)





listv = []

for var in transaction_df['card_cat_type'].unique():

    listv.append(var)



plt.figure(figsize=(45,8))



ax1 = plt.subplot(121)



g1= sns.countplot(x='card_cat_type', hue='isFraud', data=transaction_df,order=listv,palette="Set2")

gt = g1.twinx()

gt = sns.pointplot(x='card_cat_type', y='Fraud', data=per, color='black', legend=False,order=listv)

gt.set_ylabel("% of Fraud Transactions", fontsize=12)



g1.set_title("Number of Legit and Fraudelent Transaction Every Month", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





 



plt.show()
transaction_df.shape


transaction_df.drop(["Day","Day of the week","Month","Day of Month","Hour of the Day"],axis=1,inplace=True)

transaction_test_df.drop(["Day","Day of the week","Month","Day of Month","Hour of the Day"],axis=1,inplace=True)

# removing these columns after required Analysis
# Dropping card4 and card6 as we have created a column concatinating both

transaction_df.drop(["card4","card6"],axis=1,inplace=True)

transaction_test_df.drop(["card4","card6"],axis=1,inplace=True)
# dropping this columns intuitivly as they might not have any correlation with the dependent variables. 

identity_df.drop(["id_30","id_31","id_33","DeviceInfo"],axis=1,inplace=True)

identity_test_df.drop(["id_30","id_31","id_33","DeviceInfo"],axis=1,inplace=True)
print(transaction_df.shape)

print(transaction_test_df.shape)
print(identity_df.shape)

print(identity_test_df.shape)
new_paymentcard_df = pd.merge(transaction_df,identity_df,on="TransactionID",how="left")

# Doing an inner join as doind a left join would leave all the columns from transaction dataset null and imputing it will be of no use..
test_new_paymentcard_df = pd.merge(transaction_test_df,identity_test_df,on="TransactionID",how="left")

# Doing an inner join as doind a left join would leave all the columns from transaction dataset null and imputing it will be of no use..
print(new_paymentcard_df.shape)

print(test_new_paymentcard_df.shape)
new_paymentcard_df.isnull().sum()
#impute using measures of central tendency. Let us impute with mean for



#creating a list with categorical variable( as described in the data definition)

cat = ["DeviceType","DeviceInfo","id_12","id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24","id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36","id_37","id_38"]



# Imputing in iteration 

for col in new_paymentcard_df.columns:

    if new_paymentcard_df[col].isnull().sum() > 0:

        if col in cat:

            new_paymentcard_df[col].fillna(new_paymentcard_df[col].mode()[0],inplace=True)

            print("column {} has been imputed with mode".format(col))

        else:

            

            new_paymentcard_df[col].fillna(new_paymentcard_df[col].median(),inplace=True)

            print("column {} has been imputed with mean".format(col))          

            
#impute using measures of central tendency. Let us impute with mean for



#creating a list with categorical variable( as described in the data definition)

cat = ["DeviceType","DeviceInfo","id_12","id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24","id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36","id_37","id_38"]



# Imputing in iteration 

for col in test_new_paymentcard_df.columns:

    if test_new_paymentcard_df[col].isnull().sum() > 0:

        if col in cat:

            test_new_paymentcard_df[col].fillna(test_new_paymentcard_df[col].mode()[0],inplace=True)

            print("column {} has been imputed with mode".format(col))

        else:

            

            test_new_paymentcard_df[col].fillna(test_new_paymentcard_df[col].median(),inplace=True)

            print("column {} has been imputed with mean".format(col))          

            
col = []

for c in new_paymentcard_df.columns:

    if new_paymentcard_df[c].dtypes=='object':

        col.append(c)

        



new_paymentcard_dummies = pd.get_dummies(new_paymentcard_df , columns=col, drop_first=True)
X = new_paymentcard_dummies.drop(["isFraud",'TransactionID'],axis=1)

X_id = new_paymentcard_dummies["TransactionID"]

y = new_paymentcard_dummies["isFraud"]
X_Scaled = StandardScaler().fit_transform(X)

cov_matrix = np.cov(X_Scaled.T)

#print('Covariance Matrix \n%s', cov_matrix)



e_vals, e_vecs = np.linalg.eig(cov_matrix)

e_vals, e_vecs = np.linalg.eig(cov_matrix)



#print('Eigenvectors \n%s' %e_vecs)

#print('\nEigenvalues \n%s' %e_vals)



tot = sum(e_vals)

var_exp = [( i /tot ) * 100 for i in sorted(e_vals, reverse=True)]



cum_var_exp = np.cumsum(var_exp)

#print("Cumulative Variance Explained", cum_var_exp)



matplot.figure(figsize=(20 , 15))

matplot.bar(range(1, e_vals.size + 1), var_exp, alpha = 0.5, align = 'center', label = 'Individual explained variance')

matplot.step(range(1, e_vals.size + 1), cum_var_exp, where='mid', label = 'Cumulative explained variance')

matplot.ylabel('Explained Variance Ratio')

matplot.xlabel('Principal Components')

matplot.legend(loc = 'best')

matplot.tight_layout()

matplot.show()



Pricipal_comp_composition = (pd.DataFrame(cum_var_exp).reset_index())

Pricipal_comp_composition.columns = ['Pricipal Components', '% info retained']
Pricipal_comp_composition
from sklearn.decomposition import PCA as PCA

pca = PCA(n_components=143)

principalComponents = pca.fit_transform(X)
principalComponents_df = pd.DataFrame(principalComponents)
X_trainval, X_test, y_trainval, y_test = train_test_split(principalComponents_df,y , test_size=0.20, random_state=1)



X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.30, random_state=1)
lab_enc = preprocessing.LabelEncoder()

training_scores_encoded = lab_enc.fit_transform(y_train)

val_scores_encoded = lab_enc.fit_transform(y_val)

test_scores_encoded = lab_enc.fit_transform(y_test)
rfcl = RandomForestClassifier(n_estimators = 75 , random_state=1234,max_depth=25

                              ,criterion='gini',max_features='sqrt')

rfcl = rfcl.fit(X_train, y_train)

y_predict_rfcl = rfcl.predict(X_val)

print(rfcl.score(X_train , y_train))

print(rfcl.score(X_val, y_val))

print(metrics.confusion_matrix(y_val, y_predict_rfcl))

print(metrics.classification_report(y_val, y_predict_rfcl))
y_predict_rfcl = rfcl.predict(X_test)

print(rfcl.score(X_test, y_test))

print(metrics.confusion_matrix(y_test, y_predict_rfcl))

print(metrics.classification_report(y_test, y_predict_rfcl))
# calculate the fpr and tpr for all thresholds of the classification



# Firstly, calculate the probabilities of predictions made

probs = rfcl.predict_proba(X_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)

metrics

# method to plot

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0,1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from lightgbm import LGBMClassifier

lgb_clf = LGBMClassifier(random_state=17)
lgb_clf.fit(X_train,y_train)
y_predict_lgb = lgb_clf.predict(X_val)
print(lgb_clf.score(X_train , y_train))

print(metrics.accuracy_score(y_val,y_predict_lgb))

print(metrics.confusion_matrix(y_val,y_predict_lgb))

print(metrics.classification_report(y_val,y_predict_lgb))
y_predict_lgb = lgb_clf.predict(X_test)

print(metrics.accuracy_score(y_test,y_predict_lgb))

print(metrics.confusion_matrix(y_test,y_predict_lgb))

print(metrics.classification_report(y_test,y_predict_lgb))
# calculate the fpr and tpr for all thresholds of the classification



# Firstly, calculate the probabilities of predictions made

probs = lgb_clf.predict_proba(X_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)

metrics

# method to plot

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0,1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from xgboost import XGBClassifier
xgb_clf = XGBClassifier(random_state=17,n_estimators=175,max_depth=7)

xgb_clf.fit(X_train, y_train)
# make predictions for test data

print(xgb_clf.score(X_train , y_train))

y_predict = xgb_clf.predict(X_val)

print(metrics.accuracy_score(y_val,y_predict))

print(metrics.confusion_matrix(y_val,y_predict))

print(metrics.classification_report(y_val,y_predict))
y_predict = xgb_clf.predict(X_test)
print(metrics.accuracy_score(y_test,y_predict))

print(metrics.confusion_matrix(y_test,y_predict))

print(metrics.classification_report(y_test,y_predict))
# calculate the fpr and tpr for all thresholds of the classification



# Firstly, calculate the probabilities of predictions made

probs = lgb_clf.predict_proba(X_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)

metrics

# method to plot

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0,1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()