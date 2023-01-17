import warnings



warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
pd.set_option("display.max_columns", 500)

pd.set_option("display.max_rows", 500)

pd.set_option("display.float_format", lambda x: "%.3f" % x)
filepath = '../input/'

loan_df = pd.read_csv(filepath + 'loan-data/loan.csv')

loan_df.head(2)
loan_df.columns
def nullPercentage(df):

    print(round((df.isnull().sum() / len(df.index) * 100), 2))
nullPercentage(loan_df)
# Drop Column based on Null Percentage Criteria

temp_df = pd.DataFrame(round((loan_df.isnull().sum() / len(loan_df.index) * 100), 2))



column = (temp_df.loc[temp_df[0] > 7].index).tolist()



loan_df_clean = loan_df.drop(columns=column, axis=1)



nullPercentage(loan_df_clean)
if len(loan_df_clean["id"].value_counts()) == len(loan_df_clean):

    print("There's no duplicate record in dataframe")
loan_df_clean["title"].value_counts()
# Drop the Columns that will not be required for analysis



columns = [

    "zip_code",

    "pymnt_plan",

    "url",

    "addr_state",

    "earliest_cr_line",

    "initial_list_status",

    "last_pymnt_d",

    "collections_12_mths_ex_med",

    "policy_code",

    "application_type",

    "acc_now_delinq",

    "chargeoff_within_12_mths",

    "delinq_amnt",

    "tax_liens",

    "member_id",

    "id",

]



for column in columns:

    if column in loan_df_clean.columns.tolist():

        loan_df_clean = loan_df_clean.drop(column, axis=1)



loan_df_clean.head()
for col in loan_df_clean.columns:

    if col not in (["loan_amnt"]):

        print("column : " + col)

        print(loan_df_clean[col].unique())

        print("")
nullPercentage(loan_df_clean)
# drop null rows

loan_df_clean = loan_df_clean.dropna()
# Get Column and Data Type Info

loan_df_clean.info()
# Get Overview of Data

loan_df_clean.describe()
# Show Unique Value of Columns

for col in loan_df_clean.columns:

    if col not in (["loan_amnt"]):

        print("column : " + col)

        print(loan_df_clean[col].unique())

        print("")
# Int_rate Can be converted to float for more insights

loan_df_clean.int_rate = (

    loan_df_clean.int_rate.str.rstrip("%").astype(float)

    if loan_df_clean.int_rate.dtype != float

    else loan_df_clean.int_rate

)
sns.distplot(loan_df_clean["int_rate"])
loan_df_clean["int_rate"].describe()
# Create Int_rate Binning into Two Category - "Low", "Medium", "High" & "Very High"

loan_df_clean["int_rate_interval"] = pd.cut(

    x=loan_df_clean["int_rate"],

    bins=[0, 9, 11, 14, 25],

    include_lowest=True,

    labels=["Low", "Medium", "High", "Very High"],

)

loan_df_clean["int_rate_interval"].value_counts().plot("bar")
# Derive Month and year Column

loan_df_clean["issue_d"] = pd.to_datetime(loan_df_clean["issue_d"], format="%b-%y")

#loan_df_clean["issue_month"] = loan_df_clean["issue_d"].dt.month_name(locale="English")

loan_df_clean["issue_year"] = loan_df_clean["issue_d"].dt.year
# consider only data that are having loan-status as Full_Paid or Charged_Off

## Derived Column - loan_status_rate : that will give us defaulter rate

loan_df_clean = loan_df_clean.loc[loan_df_clean["loan_status"] != "Current"]



loan_df_clean["defaulter_rate"] = loan_df_clean["loan_status"].apply(

    lambda x: 0 if x == "Fully Paid" else 1

)
# Drop Consumer Variables as it will not help for new applicants

consumer_var = [

    "delinq_2yrs",

    "earliest_cr_line",

    "inq_last_6mths",

    "open_acc",

    "pub_rec",

    "revol_bal",

    "revol_util",

    "total_acc",

    "out_prncp",

    "out_prncp_inv",

    "total_pymnt",

    "total_pymnt_inv",

    "total_rec_prncp",

    "total_rec_int",

    "total_rec_late_fee",

    "recoveries",

    "collection_recovery_fee",

    "last_pymnt_d",

    "last_pymnt_amnt",

    "last_credit_pull_d",

    "application_type",

]



for var in consumer_var:

    if var in loan_df_clean.columns.to_list():

        loan_df_clean = loan_df_clean.drop(columns=var, axis=1)
# Create Interval for loan_amnt - LOW,MEDIUM,HIGH,VERY HIGH



loan_df_clean["loan_amnt_interval"] = pd.cut(

    x=loan_df_clean["loan_amnt"],

    bins=[0, 5000, 9300, 12000, 25000],

    include_lowest=True,

    labels=["LOW", "MEDIUM", "HIGH", "VERY HIGH"],

)

loan_df_clean["loan_amnt_interval"].unique()
# Create Interval for funded_amnt_inv - LOW,MEDIUM,HIGH,VERY HIGH





loan_df_clean["funded_amnt_inv_interval"] = pd.cut(

    x=loan_df_clean["funded_amnt_inv"],

    bins=[0, 5000, 8000, 12000, 25000],

    include_lowest=True,

    labels=["LOW", "MEDIUM", "HIGH", "VERY HIGH"],

)

loan_df_clean["funded_amnt_inv_interval"].unique()
# Create Interval for funded_amnt - LOW,MEDIUM,HIGH,VERY HIGH



loan_df_clean["funded_amnt_interval"] = pd.cut(

    x=loan_df_clean["funded_amnt"],

    bins=[0, 5000, 8000, 12000, 25000],

    include_lowest=True,

    labels=["LOW", "MEDIUM", "HIGH", "VERY HIGH"],

)

loan_df_clean["funded_amnt_interval"].unique()
# Create Interval for annual_inc - LOW,MEDIUM,HIGH,VERY HIGH



loan_df_clean["annual_inc_interval"] = pd.cut(

    x=loan_df_clean["annual_inc"],

    bins=[0, 40000, 53000, 71000, 120000],

    include_lowest=True,

    labels=["LOW", "MEDIUM", "HIGH", "VERY HIGH"],

)

loan_df_clean["annual_inc_interval"].unique()
# Create Interval for installment - LOW, MEDIUM, HIGH, VERY HIGH



loan_df_clean["installment_interval"] = pd.cut(

    x=loan_df_clean["installment"],

    bins=[0, 156, 250, 368, 905],

    include_lowest=True,

    labels=["LOW", "MEDIUM", "HIGH", "VERY HIGH"],

)

loan_df_clean["installment_interval"].value_counts()
# Create Interval for installment - LOW, MEDIUM, HIGH, VERY HIGH



loan_df_clean["installment_interval"] = pd.cut(

    x=loan_df_clean["installment"],

    bins=[0, 156, 250, 368, 905],

    include_lowest=True,

    labels=["LOW", "MEDIUM", "HIGH", "VERY HIGH"],

)

loan_df_clean["installment_interval"].value_counts()
# Create Interval for dti - debt to income ratio



loan_df_clean["dti_interval"] = pd.cut(

    x=loan_df_clean["dti"],

    bins=[0, 9, 14, 19, 30],

    include_lowest=True,

    labels=["LOW", "MEDIUM", "HIGH", "VERY HIGH"],

)

loan_df_clean["dti_interval"].value_counts()
loan_df_clean["emp_length"].value_counts()
loan_df_clean.describe()
plt.figure(1, figsize=(18, 5))

plt.subplot(1, 3, 1)

sns.boxplot(y=loan_df_clean["loan_amnt"])

plt.subplot(1, 3, 2)

sns.boxplot(y=loan_df_clean["funded_amnt"])

plt.subplot(1, 3, 3)

sns.boxplot(y=loan_df_clean["annual_inc"])
def PercentageLossbyCutOffValue(dataframe, columnName, cutOff):

    print(

        len(dataframe[columnName].loc[loan_df_clean[columnName] > cutOff])

        / len(loan_df_clean)

        * 100

    )
def DropRowsByCutOffValue(dataframe, columnName, cutOffValue):

    return loan_df_clean.loc[loan_df_clean[columnName] < cutOffValue]
cutOffValue = 120000

column = "annual_inc"

PercentageLossbyCutOffValue(loan_df_clean, column, cutOffValue)
loan_df_clean = DropRowsByCutOffValue(loan_df_clean, column, cutOffValue)

# Plotting Frequency Plot

sns.distplot(loan_df_clean[column])
column = "loan_amnt"

cutOffValue = 25000

PercentageLossbyCutOffValue(loan_df_clean, column, cutOffValue)
loan_df_clean = DropRowsByCutOffValue(loan_df_clean, column, cutOffValue)

# Plotting Frequency Plot

sns.distplot(loan_df_clean[column])
loan_df_clean.describe()
len(loan_df_clean) / 36502 * 100
plt.figure(1, figsize=(17, 4))

plt.subplot(1, 2, 1)

sns.distplot(loan_df_clean["loan_amnt"], color="y")

plt.subplot(1, 2, 2)

sns.distplot(loan_df_clean["funded_amnt"], color="g")
loan_df_clean.head(4)
plt.figure(1, figsize=(15, 4))

plt.subplot(1, 3, 1)

loan_df_clean["term"].value_counts().plot("bar", title="Loan Duration", color="y")



plt.subplot(1, 3, 2)

loan_df_clean["emp_length"].value_counts().plot(

    "bar", title="employement Duration", color="olivedrab"

)



plt.subplot(1, 3, 3)

loan_df_clean["grade"].value_counts().plot("bar", title="Loan Grade")
plt.figure(1, figsize=(25, 5))

plt.subplot(1, 3, 1)

loan_df_clean["sub_grade"].value_counts().plot("bar", title="Loan Sub-Grade")



plt.subplot(1, 3, 2)

loan_df_clean["home_ownership"].value_counts().plot("pie", title="Home Ownership Staus")



plt.subplot(1, 3, 3)

loan_df_clean["verification_status"].value_counts().plot(

    "bar", title="verification status"

)
plt.figure(1, figsize=(14, 5))

plt.subplot(1, 2, 1)

loan_df_clean["loan_status"].value_counts().plot("bar", title="Loan Status")
plt.figure(figsize=(10, 4))

print(

    loan_df_clean.groupby(by=["grade"])["defaulter_rate"]

    .mean()

    .sort_values(ascending=False)

)

sns.barplot(x=loan_df_clean["grade"], y=loan_df_clean["defaulter_rate"])
plt.figure(figsize=(10, 4))

print(

    loan_df_clean.groupby(by=["sub_grade"])["defaulter_rate"]

    .mean()

    .sort_values(ascending=False)

)

sns.barplot(x=loan_df_clean["sub_grade"], y=loan_df_clean["defaulter_rate"])
plt.figure(figsize=(10, 4))

print(

    loan_df_clean.groupby(by=["loan_amnt_interval"])["defaulter_rate"]

    .mean()

    .sort_values(ascending=False)

)

sns.barplot(x=loan_df_clean["loan_amnt_interval"], y=loan_df_clean["defaulter_rate"])
plt.figure(figsize=(10, 5))

print(

    loan_df_clean.groupby(by=["funded_amnt_interval"])["defaulter_rate"]

    .mean()

    .sort_values(ascending=False)

)

sns.barplot(x=loan_df_clean["funded_amnt_interval"], y=loan_df_clean["defaulter_rate"])
plt.figure(figsize=(10, 5))

print(

    loan_df_clean.groupby(by=["int_rate_interval"])["defaulter_rate"]

    .mean()

    .sort_values(ascending=False)

)

sns.barplot(x=loan_df_clean["int_rate_interval"], y=loan_df_clean["defaulter_rate"])
plt.figure(figsize=(25, 5))

print(

    loan_df_clean.groupby(by=["purpose"])["defaulter_rate"]

    .mean()

    .sort_values(ascending=False)

)

sns.barplot(x=loan_df_clean["purpose"], y=loan_df_clean["defaulter_rate"])
plt.figure(figsize=(10, 5))

print(

    loan_df_clean.groupby(by=["emp_length"])["defaulter_rate"]

    .mean()

    .sort_values(ascending=False)

)

sns.barplot(x=loan_df_clean["emp_length"], y=loan_df_clean["defaulter_rate"])
plt.figure(figsize=(10, 5))

print(

    loan_df_clean.groupby(by=["annual_inc_interval"])["defaulter_rate"]

    .mean()

    .sort_values(ascending=False)

)

sns.barplot(x=loan_df_clean["annual_inc_interval"], y=loan_df_clean["defaulter_rate"])
plt.figure(figsize=(10, 5))

print(

    loan_df_clean.groupby(by=["home_ownership"])["defaulter_rate"]

    .mean()

    .sort_values(ascending=False)

)

sns.barplot(x=loan_df_clean["home_ownership"], y=loan_df_clean["defaulter_rate"])
plt.figure(figsize=(10, 5))

print(

    loan_df_clean.groupby(by=["verification_status"])["defaulter_rate"]

    .mean()

    .sort_values(ascending=False)

)

sns.barplot(x=loan_df_clean["verification_status"], y=loan_df_clean["defaulter_rate"])
plt.figure(figsize=(10, 5))

print(

    loan_df_clean.groupby(by=["pub_rec_bankruptcies"])["defaulter_rate"]

    .mean()

    .sort_values(ascending=False)

)

sns.barplot(x=loan_df_clean["pub_rec_bankruptcies"], y=loan_df_clean["defaulter_rate"])
plt.figure(1, figsize=(15, 5))

sns.heatmap(loan_df_clean.corr(), annot=True)



print("Correlation of Defaulter_rate with Other Variables :")

loan_df_clean.corr()["defaulter_rate"].sort_values(ascending=False)
plt.figure(1, figsize=(20, 5))

sns.countplot(loan_df_clean["purpose"])
# filterling loan-purpose

main_purpose = [

    "small_business",

    "educational",

    "debt_consolidation",

    "credit_card",

    "home_improvement",

]

loan_df_clean = loan_df_clean.loc[loan_df_clean["purpose"].isin(main_purpose)]
plt.figure(1, figsize=(15, 4))

sns.barplot(

    x=loan_df_clean["loan_amnt_interval"],

    y=loan_df_clean["defaulter_rate"],

    hue=loan_df_clean["purpose"],

)
plt.figure(1, figsize=(15, 4))

sns.barplot(

    x=loan_df_clean["int_rate_interval"],

    y=loan_df_clean["defaulter_rate"],

    hue=loan_df_clean["purpose"],

)
plt.figure(1, figsize=(15, 4))

sns.barplot(

    x=loan_df_clean["installment_interval"],

    y=loan_df_clean["defaulter_rate"],

    hue=loan_df_clean["purpose"],

)
plt.figure(1, figsize=(20, 8))

sns.barplot(

    x=loan_df_clean["emp_length"],

    y=loan_df_clean["defaulter_rate"],

    hue=loan_df_clean["purpose"],

)
plt.figure(1, figsize=(15, 4))

sns.barplot(

    x=loan_df_clean["term"],

    y=loan_df_clean["defaulter_rate"],

    hue=loan_df_clean["purpose"],

)
plt.figure(1, figsize=(15, 4))

sns.barplot(

    x=loan_df_clean["grade"],

    y=loan_df_clean["defaulter_rate"],

    hue=loan_df_clean["purpose"],

)
plt.figure(1, figsize=(15, 4))

sns.barplot(

    x=loan_df_clean["home_ownership"],

    y=loan_df_clean["defaulter_rate"],

    hue=loan_df_clean["purpose"],

)
plt.figure(1, figsize=(15, 4))

sns.barplot(

    x=loan_df_clean["pub_rec_bankruptcies"],

    y=loan_df_clean["defaulter_rate"],

    hue=loan_df_clean["purpose"],

)
def getdefaultRate(cat_col):

    default_df = (

        loan_df_clean.groupby(by=[cat_col])["defaulter_rate"]

        .mean()

        .sort_values(ascending=False)

    )

    diff = (default_df[0] - default_df[-1]) * 100

    return [cat_col, diff]
defaultrateList = []

cat_df = loan_df_clean.select_dtypes(["object", "category"])



for col in cat_df.columns.tolist():

    if col not in ["emp_title", "loan_status", "title"]:

        defaultrateList.append(getdefaultRate(col))



# Convert To Dataframe

default_rate_df = pd.DataFrame(defaultrateList, columns=["variable", "default_rate"])



default_rate_df.sort_values(by=["default_rate"], ascending=False)