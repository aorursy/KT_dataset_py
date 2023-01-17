import pandas as pd

import numpy as np

import seaborn as sns

from collections import Counter

import matplotlib.pyplot as plt

import os

import math

import sys
!pip install uszipcode

import uszipcode

from uszipcode import SearchEngine

search = SearchEngine()
#Read csv file

credit = pd.read_csv('../input/should-this-loan-be-approved-or-denied/SBAnational.csv', header ='infer')
#Quick overview of our Dataset and variables

credit.head()
#Deleting duplicates if any

credit = credit.drop_duplicates(keep = 'first')

#Shape of the data: 899164 rows and 27 columns

credit.shape
#List of columns:

print(credit.dtypes.index)
#Summary of data frame (count, mean, standart deviation, min, quartiles, max)

credit.describe()
#NA observation. A lot of NAs :(

credit.isnull().sum()
#Our target is to predict "MIS_Value". According to the document, "MIS_Status" has 2 variables: 

#Loan status charged off = CHGOFF, Paid in full = PIF

Counter(credit.MIS_Status).keys() # We have: "PIF", CHGOFF, nan in this column
Counter(credit.MIS_Status).values() # "PIF": 739609, CHGOFF: 157558, nan:1997

#Quite imbalanced. We will try to not lose our rows with CHGOFF values while we are cleaning the dataset.
                                                    #####FIRST Dealing with NAs.##### 

                                                          #Name, City, and State

# 4 columns belong to Borrower information: Name, City, State, and Zip.

#Name is unique and useless and we can drop an entire column.

credit = credit.drop(axis =1, columns = ['Name'])

#Next are City and State. As you can see, there are no NAs in Zip, so we can easily impute City and State uzing Zip values

#Creating conditions for a loop

cond = (credit.City.isnull()|credit.State.isnull())

missing_rows = credit[cond].index

#I will go through loop and impute City and State using zearch.by_zipcode function.

for i in missing_rows:

    zipcode = search.by_zipcode(credit.iloc[i,3]) # 3 corresponds to Zip code

    credit.iloc[i,1] = zipcode.major_city # 1 -> City

    credit.iloc[i,2] = zipcode.state # 2 -> State



#Check how NAs were imputed. We still have 4 NAs. I have looked through them. One zipcode = 0 

#and the other 3 are not in the list of search.by_zipcode function. I think we can delete these rows.

credit.isnull().sum()
#Next features with a lot of NAs are Bank and BankState. 

#I would like to check if State of borrower = BankState. 

#If so we can just delete one of them.

equal = 0

non_equal = 0

for i in credit.index:

    if credit.State[i] == credit.BankState[i]:

        equal = equal + 1

    else:

        non_equal = non_equal +1

print(equal, non_equal)

# Sad. The assumption is incorrect, 473949 cases coincide, while 425215 cases do not.
#Since our goal is to define a default case, I will check if there are default cases in the rows where the 

#Bank Name and Bank State are unknown. 

cond_1 = credit.Bank.isnull()|credit.BankState.isnull()

missing_rows_1 = credit[cond_1].index

yes = 0

no = 0

for i in missing_rows_1:

    if credit.MIS_Status[i] == 'CHGOFF':

        yes = yes+1

    else:

        no = no +1

print(yes,no)

#Not bad, 72 cases against 1494. I'm still not sure if I need the columns with the State of the Bank and Bank name.

#We can delete 1494 rows with not default cases, since the data is imbalanced. We'll see.
#Feature - 'ChgOffDate' corresponds to the date when a loan was declared to be in default. 

#I made little investigation about ChgOffDate feature: we have 739609 paid cases and 736465 NAs 

#seems like if credit is paid then there is an NA in this featue. default date = default case.

#So, we will just drop this feature.

credit = credit.drop(axis =1, columns = ['ChgOffDate'])
#Next, I want to clean my dependent variable - MIS_Status drop NAs and change dtype from object to integer.

#1 - grant a loan (low risk of default), 0 - do not grant a loan (high risk of default)

#And, of course, drop 4 rows with NAs in City and State columns 

credit = credit.dropna(axis =0, subset=['City','State','MIS_Status'])



loan_status = {'P I F': 1,'CHGOFF': 0} 

credit.MIS_Status = [loan_status[item] for item in credit.MIS_Status] 



Counter(credit.MIS_Status).keys() 
Counter(credit.MIS_Status).values() # count values 739607 = "1" against 157556 = "0"
#According to the document we have, 5 columns are with currency values. I want to change all of them to the float.

#This will help us see the corealation of currency to our target value.

currency = [19,20,22,23,24] #To convert to float.

for i in currency:

    credit[credit.columns[i]] = credit[credit.columns[i]].replace('[\$,]', '', regex=True).astype(float) 
#Looks good.

credit.isnull().sum()
#Next feature is New Exist. According to the document

#1 = Existing business, 2 = New business 

#And at this moment we have 134 NAs

Counter(credit.NewExist).keys() # unique values
Counter(credit.NewExist).values() # count values 

#In fact, we have more than 136 NAs. 

#252559 = New business

#643443 = Existing business

#1027 = "0" whatever it means

#134 = nan as we can see from previous output
#We have a feature 'RetainedJob' - From the document, it shows number of jobs retained. 

#I can assume that if loan retains some jobs it is an existing business.

# I will create a condition and iterate through loop to 

#assign new value "1" which is Existing business to those rows where Retained Job is >= 1

cond_2 = credit[(credit['NewExist'] == 0) & (credit['RetainedJob'] >=1)].index

for i in cond_2:

    credit.loc[i,['NewExist']] = 1 
#Next we will do the same thing with another condition: isnull and Retained Job >=1

cond_3 = credit[(credit.NewExist.isnull()) & (credit['RetainedJob'] >=1)].index

for i in cond_3:

    credit.loc[i,['NewExist']] = 1    
#I will check if there are default cases in these rows.

credit[(credit['NewExist'] == 0) & (credit['MIS_Status'] == 0)] #60 rows

credit[(credit.NewExist.isnull()) & (credit['MIS_Status'] == 0)] #1 row

#Our goal is to impute 61 rows in NewExist and we can delete others.
#Ok, we continue our investigation.

#Interesting column - "FranchiseCode":

#Nofranchise = 0 or 1

#Franchise code = other numbers

#My assumption is if Franchise code != 0 and != 1, maybe it is a New Business, not the existing one.

#Someone gets a franchise and opens a Starbucks in the city.

Counter(credit.FranchiseCode).keys() #51732 Franchise Loans
Counter(credit.FranchiseCode).values() # 845431 non Franchise Loans 
#Let's check our assumption, if Franchise Code is with digits = New business

cond_4 = credit[(credit['FranchiseCode'] != 0) & (credit['FranchiseCode'] != 1)] #Lets store our Franchise cases

Counter(cond_4.NewExist).values()

# We have:

# 27940 rows - New Business

# 23725 rows - Exisitng Business

# 67 rows - NA in Existing Business

# Our assumption is not correct!
#Counter(credit.NewExist).keys() # unique values

#Counter(credit.NewExist).values() # count values 





# I think I will stop my investigation on this stage and will drop NAs in NewExist feature.

#Before it was 134 NAs + 1027 with zero value, after imputation we have 19 rows = NA and 874 = "0". In total, 893.

#To check this you can run code above this cell



#First, I will assign NA to 0 values and drop all NAs in this feature.

cond_5 = credit[(credit['NewExist'] == 0)].index

for i in cond_5:

    credit.loc[i,['NewExist']] = np.nan #11 corresponds to NewExist column

    

credit = credit.dropna(axis =0, subset=['NewExist'])
#Much better.

credit.isnull().sum()
#I decided to drop Bank Name and Bank State on this stage. My idea was to use state as a predictor, because 

#different states have different economic environments.

#According to documentation, State of Borrower is a right Feature to use for this goal.

#Also, Bank Name (> 5000 names) as a Borrowers Name is a unique value, so we can delete it, too.

#Columns "Disbursment Date", "DisbursementGross", "BalanceGross" and "ChgOffPrinGr" 

#contain information that is important after default is declared, so we can't use these columns for predicting

#default risks. I delete them.

credit = credit.drop(axis =1, columns = ['Bank', 'BankState', 'DisbursementDate', 'DisbursementGross', 

                                         'BalanceGross','ChgOffPrinGr'])
#Now we have only 2 features left: LowDoc and RevLineCr,

#We will start from - LowDoc. Loan Program: Y = Yes, N = No

Counter(credit.LowDoc).keys()
Counter(credit.LowDoc).values()
#I made some research about "LowDoc program". So, these are less than 150 000$ short-term loans. 

#To get this loan you need less documents

#And it was a very popular loan program in the USA in 2000-2007. And seems like it's a strong predictor

#I will select rows with this condition and check values in LowDoc feauture

#But first I should convert ApprovalDate from object to DateTime format

from datetime import date

credit['ApprovalDate'] = credit['ApprovalDate'].astype(str)

credit['ApprovalDate'] = pd.to_datetime(credit['ApprovalDate'])
cond_6 = credit[(credit['LowDoc'] != "Y") & (credit['LowDoc'] != "N")]

cond_6

#5997 rows with NAs and other different values. (1404)

#Seems like it's a very important value, we will not drop rows with NAs. 

#We will try to impute LowDoc value using other features
Counter(cond_6.LowDoc).values() #dict_values([757, 1, 2578, 603, 74, 494, 1490])

Counter(cond_6.LowDoc).keys() #dict_keys(['C', '1', nan, 'S', 'R', 'A', '0'])
#Let's see how many rows in cond_6 are default cases

Counter(cond_6.MIS_Status).values()

#4420 rows - not default

#1577 - default cases. 

#Would be nice to impute these rows with values.
#Let's examine LowDoc loans and Not LowDoc loans and try to find any patterns

low_doc = credit[credit['LowDoc'] == "Y"]
low_doc['GrAppv'].describe() # 75% of loans are =< $100 000
low_doc['Term'].describe() #75% or loans <= 93 months
Counter(low_doc.MIS_Status).keys()

Counter(low_doc.MIS_Status).values() #MIS_Status 1 = 100153, 0 = 9893 (1 is 10 times more than 0)
#Let's check if we have some rows in cond_6 with these conditions

cond_7 = cond_6[(cond_6['GrAppv'] <= 100000) & (cond_6['Term'] <= 93) & (cond_6['MIS_Status'] == 1)]

cond_7

#Ok, we can assign 1 to 1565 rows in LowDoc program. It's better to then just delete the rows.
#I will iterate a loop to assign value 'Y' to these rows in column 15 (LowDoc)

for i in cond_7.index:

    credit.loc[i,['LowDoc']] = 'Y'
#Examine results 4432 rows. I will drop other values

cond_8 = credit[(credit['LowDoc'] != "Y") & (credit['LowDoc'] != "N")]

cond_8
#I will assign nan to all values not equal to "Y" and "N" and than drop NAs

for i in cond_8.index:

    credit.loc[i,'LowDoc'] = np.nan



credit = credit.dropna(axis =0, subset=['LowDoc'])
#Next is RevLineCr. According to the documentation, revolving line of credit: Y = Yes, N = No

Counter(credit.RevLineCr).keys()

Counter(credit.RevLineCr).values()
#Again we will select rows not equal to Y and N

cond_9 = credit[(credit['RevLineCr'] != "Y") & (credit['RevLineCr'] != "N")] #277188 rows
Counter(cond_9.MIS_Status).values() # 1:231537 0:44978 rows.
#We will select all rows with RevLine Yes and RevLine No and try to find any patterns to impute NAs

RevLine_yes = credit[credit['RevLineCr'] == "Y"]

RevLine_no = credit[credit['RevLineCr'] == "N"]
RevLine_yes.Term.describe() #highest term 312
RevLine_no.Term.describe() #highest term 527
#number of emp

RevLine_no.NoEmp.describe()
RevLine_yes.NoEmp.describe()
#I can't find any clear pattern and there are a lot of NAs in this row, so I decided to delete this column on this stage.

#Also, I found out that column #0 is unique, too. It's an ID number of Identifier – Primary key. We will drop it, too.

#We don't need columns with dates anymore: ApprovalDate, ApprovalFY

#SBA_Appv - a guaranteed ammount from the US government. Useless, too. 
credit = credit.drop(axis =1, columns = ['Zip','LoanNr_ChkDgt','ApprovalDate', 'ApprovalFY','RevLineCr','SBA_Appv'])
#Counter(credit.City).keys() # 2 = 125348 1 = 292878 

#We can try to create a feature according to cities (Small, Big, Medium), but maybe later for improving the model. 

#On this stage I think we can drop this column, too many unique values.
Counter(credit.State).keys() 

#My idea is to create new features using States and NAICS. For each State and for each Sector of business I will give points.

#Lower default rate - higher points.
#Function extracts first 2 digits from variable, if variables = 0, returns 0

def first_two(d):

    if d <= 0:

        return 0

    return (d // 10 ** (int(math.log(d, 10)) - 1))

#Function returns points according to the given rate. Lower rate, higher points.

def point_def(rate):

    if rate <= 12:

        return 5

    elif  12 < rate <= 17: 

        return 4

    elif 17 < rate <= 21:

        return 3

    elif 21 < rate <= 25:

        return 2

    elif rate > 25:

        return 1
#Function returns points according to the default rate of each sector.

def apply_score(i):

    sector_default = {21 : 8, 11 : 9, 55 : 10, 62:10, 22:14, 92:15,54:19, 42:19,31:19,32:16,33:14,81:20,71:21,72:22,44:22,45:23,23:23,56:24,61:24,51:25,48:27,49:23,52:28,53:29}

    if i > 0:

        defrate = None

        if i in sector_default:

            defrate = sector_default[i]

            return point_def(defrate)

    return 0
#Once the functions are ready I will create a new column - "Sector_Points" and apply functions to get points

credit['Sector_Points'] = credit.NAICS.apply(first_two).apply(apply_score)
#Next, we will give points to each State.

#I indicated only the highest and the lowest default rates in the library and assigned def rate = 18 (mean) to the states left

#Function for the state scores 

def apply_score_state(i):

    state_default = {'MT':8, 'ND': 8, 'WY':8, 'SD':8, 'VT':8, 'ME':10,'NH':10, 'NM':10, 'AK':10, 'WA':13,'AD':13, 'MN':13, 

                     'WI':13, 'IA':13,'NE':13, 'KS':13, 'MA':13,'CT':13,'RI':13,'PA':13, 'NV':23, 'IL':23, 'MI':23, 'KY':23,

                     'GA':23, 'FL':28}

    temp_defrate = None

    average_def_rate = 18

    if i in state_default:

        temp_defrate = state_default[i]

        return point_def(temp_defrate)

    return point_def(average_def_rate)

#Apply function and create new feature

credit['State_Points'] = credit.State.apply(apply_score_state)
#I think on this stage we can drop columns:"City","State" and "NAICS"

#Because when I dummify these columns, especially State, 50 more features will be created.

credit = credit.drop(axis =1, columns = ['City', 'State', 'NAICS'])
#Next, I want to change some Columns this way:

#Term to Years 12 = 1 etc

#CreateJob to IscreateJob (1,0)

#Retained Job to IsRetainedJob (1,0)

#FranchiseCode to IsFranchise (1,0)



#For this I will create a simple function, which I can apply to several columns and create new features.

def yes_no(i):

    if i > 0:

        return 1

    return 0



credit.Term = credit.Term//12

credit['IscreateJob'] = credit.CreateJob.apply(yes_no)

credit['IsRetained'] = credit.RetainedJob.apply(yes_no)

credit['IsFranchise'] = credit.FranchiseCode.apply(yes_no)
#I will recode LowDoc Yes and No to 1 and 0.

def lowdoc(i):

    if i == "Y":

        return 1

    return 0

credit.LowDoc = credit.LowDoc.apply(lowdoc)
#This column is totally OK :)

Counter(credit.UrbanRural).keys() #1 = Urban, 2 =Rural, 0 = Undefined

Counter(credit.UrbanRural).values() #1 = Urban, 2 =Rural, 0 = Undefined
#Since we don't need FrancshiseCode column I will drop it

credit = credit.drop(axis =1, columns = ['FranchiseCode'])
#Let's check our dataframe

credit
#As we can see, there is skewness.

plt.figure(figsize=(15, 8))

sns.distplot(credit.GrAppv, color="g", kde=False)

plt.ylabel('Density')

plt.title('Distribution of Approved ammount')

plt.show()
#Fix skewness of GrAppv using log

credit['GrAppv'] = np.log(credit['GrAppv']) 
                                                    #### SIMPLE MODEL ####

#Split data into train and test sets + label target value

from sklearn.model_selection import train_test_split

y = credit.MIS_Status

X = credit.drop(['MIS_Status'], axis=1)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0)
#We will apply Simple Imputer and Standart Scaler from sklearn package

from sklearn.impute import SimpleImputer 

my_imputer = SimpleImputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)



#Scaling features with Standart Scaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_sc =scaler.fit_transform(train_X)

test_sc = scaler.transform(test_X)
#We will train xgboost without any tunning and check results.

import xgboost as xgb

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Train the XGboost Model for Classification

#Model with default parameters

model = xgb.XGBClassifier()

train_model = model.fit(train_sc, train_y)
#Prediction

from sklearn.metrics import classification_report

pred = train_model.predict(test_X)

print('Model XGboost Report %r' % (classification_report(test_y, pred)))
#Let's use accuracy score

from sklearn.metrics import accuracy_score

print("Accuracy for model: %.2f" % (accuracy_score(test_y, pred) * 100))