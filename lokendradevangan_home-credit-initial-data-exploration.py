# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
##Initial data understanding 
dataset=pd.read_csv("../input/application_train.csv")
test=pd.read_csv("../input/application_test.csv")
dataset.head()
##summary statistics
dataset.describe()
# Number of each type of column
dataset.dtypes.value_counts()
#Function
#Getting the percentage of missing values by column in the dataframe dataset 

def export(Path,Object1,Object2,Sheetname1,Sheetname2):
    
    # Create a Pandas Excel writer using XlsxWriter
    writer = pd.ExcelWriter(Path)
    # Convert the dataframe to an XlsxWriter Excel object
    Object1.to_excel(writer,Sheetname1)                        #DataFrame.to_excel(excel_writer, sheet_name='Sheet1') : Write DataFrame to an excel sheet
    Object2.to_excel(writer,Sheetname2)
    # Close the Pandas Excel writer and output the Excel file
    writer.save()
data_f=dataset.select_dtypes(include=['float64'])              #Dataframe containing only float variables
data_o=dataset.select_dtypes(include=['object','category'])    #Dataframe containing object and categorical variables
summary_f=data_f.describe() 
summary_o=data_o.describe() 
summary_f=summary_f.transpose()
summary_o=summary_o.transpose()
# of records
total_rows = dataset["SK_ID_CURR"].count()
#Getting the missing percentage 
summary_o["MissingCount"] = total_rows-summary_o["count"] 
summary_o["MissingPerc"] = summary_o["MissingCount"]/total_rows
summary_f["MissingCount"] = total_rows-summary_f["count"] 
summary_f["MissingPerc"] = summary_f["MissingCount"]/total_rows
#Missig in some field may be actually zero. That is a judgement we have to make
summary_f
#Missing percentage in categrical field
summary_o
#listing the count of values in each category
dataset.select_dtypes(include=['object']).apply(pd.Series.nunique, axis = 0)
# can we combine in less number of groups
dataset.groupby(['NAME_TYPE_SUITE']).SK_ID_CURR.count()
dataset.groupby(['NAME_TYPE_SUITE']).TARGET.mean() 
#reducing the unique values in occupation by grouping by skill level. This grouping can differ based on more information about each occupation
dataset['NAME_TYPE_SUITE'].replace({'Children':'Family',
                                    'Group of people':'Other',
                                    'Other_A':'Other',
                                    'Other_B':'Other',
                                    'Spouse, partner':'Family'},inplace=True)

test['NAME_TYPE_SUITE'].replace({'Children':'Family',
                                    'Group of people':'Other',
                                    'Other_A':'Other',
                                    'Other_B':'Other',
                                    'Spouse, partner':'Family'},inplace=True)
dataset.groupby(['OCCUPATION_TYPE']).SK_ID_CURR.count()

dataset.groupby(['NAME_EDUCATION_TYPE']).SK_ID_CURR.count()
dataset.groupby(['NAME_EDUCATION_TYPE']).TARGET.mean() 
dataset['NAME_EDUCATION_TYPE'].replace({'Academic degree':'Higher education '},inplace=True)
test['NAME_EDUCATION_TYPE'].replace({'Academic degree':'Higher education '},inplace=True)
dataset.groupby(['WEEKDAY_APPR_PROCESS_START']).SK_ID_CURR.count() 
#does not have any interesting information. Lets check mean of target value
dataset.groupby(['WEEKDAY_APPR_PROCESS_START']).TARGET.mean() 
#even target does not show any pattern
dataset.groupby(['OCCUPATION_TYPE']).SK_ID_CURR.count()
#group low skilled, medium skilled and high skillled profession to reduce the unique values
#reducing the unique values in occupation by grouping by skill level. This grouping can differ based on more information about each occupation
dataset['OCCUPATION_TYPE'].replace({'High skill tech staff':'High_Skill',
                                    'Managers':'High_Skill',
                                    'Accountants':'High_Med_Skill',
                                    'HR staff':'High_Med_Skill',
                                    'Core staff':'Med_Skill',
                                   'Cooking staff':'Med_Skill',
                                    'Realty agents':'Med_Skill',
                                    'Sales staff':'Med_Skill',
                                    'IT staff':'High_Med_Skill',
                                    'Medicine staff':'High_Med_Skill',
                                    'Secretaries':'Med_Skill',
                                    'Security staff':'Med_Skill',
                                    'Cleaning staff':'Low_Skill',
                                      'Laborers':'Low_Skill',
                                      'Low-skill Laborers':'Low_Skill',
                                      'Cleaning staff':'Low_Skill',
                                    'Waiters/barmen staff':'Low_Skill',
                                    'Private service staff':'Low_Skill',
                                    'Drivers':'Med_Skill'
                                   },inplace=True)
test['OCCUPATION_TYPE'].replace({'High skill tech staff':'High_Skill',
                                    'Managers':'High_Skill',
                                    'Accountants':'High_Med_Skill',
                                    'HR staff':'High_Med_Skill',
                                    'Core staff':'Med_Skill',
                                   'Cooking staff':'Med_Skill',
                                    'Realty agents':'Med_Skill',
                                    'Sales staff':'Med_Skill',
                                    'IT staff':'High_Med_Skill',
                                    'Medicine staff':'High_Med_Skill',
                                    'Secretaries':'Med_Skill',
                                    'Security staff':'Med_Skill',
                                    'Cleaning staff':'Low_Skill',
                                      'Laborers':'Low_Skill',
                                      'Low-skill Laborers':'Low_Skill',
                                      'Cleaning staff':'Low_Skill',
                                    'Waiters/barmen staff':'Low_Skill',
                                    'Private service staff':'Low_Skill',
                                    'Drivers':'Med_Skill'
                                   },inplace=True)
dataset.groupby(['OCCUPATION_TYPE']).SK_ID_CURR.count()
import matplotlib.pyplot as plt
import seaborn as sns
#fig, ax = plt.subplots(3, 1, figsize = (15, 10))
sns.countplot(dataset.CODE_GENDER)
plt.show()
sns.countplot(dataset.NAME_INCOME_TYPE)
plt.show()
sns.countplot(dataset.NAME_FAMILY_STATUS)
plt.show()
sns.countplot(dataset.NAME_HOUSING_TYPE)
plt.show()

dataset.groupby(['NAME_INCOME_TYPE']).TARGET.mean()
dataset.groupby(['NAME_INCOME_TYPE']).SK_ID_CURR.count() 

# Group the data frame by NAME_INCOME_TYPE  extract a number of stats from each group
dataset.groupby(['NAME_INCOME_TYPE']).agg({'TARGET': ["mean", "count"], 'AMT_GOODS_PRICE': ["mean", "median"]   
                                    })  
#grouping
dataset['NAME_INCOME_TYPE'].replace({'Businessman':'Other','Student':'Other','Maternity leave':'Other'},inplace=True)
test['NAME_INCOME_TYPE'].replace({'Businessman':'Other','Student':'Other','Maternity leave':'Other'},inplace=True)
#exploring Housing type
dataset.groupby(['NAME_HOUSING_TYPE']).agg({'TARGET': ["mean", "count"], 'AMT_GOODS_PRICE': ["mean", "median"]   
                                    }) 

dataset.groupby(['NAME_FAMILY_STATUS']).agg({'TARGET': ["mean", "count"], 'AMT_CREDIT': ["mean", "median"] ,'AMT_INCOME_TOTAL': ["mean", "median"]   
                                    }) 
dataset.groupby(['ORGANIZATION_TYPE']).agg({'TARGET': ["mean", "count"], 'AMT_CREDIT': ["mean", "median"] ,'AMT_INCOME_TOTAL': ["mean", "median"]   
                                    }) 
#lets not do anything for now. Later, we will be encoding all the categorical variables
# Create a label encoder object
app_train=dataset
app_test=test
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            #app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Training Features shape: ', app_test.shape)
#print('Testing Features shape: ', app_test.shape)
#distribution of the AMT_GOODS_PRICE
#histogram of price by condition and brand
# Histogram
# bins = number of bar in figure
dataset.AMT_GOODS_PRICE.plot(kind = 'hist',bins = 25,figsize = (15,15))
#distribution of other variables 
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(1, 2, figsize = (15, 10))
sns.boxplot(dataset.AMT_INCOME_TOTAL, showfliers = False, ax = ax[0])
dataset.AMT_CREDIT.plot(kind = 'hist',bins = 25,figsize = (15,10))
plt.show()
#distribution of other variables 
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(1, 2, figsize = (15, 10))
sns.boxplot(dataset.AMT_ANNUITY, showfliers = False, ax = ax[0])
dataset.DAYS_EMPLOYED.plot(kind = 'hist',bins = 15,figsize = (15,10))
plt.show()
#distribution of the DAYS_EMPLOYED has extreme values. Does noot look correct
app_train['DAYS_EMPLOYED'].describe()
# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_train.fillna(dataset.median(),inplace = True)

app_test['DAYS_EMPLOYED_ANOM'] = test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
app_test.fillna(dataset.median(),inplace = True)
app_train.DAYS_EMPLOYED.plot(kind = 'hist',bins = 25,figsize = (15,15))
#how is age variable
(dataset['DAYS_BIRTH'] / -365).describe()
#replace all  NaN in the var_list with zero
Var_List=('OBS_30_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
        'DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK',
         'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR')
def missing_val_replace(data,Var_List):
    for col in data:
        for i in Var_List:
            if col==i:
                data[col].fillna(0)
                print (col)
    return data
dataset=missing_val_replace(dataset,Var_List) 
#replace all other NaN with median values
dataset=dataset.fillna(dataset.median)


#check if missing values got replaced 
#describe 
dataset.describe()