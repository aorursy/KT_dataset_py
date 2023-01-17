%matplotlib inline



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



sal = pd.read_csv("../input/Salaries.csv")



sal.head()
sal.dtypes
sal['Year'].value_counts()
sal['Notes'].value_counts()
sal['Agency'].value_counts()
sal['Status'].value_counts()
sal = sal.drop([148646, 148650 , 148651 , 148652])  ## All rows which contain 'NOT Provided' will be droped 
sal = sal.drop(columns = ['Notes'])
sal['Benefits'] = sal['Benefits'].fillna(0)  
## JTTO = Job Title Transit Operator



JTTO = sal.loc[sal['JobTitle'] == 'Transit Operator']



## Total no. FT status for transit operator



Status_FT = JTTO.loc[JTTO['Status'] == 'FT']



## Mean value of Base Pay associated with FT 

Mean_FT= Status_FT['BasePay'].astype('Float64').mean()

print("Mean_of_FT=", Mean_FT)



## Total no. NaN status for transit operator

Status_NaN = JTTO[JTTO['Status'].isna()]



## Variable a contains basepay associated with NaN Status 

a = Status_NaN['BasePay'].astype('Float64')



## Variable b contain NaN Status

b = Status_NaN['Status']

for i in range(0, len(a.index)):

    

    if (a.iloc[i] > Mean_FT):

        b.iloc[i] = 'FT'

    else:

        b.iloc[i] = 'PT'
New_df= pd.DataFrame(Status_NaN['Status'])



JTTO.update(New_df)
JTTO['Status'].value_counts()
JTTO.head()
data_FT = JTTO[JTTO['Status'] == 'FT']

data_PT = JTTO[JTTO['Status'] == 'PT']



fig = plt.figure(figsize=(8, 8))

ax = sns.kdeplot(data_PT['TotalPayBenefits'], color = 'Orange', label='Part Time Employees', shade=True)

ax = sns.kdeplot(data_FT['TotalPayBenefits'], color = 'Green', label='Full Time Employees', shade=True)

plt.yticks([])



plt.title('Part Time Employees vs. Full Time Employees')

plt.ylabel('Density of Employees')

plt.xlabel('Total Pay + Benefits ($)')

plt.xlim(0, 200000)

plt.show()
ax = plt.scatter(JTTO['BasePay'], JTTO['Benefits'])

plt.ylabel('Benefits')

plt.xlabel('BasePay')
def fill_status(X):

    Status_FT = sal.loc[sal['Status'] == 'FT']

    Mean_FT= Status_FT['BasePay'].astype('Float64').mean()              ## Mean value of Base Pay associated with FT

    print("Mean_of_FT=", Mean_FT)

    Status_NaN = sal[sal['Status'].isna()]                                ## NaN status

    a = Status_NaN['BasePay'].astype('Float64')                         ## Variable a contains basepay associated with NaN Status 

    b = Status_NaN['Status']                                            ## Variable b contain NaN Status

    for i in range(0, len(a.index)):

        if (a.iloc[i] > Mean_FT):

            b.iloc[i] = 'FT'

        else:

            b.iloc[i] = 'PT'
JobTitle = sal.groupby('JobTitle')       ## Group by job title 
TotalPayMean = JobTitle.mean()['TotalPay']  ## take a mean of grouped job titles for TotalPay columns

TotalPayMean.head()
NaN_Status = sal[sal['Status'].isna()]      ## Get a NaN status by isna() method

NaN_Status['Status']
def fill_status(X):

    job_title = X[2]                             ## X[2] is index column 2 which is job title

    totle_Pay = X[7]                             ## X[7] is index column 7 which is Total Pay

    mean = TotalPayMean[job_title]               ## mean of perticular job title (mean will change as job title changes)

    

    if (totle_Pay > mean):                       ## Value in total pay is being comparing with mean value of perticular job title

        return "FT"                              ## if greater it will retuns FT in Status column

    else:

        return "PT"

    

NaN_Status.iloc[:110531,-1] = NaN_Status.iloc[:110531,].apply(fill_status, axis = 1)     ## 110531 is the lenght, -1 is last column

                                                                                         ## here we pass a function fill_status
NaN_Status['Status']
new_df = pd.DataFrame(NaN_Status['Status'])
sal.update(new_df )

sal['Status'].value_counts()
sal
data_FT = sal[sal['Status'] == 'FT']

data_PT = sal[sal['Status'] == 'PT']



fig = plt.figure(figsize=(8, 8))

ax = sns.kdeplot(data_PT['TotalPayBenefits'], color = 'Orange', label='Part Time Employees', shade=True)

ax = sns.kdeplot(data_FT['TotalPayBenefits'], color = 'Green', label='Full Time Employees', shade=True)

plt.yticks([])



plt.title('Part Time Employees vs. Full Time Employees')

plt.ylabel('Density of Employees')

plt.xlabel('Total Pay + Benefits ($)')

plt.xlim(0, 600000)

plt.show()
sal.corr()
ax = plt.scatter(sal['TotalPay'], sal['TotalPayBenefits'])

plt.ylabel('TotalPayBenefits')

plt.xlabel('TotalPay')

plt.show()