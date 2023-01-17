import pandas as pd 



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df2018 = pd.read_excel('../input/H-1B_Disclosure_Data_FY2018_EOY.xlsx')

df2017 = pd.read_excel('../input/H-1B_Disclosure_Data_FY17.xlsx')
df2018.columns
df2018.CASE_STATUS.value_counts() #exclude certified-withdrawn and withdrawn
df2018.VISA_CLASS.value_counts() #exclude E-3, H-1B1s
df2018.CHANGE_EMPLOYER.value_counts() #exclude when change_employer is 0, so that we only analyze H1b transfers
df2018.EMPLOYER_COUNTRY.value_counts() #exclude petitions from Non-US companies
df2018.FULL_TIME_POSITION.value_counts() #exclude part-time jobs
#Cleaning Google Job Titles

df2018.JOB_TITLE[df2018.EMPLOYER_NAME.str.contains("google",case=False)==True] = df2018.JOB_TITLE[df2018.EMPLOYER_NAME

                                                                                                  .str.contains("google",case=False)==True].str.split("1615.",n=1).str[0].str.replace(" \(","").str.strip()



df2017.JOB_TITLE[df2017.EMPLOYER_NAME.str.contains("google",case=False)==True] = df2017.JOB_TITLE[df2017.EMPLOYER_NAME

                                                                                                  .str.contains("google",case=False)==True].str.split("1615.",n=1).str[0].str.replace(" \(","").str.strip()



#Creating a new dataframe for h1b transfer cases



use_cols= ['CASE_NUMBER', 'CASE_STATUS', 'CASE_SUBMITTED', 'DECISION_DATE',

       'EMPLOYMENT_START_DATE', 'EMPLOYER_NAME',  'EMPLOYER_CITY', 'EMPLOYER_STATE',

       'JOB_TITLE', 'SOC_CODE', 'SOC_NAME',  'PREVAILING_WAGE', 'PW_UNIT_OF_PAY',

       'PW_WAGE_LEVEL', 'WAGE_RATE_OF_PAY_FROM', 'WAGE_RATE_OF_PAY_TO', 'WAGE_UNIT_OF_PAY',

       'WORKSITE_CITY', 'WORKSITE_STATE']





df2018_light = df2018[use_cols][(((((df2018.CASE_STATUS =='CERTIFIED') |(df2018.CASE_STATUS =='DENIED')) 

                             & (df2018.VISA_CLASS =='H-1B'))

                            &(df2018.CHANGE_EMPLOYER !=0))

                           &(df2018.EMPLOYER_COUNTRY =='UNITED STATES OF AMERICA'))

                                &(df2018.FULL_TIME_POSITION =='Y')

                               ]

df2017_light = df2017[use_cols][(((((df2017.CASE_STATUS =='CERTIFIED') |(df2017.CASE_STATUS =='DENIED')) 

                             & (df2017.VISA_CLASS =='H-1B'))

                            &(df2017.CHANGE_EMPLOYER !=0))

                           &(df2017.EMPLOYER_COUNTRY =='UNITED STATES OF AMERICA'))

                                &(df2017.FULL_TIME_POSITION =='Y')

                               ]
df2018_light.head(3)
print("Denial Rates and Sample Size:")

print("2018, Total: {:2.2%} - {:,}".format(1-df2018_light['CASE_STATUS'].value_counts(normalize=True)[0],df2018_light.CASE_NUMBER.count()))

print("2017, Total: {:2.2%} - {:,}".format(1-df2017_light['CASE_STATUS'].value_counts(normalize=True)[0],df2017_light.CASE_NUMBER.count()))
df2018_light2 = df2018[use_cols][(((df2018.CASE_STATUS =='CERTIFIED') |(df2018.CASE_STATUS =='DENIED')) 

                             & (df2018.VISA_CLASS =='H-1B'))

                                &(df2018.FULL_TIME_POSITION =='Y')

                               ]

df2017_light2 = df2017[use_cols][(((df2017.CASE_STATUS =='CERTIFIED') |(df2017.CASE_STATUS =='DENIED')) 

                             & (df2017.VISA_CLASS =='H-1B'))

                                &(df2017.FULL_TIME_POSITION =='Y')

                                ]
emp_list2=[

'COGNIZANT TECHNOLOGY SOLUTIONS US CORP',

'TATA CONSULTANCY SERVICES LIMITED',

'INFOSYS LIMITED',

'DELOITTE CONSULTING LLP',

'CAPGEMINI AMERICA INC',

'MICROSOFT CORPORATION',

'AMAZON.COM SERVICES, INC.',

'WIPRO LIMITED',

'ACCENTURE LLP',

'APPLE INC',

'HCL AMERICA',

'TECH MAHINDRA',

'ERNST & YOUNG U.S. LLP',

'GOOGLE',

'JPMORGAN CHASE & CO.',

'INTEL CORPORATION',

'FACEBOOK, INC.',

'IBM INDIA PRIVATE LIMITED',

'CISCO SYSTEMS, INC',

'LARSEN & TOUBRO INFOTECH LIMITED',

'L&T TECHNOLOGY SERVICES',

'MPHASIS CORPORATION',

'SYNTEL INC',

'WAL-MART ASSOCIATES',

'PRICEWATERHOUSECOOPERS ADVISORY SERVICES',

'IBM CORPORATION',

'MINDTREE',

'AMAZON CORPORATE',

'CUMMINS INC',

'RANDSTAD TECHNOLOGIES'

]
df2018.EMPLOYER_NAME[df2018.EMPLOYER_NAME.str.contains('CISCO SYSTEMS',case=False) == True].value_counts()
print("Denial Rates and Sample Size:")

for i in emp_list2:

    print("2018",i.title(),": {:2.2%} - {:,}".format(df2018_light2.CASE_STATUS[df2018_light2.EMPLOYER_NAME.str.contains(i,case=False) == True]

                                                 .value_counts(normalize=True)[0],df2018_light2.CASE_STATUS[df2018_light2.EMPLOYER_NAME.str.contains(i,case=False) == True].count()))

#df2018.EMPLOYER_NAME[df2018.EMPLOYER_NAME.str.contains('infosys',case=False) == True].value_counts()
emp_list = ['google','amazon','facebook','apple inc','netflix','microsoft','airbnb', 'uber technologies','lyft',

            'oracle','linkedin','square, inc','stripe','vmware','doordash','salesforce','postmates','tata','cognizant']

job_list = ['program manager','product manager','software engineer','engineering manager','analyst','consultant','developer','data scientist']
#double checking if the keyword search works well

for i in emp_list:

    print(df2018_light.EMPLOYER_NAME[df2018_light.EMPLOYER_NAME.str.contains(i,case=False) == True].value_counts(),"\n")
print("Denial Rates and Sample Size:")

for i in emp_list:

    print("2018",i.title(),": {:2.2%} - {:,}".format(1-df2018_light2.CASE_STATUS[df2018_light2.EMPLOYER_NAME.str.contains(i,case=False) == True]

                                                 .value_counts(normalize=True)[0],df2018_light2.CASE_STATUS[df2018_light2.EMPLOYER_NAME.str.contains(i,case=False) == True].count()))

#     print("2017",i.title(),": {:2.2%} - {:,}".format(1-df2017_light.CASE_STATUS[df2017_light.EMPLOYER_NAME.str.contains(i,case=False) == True]

#                                                  .value_counts(normalize=True)[0],df2017_light.CASE_STATUS[df2017_light.EMPLOYER_NAME.str.contains(i,case=False) == True].count()))
print("Denial Rates and Sample Size:")

for j in job_list:

    print("2018",j.title(),": {:2.2%} - {:,}".format(1-df2018_light.CASE_STATUS[df2018_light.JOB_TITLE.str.contains(j,case=False) == True]

                                                 .value_counts(normalize=True)[0],df2018_light.CASE_STATUS[df2018_light.JOB_TITLE.str.contains(j,case=False) == True].count()))

#     print("2017",j.title(),": {:2.2%} - {:,}".format(1-df2017_light.CASE_STATUS[df2017_light.JOB_TITLE.str.contains(j,case=False) == True]

#                                                  .value_counts(normalize=True)[0],df2017_light.CASE_STATUS[df2017_light.JOB_TITLE.str.contains(j,case=False) == True].count()))
# Google H1b Transfer denials by Job Title

df2018_light.JOB_TITLE[(df2018_light.EMPLOYER_NAME.str.contains("google",case=False)==True)&(df2018_light.CASE_STATUS =='DENIED')].value_counts()
# Google H1b Transfer denials by SOC Name

df2018_light.SOC_NAME[(df2018_light.EMPLOYER_NAME.str.contains("google",case=False)==True)&(df2018_light.CASE_STATUS =='DENIED')].value_counts()
# Google All H1b denials by Job Title

df2018.JOB_TITLE[(df2018.EMPLOYER_NAME.str.contains("google",case=False)==True)&(df2018.CASE_STATUS =='DENIED')].value_counts()
# Google All H1b denials by SOC Name

df2018.SOC_NAME[(df2018.EMPLOYER_NAME.str.contains("google",case=False)==True)&(df2018.CASE_STATUS =='DENIED')].value_counts()