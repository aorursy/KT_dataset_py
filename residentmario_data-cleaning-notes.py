import pandas as pd

payroll = pd.read_csv("../input/Citywide_Payroll_Data__Fiscal_Year_.csv")
payroll.head()
payroll['Agency Name'].value_counts()[['POLICE DEPARTMENT', 'Police Department']]
payroll['Agency Name'] = payroll['Agency Name'].str.upper()

payroll['Work Location Borough'] = payroll['Work Location Borough'].str.upper()
payroll['Work Location Borough'].value_counts()[::-1].head()
payroll['Title Description'].value_counts().index[11]
payroll['Title Description'] = payroll['Title Description'].str.strip()
[(d, count) for (d, count) in payroll['Agency Name'].value_counts().iteritems()\

    if ' ED ' in d or 'EDUCATION' in d]
len(payroll[payroll['Title Description'] == 'JOB TRAINING PARTICIPANT'])