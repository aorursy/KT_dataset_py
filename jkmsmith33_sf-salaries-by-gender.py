import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sexmachine.detector as gender

data = pd.read_csv('../input/Salaries.csv', header=0)
data = data.drop(['Id', 'Notes', 'Agency', 'Status'], axis=1) #Either too incomplete or
#do not provide any substantive information

data['Gender'] = data['EmployeeName']
d = gender.Detector(case_sensitive=False)

data["SplitName"] = data['EmployeeName']
data['SplitName'] = data['SplitName'].str.split()
data['FirstName'] = data['EmployeeName']

for i in range(0, 148654): 
	data['FirstName'][i] = data['SplitName'][i].pop(0)
	print(i)
	
for i in range(0, 148654):
	data['Gender'][i] = d.get_gender(data['FirstName'][i])
	print(i)
	
pandy = float(sum(data["Gender"] == "andy")) / sum(data["Gender"].notnull())
pmale = float(sum(data["Gender"] == "male")) / sum(data["Gender"].notnull())
pfemale = float(sum(data["Gender"] == "female")) / sum(data["Gender"].notnull())
pmmale = float(sum(data["Gender"] == "mostly_male")) / sum(data["Gender"].notnull())
pmfemale = float(sum(data["Gender"] == "mostly_female")) / sum(data["Gender"].notnull())

print(pandy) #0.13
print(pmale) #0.47 
print(pfemale) #0.33
print(pmmale) #0.02
print(pmfemale) #0.03
print(pandy + pmale + pfemale + pmmale + pmfemale) #1.00

data_copy = data #just in case

data = data[data.Gender != 'andy']
data['Gender'] = data['Gender'].map({'male': 'male', 'female': 'female', 'mostly_male': 'male','mostly_female': 'female'})

pandy = float(sum(data["Gender"] == "andy")) / sum(data["Gender"].notnull())
pmale = float(sum(data["Gender"] == "male")) / sum(data["Gender"].notnull())
pfemale = float(sum(data["Gender"] == "female")) / sum(data["Gender"].notnull())
pmmale = float(sum(data["Gender"] == "mostly_male")) / sum(data["Gender"].notnull())
pmfemale = float(sum(data["Gender"] == "mostly_female")) / sum(data["Gender"].notnull())

print(pandy) #0.00
print(pmale) #0.57
print(pfemale) #0.43
print(pmmale) #0.00
print(pmfemale) #0.00
print(pmale + pfemale) #1.00

data = data.drop(['EmployeeName', 'SplitName', 'FirstName'], axis=1)
data.to_csv('genderdata.csv', sep=',') #save data that has gender instead of name

def find_job_title(row): #give credit to MattEvanoff because he made this function
    
    police_title = ['police', 'sherif', 'probation', 'sergeant', 'officer', 'lieutenant']
    fire_title = ['fire']
    transit_title = ['mta', 'transit']
    medical_title = ['anesth', 'medical', 'nurs', 'health', 'physician', 'orthopedic', 'pharm', 'care']
    court_title = ['court', 'legal']
    automotive_title = ['automotive', 'mechanic', 'truck']
    engineer_title = ['engineer', 'engr', 'eng', 'program']
    general_laborer_title = ['general laborer', 'painter', 'inspector', 'carpenter', 
                             'electrician', 'plumber', 'maintenance']
    aide_title = ['aide', 'assistant', 'secretary', 'attendant']
    
    for police in police_title:
        if police in row.lower():
            return 'police'    
    for fire in fire_title:
        if fire in row.lower():
            return 'fire'
    for aide in aide_title:
        if aide in row.lower():
            return 'assistant'
    for transit in transit_title:
        if transit in row.lower():
            return 'transit'
    for medical in medical_title:
        if medical in row.lower():
            return 'medical'
    if 'airport' in row.lower():
        return 'airport'
    if 'worker' in row.lower():
        return 'social worker'
    if 'architect' in row.lower():
        return 'architect'
    for court in court_title:
        if court in row.lower():
            return 'court'
    if 'mayor' in row.lower():
        return 'mayor'
    if 'librar' in row.lower():
        return 'library'
    if 'guard' in row.lower():
        return 'guard'
    if 'public' in row.lower():
        return 'public works'
    if 'attorney' in row.lower():
        return 'attorney'
    if 'custodian' in row.lower():
        return 'custodian'
    if 'account' in row.lower():
        return 'account'
    if 'garden' in row.lower():
        return 'gardener'
    if 'recreation' in row.lower():
        return 'recreation leader'
    for automotive in automotive_title:
        if automotive in row.lower():
            return 'automotive'
    for engineer in engineer_title:
        if engineer in row.lower():
            return 'engineer'
    for general_laborer in general_laborer_title:
        if general_laborer in row.lower():
            return 'general laborer'
    if 'food serv' in row.lower():
        return 'food service'
    if 'clerk' in row.lower():
        return 'clerk'
    if 'porter' in row.lower():
        return 'porter' 
    if 'analy' in row.lower():
        return 'analyst'
    if 'manager' in row.lower():
        return 'manager'
    else:
        return 'other'
    
data['BinnedJob'] = data['JobTitle'].map(find_job_title)
data = data.drop(['JobTitle'], axis=1)

data.BasePay = data.BasePay.astype(float).fillna(0.0)
data.OvertimePay = data.OvertimePay.astype(float).fillna(0.0)
data.OtherPay = data.OtherPay.astype(float).fillna(0.0)
data.Benefits = data.Benefits.astype(float).fillna(0.0)

data.to_csv('genderjobdata.csv', sep=',')

data_male = data[data.Gender != 'female']
data_female = data[data.Gender != 'male']

data_male.describe()
data_female.describe()
#can see that males make approximately $20,000 more than females on average

data_male = data_male.rename(columns={'BasePay': 'BasePayMale', 'OvertimePay': 'OvertimePayMale',
'OtherPay': 'OtherPayMale', 'Benefits': 'BenefitsMale', 'TotalPay': 'TotalPayMale', 
'TotalPayBenefits': 'TotalPayBenefitsMale','Year': 'YearMale', 'Gender': 'GenderMale',
'BinnedJob': 'BinnedJobMale'})

data_female = data_female.rename(columns={'BasePay': 'BasePayFemale', 'OvertimePay': 'OvertimePayFemale',
'OtherPay': 'OtherPayFemale', 'Benefits': 'BenefitsFemale', 'TotalPay': 'TotalPayFemale', 
'TotalPayBenefits': 'TotalPayBenefitsFemale','Year': 'YearFemale', 'Gender': 'GenderFemale',
'BinnedJob': 'BinnedJobFemale'})

plt.figure(figsize=(16,5))
sns.countplot('BinnedJob', data = data, hue = 'Gender')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

#shows counts of males and females for each of the job cateogies
#men dominate transit, police, automotive, fire, general laborer, engineer
#women dominate medical, clerk

plt.figure(figsize=(16,10))
sns.boxplot(x='BinnedJob', y='TotalPayBenefits', data = data, hue = 'Gender')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()


