import pandas as pd
import datetime as dt
df_task1 = pd.read_csv('../input/dataset1_with_error.csv')
df_task1
df_task1['Id'] = df_task1['Id'].astype('str')
len_check = (df_task1['Id'].str.len() < 8) & (df_task1['Id'].str.len() > 8)
df_id_check = df_task1.loc[len_check]
print(df_id_check)
df_task1['Location']
df_task1['Location'].replace({'Surey':'Surrey',
                              'Nottinham':'Nottingham',
                              'Reeding':'Reading',
                              'oxfords':'oxford',
                              'Leads':'leeds'},inplace = True)
df_task1['ContractType'].unique()
df_task1['ContractType'].replace({'not available':'non-specified','full_time':'full-time', 'part_time':'part-time'},inplace = True)

df_task1['ContractType'].unique()
df_task1['ContractTime'].unique()
df_task1['ContractTime'].replace({'not available':'non-specified'},inplace = True)
df_task1['ContractTime'].unique()
df_task1['Salary per annum']
task1_sal =[]
for element in df_task1["Salary per annum"]:
    salary = ''
    if element.endswith('K'):
        salary = int(element.replace('K','000'))
    elif '-' in element:
        i = element.split()
        i = ((float(i[0])+float(i[2]))/2)
        salary = i
    else:
        salary = int(element)     
    task1_sal.append(salary)    
df_task1["Salary per annum"] = task1_sal
for element in df_task1['SourceName']:
    if not '.' in element:
        print(element)
df_task1['SourceName'].replace({'monashstudent':'monashstudent.co.uk','jobcareer':'jobcareer.com','admin@caterer.com':'caterer.com'},inplace = True)
OpenDate =[]
i = 0
for each in df_task1["OpenDate"]:
    Date = each
    if int(Date[4:6])>12:
        Date =Date[0:4]+Date[6:8]+Date[4:6]+Date[8:]
    else:
        Date
    Date = pd.to_datetime(Date, format='%Y%m%dT%H%M%S')
    #print(Date)
    OpenDate.append(Date)

df_task1["OpenDate"] = OpenDate  
# Convert Open Date to required format
CloseDate =[]
for each in df_task1["CloseDate"]:
    
    Date = each
    Date = pd.to_datetime(Date, format='%Y%m%dT%H%M%S')
    
    CloseDate.append(Date)

df_task1["CloseDate"] = CloseDate 
df_task1['Company']= df_task1['Company'].fillna('non specified')
df_task1.to_csv('dataset1_solution.csv',header= True, index = False)
df_task1.isna().sum()