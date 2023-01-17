import pandas as pd

import numpy as np         #importing libraries
df=pd.read_excel("../input/grades-for-four-years/datadisciplinary.xlsx")        #Loading the dataset

df.head()
df.drop(df.index[[0]],inplace=True)         #removing the zeroth row because it does not contain any data
df1=df.groupby('ENROLLMENTNO').agg(lambda x: x.tolist())        #making another dataframe (df1) where data from four rows of each student is emrged into one
df1.head()
df2 = pd.DataFrame(df1)
df2[['first','second','third','fourth']] = pd.DataFrame(df2.DISCIPLINARYGRADE.tolist(), index= df2.index)

#making 4 new columns to store the grade for each semester
df2.head()
#breaking the branch list of four elemtents in each row into four separate columns

df2[['branch','bran','braa','br']] = pd.DataFrame(df2.BRANCHCODE.tolist(), index= df2.index)

#breaking the program code list of four elemtents in each row into four separate columns

df2[['program_code','program','progra','pro']] = pd.DataFrame(df2.PROGRAMCODE.tolist(), index= df2.index)

df2.head()
df2.drop(['PROGRAMCODE','BRANCHCODE','SEMESTER','DISCIPLINARYGRADE'],inplace=True, axis=1)

df2.drop(['bran','braa','br','program','progra','pro'],inplace=True, axis=1)

df2.head()
df2['firstp'] = [0 if x=="A+" else (1 if x=="A" else (2 if x=="B+" else (3 if x=="B" else (4 if x=="C+" else(5 if x=="C" else 6))))) for x in df2['first']]

df2['secondp'] = [0 if x=="A+" else (1 if x=="A" else (2 if x=="B+" else (3 if x=="B" else (4 if x=="C+" else(5 if x=="C" else 6))))) for x in df2['second']]

df2['thirdp'] = [0 if x=="A+" else (1 if x=="A" else (2 if x=="B+" else (3 if x=="B" else (4 if x=="C+" else(5 if x=="C" else 6))))) for x in df2['third']]

df2['fourthp'] = [0 if x=="A+" else (1 if x=="A" else (2 if x=="B+" else (3 if x=="B" else (4 if x=="C+" else(5 if x=="C" else 6))))) for x in df2['fourth']]
df2['average'] = df2[['firstp', 'secondp','thirdp','fourthp']].mean(axis=1) 

#creating a new column to store average grade point

#storing average of grade points in the average column using mean function
df2['final_grade'] = ['A+' if x<=0.25 else 'A' for x in df2['average']]

#only two conditions specified because the average grade point loss does not go below 0.5

df2.head()
df2['Enrollment No.'] = df2.index
df2.to_excel(r'AverageGrage.xlsx', index = False) #final excel sheet consisting of average disciplinary grade obtained by the student at the end of four years