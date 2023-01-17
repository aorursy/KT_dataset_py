# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Suicides in India 2001-2012.csv")

data.head()
data.replace('A & N Islands', 'A & N Islands (Ut)', inplace=True)

data.replace('Chandigarh', 'Chandigarh (Ut)', inplace=True)

data.replace('D & N Haveli', 'D & N Haveli (Ut)', inplace=True)

data.replace('Daman & Diu', 'Daman & Diu (Ut)', inplace=True)

data.replace('Lakshadweep', 'Lakshadweep (Ut)', inplace=True)

data.replace('Puducherry', 'Puducherry (Ut)', inplace=True)



data.replace('Bankruptcy or Sudden change in Economic', 'Bankruptcy or Sudden change in Economic Status', inplace=True)

data.replace('Not having Children(Barrenness/Impotency', 'Not having Children (Barrenness/Impotency', inplace=True)

data.replace('By Other means (please specify)', 'By Other means', inplace=True)
#Feature Engineering

data['State_Type'] = 'S'

data.loc[(data.State.str.contains('\(Ut\)')),'State_Type'] = 'U'

data.loc[(data.State.str.contains('Total')),'State_Type'] = 'T'
#Splitting the dataset horizontally as we can't make educational type_code as different columns
data_Causes = data.loc[data.Type_code == 'Causes']

data_Education_Status = data.loc[data.Type_code == 'Education_Status']

data_Means_adopted = data.loc[data.Type_code == 'Means_adopted']

data_Professional_Profile = data.loc[data.Type_code == 'Professional_Profile']

data_Social_Status = data.loc[data.Type_code == 'Social_Status']

states = [s for s in list(data.State.unique()) if ("(" not in s)]

uts = [s for s in list(data.State.unique()) if "(Ut)" in s]

totals = [s for s in list(data.State.unique()) if "Total" in s]



n = 5 #number of datasets

states_total = [0 for i in range(0,n)]

uts_total = [0 for i in range(0,n)]

totals_total = [[0 for t in range(n)] for i in range(len(totals))]



for i in states:

    states_total[0] += sum(data_Causes.loc[data_Causes.State == i]['Total'])

    states_total[1] += sum(data_Education_Status.loc[data_Education_Status.State == i]['Total'])

    states_total[2] += sum(data_Means_adopted.loc[data_Means_adopted.State == i]['Total'])

    states_total[3] += sum(data_Professional_Profile.loc[data_Professional_Profile.State == i]['Total'])

    states_total[4] += sum(data_Social_Status.loc[data_Social_Status.State == i]['Total'])

for i in uts:

    uts_total[0] += sum(data_Causes.loc[data_Causes.State == i]['Total'])

    uts_total[1] += sum(data_Education_Status.loc[data_Education_Status.State == i]['Total'])

    uts_total[2] += sum(data_Means_adopted.loc[data_Means_adopted.State == i]['Total'])

    uts_total[3] += sum(data_Professional_Profile.loc[data_Professional_Profile.State == i]['Total'])

    uts_total[4] += sum(data_Social_Status.loc[data_Social_Status.State == i]['Total'])

for idx, t in enumerate(totals):

    totals_total[idx][0] += sum(data_Causes.loc[data_Causes.State == t]['Total'])

    totals_total[idx][1] += sum(data_Education_Status.loc[data_Education_Status.State == t]['Total'])

    totals_total[idx][2] += sum(data_Means_adopted.loc[data_Means_adopted.State == t]['Total'])

    totals_total[idx][3] += sum(data_Professional_Profile.loc[data_Professional_Profile.State == t]['Total'])

    totals_total[idx][4] += sum(data_Social_Status.loc[data_Social_Status.State == t]['Total'])

print("states_total: ")

print(states_total)

print(uts_total)

print(uts_total)

print(totals_total)

#Checking if the 'Total'and the summation of Education_Status tally for States and Uts

tot_state = pd.pivot_table(data.loc[(data.Type_code == 'Education_Status') & (data.State.str.contains('Total \(S') )],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values

sum_state = pd.pivot_table(data.loc[(data.Type_code == 'Education_Status') & (data.State_Type == 'S')],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values



tot_ut = pd.pivot_table(data.loc[(data.Type_code == 'Education_Status') & (data.State.str.contains('Total \(U') )],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values

sum_ut = pd.pivot_table(data.loc[(data.Type_code == 'Education_Status') & (data.State_Type == 'U')],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values



missing_data = pd.DataFrame(np.hstack((np.array([[x] for x in data.Year.unique()]), np.hstack((tot_state,sum_state)), np.hstack((tot_ut,sum_ut)) )))

missing_data.columns = ['Year', 'State_Total_Female', 'State_Total_Male', 'State_Sum_Female', 'State_Sum_Male',

                       'Ut_Total_Female', 'Ut_Total_Male', 'Ut_Sum_Female', 'Ut_Sum_Male',]

missing_data['State_Missing_Female'] = missing_data.State_Total_Female - missing_data.State_Sum_Female

missing_data['State_Missing_Male'] = missing_data.State_Total_Male - missing_data.State_Sum_Male

missing_data['Ut_Missing_Female'] = missing_data.Ut_Total_Female - missing_data.Ut_Sum_Female

missing_data['Ut_Missing_Male'] = missing_data.Ut_Total_Male - missing_data.Ut_Sum_Male

missing_data.head(20)

#No missing values
#Checking if the 'Total'and the summation of Social_Status tally for States and Uts

tot_state = pd.pivot_table(data.loc[(data.Type_code == 'Social_Status') & (data.State.str.contains('Total \(S') )],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values

sum_state = pd.pivot_table(data.loc[(data.Type_code == 'Social_Status') & (data.State_Type == 'S')],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values



tot_ut = pd.pivot_table(data.loc[(data.Type_code == 'Social_Status') & (data.State.str.contains('Total \(U') )],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values

sum_ut = pd.pivot_table(data.loc[(data.Type_code == 'Social_Status') & (data.State_Type == 'U')],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values



missing_data = pd.DataFrame(np.hstack((np.array([[x] for x in data.Year.unique()]), np.hstack((tot_state,sum_state)), np.hstack((tot_ut,sum_ut)) )))

missing_data.columns = ['Year', 'State_Total_Female', 'State_Total_Male', 'State_Sum_Female', 'State_Sum_Male',

                       'Ut_Total_Female', 'Ut_Total_Male', 'Ut_Sum_Female', 'Ut_Sum_Male',]

missing_data['State_Missing_Female'] = missing_data.State_Total_Female - missing_data.State_Sum_Female

missing_data['State_Missing_Male'] = missing_data.State_Total_Male - missing_data.State_Sum_Male

missing_data['Ut_Missing_Female'] = missing_data.Ut_Total_Female - missing_data.Ut_Sum_Female

missing_data['Ut_Missing_Male'] = missing_data.Ut_Total_Male - missing_data.Ut_Sum_Male

missing_data.head(20)
#Checking for missing Causes values for States and Uts

tot_state = pd.pivot_table(data.loc[(data.Type_code == 'Education_Status') & (data.State.str.contains('Total \(S') )],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values

sum_state = pd.pivot_table(data.loc[(data.Type_code == 'Causes') & (data.State_Type == 'S')],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values



tot_ut = pd.pivot_table(data.loc[(data.Type_code == 'Education_Status') & (data.State.str.contains('Total \(U') )],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values

sum_ut = pd.pivot_table(data.loc[(data.Type_code == 'Causes') & (data.State_Type == 'U')],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values



missing_data = pd.DataFrame(np.hstack((np.array([[x] for x in data.Year.unique()]), np.hstack((tot_state,sum_state)), np.hstack((tot_ut,sum_ut)) )))

missing_data.columns = ['Year', 'State_Total_Female', 'State_Total_Male', 'State_Sum_Female', 'State_Sum_Male',

                       'Ut_Total_Female', 'Ut_Total_Male', 'Ut_Sum_Female', 'Ut_Sum_Male',]

missing_data['State_Missing_Female'] = missing_data.State_Total_Female - missing_data.State_Sum_Female

missing_data['State_Missing_Male'] = missing_data.State_Total_Male - missing_data.State_Sum_Male

missing_data['Ut_Missing_Female'] = missing_data.Ut_Total_Female - missing_data.Ut_Sum_Female

missing_data['Ut_Missing_Male'] = missing_data.Ut_Total_Male - missing_data.Ut_Sum_Male

missing_data.head(20)

#Missing values only for the Year 2012 from the States
#Checking for missing Means_adopted values for States and Uts

tot_state = pd.pivot_table(data.loc[(data.Type_code == 'Education_Status') & (data.State.str.contains('Total \(S') )],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values

sum_state = pd.pivot_table(data.loc[(data.Type_code == 'Means_adopted') & (data.State_Type == 'S')],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values



tot_ut = pd.pivot_table(data.loc[(data.Type_code == 'Education_Status') & (data.State.str.contains('Total \(U') )],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values

sum_ut = pd.pivot_table(data.loc[(data.Type_code == 'Means_adopted') & (data.State_Type == 'U')],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values



missing_data = pd.DataFrame(np.hstack((np.array([[x] for x in data.Year.unique()]), np.hstack((tot_state,sum_state)), np.hstack((tot_ut,sum_ut)) )))

missing_data.columns = ['Year', 'State_Total_Female', 'State_Total_Male', 'State_Sum_Female', 'State_Sum_Male',

                       'Ut_Total_Female', 'Ut_Total_Male', 'Ut_Sum_Female', 'Ut_Sum_Male',]

missing_data['State_Missing_Female'] = missing_data.State_Total_Female - missing_data.State_Sum_Female

missing_data['State_Missing_Male'] = missing_data.State_Total_Male - missing_data.State_Sum_Male

missing_data['Ut_Missing_Female'] = missing_data.Ut_Total_Female - missing_data.Ut_Sum_Female

missing_data['Ut_Missing_Male'] = missing_data.Ut_Total_Male - missing_data.Ut_Sum_Male

missing_data.head(20)

#No missing data
#Checking for missing Professional_Profile values for States and Uts

tot_state = pd.pivot_table(data.loc[(data.Type_code == 'Education_Status') & (data.State.str.contains('Total \(S') )],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values

sum_state = pd.pivot_table(data.loc[(data.Type_code == 'Professional_Profile') & (data.State_Type == 'S')],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values



tot_ut = pd.pivot_table(data.loc[(data.Type_code == 'Education_Status') & (data.State.str.contains('Total \(U') )],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values

sum_ut = pd.pivot_table(data.loc[(data.Type_code == 'Professional_Profile') & (data.State_Type == 'U')],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values



missing_data = pd.DataFrame(np.hstack((np.array([[x] for x in data.Year.unique()]), np.hstack((tot_state,sum_state)), np.hstack((tot_ut,sum_ut)) )))

missing_data.columns = ['Year', 'State_Total_Female', 'State_Total_Male', 'State_Sum_Female', 'State_Sum_Male',

                       'Ut_Total_Female', 'Ut_Total_Male', 'Ut_Sum_Female', 'Ut_Sum_Male',]

missing_data['State_Missing_Female'] = missing_data.State_Total_Female - missing_data.State_Sum_Female

missing_data['State_Missing_Male'] = missing_data.State_Total_Male - missing_data.State_Sum_Male

missing_data['Ut_Missing_Female'] = missing_data.Ut_Total_Female - missing_data.Ut_Sum_Female

missing_data['Ut_Missing_Male'] = missing_data.Ut_Total_Male - missing_data.Ut_Sum_Male

missing_data.head(20)

#Missing values maximum for the Year 2012 from the States
data.loc[(data.Year == 2001) & (data.State_Type == 'U') & (data.Gender == 'Male')].groupby(['State','Type']).Total.aggregate(sum)
data.loc[(data.Year == 2001) & (data.State_Type == 'U') & (data.State == 'Daman & Diu (Ut)') & (data.Gender == 'Male')].groupby(['State','Type_code','Age_group']).Total.aggregate(sum)
data.loc[ (data.State_Type == 'U') & 

         (data.State == 'Daman & Diu (Ut)') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile')].groupby(['Age_group','Type',]).Total.aggregate(sum)
data.loc[(data.State_Type == 'U') & (data.Year == 2001) & (data.Age_group == '15-29') & 

         (data.State == 'Daman & Diu (Ut)') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Service (Private)'), 'Total'] += 1
data.loc[(data.State_Type == 'U') & (data.Gender == 'Male') & 

        (data.State == 'Daman & Diu (Ut)') & #found out

         ((data.Age_group == '30-44') | (data.Age_group == '45-59')) & #found out 1 missing from each

        (data.Type_code == 'Professional_Profile')].groupby(['State','Age_group','Type',]).Total.aggregate(sum)

data.loc[(data.State_Type == 'U') & (data.Year == 2003) & (data.Age_group == '30-44') & 

         (data.State == 'Daman & Diu (Ut)') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Service (Private)'), 'Total'] += 1

data.loc[(data.State_Type == 'U') & (data.Year == 2003) & (data.Age_group == '45-59') & 

         (data.State == 'Daman & Diu (Ut)') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Others (Please Specify)'), 'Total'] += 1
data.loc[(data.State_Type == 'U') & (data.Year == 2004) & (data.Age_group == '0-14') & 

         (data.State == 'Delhi (Ut)') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Student'), 'Total'] += 1

data.loc[(data.State_Type == 'U') & (data.Year == 2005) & (data.Age_group == '30-44') & 

         (data.State == 'Daman & Diu (Ut)') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Service (Private)'), 'Total'] += 1

data.loc[(data.State_Type == 'U') & (data.Year == 2007) & (data.Age_group == '45-59') & 

         (data.State == 'A & N Islands (Ut)') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Service (Private)'), 'Total'] += 4 #4 missing

data.loc[(data.State_Type == 'U') & (data.Year == 2007) & (data.Age_group == '60+') & 

         (data.State == 'A & N Islands (Ut)') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Service (Private)'), 'Total'] += 1

data.loc[(data.State_Type == 'U') & (data.Year == 2007) & (data.Age_group == '15-29') & 

         (data.State == 'Chandigarh (Ut)') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Unemployed'), 'Total'] += 2 #2 missing records

data.loc[(data.State_Type == 'U') & (data.Year == 2008) & (data.Age_group == '60+') & 

         (data.State == 'A & N Islands (Ut)') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Service (Private)'), 'Total'] += 1

data.loc[(data.State_Type == 'U') & (data.Year == 2009) & (data.Age_group == '15-29') & 

         (data.State == 'Chandigarh (Ut)') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Unemployed'), 'Total'] += 1



######Women######

data.loc[(data.State_Type == 'U') & (data.Year == 2004) & (data.Age_group == '30-44') & 

         (data.State == 'Puducherry (Ut)') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 1

data.loc[(data.State_Type == 'U') & (data.Year == 2004) & (data.Age_group == '45-59') & 

         (data.State == 'Delhi (Ut)') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 2 ##

data.loc[(data.State_Type == 'U') & (data.Year == 2005) & (data.Age_group == '45-59') & 

         (data.State == 'Delhi (Ut)') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 1

data.loc[(data.State_Type == 'U') & (data.Year == 2006) & (data.Age_group == '45-59') & 

         (data.State == 'Delhi (Ut)') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 2 ##

data.loc[(data.State_Type == 'U') & (data.Year == 2008) & (data.Age_group == '30-44') & 

         (data.State == 'Delhi (Ut)') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 4 ##

data.loc[(data.State_Type == 'U') & (data.Year == 2010) & (data.Age_group == '45-59') & 

         (data.State == 'Delhi (Ut)') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 1

data.loc[(data.State_Type == 'U') & (data.Year == 2010) & (data.Age_group == '60+') & 

         (data.State == 'Delhi (Ut)') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 1

data.loc[(data.State_Type == 'U') & (data.Year == 2012) & (data.Age_group == '60+') & 

         (data.State == 'Delhi (Ut)') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 1

data.loc[(data.State_Type == 'U') & (data.Year == 2012) & (data.Age_group == '15-29') & 

         (data.State == 'Chandigarh (Ut)') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 3 ##
####Women of States###

data.loc[(data.State_Type == 'S') & (data.Year == 2005) & (data.Age_group == '60+') & 

         (data.State == 'Rajasthan') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 1

data.loc[(data.State_Type == 'S') & (data.Year == 2006) & (data.Age_group == '15-29') & 

         (data.State == 'Jammu & Kashmir') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 8 ##

data.loc[(data.State_Type == 'S') & (data.Year == 2006) & (data.Age_group == '0-14') & 

         (data.State == 'Gujarat') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Student'), 'Total'] += 3 ##



data.loc[(data.State_Type == 'S') & (data.Year == 2003) & (data.Age_group == '30-44') & 

         (data.State == 'Bihar') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 7 ##

data.loc[(data.State_Type == 'S') & (data.Year == 2003) & (data.Age_group == '45-59') & 

         (data.State == 'Bihar') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 4 ##

data.loc[(data.State_Type == 'S') & (data.Year == 2003) & (data.Age_group == '30-44') & 

         (data.State == 'Uttarakhand') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 1

data.loc[(data.State_Type == 'S') & (data.Year == 2003) & (data.Age_group == '60+') & 

         (data.State == 'Uttarakhand') & (data.Gender == 'Female') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'House Wife'), 'Total'] += 1
####Men of States###

data.loc[(data.State_Type == 'S') & (data.Year == 2006) & (data.Age_group == '60+') & 

         (data.State == 'Jammu & Kashmir') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Others (Please Specify)'), 'Total'] += 1

data.loc[(data.State_Type == 'S') & (data.Year == 2006) & (data.Age_group == '60+') & 

         (data.State == 'Mizoram') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Others (Please Specify)'), 'Total'] += 2 ##

data.loc[(data.State_Type == 'S') & (data.Year == 2006) & (data.Age_group == '45-59') & 

         (data.State == 'Meghalaya') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Others (Please Specify)'), 'Total'] += 1

data.loc[(data.State_Type == 'S') & (data.Year == 2001) & (data.Age_group == '0-14') & 

         (data.State == 'Odisha') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Others (Please Specify)'), 'Total'] += 6 ##

data.loc[(data.State_Type == 'S') & (data.Year == 2010) & (data.Age_group == '45-59') & 

         (data.State == 'Meghalaya') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Others (Please Specify)'), 'Total'] += 1

data.loc[(data.State_Type == 'S') & (data.Year == 2010) & (data.Age_group == '60+') & 

         (data.State == 'Jharkhand') & (data.Gender == 'Male') & 

        (data.Type_code == 'Professional_Profile') & (data.Type == 'Others (Please Specify)'), 'Total'] += 2 ##
#Checking for missing Professional_Profile values for States and Uts

tot_state = pd.pivot_table(data.loc[(data.Type_code == 'Education_Status') & (data.State.str.contains('Total \(S') )],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values

sum_state = pd.pivot_table(data.loc[(data.Type_code == 'Professional_Profile') & (data.State_Type == 'S')],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values



tot_ut = pd.pivot_table(data.loc[(data.Type_code == 'Education_Status') & (data.State.str.contains('Total \(U') )],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values

sum_ut = pd.pivot_table(data.loc[(data.Type_code == 'Professional_Profile') & (data.State_Type == 'U')],index=['Year'],values=["Total"],aggfunc=np.sum, columns='Gender').values



missing_data = pd.DataFrame(np.hstack((np.array([[x] for x in data.Year.unique()]), np.hstack((tot_state,sum_state)), np.hstack((tot_ut,sum_ut)) )))

missing_data.columns = ['Year', 'State_Total_Female', 'State_Total_Male', 'State_Sum_Female', 'State_Sum_Male',

                       'Ut_Total_Female', 'Ut_Total_Male', 'Ut_Sum_Female', 'Ut_Sum_Male',]

missing_data['State_Missing_Female'] = missing_data.State_Total_Female - missing_data.State_Sum_Female

missing_data['State_Missing_Male'] = missing_data.State_Total_Male - missing_data.State_Sum_Male

missing_data['Ut_Missing_Female'] = missing_data.Ut_Total_Female - missing_data.Ut_Sum_Female

missing_data['Ut_Missing_Male'] = missing_data.Ut_Total_Male - missing_data.Ut_Sum_Male

missing_data.head(20)

#Missing values maximum for the Year 2012 from the States