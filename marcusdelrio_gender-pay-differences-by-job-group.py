# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import statsmodels.api as sm
rd=pd.read_csv("../input/Salaries.csv",low_memory=False)
rd=rd[rd.BasePay!="Not Provided"]

rd.BasePay=pd.to_numeric(rd.BasePay)

firstnames=rd.EmployeeName.str.lower().str.split(" ").str.get(0)
# Add some new categories to our dataset - gender and Job Group

def get_gender(row):

    firstName= row.lower().split(" ")[0]

    maleNames=["michael","john","david","james","robert","joseph","william","richard",

               "daniel","mark","thomas","kevin","christopher","jose","steven","anthony","paul",

               "brian","kenneth","charles","eric","peter","matthew","patrick","edward","stephen",

               "jason"] 

    femaleNames=["maria","mary","jennifer","lisa","linda","susan"]

    if firstName in maleNames:

        return "Male"

    if firstName in femaleNames:

        return "Female"

    return "Not Classified"

    

rd["gender"]=rd["EmployeeName"].map(get_gender)
def find_job_title(row):

    

    police_title = ['police', 'sherif', 'probation', 'sergeant', 'officer', 'lieutenant']

    fire_title = ['fire']

    transit_title = ['mta', 'transit']

    medical_title = ['anesth', 'medical', 'nurs', 'health', 'physician', 'orthopedic', 'pharm', 'care']

    court_title = ['court', 'legal']

    automotive_title = ['automotive', 'mechanic', 'truck']

    engineer_title = ['engineer', 'engr', 'eng', 'program']

    general_laborer_title = ['general laborer', 'painter', 'inspector', 'carpenter', 'electrician', 'plumber', 'maintenance']

    aide_title = ['aide', 'assistant', 'secretary', 'attendant']

    

    for title in police_title:

        if title in row.lower():

            return 'police'    

    for title in fire_title:

        if title in row.lower():

            return 'fire'

    for title in aide_title:

        if title in row.lower():

            return 'assistant'

    for title in transit_title:

        if title in row.lower():

            return 'transit'

    for title in medical_title:

        if title in row.lower():

            return 'medical'

    if 'airport' in row.lower():

        return 'airport'

    if 'worker' in row.lower():

        return 'social worker'

    if 'architect' in row.lower():

        return 'architect'

    for title in court_title:

        if title in row.lower():

            return 'court'

    if 'major' in row.lower():

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

    for title in automotive_title:

        if title in row.lower():

            return 'automotive'

    for title in engineer_title:

        if title in row.lower():

            return 'engineer'

    for title in general_laborer_title:

        if title in row.lower():

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



rd["Job_Group"]=rd["JobTitle"].map(find_job_title)
rd2=rd[rd.gender!="Not Classified"]

WageDifference=rd2.groupby(by="gender").mean().TotalPay

rd2_pivot=rd2.pivot_table(values=["TotalPay"],columns="gender",

                          index="Job_Group",aggfunc=[np.mean,len])

            
#some illuration of wage differences              

mask=rd2_pivot.iloc[:,2]>50

rd2_pivot_50=rd2_pivot[mask]

wageDifference_toPlot=rd2_pivot_50.iloc[:,1]-rd2_pivot_50.iloc[:,0]

wageDifference_toPlot.plot(kind="barh",title="Wage Difference (M-F wage)")

female_weights=rd2_pivot_50.iloc[:,2]/rd2_pivot_50.iloc[:,2].sum()

male_weights=rd2_pivot_50.iloc[:,3]/rd2_pivot_50.iloc[:,3].sum()

female_weighted_salary=female_weights.mul(rd2_pivot_50.iloc[:,0]).sum()

male_weighted_salary=male_weights.mul(rd2_pivot_50.iloc[:,1]).sum()

print (female_weighted_salary,male_weighted_salary)
rd2["gender_num"]=rd2.gender.map({"Male":1,"Female":0})

rd2["Status_num"]=rd2.Status.map({"PT":0,"FT":1})
def regress(data,yvar,xvars):

    Y=data[yvar]

    X=data[xvars]

    X['intercept']=1

    result=sm.OLS(Y,X).fit()

    return result.params
for_analysis=rd2[["BasePay","TotalPay","OvertimePay","gender","gender_num","Job_Group"]].dropna()

grouped_by_job=for_analysis.groupby("Job_Group")
reg_results=grouped_by_job.apply(regress,"TotalPay",["gender_num","BasePay",])

reg_results["Total_cases"]=grouped_by_job.count()["gender"]

reg_results["Male_cases"]=grouped_by_job.sum()["gender_num"]

reg_results["Female_cases"]=reg_results["Total_cases"] - reg_results["Male_cases"]

print(reg_results)