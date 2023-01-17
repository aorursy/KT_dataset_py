import numpy as np

import pandas as pd
data = pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv").dropna()

data.head()
newData = pd.DataFrame(index=data.index, columns=["Job Title", "Min salary(K)", "Max salary(K)", "Rating", "Location", "Industry"])

newData.astype('float32')

newData.head()
for i in data.index:

    newData.loc[i]["Job Title"] = data.loc[i]["Job Title"]

    newData.loc[i]["Rating"]    = data.loc[i]["Rating"]

    newData.loc[i]["Location"]  = data.loc[i]["Location"]

    newData.loc[i]["Industry"]  = data.loc[i]["Industry"]

newData.head()
def selectSalary(value=""):

    k = 0

    valueLst = list(value)

    result = []

    res = ""

    

    for i in range( len(valueLst) ):

        if(k<2):

            if( valueLst[i].isdigit() ):

                res += valueLst[i]

            elif( (not valueLst[i].isdigit()) and valueLst[i-1].isdigit() ):

                result.append(res)

                res = ""

                k += 1

        else:

            return result

            


salaries = ['0', '0']

i = 0



for i in data.index:

    try:

        salaries = selectSalary( data.loc[i]["Salary Estimate"] )

        newData.loc[i]["Min salary(K)"] = float(salaries[0])

        newData.loc[i]["Max salary(K)"] = float(salaries[1])

        i += 1

    except:

        continue

newData = newData.dropna()

newData.head()
data = newData

del newData



data.head()
maxMinSalary = min(data["Min salary(K)"])

maxMaxSalary = min(data["Max salary(K)"])
jobs = list( data[ data["Min salary(K)"] == maxMinSalary ][ data["Rating"]>=4 ]["Job Title"].unique() )

for i in range(len(jobs)):

    print(jobs[i])
jobs = list( data[ data["Min salary(K)"] == maxMaxSalary ][ data["Rating"]>=4 ]["Job Title"].unique() )

for i in range(len(jobs)):

    print(jobs[i])