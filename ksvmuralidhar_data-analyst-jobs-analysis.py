import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
src = pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv")

src.head()
src.drop(columns="Unnamed: 0",inplace=True)

src.head()
src.insert(2,"Average Salary",0)



#calculating avg salary

def avg_sal(x):

    x = x.replace("(Glassdoor est.)","")

    x = x.replace("$","")

    x = x.replace("K","000")

    if x.find("0-")!=-1:

        avg = (int(x.split("-")[0])+int(x.split("-")[1]))/2

    else:

        avg=0

    return avg



src["Average Salary"] = src["Salary Estimate"].apply(avg_sal)



#Max salary per location

max_sal = src[["Location","Average Salary"]].copy()

max_sal = max_sal.groupby(["Location"]).max()

max_sal = max_sal.reset_index()

max_sal.columns = ["Location", "Max Salary"]





#Max rating per location

max_rat = src[["Location","Rating"]].copy()

max_rat = max_rat.groupby(["Location"]).max()

max_rat = max_rat.reset_index()

max_rat.columns = ["Location", "Max Rating"]



#Adding to main dataframe

src = src.merge(max_sal,on="Location",how="left")

src = src.merge(max_rat,on="Location",how="left")
best_rating = src.loc[(src["Rating"]==src["Max Rating"]),:].copy()

best_rating.sort_values(by=["Location","Rating"],ascending=[True,False])
best_salary = src.loc[(src["Average Salary"]==src["Max Salary"]),:].copy()

best_salary.sort_values(by=["Location","Average Salary"],ascending=[True,False])
best_salary_rating = src.loc[(src["Average Salary"]==src["Max Salary"]) & (src["Rating"]==src["Max Rating"]),:].copy()

best_salary_rating.sort_values(by=["Location","Rating","Average Salary"],ascending=[True,False,False])
src1 = src[["Rating", "Average Salary"]].copy()

src1 = src1.loc[(src1["Rating"]>0) & (src1["Average Salary"]>0)].copy()
src1.corr()
sns.regplot(data=src1,x="Rating",y="Average Salary")

plt.show()
fig = plt.figure(figsize=(15,7))

sns.boxplot(x="Rating",y="Average Salary",data=src1)