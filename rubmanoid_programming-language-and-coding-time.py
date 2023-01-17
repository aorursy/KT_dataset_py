import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # operating system
import seaborn as sns # For plots
data = pd.read_csv("../input/multipleChoiceResponses.csv",low_memory=False)
# Delete first row with questions
data = data.iloc[1:]
# Delete excess columns
col_list = ["Q17","Q23"]
data=data[col_list]
print(data.shape)
# Delete nan rows for question 17 and 23
data = data.dropna(subset=["Q17","Q23"])
print(data.shape)
# Replace string values for actively time coding to integer values
data["Q23"] = data["Q23"].map({'100% of my time' : 100.0 , '0% of my time' : 0.0 , '25% to 49% of my time' : 37.0,'75% to 99% of my time' : 85.0 , '50% to 74% of my time': 62.0 ,  '1% to 25% of my time' : 12.0 })
grouped = data.groupby("Q17").mean().reset_index()
grouped = grouped.sort_values(by="Q23",ascending=False)

print (grouped) 

plotdata = pd.DataFrame({"Programming language" : grouped["Q17"]  , "Time coding [percent]" : grouped["Q23"] })
bar = sns.barplot(y="Programming language",x="Time coding [percent]" ,data = plotdata)
# Any results you write to the current directory are saved as output.