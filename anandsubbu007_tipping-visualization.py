import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
data = pd.read_csv("../input/tipping/tips.csv")
data.head()
data['tip'].mean()
data['tip'].median()
sns.boxplot(x="tip", data=data)
q3, q1 = np.percentile(data.tip, [75,25])



iqr = q3 - q1

iqr = round(iqr,2)



print ("Lower Quatile:- ", q1 )

print ("Lower Quatile:- ", q3 )

print ("IQR:- ", iqr )

l = q1 - (1.5*iqr)

u = q1 + (1.5*iqr)

l = round(l,2)

u = round(u,2)

print("Lower range in boxplot is {}, & the upper range is, {}".format(l,u))
sns.boxplot(x="total_bill", data = data)
q3, q1 = np.percentile(data.total_bill, [75,25])



iqr = q3 - q1

iqr = round(iqr,2)



print ("Lower Quatile:- ", q1 )

print ("Lower Quatile:- ", q3 )

print ("IQR:- ", iqr )

l = q1 - (1.5*iqr)

u = q1 + (1.5*iqr)

l = round(l,2)

u = round(u,2)

print("Lower range in boxplot is {}, & the upper range is, {}".format(l,u))
data.groupby('sex').size()
x = data.groupby("sex").size()

t = data["sex"].count()

p = x/t * 100

p[0]
cnt = data.groupby(['sex']).count().reset_index()

cnt
cnt['count_perc'] = (cnt['total_bill']/ len(data)) *100

cnt
sns.barplot(x="sex",y='count_perc',

            hue = 'count_perc'

            ,data = cnt)
cnt = data.groupby(['sex']).count().reset_index()

cnt

cnt['count_perc'] = (cnt['total_bill']/ len(data)) *100 



plt.pie(x='count_perc',data=cnt,labels=['Female', 'Male'], autopct='%1.1f%%',

       shadow=True, startangle=90)

data.groupby(["sex"]).mean()['tip']
data.groupby(["day","time"]).mean()['tip']
data.groupby(["day"]).mean()['tip']
data.groupby(["time"]).mean()['tip']
data.groupby('size').mean()['tip']
data.groupby('smoker').sum()['tip']
data.groupby(['sex','smoker']).mean()['tip']
data['pct_tip'] = data['tip']/data['total_bill']
data.groupby(["sex"]).sum()['pct_tip']
data.groupby(["size"]).sum()['pct_tip']
data['sex'].groupby(data["smoker"]).value_counts(normalize=True).rename('pct_tip').reset_index()





x,y,hue = 'sex','pct_tip','smoker'



sns.barplot(x,y,hue,data=data)

sns.scatterplot(x="total_bill", y = "tip",

               data = data)
sns.scatterplot(x="total_bill", y = "pct_tip",

               data = data)
data["smoker"].count()