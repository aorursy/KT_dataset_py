import pandas as pd

import numpy as np

import os
df=pd.read_csv("../input/bank-marketing-dataset/bank.csv")

df.tail(10)
df.info()
df['deposit']=(df['deposit']=='yes').astype(int)
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(20,16))

df['job'].value_counts().plot.pie(autopct='%1.1f%%')
sns.set(style="darkgrid")

plt.figure(figsize=(15,12))

sns.countplot(df['job'],hue='deposit',data=df)

plt.show()
sns.set(style="darkgrid")

plt.figure(figsize=(16,12))

sns.catplot(x="education", hue="marital", col="deposit",

                data=df, kind="count",

                height=10, aspect=.7);

plt.show()
df['deposit'].value_counts()/df.shape[0]
df.describe()
numerical_features=(df.select_dtypes(exclude=['object'])).columns

numerical_features=list(numerical_features)



# drop the target variable 'deposit'

numerical_features.remove('deposit')
for features in numerical_features:

    mean=df[features].mean(axis=0)

    std=df[features].std(axis=0)

    upper_thres=mean+3*std

    lower_thres=mean-3*std

    outliers=0

    for i in range(df.shape[0]):

        if df[features].iloc[i]>=upper_thres or  df[features].iloc[i]<=lower_thres:

            outliers+=1

    print("{features}->>  outliers: {outlier}".format(features=features,outlier=outliers))
corr=df.corr()

plt.figure(figsize=(14,8))

sns.heatmap(corr,annot=True)
# Day and month columns appear to be uncorrelated.

df['month'][df['deposit']==1].value_counts()
df['month'][df['deposit']==0].value_counts()
import warnings

warnings.simplefilter("ignore")





subscribers_w_loan=df[df['loan']=='yes'][df['deposit']==1].shape[0]

subscribers_wo_loan=df[df['loan']=='no'][df['deposit']==1].shape[0]



subscribers=df[df['deposit']==1]



print("subscribers_w_loan ->>",subscribers_w_loan/subscribers.shape[0])

print("\n")

print("subscribers_wo_loan ->>",subscribers_wo_loan/subscribers.shape[0])
df['contact'].value_counts()
# subscribers who are contacted by various modes.

cellular=0

unknown=0

telephone=0

for x in range(df.shape[0]):

    if df['deposit'].iloc[x]==1:

        if df['contact'].iloc[x]=='cellular':

            cellular+=1

        if df['contact'].iloc[x]=='unknown':

            unknown+=1

        if df['contact'].iloc[x]=='telephone':

            telephone+=1





plt.figure(figsize=(15,8))

contacts = ['cellular', 'unknown', 'telephone']

sizes = [cellular,unknown,telephone]

plt.pie(sizes, labels = contacts,autopct='%1.2f%%')

plt.show()
plt.figure(figsize=(18,10))

sns.violinplot(x="balance",y="job",hue='deposit',data=df)

plt.title("Job distribution with Deposit status",fontsize=16)

plt.show()
df['campaign'].value_counts()
# Duration of the campaign also plays a great role,has a good correlation with deposit subscription.



# this code snippet is forked.



sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.set_style('whitegrid')

avg_duration = df['duration'].mean()



lst = [df]

df["duration_status"] = np.nan



for col in lst:

    col.loc[col["duration"] < avg_duration, "duration_status"] = "below_average"

    col.loc[col["duration"] > avg_duration, "duration_status"] = "above_average"

    

pct_term = pd.crosstab(df['duration_status'], df['deposit']).apply(lambda r: round(r/r.sum(), 2) * 100, axis=1)





ax = pct_term.plot(kind='bar', stacked=False, cmap='RdBu')

plt.title("The Impact of Duration \n in Opening a Term Deposit", fontsize=18)

plt.xlabel("Duration Status", fontsize=18);

plt.ylabel("Percentage (%)", fontsize=18)



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

    



plt.show()
df['poutcome'].value_counts()    
# housing with marital status

plt.figure(figsize=(15,8))

sns.catplot(x="housing", hue="marital", col="deposit",

                data=df, kind="count",

                height=10, aspect=.7);

plt.show()
plt.figure(figsize=(16,20))

sns.set_style("darkgrid")

sns.boxplot(x='job',y='balance',data=df,hue='deposit',whis=1.5)

plt.show()
plt.figure(figsize=(15,8))

sns.set_style("darkgrid")

sns.boxplot(x='deposit',y='age',data=df,whis=1.5)

plt.show()
# drop irrelevant features

df=df.drop(['day','month','poutcome'],axis=1)

df.head()