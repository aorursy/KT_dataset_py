import pandas as pd

df = pd.read_csv("../input/attrition.csv")

df1=df
for i in df.columns:

    print(i+": "+str(len(df[i].value_counts())))
df['Designation'] = df['Designation'].str.lower()

for i in range(df['Designation'].shape[0]):

    if "sr" not in df['Designation'][i]:

        if "sales executive" in df['Designation'][i]:

            df['Designation'][i]="sales executive"

    else:

        df['Designation'][i] = "sr sales executive"
df['Grade'].value_counts()

#from the box below it is clear that there is no duplicacy 
df['Gender'].value_counts()

#from the box below it is clear that there is no duplicacy
df['Education'].value_counts()

#from the box below it is clear that there is no duplicacy
df['Last Rating'].value_counts()

#from the box below it is clear that there is no duplicacy
df['Marital Status'].value_counts()

#from the box below it is clear that there is no duplicacy
print("Before Handling")

print(df['Zone'].value_counts())

#This contains duplicacy, we have to handle it 

df['Zone']=df['Zone'].str.lower()

print('\n')

print("After Handling")

print(df['Zone'].value_counts())
df['Remarks'].value_counts()

#from the box below it is clear that there is no duplicacy
l = ['Designation', 'Grade', 'Gender', 'Education', 'Last Rating', 'Marital Status', 'Zone', 'Remarks']
import seaborn as sns

import matplotlib.pyplot as plt

for i in l:

    plt.rcParams.update({'font.size': 35})

    plt.figure(figsize=(40,15))

    plt.xticks(rotation=90)

    ax = sns.countplot(x=i, data=df)

    plt.title(i+' wise count of attrition')

    plt.xlabel(i)

    plt.ylabel('Count')

    for p in ax.patches:

        ax.annotate('%{:.1f}'.format(100*(p.get_height()/327)), (p.get_x()+0.1, p.get_height()+1))
df.columns
i='Grade'

plt.rcParams.update({'font.size': 10})

plt.figure(figsize=(15,6))

plt.xticks(rotation=90)

ax = sns.countplot(x=i, data=df,hue='Zone')

plt.title(i+' wise count of attrition')

plt.xlabel(i)

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate('{:.1f}%'.format(100*(p.get_height()/327)), (p.get_x(), p.get_height()+1))
i='Grade'

plt.rcParams.update({'font.size': 10})

plt.figure(figsize=(15,6))

plt.xticks(rotation=90)

ax = sns.countplot(x=i, data=df,hue='Remarks')

plt.title(i+' wise count of attrition')

plt.xlabel(i)

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate('{:.1f}%'.format(100*(p.get_height()/327)), (p.get_x(), p.get_height()+1))
i='Zone'

plt.rcParams.update({'font.size': 10})

plt.figure(figsize=(15,6))

plt.xticks(rotation=90)

ax = sns.countplot(x=i, data=df,hue='Education')

plt.title(i+' wise count of attrition')

plt.xlabel(i)

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate('{:.1f}%'.format(100*(p.get_height()/327)), (p.get_x(), p.get_height()+1))
i='Zone'

plt.rcParams.update({'font.size': 10})

plt.figure(figsize=(15,6))

plt.xticks(rotation=90)

ax = sns.countplot(x=i, data=df,hue='Remarks')

plt.title(i+' wise count of attrition')

plt.xlabel(i)

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate('{:.1f}%'.format(100*(p.get_height()/327)), (p.get_x(), p.get_height()+1))
i='Zone'

plt.rcParams.update({'font.size': 10})

plt.figure(figsize=(15,6))

plt.xticks(rotation=90)

ax = sns.countplot(x=i, data=df,hue='Grade')

plt.title(i+' wise count of attrition')

plt.xlabel(i)

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate('{:.1f}%'.format(100*(p.get_height()/327)), (p.get_x(), p.get_height()+1))
i='Zone'

plt.rcParams.update({'font.size': 10})

plt.figure(figsize=(15,6))

plt.xticks(rotation=90)

ax = sns.countplot(x=i, data=df,hue='Grade')

plt.title(i+' wise count of attrition')

plt.xlabel(i)

plt.ylabel('Count')

for p in ax.patches:

    ax.annotate('{:.1f}%'.format(100*(p.get_height()/327)), (p.get_x(), p.get_height()+1))
df['years']=df['Tenure'].str.split('.').map(lambda x: int(x[0]))

df['month1']=df['Tenure'].str.split('.').map(lambda x: int(x[1]))

df['total_months']=df['years']*12+df['month1']
l = ['Gender', 'Zone', 'Grade', 'Marital Status', 'Education']
l = ['Gender', 'Zone', 'Grade']

#these were found to be significant

for j in l:

    i='total_months'

    plt.rcParams.update({'font.size': 10})

    plt.figure(figsize=(15,6))

    plt.xticks(rotation=90)

    ax = sns.boxplot(x=j, y=i,data=df)

    plt.title(i+' wise count of attrition')

    plt.xlabel(j)

    plt.ylabel(i)
l = ['Zone', 'Grade', 'Education']

#these were found to be significant

for j in l:

    i='Age'

    plt.rcParams.update({'font.size': 10})

    plt.figure(figsize=(15,6))

    plt.xticks(rotation=90)

    ax = sns.boxplot(x=j, y=i,data=df)

    plt.title(i+' wise count of attrition')

    plt.xlabel(j)

    plt.ylabel(i)
df['Engagement Score (% Satisfaction)']=df['Engagement Score (% Satisfaction)'].str.split('%').map(lambda x: int(x[0]))
l = ['Zone', 'Grade', 'Education']

#these were found to be significant

for j in l:

    i='Engagement Score (% Satisfaction)'

    plt.rcParams.update({'font.size': 10})

    plt.figure(figsize=(15,6))

    plt.xticks(rotation=90)

    ax = sns.boxplot(x=j, y=i,data=df)

    plt.title(i+' wise attrition')

    plt.xlabel(j)

    plt.ylabel(i)
df3=df[['Age','Engagement Score (% Satisfaction)','total_months','Monthly Income']]
sns.pairplot(df3,kind='reg')
sns.heatmap(df3.corr(),annot=True)
df1 = df

df1['cumul'] = 1

df['cumul'] = 1
df['In Active Date']=pd.to_datetime(df['In Active Date'])

df['DOJ']=pd.to_datetime(df['DOJ'])
df.index=df['In Active Date']

k = df['cumul']

g = k.groupby(pd.Grouper(freq="M")).sum()

g = pd.DataFrame(g)

g=g[:-1]

plt.figure(figsize=(40,15))

plt.rcParams.update({'font.size': 20})

sns.lineplot(data=g,x=g.index,y='cumul')

plt.ylabel('Attritions')