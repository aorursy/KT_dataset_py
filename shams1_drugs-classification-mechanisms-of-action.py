import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
a= pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

b= pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

c=pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

d=pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')



merged=pd.concat([a,b])



#Datasets for treated and control experiments

treated= a[a['cp_type']=='trt_cp']

control= a[a['cp_type']=='ctl_vehicle']



#Treatment time datasets

cp24= a[a['cp_time']== 24]

cp48= a[a['cp_time']== 48]

cp72= a[a['cp_time']== 72]



#Merge scored and nonscored labels

all_drugs= pd.merge(d, c, on='sig_id', how='inner')



#Treated drugs without control

treated_list = treated['sig_id'].to_list()

drugs_tr= d[d['sig_id'].isin(treated_list)]



#adt= All Drugs Treated

adt= all_drugs[all_drugs['sig_id'].isin(treated_list)]
def plotd(f1):

    plt.style.use('seaborn')

    sns.set_style('whitegrid')

    fig = plt.figure(figsize=(15,5))

    #1 rows 2 cols

    #first row, first col

    ax1 = plt.subplot2grid((1,2),(0,0))

    plt.hist(control[f1], bins=4, color='mediumpurple',alpha=0.5)

    plt.title(f'control: {f1}',weight='bold', fontsize=18)

    #first row sec col

    ax1 = plt.subplot2grid((1,2),(0,1))

    plt.hist(treated[f1], bins=4, color='darkcyan',alpha=0.5)

    plt.title(f'Treated with compound: {f1}',weight='bold', fontsize=18)

    plt.show()

    

def plott(f1):

    plt.style.use('seaborn')

    sns.set_style('whitegrid')

    fig = plt.figure(figsize=(15,5))

    #1 rows 2 cols

    #first row, first col

    ax1 = plt.subplot2grid((1,3),(0,0))

    plt.hist(cp24[f1], bins=3, color='deepskyblue',alpha=0.5)

    plt.title(f'Treatment duration 24h: {f1}',weight='bold', fontsize=14)

    #first row sec col

    ax1 = plt.subplot2grid((1,3),(0,1))

    plt.hist(cp48[f1], bins=3, color='lightgreen',alpha=0.5)

    plt.title(f'Treatment duration 48h: {f1}',weight='bold', fontsize=14)

    #first row 3rd column

    ax1 = plt.subplot2grid((1,3),(0,2))

    plt.hist(cp72[f1], bins=3, color='gold',alpha=0.5)

    plt.title(f'Treatment duration 72h: {f1}',weight='bold', fontsize=14)

    plt.show()



def plotf(f1, f2, f3, f4):

    plt.style.use('seaborn')

    sns.set_style('whitegrid')



    fig= plt.figure(figsize=(15,10))

    #2 rows 2 cols

    #first row, first col

    ax1 = plt.subplot2grid((2,2),(0,0))

    plt.hist(a[f1], bins=3, color='orange', alpha=0.7)

    plt.title(f1,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    #first row sec col

    ax1 = plt.subplot2grid((2,2), (0, 1))

    plt.hist(a[f2], bins=3, alpha=0.7)

    plt.title(f2,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    #Second row first column

    ax1 = plt.subplot2grid((2,2), (1, 0))

    plt.hist(a[f3], bins=3, color='red', alpha=0.7)

    plt.title(f3,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    #second row second column

    ax1 = plt.subplot2grid((2,2), (1, 1))

    plt.hist(a[f4], bins=3, color='green', alpha=0.7)

    plt.title(f4,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')



    return plt.show()



def ploth(data, w=15, h=9):

    plt.figure(figsize=(w,h))

    sns.heatmap(data.corr(), cmap='hot')

    plt.title('Correlation between the drugs', fontsize=18, weight='bold')

    return plt.show()
a.head()
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))

#1 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((1,2),(0,0))

sns.countplot(x='cp_type', data=a, palette='pastel')

plt.title('Train: Control and treated samples', fontsize=15, weight='bold')

#first row sec col

ax1 = plt.subplot2grid((1,2),(0,1))

sns.countplot(x='cp_dose', data=a, palette='Purples')

plt.title('Train: Treatment Doses: Low and High',weight='bold', fontsize=18)

plt.show()
plt.figure(figsize=(15,5))

sns.distplot( a['cp_time'], color='red', bins=5)

plt.title("Train: Treatment duration ", fontsize=15, weight='bold')

plt.show()
plotf('c-10', 'c-50', 'c-70', 'c-90')
plotd("c-30")
plott('c-30')
#Select the columns c-

c_cols = [col for col in a.columns if 'c-' in col]

#Filter the columns c-

cells=treated[c_cols]

#Plot heatmap

plt.figure(figsize=(12,6))

sns.heatmap(cells.corr(), cmap='coolwarm', alpha=0.9)

plt.title('Correlation: Cell viability', fontsize=15, weight='bold')

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
plotf('g-10','g-100','g-200','g-400')
plotd('g-510')
plott('g-510')
#Select the columns g-

g_cols = [col for col in a.columns if 'g-' in col]

#Filter the columns g-

genes=treated[g_cols]

#Plot heatmap

plt.figure(figsize=(15,7))

sns.heatmap(genes.corr(), cmap='coolwarm', alpha=0.9)

plt.title('Gene expression: Correlation', fontsize=15, weight='bold')

plt.show()
#Count unique values per column

cols = drugs_tr.columns.to_list() # specify the columns whose unique values you want here

uniques = {col: drugs_tr[col].nunique() for col in cols}

uniques=pd.DataFrame(uniques, index=[0]).T

uniques=uniques.rename(columns={0:'count'})

uniques= uniques.drop('sig_id', axis=0)





#Calculate the mean values

average=d.mean()

average=pd.DataFrame(average)

average=average.rename(columns={ 0: 'mean'})

average['percentage']= average['mean']*100

#Filter just the drugs with mean >0.01

average_filtered= average[average['mean'] > 0.01]

average_filtered= average_filtered.reset_index()

average_filtered= average_filtered.rename(columns={'index': 'drug'})
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))

#1 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((1,2),(0,0))

sns.countplot(uniques['count'], color='deepskyblue', alpha=0.75)

plt.title('Unique elements per drug [0,1]', fontsize=15, weight='bold')

#first row sec col

ax1 = plt.subplot2grid((1,2),(0,1))

sns.distplot(average['percentage'], color='orange', bins=40)

plt.title("The drugs mean distribution", fontsize=15, weight='bold')

plt.show()
plt.figure(figsize=(7,7))

plt.scatter(average_filtered['percentage'].sort_values(), average_filtered['drug'], color=sns.color_palette('Reds',22))

plt.title('Drugs with higher presence in train samples', weight='bold', fontsize=15)

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.xlabel('Percentage', fontsize=13)

plt.show()
ploth(drugs_tr)
#Correlation between drugs

corre= drugs_tr.corr()

#Unstack the dataframe

s = corre.unstack()

so = s.sort_values(kind="quicksort", ascending=False)

#Create new dataframe

so2= pd.DataFrame(so).reset_index()

so2= so2.rename(columns={0: 'correlation', 'level_0':'Drug 1', 'level_1': 'Drug2'})

#Filter out the coef 1 correlation between the same drugs

so2= so2[so2['correlation'] != 1]

#Drop pair duplicates

so2= so2.reset_index()

pos = [1,3,5,7,9]

so2= so2.drop(so2.index[pos])

so2= so2.round(decimals=4)

so3=so2.head(4)

#Show the first 10 high correlations

cm = sns.light_palette("Red", as_cmap=True)

s = so2.head().style.background_gradient(cmap=cm)

s
plt.figure(figsize=(8,10))

the_table =plt.table(cellText=so3.values,colWidths = [0.35]*len(so3.columns),

          rowLabels=so3.index,

          colLabels=so3.columns

          ,cellLoc = 'center', rowLoc = 'center',

          loc='left', edges='closed', bbox=(1,0, 1, 1)

         ,rowColours=sns.color_palette('Reds',10))

the_table.auto_set_font_size(False)

the_table.set_fontsize(10.5)

the_table.scale(2, 2)

plt.scatter(average_filtered['percentage'].sort_values(), average_filtered['drug'], color=sns.color_palette('Reds',22))

plt.title('Drugs with higher presence in train samples', weight='bold', fontsize=15)

plt.xlabel('Percentage', weight='bold')

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.axhline(y=13.5, color='black', linestyle='-')

plt.axhline(y=18.5, color='black', linestyle='-')

plt.show()
#Correlation between drugs

corre= adt.corr()

#Unstack the dataframe

s = corre.unstack()

so = s.sort_values(kind="quicksort", ascending=False)

#Create new dataframe

so2= pd.DataFrame(so).reset_index()

so2= so2.rename(columns={0: 'correlation', 'level_0':'Drug 1', 'level_1': 'Drug2'})

#Filter out the coef 1 correlation between the same drugs

so2= so2[so2['correlation'] != 1]

#Drop pair duplicates

so2= so2.reset_index()

pos = [1,3,5,7,9, 11,13,15,17,19,21,23, 25, 27, 29, 31,33,35, 37, 39, 41, 43, 45]

so2= so2.drop(so2.index[pos])

#so2= so2.round(decimals=4)

so3=so2.head()

#Show the first 10 high correlations

cm = sns.light_palette("Red", as_cmap=True)

s = so2.head(16).style.background_gradient(cmap=cm)

s
#High correlation adt 22 pairs

adt15= so2.head(22)

#Filter the drug names

adt_1=adt15['Drug 1'].values.tolist()

adt_2=adt15['Drug2'].values.tolist()

#Join the 2 lists

adt3= adt_1 + adt_2

#Keep unique elements and drop duplicates

adt4= list(dict.fromkeys(adt3))

#Filter out the selected drugs from the "all drugs treated" adt dataset

adt5= adt[adt4]
ploth(adt5)
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))

#1 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((1,2),(0,0))

sns.countplot(x='cp_type', data=b, palette='rainbow')

plt.title('Test: Control and treated samples', fontsize=15, weight='bold')

#first row sec col

ax1 = plt.subplot2grid((1,2),(0,1))

sns.countplot(x='cp_dose', data=b, palette='PiYG')

plt.title('Test: Treatment Doses: Low and High',weight='bold', fontsize=18)

plt.show()
plt.figure(figsize=(15,5))

sns.distplot( a['cp_time'], color='gold', bins=5)

plt.title("Test: Treatment duration ", fontsize=15, weight='bold')

plt.show()
#Filter out just the treated samples

treated2= b[b['cp_type']=='trt_cp']

treated_list2 = treated2['sig_id'].to_list()

full_tr= b[b['sig_id'].isin(treated_list2)]



#Select the columns c-

c_cols2 = [col for col in full_tr.columns if 'g-' in col]

#Filter the columns c-

cells2=treated2[c_cols2]

#Plot heatmap

plt.figure(figsize=(15,6))

sns.heatmap(cells2.corr(), cmap='coolwarm', alpha=0.9)

plt.title('Test: Correlation gene expression', fontsize=15, weight='bold')

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
#Select the columns c-

c_cols3 = [col for col in b.columns if 'c-' in col]

#Filter the columns c-

cells3=treated[c_cols3]

#Plot heatmap

plt.figure(figsize=(12,6))

sns.heatmap(cells3.corr(), cmap='coolwarm', alpha=0.9)

plt.title('Correlation: Cell viability', fontsize=15, weight='bold')

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()