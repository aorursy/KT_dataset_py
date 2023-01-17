import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

############################################

### DATASETS:



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



#Treated drugs:

nonscored= c[c['sig_id'].isin(treated_list)]

scored= d[d['sig_id'].isin(treated_list)]



#adt= All Drugs Treated

adt= all_drugs[all_drugs['sig_id'].isin(treated_list)]



#Select the columns c-

c_cols = [col for col in a.columns if 'c-' in col]

#Filter the columns c-

cells=treated[c_cols]



#Select the columns g-

g_cols = [col for col in a.columns if 'g-' in col]

#Filter the columns g-

genes=treated[g_cols]







#####################################################

#### HELPER FUNCTIONS



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

    plt.title(f'Treated with drugs: {f1}',weight='bold', fontsize=18)

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

    sns.distplot(a[f1], color='crimson')

    plt.title(f1,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    #first row sec col

    ax1 = plt.subplot2grid((2,2), (0, 1))

    sns.distplot(a[f2], color='gainsboro')

    plt.title(f2,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    #Second row first column

    ax1 = plt.subplot2grid((2,2), (1, 0))

    sns.distplot(a[f3], color='deepskyblue')

    plt.title(f3,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    #second row second column

    ax1 = plt.subplot2grid((2,2), (1, 1))

    sns.distplot(a[f4], color='black')

    plt.title(f4,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')



    return plt.show()



def ploth(data, w=15, h=9):

    plt.figure(figsize=(w,h))

    sns.heatmap(data.corr(), cmap='hot')

    plt.title('Correlation between targets', fontsize=18, weight='bold')

    return plt.show()



# corrs function: Show dataframe of high correlation between features

def corrs(data, col1='Gene 1', col2='Gene 2',rows=5,thresh=0.8, pos=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53]):

        #Correlation between genes

        corre= data.corr()

         #Unstack the dataframe

        s = corre.unstack()

        so = s.sort_values(kind="quicksort", ascending=False)

        #Create new dataframe

        so2= pd.DataFrame(so).reset_index()

        so2= so2.rename(columns={0: 'correlation', 'level_0':col1, 'level_1': col2})

        #Filter out the coef 1 correlation between the same drugs

        so2= so2[so2['correlation'] != 1]

        #Drop pair duplicates

        so2= so2.reset_index()

        pos = pos

        so3= so2.drop(so2.index[pos])

        so3= so3.drop('index', axis=1)

        #Show the first 10 high correlations

        cm = sns.light_palette("Red", as_cmap=True)

        s = so3.head(rows).style.background_gradient(cmap=cm)

        print(f"{len(so2[so2['correlation']>thresh])/2} {col1} pairs have +{thresh} correlation.")

        return s



def plotgene(data):

    sns.set_style('whitegrid')    

    data.plot.bar(color=sns.color_palette('Reds',885), edgecolor='black')

    set_size(13,5)

    #plt.xticks(rotation=90)

    plt.tick_params(

        axis='x',          # changes apply to the x-axis

        which='both',      # both major and minor ticks are affected

        bottom=False,      # ticks along the bottom edge are off

        top=False,         # ticks along the top edge are off

        labelbottom=False) # labels along the bottom edge are off

    plt.ylabel('Gene expression values', weight='bold')

    plt.title('Mean gene expression of the 772 genes', fontsize=15)

    return plt.show()



def mean(row):

   

    return row.mean()



def set_size(w,h, ax=None):

    """ w, h: width, height in inches """

    if not ax: ax=plt.gca()

    l = ax.figure.subplotpars.left

    r = ax.figure.subplotpars.right

    t = ax.figure.subplotpars.top

    b = ax.figure.subplotpars.bottom

    figw = float(w)/(r-l)

    figh = float(h)/(t-b)

    ax.figure.set_size_inches(figw, figh)
a.head()
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))

#1 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((1,2),(0,0))

sns.countplot(x='cp_type', data=a, palette='rainbow', alpha=0.75)

plt.title('Train: Control and treated samples', fontsize=15, weight='bold')

#first row sec col

ax1 = plt.subplot2grid((1,2),(0,1))

sns.countplot(x='cp_dose', data=a, palette='Purples', alpha=0.75)

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
corrs(cells, 'Cell', 'Cell 2', rows=7)
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
corrs(genes, 'Gene', 'Gene 2')
#Correlation between drugs

corre= genes.corr()

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

so2= so2.sort_values(by=['correlation'])

pos = [1,3,5,7,9,11,13,15,17,19,21]

so2= so2.drop(so2.index[pos])

so2= so2.round(decimals=4)

so2=so2.drop('index', axis=1)

so3=so2.head(4)

#Show the first 10 high correlations

cm = sns.light_palette("Red", as_cmap=True)

s = so2.head().style.background_gradient(cmap=cm)

s
#Transpose the dataframe

genesT=genes.T

#Calculate the mean of each g_xxx feature

genesT['mean'] = genesT.apply (lambda row: mean(row), axis=1)

#Plot the mean values

genesTm=genesT.reset_index()

genesTm=genesTm[['index', 'mean']]

plotgene(genesTm)
#Filter out just the treated samples

scored= d[d['sig_id'].isin(treated_list)]



#Count unique values per column

cols = drugs_tr.columns.to_list() # specify the columns whose unique values you want here

uniques = {col: drugs_tr[col].nunique() for col in cols}

uniques=pd.DataFrame(uniques, index=[0]).T

uniques=uniques.rename(columns={0:'count'})

uniques= uniques.drop('sig_id', axis=0)





#Calculate the mean values

average=scored.mean()

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

plt.title('Unique elements per target [0,1]', fontsize=15, weight='bold')

#first row sec col

ax1 = plt.subplot2grid((1,2),(0,1))

sns.distplot(average['percentage'], color='orange', bins=20)

plt.title("The targets mean distribution", fontsize=15, weight='bold')

plt.show()
plt.figure(figsize=(7,7))

average_filtered.sort_values('percentage', inplace=True) 

plt.scatter(average_filtered['percentage'], average_filtered['drug'], color=sns.color_palette('Reds',len(average_filtered)))

plt.title('Targets with higher presence in train samples', weight='bold', fontsize=15)

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.xlabel('Percentage', fontsize=13)

plt.show()
inhibitors = [col for col in d.columns if 'inhibitor' in col]

activators = [col for col in d.columns if 'activator' in col]

antagonists = [col for col in d.columns if 'antagonist' in col]

agonists = [col for col in d.columns if 'agonist' in col]

modulators = [col for col in d.columns if 'modulator' in col]

receptors = [col for col in d.columns if 'receptor' in col]

receptors_ago = [col for col in d.columns if 'receptor_agonist' in col]

receptors_anta = [col for col in d.columns if 'receptor_antagonist' in col]





labelss= {'Drugs': ['inhibitors', 'activators', 'antagonists', 'agonists', 'receptors', 'receptors_ago', 'receptors_anta'],

          'Count':[112,5,32,60, 53, 24, 26]}





labels= pd.DataFrame(labelss)

labels=labels.sort_values(by=['Count'])

plt.figure(figsize=(15,5))

plt.bar(labels['Drugs'], labels['Count'], color=sns.color_palette('Reds',len(labels)))

plt.xticks(weight='bold')

plt.title('Target types', weight='bold', fontsize=15)

plt.show()
ploth(drugs_tr)
#Correlation between drugs

corre= drugs_tr.corr()

#Unstack the dataframe

s = corre.unstack()

so = s.sort_values(kind="quicksort", ascending=False)

#Create new dataframe

so2= pd.DataFrame(so).reset_index()

so2= so2.rename(columns={0: 'correlation', 'level_0':'Target 1', 'level_1': 'Target 2'})

#Filter out the coef 1 correlation between the same drugs

so2= so2[so2['correlation'] != 1]

#Drop pair duplicates

so2= so2.reset_index()

pos = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35]

so2= so2.drop(so2.index[pos])

so2= so2.round(decimals=4)

so2=so2.drop('index', axis=1)

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

average_filtered.sort_values('percentage', inplace=True) 

plt.scatter(average_filtered['percentage'], average_filtered['drug'], color=sns.color_palette('Reds',len(average_filtered)))

plt.title('Targets with higher presence in train samples', weight='bold', fontsize=15)

plt.xlabel('Percentage', weight='bold')

plt.xticks(weight='bold')

plt.yticks(weight='bold')

plt.show()
#Extract unique elements per column

cols2 = nonscored.columns.to_list() # specify the columns whose unique values you want here

uniques2 = {col: nonscored[col].nunique() for col in cols2}

uniques2=pd.DataFrame(uniques2, index=[0]).T

uniques2=uniques2.rename(columns={0:'count'})

uniques2= uniques2.drop('sig_id', axis=0)



#############################

### PLOT

plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))

#1 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((1,2),(0,0))

sns.countplot(uniques2['count'], palette='Blues', alpha=0.75)

plt.title('Nonscored: Unique elements per target [0,1]', fontsize=13, weight='bold')

#first row sec col

ax1 = plt.subplot2grid((1,2),(0,1))

sns.countplot(uniques['count'], color='cyan', alpha=0.75)

plt.title('Scored: Unique elements per target [0,1]', fontsize=13, weight='bold')

plt.show()
print(f"{len(uniques2[uniques2['count']==1])} targets without ANY mechanism of action in the nonscored dataset")
#Filter out just the treated samples

#Calculate the mean values

average2=nonscored.mean()

average2=pd.DataFrame(average2)

average2=average2.rename(columns={ 0: 'mean'})

average2['percentage']= average2['mean']*100

#Filter just the drugs with mean >0.01

average_filtered2= average2[average2['mean'] > 0.01]

average_filtered2= average_filtered2.reset_index()

average_filtered2= average_filtered2.rename(columns={'index': 'drug'})



#####################

#Plot the percentage of MoAs

plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))

#1 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((1,2),(0,0))

sns.distplot(average2['percentage'], color='blue', bins=20)

plt.title('Percentage of the nonscored MoAs in the samples',weight='bold', fontsize=13)

#first row sec col

ax1 = plt.subplot2grid((1,2),(0,1))

sns.distplot(average['percentage'], color='gold', bins=20)

plt.title('Percentage of the scored MoAs in the samples',weight='bold', fontsize=13)

plt.show()
corrs(adt, 'target', 'target 2', 15, thresh= 0.7)
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

sns.countplot(x='cp_type', data=b, palette='rainbow', alpha=0.75)

plt.title('Test: Control and treated samples', fontsize=15, weight='bold')

#first row sec col

ax1 = plt.subplot2grid((1,2),(0,1))

sns.countplot(x='cp_dose', data=b, palette='rainbow', alpha=0.75)

plt.title('Test: Treatment Doses: Low and High',weight='bold', fontsize=18)

plt.show()
plt.figure(figsize=(13,3))

sns.distplot( b['cp_time'], color='gold', bins=5)

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
#Correlation between drugs

corre= cells2.corr()

#Unstack the dataframe

s = corre.unstack()

so = s.sort_values(kind="quicksort", ascending=False)

#Create new dataframe

so2= pd.DataFrame(so).reset_index()

so2= so2.rename(columns={0: 'correlation', 'level_0':'Gene 1', 'level_1': 'Gene 2'})

#Filter out the coef 1 correlation between the same drugs

so2= so2[so2['correlation'] != 1]

#Drop pair duplicates

so2= so2.reset_index()

#so2= so2.sort_values(by=['correlation'])

pos = [1,3,5,7,9,11,13,15,17,19,21]

so2= so2.drop(so2.index[pos])

so2= so2.round(decimals=4)

so2=so2.drop('index', axis=1)

so4=so2.head(10)

cm = sns.light_palette("Red", as_cmap=True)

s = so2.head(10).style.background_gradient(cmap=cm)

s
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
def plotf2(f1, f2, f3, f4):

    plt.style.use('seaborn')

    sns.set_style('whitegrid')



    fig= plt.figure(figsize=(15,10))

    #2 rows 2 cols

    #first row, first col

    ax1 = plt.subplot2grid((2,2),(0,0))

    sns.distplot(cells3[f1], color='crimson')

    plt.title(f1,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    #first row sec col

    ax1 = plt.subplot2grid((2,2), (0, 1))

    sns.distplot(cells3[f2], color='gainsboro')

    plt.title(f2,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    #Second row first column

    ax1 = plt.subplot2grid((2,2), (1, 0))

    sns.distplot(cells3[f3], color='darkred')

    plt.title(f3,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    #second row second column

    ax1 = plt.subplot2grid((2,2), (1, 1))

    sns.distplot(cells3[f4], color='black')

    plt.title(f4,weight='bold', fontsize=18)

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')



    return plt.show()
plotf2('c-5', 'c-12', 'c-50', 'c-76')