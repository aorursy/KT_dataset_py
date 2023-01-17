import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
sns.set_style('whitegrid')
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
inpath = "../input/224_v2/224_v2/"
print(os.listdir("../input/224_v2/224_v2"))
# reading the data
data = pd.read_csv("../input/224_v2/224_v2/Data_Entry_2017.csv")
data.head()
# Dimensions
print(data.shape)
# Columns Titles
print(data.columns)
# Statistics
data.describe()
#drop unused columns
data = data[['Image Index','Finding Labels','Follow-up #','Patient ID','Patient Age','Patient Gender']]
data.head()
# removing the rows which have patient_age >100
total = len(data)
print('No. of rows before removing rows having age >100 : ',len(data))
data = data[data['Patient Age']<100]
print('No. of rows after removing rows having age >100 : ',len(data))
print('No. of datapoints having age > 100 : ',total-len(data))
print(data.shape)
data.head()
# rows having no. of disease (Adding a new column)
data['Labels_Count'] = data['Finding Labels'].apply(lambda text: len(text.split('|')) if(text != 'No Finding') else 0)
data.head()
label_counts = data['Finding Labels'].value_counts()[:15]
label_counts
# Space
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
# Blue Bars and separation 0.5
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)

# Set Position of Labels
ax1.set_xticks(np.arange(len(label_counts))+0.5)

# Labels with a rotation 90° 
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)
#plt.figure(figsize=(20,15))
sns.FacetGrid(data,hue='Patient Gender',size=5).map(sns.distplot,'Patient Age').add_legend()
plt.show()
g = sns.factorplot(x="Patient Age", col="Patient Gender",data=data, kind="count",size=10, aspect=0.8,palette="GnBu_d");
g.set_xticklabels(np.arange(0,100));
g.set_xticklabels(step=10);
g.fig.suptitle('Age distribution by sex',fontsize=22);
g.fig.subplots_adjust(top=.9)
# To create the empty 14 plots
f, axarr = plt.subplots(7, 2, sharex=True,figsize=(15, 20))

# List of Deseases
pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis',
                  'Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']

# Take all the Deseases without 'No finding'
df = data[data['Finding Labels'] != 'No Finding']

# Set some needed values
i=0
j=0
x=np.arange(0,100,10)
print(x)

for pathology in pathology_list :
    index = []
    for k in range(len(df)):
        # organizar por enfermedad
        if pathology in df.iloc[k]['Finding Labels']:
            index.append(k)
    
    #add the gender
    g=sns.countplot(x='Patient Age', hue="Patient Gender",data=df.iloc[index], ax=axarr[i, j])
    
    # Distribute to each label
    
    axarr[i, j].set_title(pathology)
    g.set_xlim(0,90)
    
    g.set_xticks(x)
    g.set_xticklabels(x)
    j=(j+1)%2
    if j==0:
        i=(i+1)%7

f.subplots_adjust(hspace=0.3)
# Generate the label for each pathology find in a single image
for pathology in pathology_list :
    data[pathology] = data['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)
print(data.columns)
data.head()
# Instance Space
plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(8,1)

#Add Axes
# Space Used to show Each pathology separate by gender
ax1 = plt.subplot(gs[:7, :])
# Space Used to show all the pathology separate by gender
ax2 = plt.subplot(gs[7, :])

# Separate each pathology on the image ALL EXCEPT NO FINDING
data1 = pd.melt(data,
             id_vars=['Patient Gender'],
             value_vars = list(pathology_list),
             var_name = 'Category',
             value_name = 'Count')
data1 = data1.loc[data1.Count>0]

# Bars for the first Space ALL EXCEPT NO FINDING
g=sns.countplot(y='Category',
                hue='Patient Gender',
                data=data1, 
                ax=ax1, 
                order = data1['Category'].value_counts().index)

# Title for the first Space
ax1.set( ylabel="",xlabel="")
ax1.legend(fontsize=20)
ax1.set_title('X Ray partition (total number = 121120)',fontsize=18);

# Adding The no finding Label *** Important ***
data['Nothing']=data['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)
data.head()

# Just the NO FINDING pathology
data2 = pd.melt(data,
             id_vars=['Patient Gender'],
             value_vars = list(['Nothing']),
             var_name = 'Category',
             value_name = 'Count')
data2 = data2.loc[data2.Count>0]

# Bars for the second space NO FINDING
g=sns.countplot(y='Category',hue='Patient Gender',data=data2,ax=ax2)

# Label of botton
ax2.set( ylabel="",xlabel="Number of decease")
ax2.legend('')

# Size
plt.subplots_adjust(hspace=.5)


# Set Two Spaces
f, (ax1,ax2) = plt.subplots( 2, figsize=(15, 10))

# will show patients with up to 15 Follow-up images
df = data[data['Follow-up #']<15]
print(df.shape)

# Bars for the first space
g = sns.countplot(x='Follow-up #',data=df,palette="GnBu_d",ax=ax1);
# Title for the firts space
ax1.set_title('Follow-up distribution');

# will show patients with more than 14 Follow-up images
df = data[data['Follow-up #']>14]
print(df.shape)

# Bars for the second space
g = sns.countplot(x='Follow-up #',data=df,palette="GnBu_d",ax=ax2);

# Limit
x=np.arange(15,100,10)
g.set_ylim(15,450)
g.set_xlim(15,100)
g.set_xticks(x)
g.set_xticklabels(x)

f.subplots_adjust(top=1)

# Group
df=data.groupby('Finding Labels').count().sort_values('Patient ID',ascending=False)
df.head()
# Multi pathology
df1=df[['|' in index for index in df.index]].copy()
df1.head()
# Single Pathology
df2=df[['|' not in index for index in df.index]]
df2.head()
# Single Pathology without NO FINDING
df2=df2[['No Finding' not in index for index in df2.index]]
df2.head()
# Set FINDING LABELS AS index
df2['Finding Labels']=df2.index.values
df2.head()
# Set FINDING LABELS AS index
df1['Finding Labels']=df1.index.values
df1.head()
# A Single Space
f, ax = plt.subplots(sharex=True,figsize=(15, 10))

sns.set_color_codes("pastel")

# Multi decease
g=sns.countplot(y='Category',
                data=data1, 
                ax=ax, 
                order = data1['Category'].value_counts().index,
                color='b',
                label="Multiple Pathologies")

sns.set_color_codes("muted")

# Single decease
g=sns.barplot(x='Patient ID',
              y='Finding Labels',
              data=df2, 
              ax=ax, 
              color="b",
              label="Simple Pathology")

# Conventions
ax.legend(ncol=2, loc="center right", frameon=True,fontsize=20)

# X label
ax.set( ylabel="",xlabel="Number of decease")
       
# Title
ax.set_title("Comparaison between simple or multiple decease",fontsize=20)      
sns.despine(left=True)
#we just keep groups of pathologies which appear more than 30 times
df3=df1.loc[df1['Patient ID']>30,['Patient ID','Finding Labels']]
df3.head()
for pathology in pathology_list:
    df3[pathology]=df3.apply(lambda x: x['Patient ID'] if pathology in x['Finding Labels'] else 0, axis=1)

df3.head()
#'Hernia' has not enough values to figure here
df4=df3[df3['Hernia']>0]
df4.head()
#remove 'Hernia' from list
pat_list=[elem for elem in pathology_list if 'Hernia' not in elem]
pat_list
# 13 Spaces
# Each decease with all combinations
f, axarr = plt.subplots(13, sharex=True,figsize=(10, 140))
i=0
for pathology in pat_list :
    df4=df3[df3[pathology]>0]
    if df4.size>0:  #'Hernia' has not enough values to figure here
        # Pie
        axarr[i].pie(df4[pathology],labels=df4['Finding Labels'], autopct='%1.1f%%')
        # Title
        axarr[i].set_title('main desease : '+pathology,fontsize=14)   
        i +=1
