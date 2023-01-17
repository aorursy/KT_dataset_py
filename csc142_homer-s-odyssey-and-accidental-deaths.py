%matplotlib inline



# numbers

import numpy as np

import pandas as pd



# stats

import statsmodels.api as sm

import scipy.stats as stats



# plots

import matplotlib.pyplot as plt

import seaborn as sns



# quiet

pd.options.mode.chained_assignment = None  # default='warn'



# utils

import os, re, io

from pprint import pprint
# Load death records

dr_df = pd.read_csv('../input/DeathRecords.csv', header=0)



## (Also useful to make a smaller version of 10k or 100k records, to work out details)

#dr_df = pd.read_csv('data/SmallDeathRecords.csv', header=0)



# Load manner of death keys

mod = pd.read_csv('../input/MannerOfDeath.csv',header=0,index_col=0)
# All ICD 10 code prefixes.

Icd10CodePrefixes_csv = """

Start Code,End Code,Description,Long Description

A00,B99,"Infectious/Parasitic Diseases","Certain infectious and parasitic diseases"

C00,D49,"Neoplasms","Neoplasms"

D50,D89,"Blood/Immuno Disorders","Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism"

E00,E89,"Endocrine/Nutritional/Metabolic","Endocrine, nutritional and metabolic diseases"

F01,F99,"Mental/Behavioral","Mental, Behavioral and Neurodevelopmental disorders"

G00,G99,"Nervous System","Diseases of the nervous system"

H00,H59,"Eye","Diseases of the eye and adnexa"

H60,H95,"Ear","Diseases of the ear and mastoid process"

I00,I99,"Circulatory","Diseases of the circulatory system"

J00,J99,"Respiratory","Diseases of the respiratory system"

K00,K95,"Digestive","Diseases of the digestive system"

L00,L99,"Skin","Diseases of the skin and subcutaneous tissue"

M00,M99,"Musculoskeletal","Diseases of the musculoskeletal system and connective tissue"

N00,N99,"Genitourinary","Diseases of the genitourinary system"

O00,O9A,"Pregancy/Childbirth","Pregnancy, childbirth and the puerperium"

P00,P96,"Perinatal","Certain conditions originating in the perinatal period"

Q00,Q99,"Malformations/Deformations","Congenital malformations, deformations and chromosomal abnormalities"

R00,R99,"Unclassified","Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"

S00,T88,"External Injury","Injury, poisoning and certain other consequences of external causes"

V00,Y99,"External Morbidity","External causes of morbidity"

Z00,Z99,"Health Services Contact","Factors influencing health status and contact with health services"

"""

icd10prefixes = pd.read_csv(io.StringIO(Icd10CodePrefixes_csv))





# Injury, poisoning and certain other consequences of external causes

Icd10CodePrefix_csv = """

Start Code,End Code,Description,Long Description

S00,T88,"External Injury","Injury, poisoning and certain other consequences of external causes"

"""

icd10prefix = pd.read_csv(io.StringIO(Icd10CodePrefix_csv))
# Load the full list of 12,000+ ICD 10 codes

icd10codes = pd.read_csv('../input/Icd10Code.csv',header=0)



# Load the record axis condition data set (small)

# (This takes a bit)

record_axis = pd.read_csv('../input/RecordAxisConditions.csv', header=0)
print(dr_df.head())
print(record_axis.head(10))
# Define a comparison function for ICD10 codes

def icd10_comes_before(a,b):

    a_prefix0 = a[0]

    b_prefix0 = b[0]

    if(a_prefix0 < b_prefix0):

        return True

    elif(b_prefix0 < a_prefix0):

        return False

    else:

        a_prefix1 = int(a[1:2])

        b_prefix1 = int(b[1:2])

        if(a_prefix1 < b_prefix1):

            return True

        elif(b_prefix1 < a_prefix1):

            return False

        else:

            # if both are equal, return True

            # (no ternary logic here.)

            return True
# Tag deaths in which external injury was a factor

# Iterate through dr_df dataframe 

# For each death id, cross-reference the ICD 10 codes in record_axis dataframe

# If death ICD 10 code is between S00 and T88, it's an external injury







def record_axis_crossref(x):

    

    print(snip)

    

#    snip = record_axis[ record_axis['DeathRecordId']==x ]

#    for (i,row) in snip.iterrows():

#        if( icd10_comes_before("S00",row['Icd10Code']) and icd10_comes_before(row['Icd10Code'],"T88") ):

#            return True

#    return False



# Since S00 is first S, and T88 is last T,

# just look for S and T prefixes. Way faster.

snip = record_axis['DeathRecordId'].loc[record_axis['Icd10Code'].map(lambda x : x[0] in ['S','T'])].unique()



# If we use a set, this will go way faster

snip = set(snip)
# We are looking at about 245k records... 

print(len(snip))

print(type(snip))

# Out of a total 2.6M

print(len(dr_df))

print(type(dr_df))
# Deaths involving an external injury

external_injury_df = dr_df[ dr_df['Id'].apply(lambda x : x in snip)]
total_deaths = dr_df.shape[0]

external_injury_deaths = external_injury_df.shape[0]



print("Ratio of death records tagged with external injury to total death records: \n%0.4f\n"%( 1.0*external_injury_deaths/total_deaths ))
counts = external_injury_df[['Id','Sex']].groupby('Sex').count()

print(counts)
print(1.0*counts.ix['M'].values[0] / counts.ix['F'].values[0])
fig = plt.figure(figsize=(6,4))

ax1 = fig.add_subplot(111)



sns.barplot(x=counts.index, y=counts.Id,

            ax=ax1)



ax1.set_title('Deaths Involving External Injury')

ax1.set_ylabel('Count')



plt.show()
# Revisit method of death from prior notebook

print(mod)
#print(external_injury_df.columns.tolist())
count = external_injury_df[['Id','MannerOfDeath']].groupby('MannerOfDeath').count()

print(count)
fig = plt.figure(figsize=(10,4))

ax = fig.add_subplot(111)



sns.barplot( x=count.index, y=count['Id'], 

            ax=ax,

            palette="Blues_d")



labelz = mod['Description']# placeofinjury['Description'][placeofinjury['Code']==j].values[0] for j in counts.index.tolist()]

ax.xaxis.set_ticklabels(labelz)



ax.set_xlabel('Manner of Death')

ax.set_ylabel('Count')



ax.set_title('Histogram: External Injuries by Manner of Death')



plt.show()
accident_df = external_injury_df[external_injury_df['MannerOfDeath'].eq(1)]

#accident_df.head()
#total_deaths = dr_df.shape[0]

accidental_injury_deaths = accident_df.shape[0]



print("Ratio of accidental deaths by external injury to total deaths:")

print("%0.4f"%( 1.0*accidental_injury_deaths/total_deaths ))
accident_counts = accident_df[['Id','Sex']].groupby('Sex').count()
print(1.0*accident_counts.ix['M'].values[0]/accident_counts.ix['F'].values[0])
fig = plt.figure(figsize=(6,4))

ax1 = fig.add_subplot(111)



sns.barplot(x=accident_counts.index, y=accident_counts.Id,

            ax=ax1)



ax1.set_title('Accidental Deaths Involving External Injury')

ax1.set_ylabel('Count')



plt.show()
accident_counts = accident_df[['Id','MaritalStatus']].groupby('MaritalStatus').count()



fig = plt.figure(figsize=(6,4))

ax1 = fig.add_subplot(111)



sns.barplot(x=accident_counts.index, y=accident_counts.Id,

            ax=ax1,

            palette="Blues_d")



ax1.set_title('Deaths Involving Accidental Injury')

ax1.set_ylabel('Count')



plt.show()
placeofinjury = pd.read_csv('../input/PlaceOfInjury.csv',header=0)
counts = accident_df[['Id','PlaceOfInjury']].groupby('PlaceOfInjury').count()



fig = plt.figure(figsize=(4,4))

ax1 = fig.add_subplot(111)



sns.barplot(y=counts.index, x=counts.Id,

            ax=ax1,

            palette="Set1",

            orient='h')



ax1.set_title('Deaths Involving Accidental Injury')



labelz = [placeofinjury['Description'][placeofinjury['Code']==j].values[0] for j in counts.index.tolist()]

ax1.yaxis.set_ticklabels(labelz)

#yt = ax1.yaxis.get_majorticklabels()



ax1.set_xlabel('Count')



plt.show()
accident_df = accident_df[accident_df['AgeType'].eq(1)]
# Group accidents involving external injury by age and by place of injury

grp = accident_df[['Id','Age','PlaceOfInjury']].groupby(['Age','PlaceOfInjury'])



# mat is now a dataframe with a two-level index, age and place of injury

mat = grp.count()
print(accident_df.shape)

print(pd.cut( accident_df['Age'], [i*7 for i in range(14)] ).shape)



accident_df['Age Bin'] = pd.cut( accident_df['Age'], [i*7 for i in range(15)] )



## Bingo!

#print accident_df['Age Bin']
# Now group by age bins, instead of age.

# Group accidents involving external injury by age and by place of injury

grp2 = accident_df[['Id','Age Bin','PlaceOfInjury']].groupby(['Age Bin','PlaceOfInjury'])



# mat is now a dataframe with a two-level index, age and place of injury

mat2 = grp2.count().replace(np.nan,0.0)
#print(mat2[:10])
print(placeofinjury)
## This is how to 

## swap index levels for mat2

#print(mat2.swaplevel('PlaceOfInjury','Age Bin').head())
fig = plt.figure(figsize=(10,4))

sns.heatmap(mat2.swaplevel('PlaceOfInjury','Age Bin').unstack(),

           cmap="RdPu")

ax = plt.gca()



ax.set_xlabel('')

ax.set_ylabel('')



xlabelz = mat2.index.levels[0].map(lambda x : str(x))

ax.xaxis.set_ticklabels(xlabelz)

xt = ax.xaxis.get_majorticklabels()

plt.setp(xt,rotation=320)



# This is absurdly difficult to work out. Thank you matplotlib.

ylabelz_lookupdf = placeofinjury[['Code','Description']].set_index('Code')

ylabelz = mat2.unstack().columns.map(lambda x : ylabelz_lookupdf.ix[ x[1] ].values[0] )

ylabelz = ylabelz[::-1]



ax.yaxis.set_ticklabels(ylabelz)

yt = ax.yaxis.get_majorticklabels()

plt.setp(yt,rotation=0)



plt.title('Heatmap: Location of Injury/Death vs Age')



plt.show()
# Now, we'll fix the order in which these rows occur.

# We're going to use a big hairy one-liner to get the y-axis 

# sorted in order of greatest to least cumulative sum (over all age brackets).

# That's the apply(sum,axis=1) call.

# Once we've computed the cumulative sum, we sort by that value,

# save the order of the y-axis labels, and use it to reindex 

# the matrix we visualized above.



# Fix the order so things occur in order.

ordered_yindex = mat2.swaplevel('PlaceOfInjury','Age Bin').unstack().apply(sum,axis=1).sort_values().index

print("Order of y-axis labels, from least to greatest cumulative sum:")

print(ordered_yindex)
mat3 = mat2.swaplevel('PlaceOfInjury','Age Bin').unstack()

mat4 = mat3.reindex(ordered_yindex)
fig = plt.figure(figsize=(12,4))

sns.heatmap(mat4,

           cmap="RdPu")

ax = plt.gca()



ax.set_xlabel('')

ax.set_ylabel('')



xlabelz = mat4.columns.map(lambda x : str(x[1]))

ax.xaxis.set_ticklabels(xlabelz)

xt = ax.xaxis.get_majorticklabels()

plt.setp(xt,rotation=320)





# This was absurdly difficult to work out... 

ylabelz_lookupdf = placeofinjury[['Code','Description']].set_index('Code')

ylabelz = mat4.index.map( lambda y : ylabelz_lookupdf.ix[ y ].values[0] )

ylabelz = ylabelz[::-1]



ax.yaxis.set_ticklabels(ylabelz)

yt = ax.yaxis.get_majorticklabels()

plt.setp(yt,rotation=0)



plt.title('Heatmap: Location of Injury/Death vs Age')



plt.show()
icd10prefixes.loc[19,:]
code = 'W13'

descr = icd10codes['Description'][icd10codes['Code']==code].values[0]

print("Code %s: %s"%(code,descr))



maskW = icd10codes['Code'].apply(lambda s : s[0]=='W')

print(icd10codes[maskW][:15])
maskY = icd10codes['Code'].apply(lambda s : s[0]=='Y')

print("Category Y details %d different circumstances of death."%( maskY.sum() ))

icd10codes[maskY][:50]
code = 'Y02'

descr = icd10codes['Description'][icd10codes['Code']==code].values[0]

print("Code %s: %s"%(code,descr))
print(record_axis.shape)

#record_axis.head()
print(accident_df.shape)

#accident_df.head()
merge = pd.merge( accident_df, record_axis, left_on='Id', right_on='DeathRecordId', how='inner', suffixes=('','_dupe'))

print( merge.shape )

#merge.head()
#pprint(merge.columns.tolist())
merge['Icd10Code_dupe'][:15]
print(icd10prefixes.ix[18:19])
maskW = icd10codes['Code'].apply(lambda s : s[0]=='W')

print(len(icd10codes[maskW]))

print("-"*20)

print(icd10codes[maskW][:30])

print("-"*20)

print(icd10codes[maskW][-20:-1])
def fmask(arg):

    start_code = "W00"

    end_code = "W50"

    if( icd10_comes_before(start_code,arg) and icd10_comes_before(arg,end_code) ):

        return True

    

    return False



merge_mask = merge['Icd10Code_dupe'].apply(fmask)



# Mask looks only at first 30 ICD 10 codes starting with W

short = merge[merge_mask]
#del short['Age Bin']

short.loc[:,('Age Bin')] = pd.cut( short['Age'], [i*7 for i in range(15)] )
W30count = short[['Id','Icd10Code_dupe','Age Bin']].groupby(['Icd10Code_dupe','Age Bin']).count().replace(np.nan,0.0)

W30count = W30count.unstack()
print(W30count.head())
fig = plt.figure(figsize=(10,9))

sns.heatmap(W30count,

           cmap="RdPu", vmin=0 )

ax = plt.gca()



ax.set_xlabel('')

ax.set_ylabel('')



xlabelz = W30count.columns.map(lambda x : str(x[1]))

ax.xaxis.set_ticklabels(xlabelz)

xt = ax.xaxis.get_majorticklabels()

plt.setp(xt,rotation=290)





# Many Bothans died to work out these lines of code.

ylabelz_lookupdf = icd10codes[['Code','Description']].set_index('Code')

ylabelz = W30count.index.map( lambda y : ylabelz_lookupdf.ix[ y ].values[0] )

ylabelz = ylabelz[::-1]



ax.yaxis.set_ticklabels(ylabelz)

yt = ax.yaxis.get_majorticklabels()

plt.setp(yt,rotation=0)



plt.title('Heatmap: Category of Death vs. Age')



plt.show()


#########

# Now stuff that into a function:



def heat_map_icd10_age(start_code, end_code,myfigsize=(0,0),myvmax=0):



    def f_mask(arg):

        # start_code and end_code are 

        if( icd10_comes_before(start_code,arg) and icd10_comes_before(arg,end_code) ):

            return True



        return False



    merge_mask = merge['Icd10Code_dupe'].apply(f_mask)

    

    if( merge_mask.sum() < 1 ):

        raise Exception("Error: no death records were matched in the code range specified.")



    # Mask looks only at first 30 ICD 10 codes starting with W

    short = merge[merge_mask]

    

    short.loc[:,('Age Bin')] = pd.cut( short['Age'], [i*7 for i in range(15)] )



    # Note: we have already filtered the data so AgeType==1 (years)

    count = short[['Id','Icd10Code_dupe','Age Bin']].groupby(['Icd10Code_dupe','Age Bin']).count().replace(np.nan,0.0)

    count = count.unstack()



    if(myfigsize==(0,0)):

        myfigsize = (12,10)

    fig = plt.figure(figsize=myfigsize)

    

    if(myvmax==0):

        myvmax = count.max().max()

        

    sns.heatmap(count,

               cmap="RdPu", vmin=0, vmax = myvmax)

    ax = plt.gca()



    ax.set_xlabel('')

    ax.set_ylabel('')



    xlabelz = count.columns.map(lambda x : str(x[1]))

    ax.xaxis.set_ticklabels(xlabelz)

    xt = ax.xaxis.get_majorticklabels()

    plt.setp(xt,rotation=290)





    ylabelz_lookupdf = icd10codes[['Code','Description']].set_index('Code')

    ylabelz = count.index.map( lambda y : ylabelz_lookupdf.ix[ y ].values[0] )

    ylabelz = ylabelz[::-1]



    ax.yaxis.set_ticklabels(ylabelz)

    yt = ax.yaxis.get_majorticklabels()

    plt.setp(yt,rotation=0)



    plt.title('Heatmap: Accidental Deaths, ICD 10 Codes vs. Age')



    plt.show()
heat_map_icd10_age("W00","W15",myfigsize=(10,8))
heat_map_icd10_age("W00","W15",myfigsize=(10,8),myvmax=500)
heat_map_icd10_age("W20","W25",myfigsize=(10,4))
heat_map_icd10_age("W30","W50",myfigsize=(10,6))
heat_map_icd10_age("W60","W75",myfigsize=(10,4))
heat_map_icd10_age("W80","W99",myfigsize=(10,2))
heat_map_icd10_age("Y00","Y49",myfigsize=(8,6))
heat_map_icd10_age("Y50","Y75",myfigsize=(10,8))
maskT = icd10codes['Code'].apply(lambda s : s[0]=='T')

print("Category T details %d different circumstances of death."%( maskT.sum() ))

icd10codes[maskT][:15]
heat_map_icd10_age("T00","T09",myfigsize=(10,8))
heat_map_icd10_age("T00","T09",myfigsize=(10,8),myvmax=100)
heat_map_icd10_age("T10","T15",myfigsize=(10,12))
heat_map_icd10_age("T10","T15",myfigsize=(10,12),myvmax=400)
heat_map_icd10_age("T10","T15",myfigsize=(10,12),myvmax=10)
heat_map_icd10_age("T20","T29",myfigsize=(10,8))
heat_map_icd10_age("T30","T39",myfigsize=(10,8))
heat_map_icd10_age("T30","T39",myfigsize=(10,8),myvmax=25)
heat_map_icd10_age("F00","F39",myfigsize=(14,16))
heat_map_icd10_age("F00","F39",myfigsize=(14,16),myvmax=200)
heat_map_icd10_age("F40","F69",myfigsize=(10,6))
heat_map_icd10_age("F70","F99",myfigsize=(10,3))