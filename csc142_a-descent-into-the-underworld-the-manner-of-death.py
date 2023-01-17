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



# utils

import os, re, io

from pprint import pprint
os.listdir('../input/')
# I made a SmallDeathRecords.csv 

# consisting of 10,000 and later 100,000 death records.

# This made loading, testing, and developing

# a lot faster.



#df = pd.read_csv('data/SmallDeathRecords.csv', header=0)



# Now that we've worked out the details, 

# bite the bullet and load all 2.6M records

df = pd.read_csv('../input/DeathRecords.csv', header=0)



df.head()
print("Number of columns: %d"%(len(df.columns)))
pprint(df.columns.sort_values().tolist())
mod = pd.read_csv('../input/MannerOfDeath.csv',header=0,index_col=0)

print(mod)
# Count up the number of deaths of each type:

grp = df[['Id','MannerOfDeath']].groupby('MannerOfDeath')

count_data = grp.count()

print("There are %d different manners of death."%(len(count_data)))



sorted_counts = pd.merge(count_data,mod,left_index=True,right_index=True).sort_values('Id',ascending=False)



print("Counts:")

print(sorted_counts)
fig = plt.figure(figsize=(6,8))

ax1 = fig.add_subplot(211)

ax2 = fig.add_subplot(212)



# Bar plot #1: 

# Number of deaths by manner of death



sns.barplot(x=count_data.index, y=count_data.Id, ax=ax1, palette="Blues_d")



ax1.set_ylabel('Count')

ax1.set_title('Histogram: Manner of Death')



# relabel the axis

ax1.xaxis.set_ticklabels(mod.values)



# rotate the frickin' axis

xt = ax1.xaxis.get_majorticklabels()

plt.setp(xt,rotation=320)







# Bar plot #2:

# Log(Number of deaths) by manner of death



sns.barplot(x=count_data.index, y=count_data.Id.apply(lambda x : np.log10(x)), ax=ax2, palette="Blues_d")



ax2.set_ylabel('Log(Count)')

ax2.set_title('(Log) Histogram: Manner of Death')



# fix labels

ax2.xaxis.set_ticklabels(mod.values)

xt = ax2.xaxis.get_majorticklabels()

plt.setp(xt,rotation=320)





# Space out plots

plt.subplots_adjust(hspace=0.80)





plt.show()
# Load the record axis condition data set

# (Another case where creating a smaller version with only 10,000 records 

# helps save in development and loading time)

record_axis = pd.read_csv('../input/RecordAxisConditions.csv', header=0)



# Load the list of ICD10 codes

icd10 = pd.read_csv('../input/Icd10Code.csv', header=0)
record_axis.head()
# Count number of deaths using an A-code ICD10 code (refers to bacerial/viral deaths)

numAdeaths = record_axis['Icd10Code'].apply( lambda x : 'A' in x).sum()

print("Number of individual deaths due to ICD 10 codes prefixed with A: %d"%(numAdeaths))
# List unique A-codes

typeAmatches = record_axis['Icd10Code'][record_axis['Icd10Code'].apply( lambda x : 'A' in x)].unique()



print( "Number of unique A code matches: %d"%(len(typeAmatches)) )

print("")

print( typeAmatches )
print(len(icd10))
for d in typeAmatches:

    result = icd10[['Code','Description']][icd10['Code'].eq(d)]

    code = result['Code'].values[0]

    descr = result['Description'].values[0]

    print("%s : %s"%( code, descr ))
import string

print(string.ascii_uppercase)
def code_count(df,letter_code):

    # Count number of deaths using an A-code ICD10 code (refers to bacerial/viral deaths)

    numAdeaths = df['Icd10Code'].apply( lambda x : letter_code in x).sum()

    print("Number of individual deaths due to %s codes: %d"%(letter_code,numAdeaths))



# Count each occurrence of each lettered category

for code_prefix in string.ascii_uppercase:

    code_count(record_axis, code_prefix)
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

icd10prefixes.head()
record_axis.head()


# Define the function

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



# Test the function

print("Testing... these should all return True:")



a = "I250"

b = "I99"

print(icd10_comes_before(a,b))



a = "I00"

b = "I250"

print(icd10_comes_before(a,b))



a = "A00"

b = "B4702"

print(icd10_comes_before(a,b))



a = "B4702"

b = "B50"

print(icd10_comes_before(a,b))

test = "I5502"



print("Testing %s lookup, should return shortcode for \"Diseases of the circulatory system\":"%(test))



for (i,row) in icd10prefixes.iterrows():

    istart = row['Start Code']

    iend = row['End Code']

    descr = row['Description']

        

    if( icd10_comes_before(istart,test) and icd10_comes_before(test,iend) ):

        print(descr)
def label_icd10(code):



    result = (0,None)

    

    for (i,row) in icd10prefixes.iterrows():

        istart = row['Start Code']

        iend = row['End Code']

        descr = row['Description']

        

        if( icd10_comes_before(istart,code) and icd10_comes_before(code,iend) ):

            result = (i,descr)

        

    return result

# Get rid of the letter category, since we're replacing it.

try:

    del record_axis['Letters']

except KeyError:

    pass
small_record_axis = record_axis[:10000]
#

# This takes a while!

#

small_record_axis.loc[:,('Icd10Prefix')] = small_record_axis['Icd10Code'].apply(lambda x : label_icd10(x)[0])
# Need to speed up the above function.

# This took **several** minutes.

#

# Probably create a FirstThree column (first letter and first two numbers).

# Then a backwards "last one to match" to categorize it.

# Then bin it/use pd.join()
print(small_record_axis.head())
grp = small_record_axis.groupby('Icd10Prefix')

counts = grp['Id'].count()#.sort_values(ascending=False)
fig = plt.figure(figsize=(8,10))

ax1 = fig.add_subplot(211)

ax2 = fig.add_subplot(212)



sns.barplot(x=counts.values, 

            y=counts.index, 

            ax=ax1, 

            palette="gist_earth",

            orient='h')



ax1.set_xlabel('Number of Occurrences')

ax1.set_ylabel('')

ax1.set_title('Histogram: ICD10 Prefixes')



# fix labels

ylabels = counts.index.map(lambda x : icd10prefixes.ix[x,2])

ax1.yaxis.set_ticklabels(ylabels)





sns.barplot(y=counts.index, x=counts.apply(lambda x : np.log10(x)).values, 

            ax=ax2, palette="gist_earth",

            orient='h')



ax2.set_xlabel('Log( Number of Occurrences )')

ax2.set_ylabel('')

ax2.set_title('(Log) Histogram: ICD10 Prefixes')



# fix labels

ylabels = counts.index.map(lambda x : icd10prefixes.ix[x,2])

ax2.yaxis.set_ticklabels(ylabels)







# space out

plt.subplots_adjust(hspace=0.4)



plt.show()
record_axis.head()
df.head()
# record axis conditions labels

ralabs = ['DeathRecordId','Icd10Code','Icd10Prefix']



# Death record dataframe labels

drlabs = ['Id','MannerOfDeath']



merged = pd.merge(small_record_axis[ralabs], df[drlabs], left_on='DeathRecordId', right_on='Id')

merged.head()
grp = merged.groupby(['Icd10Prefix','MannerOfDeath'])

counts = grp['Id'].count()
fig = plt.figure(figsize=(4,16))

ax1 = fig.add_subplot(211)

ax2 = fig.add_subplot(212)





sns.barplot(y=counts.index.map(lambda x : x[0]), 

            x=counts.values,

            hue=counts.index.map(lambda x : x[1]),

            palette="Set2",

            ax=ax1,

            orient='h')



sns.barplot(y=counts.index.map(lambda x : x[0]), 

            x=counts.apply(lambda x : np.log10(x)).values, 

            hue=counts.index.map(lambda x : x[1]),

            palette="Set2",

            ax=ax2,

            orient='h')





# fix y-axis labels

ylabels = counts.index.levels[0].map(lambda j : icd10prefixes.ix[j,'Description'])

ax1.yaxis.set_ticklabels(ylabels)

ax2.yaxis.set_ticklabels(ylabels)



#print mod



# fix axis 1 legend 

handles, labels = ax1.get_legend_handles_labels()

for handle in handles:

    lookup = int(handle.get_label())

    new_label = mod.ix[lookup].values[0]

    handle.set_label(new_label)

ax1.legend(loc='best')

ax1.set_xlabel('Number of Occurrences')

ax1.set_title('Breakdown of ICD 10 Prefixes\nby Manner of Death')





# fix axis 2 legend

ax2.set_xlim([0,4.5])

handles, labels = ax2.get_legend_handles_labels()

for handle in handles:

    lookup = int(handle.get_label())

    new_label = mod.ix[lookup].values[0]

    handle.set_label(new_label)

ax2.legend(loc='right')

ax2.set_xlabel('Log( Number of Occurrences )')

ax2.set_title('(Log) Breakdown of ICD 10 Prefixes\nby Manner of Death')



plt.show()
