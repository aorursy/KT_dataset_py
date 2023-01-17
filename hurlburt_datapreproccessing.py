import numpy as np 
import pandas as pd 
import pylab as plt
import matplotlib as mpl
import nltk
df = pd.read_csv( "../input/Salaries.csv", low_memory=False)
df[df.EmployeeName.isin(['ZULA JONES','Zula Jones','Zula M Jones'])]
years = pd.unique(df.Year)
for y in years:
    print("YEAR: ",y)
    tempdf = df[df.Year == y].copy()
    tempdf.sort_values('EmployeeName',inplace = True)
    print(tempdf[['EmployeeName','JobTitle']].head())
    print('\n')
df['ename'] = df.EmployeeName.str.replace('.',' ').str.lower().str.split().str.join(' ')
df['jtitle'] = df.JobTitle.str.replace('.',' ').str.replace(',',', ').str.lower().str.split().str.join(' ')
jydf = df.groupby(['jtitle','Year']).size().unstack().fillna(0)
jydf['min_counts'] = jydf.min(axis = 1)
#jydf.sort_values([2011,2012,2013,2014])
jydf.loc[['library page', 'public service trainee', 'chief investment officer', 'wharfinger i','wharfinger 1','wharfinger ii','wharfinger 2']]
jydf.loc[[x for x in jydf.index if 'lib' in x]]
df[df.jtitle.str.contains('investment')]
df[df.ename.str.contains('coaker')]
df[df.ename.str.contains(r'^robert.?.?.?shaw')]
years = [2011,2012,2013,2014]
print("Excluded Year\t No. of Titles in use all other years\t No. of Titles not always in use")
for y in years:
    jy = df[df.Year != y].groupby(['jtitle','Year']).size().unstack().fillna(0)
    jy['min_counts'] = jy.min(axis = 1)
    print("%9d\t %19d\t %36d"%(y,jy[jy.min_counts > 0].shape[0],jy[jy.min_counts == 0].shape[0]))
df[df.ename.isin(['aaron craig','carolina reyes ouk','zakhary mallett','ziran zhang'])][['ename','jtitle','Year']].sort_values(['ename','Year'])
jydf.loc[[x for x in jydf.index if 'sergeant' in x and jydf.loc[x].min_counts == 0]]
#Replacement dictionaries for roman numerals
rdict = {'jtitle': {r' iii': r' 3',
                    r' ii ': r' 2 ',
                    r' i ': r' 1 ',
                    r' ii, ': r' 2, ',
                    r' i, ': r' 1, ',
                    r' iv, ': r' 4, ',
                    r' v, ': r' 5, ',
                    r' vi, ': r' 6, ',
                    r' vii, ': r' 7, ',
                    r' viii, ': r' 8, ',
                     r' v ': r' 5 ',
                    r' vi ': r' 6 ',
                    r' vii ': r' 7 ',
                    r' viii ': r' 8 ',
                    r' iv': r' 4',
                    r' xiv': r' 14',
                    r' xxii': r' 22'}}
rdict2 = {'jtitle':{r' i$': r' 1',
                    r' ii$': r' 2',
                    r' iii$': r' 3',
                    r' iv$': r' 4',
                   r' v$': r' 5',
                   r' vi$': r' 6',
                   r' vii$': r' 7',
                   r' viii$': r' 8',
                   r' ix$': r' 9',
                   r' x$': r' 10',
                   r' xi$': r' 11',
                   r' xii$': r' 12',
                   r' xiii$': r' 13',
                   r' xiv$': r' 14',
                   r' xv$': r' 15',
                   r' xvi$': r' 16',
                   r' xvii$': r' 17',
                   r' xviii$': r' 18'}}

ndf = df.replace(rdict, regex=True, inplace=False)
ndf.replace(rdict2, regex=True, inplace=True)
ndf.replace({'jtitle':{r' , ': r', '}}, regex = True, inplace = True)
#Visual check line...
pd.unique(ndf[ndf.jtitle.str.contains('^supv')].jtitle)
#Replacement dictionary for abbr. and misspellings
adict = {'jtitle': {
                   r'asst': r'assistant',
                   r'dir ': r'director ',
                   r' sprv ': r' supervisor ',
                   r' sprv$': r' supervisor',
                   r'sprv1': r'supervisor 1',
                   r'qualitytech': r'quality technician',
                   r'maint ': r'maintenance ',
                   r'asst ': r'assistant ',
                   r'emerg ': r'emergency ',
                   r'emergencycy': r'emergency',
                   r'engr': r'engineer',
                   r'coord ': r'coordinator ',
                   r'coord$': r'coordinator',
                   r' spec ': r' specialist ',
                   r' spec$': r' specialist',
                   r' emp ': r' employee ',
                   r' repr$': r' representative', 
                   r' repres$': r' representative',
                   r' representat$': r' representative', 
                   r' - municipal transportation agency': r', mta',
                   r'safetycomm': r'safety communications',
                   r'trnst': r'transit',
                   r'wrk': r'worker',
                   r'elig ': r'eligibility '}}

ndf2 = ndf.replace(adict, regex=True, inplace=False)

#Unfortunately there are enough ambiguous abbreviations that we either need to switch to 
#trying to use nltk more or we need to convert specific jobs... 

jydf4 = ndf2.groupby(['jtitle','Year']).size().unstack().fillna(0)
jydf4['min_counts'] = jydf4.min(axis = 1)
jydf3 = ndf.groupby(['jtitle','Year']).size().unstack().fillna(0)
jydf3['min_counts'] = jydf3.min(axis = 1)
print(jydf[jydf.min_counts > 0].shape)
print(jydf3[jydf3.min_counts > 0].shape)
print(jydf4[jydf4.min_counts > 0].shape)
#Replacement dictionary for titles
title_dict = {'jtitle': {'water quality technician 3': 'water quality tech 3',
                   'water construction and maintenance superintendent': 'water const&main supt',
                   'track maintenance superintendent, municipal railway': 'track maintenance supt, muni railway',
                   }}

ndf3 = ndf2.replace(title_dict, regex=True, inplace=False)


jydf5 = ndf3.groupby(['jtitle','Year']).size().unstack().fillna(0)
jydf5['min_counts'] = jydf5.min(axis = 1)
print(jydf5[jydf5.min_counts > 0].shape)
gad = ndf2[ndf2.Year.isin([2011,2012])][['ename','jtitle','Year']]
print(gad.head())
lines = gad.groupby('ename').size()
tlines = lines[lines == 2]
mate1 = gad[gad.ename.isin(tlines.index)]
mate2 = mate1.groupby(['ename','jtitle']).size()
mate3 = mate2[mate2 == 1]
cand = mate3.unstack().index
matchers = gad[gad.ename.isin(cand)]
matchers.sort_values(['ename','Year'])
def find_name_end(name):
    s = name.split()
    last = s[-1]
    if last in ['jr','ii','iii'] and len(s)>2:
        last = ' '.join(s[-2:])
    return last
ndf3['ename_start'] = ndf.ename.apply(lambda x: x.split()[0])
ndf3['ename_end'] = ndf.ename.apply(find_name_end)
three_pint = ndf3.groupby(['ename_start','ename_end','Year'])
replicates = three_pint.size().unstack()
replicates = replicates.fillna(0)
replicates[replicates.max(axis = 1) > 1]
three_pint.get_group(('zenaida','cajilig',2014)).sort_values('jtitle')
three_pint.get_group(('yu','huang',2014)).sort_values('jtitle')
replicates[2011].argmax()
ndf3[ndf3.ename_end.str.contains(r' ii$')][['ename','ename_start','ename_end']].groupby(['ename_end','ename_start']).size()
tp = ndf3.groupby(['ename_start','ename_end'])
count = 0
for name, g in tp:
    if g.shape[0] > 4:
        count = count + 1
        if count < 90 and count > 50:
            print(name)
            print(g[['ename','jtitle','Year']])
            print(' ')
replicates[replicates.max(axis = 1) < 2]
replicates[(replicates.max(axis = 1) < 2) & (replicates.sum(axis = 1) == 4)]
ndf3.groupby(['ename_start','ename_end']).apply(lambda x: x.Year.max())
def playa(g):
    return g.groupby('Year').apply(lambda x: x.BasePay.mean())
ndf3.groupby(['ename_start','ename_end']).apply(playa)



lib3 = ndf3[ndf3.jtitle.str.contains('librarian 3')]
lib3[lib3.Year == 2014].describe()
chared = lib3.sort_values(['ename_end','ename_start','Year'])[['EmployeeName','BasePay','TotalPay','Year']]
chared.shape
chared
for i in range(0,70,60):
    inds = chared.index[i:i+60]
    print(i)
    print(chared.loc[inds])
checkers = pd.unique(lib3['ename_end'])
check2 = pd.unique(lib3['ename'])
check2


for n in checkers:
    tempt = ndf3[(ndf3.ename_end == n) & ndf3.ename.isin(check2) ]
    print (tempt.sort_values(['ename_end','ename_start','Year'])[['EmployeeName','JobTitle','BasePay','OtherPay','Year','Status']])
ndf3[(ndf3.ename_start == 'camille') & (ndf3.ename_end.str.contains('arr'))]
ndf3[(ndf3.ename_start == 'richard') & (ndf3.ename_end == 'le')]

lib3
lib3[['BasePay','TotalPay','TotalPayBenefits','Year']].boxplot( by = 'Year')
lib3.describe()
