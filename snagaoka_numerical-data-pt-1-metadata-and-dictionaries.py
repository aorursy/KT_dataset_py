import pandas as pd

import numpy as np

import re



from IPython.display import display, HTML

df = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)

num_observations = len(df)

column_prefs =  df.columns.map(lambda x: re.search('(ML|[A-Z])[a-z]*(?=[A-Z]*[a-z]*)', x).group(0))



df_meta = pd.DataFrame(df.columns,columns=['ColName'])

df_meta['Prefix'] = column_prefs

df_meta['ResponseRate'] = 1-df.isnull().sum().values/len(df)

df_meta['Dtypes'] = df.dtypes.values



df_meta.set_index(['Prefix','ColName'],inplace=True)
display(df_meta.head())
display(df_meta.loc['ML'])
def get_metameta():

    grp = df_meta.groupby(['Prefix'])

    grp_mean = grp.mean()

    grp_std = grp.std()

    df_metameta = grp_mean.merge(grp_std, left_index=True, right_index=True)

    df_metameta.columns = ['Mean','Std']



    unique_vals, counts = np.unique(column_prefs,return_counts=True)



    df_metameta['Counts'] = counts;

    df_metameta.sort_values('Mean',ascending=False,inplace=True);

    return df_metameta



df_metameta = get_metameta()

display(df_metameta)
numeric_cols = df.columns[~(df.dtypes == 'object')]

display(numeric_cols)

print('Number of numeric cols:',len(numeric_cols))
# First we deal with CompensationAmount separately

# we'll also prune a few superstars and people that earn 'too little'. No offence intended.

thres_max = 400000

thres_min = 10000



def replace_if_str(s):

    if isinstance(s,str):

        return s.replace(',','').replace('-','')

    else:

        return s

        

df['CompensationAmount'] = df['CompensationAmount'].apply(lambda s: replace_if_str(s))

df['CompensationAmount'] = df['CompensationAmount'].apply(

    lambda x: pd.to_numeric(x, errors='ignore', downcast='float')).astype(float)



df_conv = pd.read_csv('../input/conversionRates.csv')



price_dict = dict(list(zip(df_conv['originCountry'],df_conv['exchangeRate'])))

df['CompensationNormalized'] = df['CompensationAmount'] * df['CompensationCurrency'].map(price_dict)



is_outside_range = ((df['CompensationNormalized']>thres_max) 

                    | (df['CompensationNormalized']<thres_min))

df.at[is_outside_range,'CompensationNormalized'] = np.nan

df_schema = pd.read_csv('../input/schema.csv')

display(df_schema.head())

display(df_schema['Asked'].unique())
df_numlis = pd.DataFrame(list(zip(['Age', 'TitleFit', 'LearningPlatformUsefulness', 

                                   'LearningDataScienceTime', 'JobSkillImportance', 

                         'TimeSpentStudying', 'FormalEducation', 'Tenure', 'LearningCategory',

                         'ParentsEducation', 'EmployerSize', 'EmployerSizeChange','EmployerMLTime',

                                   'WorkToolsFrequency', 'WorkMethodsFrequency', 

                         'Time', 'AlgorithmUnderstandingLevel', 'WorkChallengeFrequency', 'RemoteWork',

                         'CompensationAmount','SalaryChange','JobSatisfaction','JobHuntTime',

                         'JobFactor'], ['4','13','29-47','56','57-66','77','84','87','92-98',

                                       '103','106', '107', '108','127-174','185-214','221-226','228','232-252',

                                       '266','267','269','271','274','275-290'])))

df_numlis.columns = ['ColName','Idx/Indices']



df_numlis.set_index('ColName',inplace=True)
did_match = lambda s,pat: not(re.search(pat,s)==None) if isinstance(s,str) else False

def make_did_match(pat): return lambda s: did_match(s,pat)



#define dictionaries

#Age, is already numerical

df_numlis['Dict'] = None



df_numlis.at['TitleFit','Dict']  = dict(list(zip(['Poorly', 'Fine', 'Perfectly', np.nan],[0,1,2,np.nan])))



df_numlis.at['LearningPlatformUsefulness','Dict'] = dict(list(zip(

    [np.nan, 'Somewhat useful', 'Very useful', 'Not Useful'],[np.nan,1,2,0])))



df_numlis.at['LearningDataScienceTime','Dict'] = dict(list(zip(

    [np.nan, '1-2 years', '< 1 year', '3-5 years', '15+ years',

       '5-10 years', '10-15 years'],[np.nan, 1, 0, 2, 5, 3, 4])))



df_numlis.at['JobSkillImportance','Dict'] = dict(list(zip(

    [np.nan, 'Nice to have', 'Unnecessary', 'Necessary'],[np.nan,1,0,2])))



df_numlis.at['TimeSpentStudying','Dict'] = dict(list(zip([np.nan, '2 - 10 hours', '0 - 1 hour', 

                                                          '11 - 39 hours', '40+'],

                                       [np.nan,1,0,2,3])))



df_numlis.at['FormalEducation','Dict'] = dict(list(zip(["Bachelor's degree", "Master's degree", 'Doctoral degree', np.nan,

       "Some college/university study without earning a bachelor's degree",

       'I did not complete any formal education past high school',

       'Professional degree', 'I prefer not to answer'], 

                                    [2,4,5,np.nan,1,0,3,np.nan]))) # this one is kinda hard to assess



df_numlis.at['Tenure','Dict'] = dict(list(zip(['More than 10 years', 'Less than a year', '3 to 5 years',

       '6 to 10 years', '1 to 2 years', np.nan,

       "I don't write code to analyze data"],[4,0,2,3,1,np.nan,np.nan])))





#LearningCategory.* is already numerical





df_numlis.at['ParentsEducation','Dict'] = dict_TSS = dict(list(zip(['A doctoral degree', "A bachelor's degree", 'High school',

       'Primary/elementary school', "A master's degree", np.nan,

       "Some college/university study, no bachelor's degree",

       'A professional degree', 'I prefer not to answer',

       "I don't know/not sure", 'No education'],

                                                [7, 4, 2,

                                                    1, 6, np.nan,

                                                    3,

                                                    5, np.nan,

                                                 np.nan,0])))



df_numlis.at['EmployerMLTime','Dict'] = dict(list(zip(['3-5 years', np.nan, "Don't know", '6-10 years', '1-2 years',

       'More than 10 years', 'Less than one year'], [2, np.nan, np.nan, 3, 1, 4, 0])))



df_numlis.at['EmployerSize','Dict'] = dict(list(zip(['100 to 499 employees', np.nan, '5,000 to 9,999 employees',

       '500 to 999 employees', '10,000 or more employees',

       '20 to 99 employees', 'Fewer than 10 employees', "I don't know",

       '1,000 to 4,999 employees', '10 to 19 employees',

       'I prefer not to answer'],

                                  [3,np.nan,6,

                                  4, 7,

                                  2, 0, np.nan,

                                  5, 1,

                                  np.nan])))



df_numlis.at['EmployerSizeChange','Dict'] = dict(list(zip(['Increased slightly', np.nan, 'Stayed the same',

       'Increased significantly', 'Decreased significantly',

       'Decreased slightly'],

                                  [1, np.nan, 0,

                                  2, -2,

                                  -1])))





df_numlis.at['WorkToolsFrequency','Dict']=dict(list(zip([np.nan, 'Sometimes', 'Often', 'Most of the time', 'Rarely'],

                                [np.nan, 1, 2, 3, 0])))



df_numlis.at['WorkMethodsFrequency','Dict'] = df_numlis.loc['WorkToolsFrequency','Dict']



# # Time.* is already numerical



df_numlis.at['AlgorithmUnderstandingLevel','Dict'] = dict(list(zip(['Enough to explain the algorithm to someone non-technical', np.nan,

       'Enough to refine and innovate on the algorithm',

       'Enough to tune the parameters properly',

       'Enough to code it again from scratch, albeit it may run slowly',

       'Enough to run the code / standard library',

       'Enough to code it from scratch and it will run blazingly fast and be super efficient'],

                                     [3,np.nan,

                                     4,

                                     1,

                                     2,

                                     0,

                                     5])))



df_numlis.at['WorkChallengeFrequency','Dict'] = df_numlis.loc['WorkToolsFrequency','Dict']





df_numlis.at['RemoteWork','Dict'] = dict(list(zip(['Always', np.nan, 'Rarely', 'Sometimes', 

                                            'Most of the time', 'Never', "Don't know"],

                                           [4, np.nan, 1, 2, 3, 0, np.nan])))



# CompensationAmount actually has a different problem: commas



df_numlis.at['SalaryChange','Dict'] = dict(list(zip(['I am not currently employed', np.nan, 

                                                     'Has increased 20% or more',

       'I do not want to share information about my salary/compensation',

       'Has stayed about the same (has not increased or decreased more than 5%)',

       'Has increased between 6% and 19%',

       'Has decreased between 6% and 19%',

       'I was not employed 3 years ago', 'Has decreased 20% or more',

       'Other'],[np.nan, np.nan, 2,

                np.nan,

                0,

                1,

                -1,

                np.nan, -2])))





df_numlis.at['JobSatisfaction','Dict'] = dict(list(zip(['5', np.nan, '10 - Highly Satisfied', '2', '8', '7', '6', '9',

       '1 - Highly Dissatisfied', 'I prefer not to share', '3', '4'],

                                     [5, np.nan, 10, 2, 8, 7, 6, 9, 1, np.nan, 3, 4])))





df_numlis.at['JobHuntTime','Dict']= dict(list(zip([np.nan, '1-2', '0', '3-5', '11-15', '20+', '6-10', '16-20'],

                                     [np.nan, 1, 0, 2, 4, 6, 3, 5])))



df_numlis.at['JobFactor','Dict'] = dict(list(zip([np.nan, 'Very Important', 'Somewhat important', 'Not important'],

                                     [np.nan, 2, 1, 0])))



df_numlis
# A small helper function for finding substrings

did_match = lambda pat,s: not(re.search(pat,s)==None) if isinstance(s,str) else False

did_match_full = lambda pat,s: re.search(pat,s).group(0)==s if did_match(pat,s) else False

def make_did_match(pat): return lambda s: did_match(pat,s)

def make_did_match_full(pat): return lambda s: did_match_full(pat,s)



df_n = df.copy()

is_single = ~df_numlis['Idx/Indices'].apply(make_did_match('.*-.*')) #when there is a minus



# First, deal with prefixes representing one column

for idx in df_numlis.loc[is_single].index:

     if (df[idx].dtype == 'object'):

        df_n[idx] = df[idx].map(df_numlis.loc[idx,'Dict'])



all_columns = df_meta.index.get_level_values(1)



# Then, deal with the rest

for idx in df_numlis.loc[~is_single].index:    

    ix = all_columns.map(make_did_match_full(idx +'.*')).values.astype(np.bool_)



    for idx_sub in all_columns[ix]:

        if (df_n[idx_sub].dtype == 'object'):

            df_n[idx_sub] = df[idx_sub].map(df_numlis.loc[idx,'Dict'])

numeric_cols = df_n.columns[~(df_n.dtypes == 'object')]

display(numeric_cols)

print('Number of numeric cols:',len(numeric_cols))
import seaborn as sns



df_nonly = df_n[numeric_cols]



import matplotlib.pyplot as plt

plt.figure(figsize=[20, 20])

corrs=df_nonly.corr()

mh = sns.heatmap(corrs)

mh.tick_params(labelsize = 15)
#code copied from arun's answer

def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Correlations 10")

print(get_top_abs_correlations(df_nonly, 10))
respg25 = df_meta[df_meta['ResponseRate']>0.25].index.get_level_values(1)

df_nonly_pruned = df_n.loc[:,respg25.intersection(numeric_cols).insert(0,'CompensationNormalized')]





print("Top Correlations 50")

print(get_top_abs_correlations(df_nonly_pruned, 50))

df_nonly_pruned



#import matplotlib.pyplot as plt

plt.figure(figsize=[20, 20])

mh = sns.heatmap(df_nonly_pruned.corr())

mh.tick_params(labelsize = 15)



corrs = df_nonly_pruned.corr()

data = (corrs['JobSatisfaction'].sort_values(ascending=False)

.drop(['JobSatisfaction','LearningDataScienceTime']))

plt.figure(figsize=[20,25])

plt.rcParams["axes.labelsize"] = 50

bh = sns.barplot(data.values,data.index)

bh.tick_params(labelsize = 25)

bh.axes.set_title('JobSatisfaction',fontsize=40)

plt.xlabel('Correlation', fontsize=35)





data = (corrs['CompensationNormalized'].sort_values(ascending=False)

.drop(['CompensationNormalized','LearningDataScienceTime']))

plt.figure(figsize=[20,25])

plt.rcParams["axes.labelsize"] = 50

bh = sns.barplot(data.values,data.index)

bh.tick_params(labelsize = 25)

bh.axes.set_title('CompensationNormalized',fontsize=40)

plt.xlabel('Correlation', fontsize=35)





a = corrs.loc['WorkToolsFrequencyPython','JobSatisfaction']

b = corrs.loc['WorkToolsFrequencyR','JobSatisfaction']

c = corrs.loc['WorkToolsFrequencyPython','CompensationNormalized']

d = corrs.loc['WorkToolsFrequencyR','CompensationNormalized']





plt.plot([a, c])

plt.plot([b, d])

plt.legend(['Python','R'])

plt.ylabel('correlation',fontsize=14)
ffr = pd.read_csv('../input/freeformResponses.csv',low_memory=False)

ffr_gender = ffr['GenderFreeForm']

ffr_gender = ffr_gender[ffr_gender.notnull()]



num_copters = ffr_gender.apply(lambda s: re.search('.*((h|H)elicopter|AH-64).*',s)).notnull().sum()

print('Insight of the day: {0:4.4f}% of Kaggle users sexually identfy themselves as attack helicopters!'.format(100*num_copters/num_observations))
