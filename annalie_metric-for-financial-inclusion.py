import numpy as np
import pandas as pd
from numpy import log10, ceil, ones
from numpy.linalg import inv 
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
Fcol = ['Country Code', 'Indicator Name', 'MRV']
F = pd.read_csv("../input/findex-world-bank/FINDEXData.csv", usecols=Fcol)
F.head()
indicators_search = 'Borrowed any money in the past year|\
Borrowed for education or school fees|\
Borrowed for health or medical purposes|\
Borrowed from a financial institution|\
Borrowed from a private informal lender|\
Borrowed from family or friends|\
Borrowed to start, operate, or expand a farm or business|\
Coming up with emergency funds: not at all possible|\
Coming up with emergency funds: not very possible|\
Coming up with emergency funds: somewhat possible|\
Coming up with emergency funds: very possible|\
Account at a financial institution|\
Saved at a financial institution|\
Main source of emergency funds: family or friends|\
Main source of emergency funds: savings'
# Loan in the past year

group_search_income = 'richest|poorest'

def clean(findex, indicators_search, group_search=group_search_income):
    """Split Indicator Name into two columns: Indicator and Group."""

    # select only rows from wave 2 (2014) with the following indicators and groups
    findex = findex[findex['Indicator Name'].str.contains('w2')]
    findex = findex[findex['Indicator Name'].str.contains(indicators_search)]
    findex = findex[findex['Indicator Name'].str.contains(group_search)]

    # remove last words from the string
    findex['Indicator Name'] = findex['Indicator Name'].str.split().str[:-4]
    findex['Indicator Name'] = findex['Indicator Name'].apply(lambda x: ' '.join(x))

    # create a column for the indicator and one for the income group
    findex['Group'] = findex['Indicator Name'].str.split().str[-2:]
    findex['Group'] = findex['Group'].apply(lambda x: ' '.join(x))

    findex['Indicator'] = findex['Indicator Name'].str.split().str[:-3]
    findex['Indicator'] = findex['Indicator'].apply(lambda x: ' '.join(x))
    # remove last words from the string
    findex['Indicator'] = findex['Indicator'].str[:-1]

    # remove column Indicator Name
    findex.drop('Indicator Name', axis=1, inplace=True)

    # set index
    findex.set_index(['Country Code', 'Indicator', 'Group'], inplace=True)

    # create series
    findex = findex['MRV']

    # unstack series
    findex = findex.unstack(level=-1)
    
    return findex
def add_formal_informal_borrowing(df):
    
    # df['Informal'] = df['Borrowed from a private informal lender'] + df['Borrowed from family or friends']
    df['Informal'] = df['Borrowed from a private informal lender']
    df['Informal ratio'] = df['Informal']/df['Borrowed any money in the past year']
    df['Formal ratio'] = df['Borrowed from a financial institution']/df['Borrowed any money in the past year']
    df['Formal/Informal ratio'] = df['Formal ratio']/df['Informal ratio'] 
       
    return df
def compute_finexcl(df):
    RF = df.loc[df['Group'] == 'richest 60%', 'Formal ratio']
    RI = df.loc[df['Group'] == 'richest 60%', 'Informal ratio']
    PF = df.loc[df['Group'] == 'poorest 40%', 'Formal ratio']
    PI = df.loc[df['Group'] == 'poorest 40%', 'Informal ratio']
    finexcl = (float(RF-RI)-float(PF-PI))/float(RF-RI)

    df['Financial Exclusion'] = finexcl
        
    return df
def main():
    
    # return a dataframe with groups as variables
    # merge findex percentages and findex index
    F_groups = clean(F, indicators_search, group_search=group_search_income)
    
    # return a dataframe with the indicators as variables
    # add variables for formal and informal borrowing to dataframe
    F_ind = F_groups.stack().unstack(level=1)
    F_ind = add_formal_informal_borrowing(F_ind)
        
    # add indicator for financial exclusion to dataframe
    F_ind.reset_index(inplace=True)
    F_ind.rename(columns={'level_1':'Group'}, inplace=True)
    
    grouped = F_ind.groupby('Country Code')
    F_ind_copy = pd.DataFrame()
    for user, group in grouped:
        group = compute_finexcl(group)
        F_ind_copy = F_ind_copy.append(group)
    F_ind_copy.set_index(['Country Code', 'Group'], inplace=True)
    F_ind = F_ind_copy
    
    
    print ("2 Dataframes returned: findex_groups (where the variables are the groups) and findex_ind(where the indicators are the variables).")
    
    return F_groups, F_ind
F_groups, F_ind = main()
F_ind = F_ind[['Informal', 'Formal ratio', 'Informal ratio', 'Financial Exclusion']]
F_ind.head(8)
F_ind.reset_index(inplace=True)
fig, ax = plt.subplots(figsize=(20,10))
ax = sns.stripplot(ax=ax, x='Country Code', y='Financial Exclusion', data=F_ind, color='g')
plt.xticks(rotation=90)
ax.set_title("Financial Exclusion gap between rich and poor per country")
ax.set_ylim(-3,3);
def read_findex(datafile=None, interpolate=False, invcov=True, variables = ["Account", "Loan", "Emergency"], norm=True):
    """
    Returns constructed findex values for each country

    Read in Findex data - Variables include: Country ISO Code, Country Name,
                          Pct with Account at Financial institution (Poor),
                          Pct with a loan from a Financial institution (Poor),
                          Pct who say they could get an emergency loan (Poor)

    Take average of 'poorest 40%' values for each value in `variables'

     If `normalize':
        Apply the normalization function to every MPI variable
    """
    if datafile == None: datafile = "../input/findex-world-bank/FINDEXData.csv"

    F = pd.read_csv(datafile)#~ [["ISO","Country Name", "Indicator Name", "MRV"]]
    
    Fcols = {'Country Name': 'Country',
        'Country Code': 'ISO',
        'Indicator Name': 'indicator',
        'Indicator Code': 'DROP',
        '2011': 'DROP',
        '2014': 'DROP',
        'MRV': 'Val'
        }
    F = F.rename(columns=Fcols).drop("DROP",1)
    F['Val'] /= 100.
    
    indicators = {"Account at a financial institution, income, poorest 40% (% ages 15+) [ts]": "Account",
        "Coming up with emergency funds: somewhat possible, income, poorest 40% (% ages 15+) [w2]": "Emergency",
        "Coming up with emergency funds: very possible, income, poorest 40% (% ages 15+) [w2]": "Emergency",
        "Borrowed from a financial institution, income, poorest 40% (% ages 15+) [ts]": "Loan"
        }

    F['Poor'] = F['indicator'].apply(lambda ind: "Poor" if "poorest" in ind else "Rich") 
    F['indicator'] = F['indicator'].apply(lambda ind: indicators.setdefault(ind,np.nan)) 
    F = F.dropna(subset=["indicator"])
    F = F.groupby(["Poor","ISO","indicator"])["Val"].sum()
    F = 1 - F.loc["Poor"]

    F = F.unstack("indicator")
    
    # fill missing values for the emergency indicator with a predicted score from OLS regression analysis 
    if interpolate:
        results = smf.ols("Emergency ~ Loan + Account",data=F).fit()
        F['Emergency_fit'] = results.params['Intercept'] + F[['Loan','Account']].mul(results.params[['Loan','Account']]).sum(1)
        F['Emergency'].fillna(F['Emergency_fit'],inplace=True)
    if invcov: F['Findex'] = invcov_index(F[variables]) #.mean(1)
    else: F['Findex'] = F[variables].mean(1,skipna=True)
        
    flatvar = flatten(F['Findex'].dropna(), use_buckets = False, return_buckets = False)
    F = F.join(flatvar,how='left',lsuffix=' (raw)')
    
    return F

def invcov_index(indicators):
    """
    Convert a dataframe of indicators into an inverse covariance matrix index
    """
    df = indicators.copy()
    df = (df-df.mean())/df.std()
    I  = np.ones(df.shape[1])
    E  = inv(df.cov())
    s1  = I.dot(E).dot(I.T)
    s2  = I.dot(E).dot(df.T)
    try:
        int(s1)
        S  = s2/s1
    except TypeError: 
        S  = inv(s1).dot(s2)
    
    S = pd.Series(S,index=indicators.index)

    return S

def flatten(Series, outof = 10., bins = 20, use_buckets = False, write_buckets = False, return_buckets = False):
    """
    NOTE: Deal with missing values, obviously!
    Convert Series to a uniform distribution from 0 to `outof'
    use_buckets uses the bucketing rule from a previous draw.
    """

    tempSeries = Series.dropna()
    if use_buckets: #~ Use a previously specified bucketing rule
        cuts, pcts = list(rule['Buckets']), np.array(rule['Values']*(100./outof))
    else: #~ Make Bucketing rule to enforce a uniform distribution
        pcts = np.append(np.arange(0,100,100/bins),[100])
        cuts = [ np.percentile(tempSeries,p) for p in pcts ]
        while len(cuts)>len(set(cuts)):
            bins -= 1
            pcts = np.append(np.arange(0,100,100/bins),[100])
            cuts = [ np.percentile(tempSeries,p) for p in pcts ]

    S = pd.cut(tempSeries,cuts,labels = pcts[1:]).astype(float)
    S *= outof/100

    buckets = pd.DataFrame({"Buckets":cuts,"Values":pcts*(outof/100)})

    if return_buckets: return S, 
    else: return S
F = read_findex()
F.head()
temp = F_ind.merge(F.reset_index(), left_on='Country Code', right_on='ISO')
print (temp[['Financial Exclusion', 'Findex', 'Findex (raw)']].head())

fig, ax = plt.subplots(figsize=(12,6))
temp.plot(kind='scatter', x='Findex (raw)', y='Financial Exclusion', ax=ax)
plt.title("Comparison of two measures for Financial Inclusion");