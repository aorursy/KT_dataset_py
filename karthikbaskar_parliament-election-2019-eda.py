import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import LabelEncoder
DATA=pd.read_csv('../input/indian-candidates-for-general-election-2019/LS_2.0.csv')





print(DATA.size)

print(DATA.shape)
DATA.info()
DATA.rename(columns = {'TOTAL\nVOTES':'TOTALVOTES','CRIMINAL\nCASES':'CRIMINALCASES'}, inplace = True)



DATA['CRIMINALCASES'].fillna(DATA['CRIMINALCASES'].mode().values[0], inplace = True)

CONS=DATA.groupby("STATE")["CONSTITUENCY"].nunique()

CONS.plot(figsize=(8,8));

CONS.plot.bar();
DATA.loc[DATA.WINNER==1,'WINNER']="Yes"

DATA.loc[DATA.WINNER==0,'WINNER']="No"
DATA['WINNER'].value_counts()
TAMIL_NADU=DATA[DATA['STATE']== 'Tamil Nadu']

TAMIL_NADU.head()

No_Of_Cons=TAMIL_NADU['CONSTITUENCY'].nunique()

No_Of_Cons
TAMIL_NADU.head()
print("Number of Voters:",TAMIL_NADU['TOTAL ELECTORS'].sum())
TAMIL_NADU.describe().T
TAMIL_NADU['PARTY'].value_counts()
TAMIL_NADU['GENDER'].value_counts()

Gender_tn=pd.crosstab(TAMIL_NADU['PARTY'],TAMIL_NADU['GENDER'])

Gender_tn.plot.bar();

plt.title("No of Male and Female Candidates across party in Tamil Nadu 2019")

Gender_win=pd.crosstab(TAMIL_NADU['GENDER'],TAMIL_NADU['WINNER'])

Gender_win.plot.bar();

plt.title("Contestents splitup based on result 2019 Tamil Nadu")

Gender_win_ind=pd.crosstab(DATA['GENDER'],DATA['WINNER'])

Gender_win_ind.plot.bar();

plt.title("Contestents splitup based on result 2019 All Over India")
TAMIL_NADU.skew()
sns.scatterplot(TAMIL_NADU['PARTY'],TAMIL_NADU['GENDER'],hue=TAMIL_NADU['WINNER']);
#Let's define the Null Hypothesis & Alternate Hypothesis

H0="Gender does not have any impact on winning in 2019"

Ha="Gender does have an  impact on  winning in 2019"

chi,p_value,expected,dof=stats.chi2_contingency(Gender_win_ind)

print (p_value)

if p_value < 0.05:

    print(f'{Ha} as the p_value ({p_value}) < 0.05')

else:

    print(f'{H0} as the p_value ({p_value}) > 0.05')
sns.scatterplot(x=TAMIL_NADU['PARTY'],y=TAMIL_NADU['TOTALVOTES'],hue=TAMIL_NADU['EDUCATION']);
TAB=pd.crosstab(TAMIL_NADU['WINNER'],TAMIL_NADU['EDUCATION'])

TAB

H0="Education of Candidates does not influence winning"

Ha="Education of candidates does influence winning"

chi,p_value,dof,expected=stats.chi2_contingency(TAB)

print(p_value)

if p_value < 0.05:

    print(Ha)

else:

    print(H0)
Vote_share=DATA.groupby("PARTY")["TOTALVOTES"].sum().nlargest(5).index.tolist()

Vote_share

def fuc(row):

    if row["PARTY"] not in Vote_share:

        return("Others")

    else:

        return row['PARTY']

DATA['Party_new']=DATA.apply(fuc,axis=1)

DATA.head()

Top5=DATA.groupby("Party_new")["TOTALVOTES"].sum()

Top5_index=Top5.index

Top5_label=Top5.values

Top5.plot.pie(labels=Top5_index,

        shadow=True, startangle=30);
TOP5_TN=TAMIL_NADU.groupby("PARTY")["TOTALVOTES"].sum().nlargest(5).index.tolist()

TOP5_TN

def TN(row):

    if row["PARTY"] not in TOP5_TN:

        return("Others")

    else:

        return row["PARTY"]

TAMIL_NADU['TN']=TAMIL_NADU.apply(TN,axis=1)

TN_pie=TAMIL_NADU.groupby('TN')["TOTALVOTES"].sum()

TN_pie.plot.pie()