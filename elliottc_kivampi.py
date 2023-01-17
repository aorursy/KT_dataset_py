# Load libraries
%matplotlib inline
import numpy as np
import pandas as pd
from numpy import log10, ceil, ones
from numpy.linalg import inv 
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Load & Merge data
LT = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv").set_index('Loan Theme ID')
MPI = pd.read_csv("../input/mpi/MPI_national.csv")[['ISO','MPI Urban','MPI Rural']].set_index("ISO")
LT = LT.join(MPI,how='left',on="ISO")[['Partner ID','Field Partner Name','ISO','MPI Rural','MPI Urban','rural_pct','amount']].dropna()
print("Merged Loan Theme data with National MPI Scores (Rural & Urban)")
LT.head()
LT['Rural'] = LT['rural_pct']/100        #~ Convert rural percentage to 0-1
LT['MPI Natl'] = LT['Rural']*LT['MPI Rural'] + (1-LT['Rural'])*LT['MPI Urban'] #~ Compute the MPI Score for each loan theme, weighting by rural_pct
weighted_avg = lambda df: pd.Series(np.average(df['MPI Natl'],weights=df['amount']))             #~ Need a volume-weighted average for mutli-country partners. 
Scores = LT.groupby(['Partner ID','ISO']).agg({'MPI Natl': np.mean,'amount':np.sum}).groupby(level='Partner ID').apply(weighted_avg)
Scores.columns = ['MPI Natl']
Scores = Scores.join(LT.groupby('Partner ID')['Rural'].mean())
fig, ax = plt.subplots(2, 2,figsize=(8,8))
Scores['MPI Natl'].plot(kind='hist', bins=30,ax=ax[0,0], title= "Rural-weighted MPI Scores by Field Parnter")
MPI['MPI Rural'].plot(kind='hist', bins=30,ax=ax[0,1], title="Rural MPI Scores by Country")
MPI.plot(kind='scatter',x = 'MPI Rural', y = 'MPI Urban', title = "Urban vs. Rural MPI Scores by Country\n(w/ y=x line)", ax=ax[1,0])
ax[1,0].plot(ax[1,0].get_xlim(),ax[1,0].get_ylim(),label="Rural==Urban line"); ax[1,0].legend()
sns.regplot('Rural','MPI Natl', data=Scores, order=2, ax=ax[1,1]).set_title('Rural Share of Borrowers vs. Country-level MPI')
plt.tight_layout()
# Load data
MPI = pd.read_csv("../input/mpi/MPI_subnational.csv")[['Country', 'Sub-national region', 'World region', 'MPI Regional']]
MPInat = pd.read_csv("../input/mpi/MPI_national.csv")[['ISO','Country','MPI Rural', 'MPI Urban']].set_index('ISO')
LT = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")[['country','Partner ID', 'Loan Theme ID', 'region', 'mpi_region', 'ISO', 'number', 'amount','rural_pct', 'LocationName', 'Loan Theme Type']]
# Create new column mpi_region and join MPI data to Loan themes on it
MPI['mpi_region'] = MPI[['Sub-national region', 'Country']].apply(lambda x: ', '.join(x), axis=1)
MPI = MPI.set_index('mpi_region')
LT = LT.join(MPI, on='mpi_region', rsuffix='_mpi') #[['country','Partner ID', 'Loan Theme ID', 'Country', 'ISO', 'mpi_region', 'MPI Regional', 'number', 'amount','Loan Theme Type']]
#~ Pull in country-level MPI Scores for when there aren't regional MPI Scores
LT = LT.join(MPInat, on='ISO',rsuffix='_mpinat')
LT['Rural'] = LT['rural_pct']/100        #~ Convert rural percentage to 0-1
LT['MPI Natl'] = LT['Rural']*LT['MPI Rural'] + (1-LT['Rural'])*LT['MPI Urban']
LT['MPI Regional'] = LT['MPI Regional'].fillna(LT['MPI Natl'])
#~ Get "Scores": volume-weighted average of MPI Region within each loan theme.
Scores = LT.groupby('Loan Theme ID').apply(lambda df: np.average(df['MPI Regional'], weights=df['amount'])).to_frame()
Scores.columns=["MPI Score"]
#~ Pull loan theme details
LT = LT.groupby('Loan Theme ID').first()[['country','Partner ID','Loan Theme Type','MPI Natl','Rural','World region']].join(Scores)#.join(LT_['MPI Natl'])
notmissing = LT['MPI Score'].count()
notmissing_pct = round(100*notmissing/float(LT.shape[0]),1)
print("Now we've made Subnational MPI Scores for each loan theme.\nNote we only have scores for {}% ({}) of Loan Themes.".format(notmissing_pct, notmissing))
fig, ax = plt.subplots(2, 2, figsize=(10,10))
# Compare distributions 
LT[['MPI Score','MPI Natl']].plot(kind='kde', ax=ax[0,0], title = "Distribution of National & Regional MPI")

# Compare Regions
sns.boxplot(y='MPI Score',x='World region',data=LT,ax=ax[0,1])
for tick in ax[0,1].get_xticklabels(): tick.set_rotation(35)

#~ Rural/Urban vs Sub-national MPI Scores
colors = dict(zip(set(LT['World region']),'red,blue,green,orange,black,purple,yellow'.split(",")))
for area,df in LT.groupby('World region'): ax[1,0].scatter(df['MPI Score'],df['MPI Natl'],c=colors[area],label=area,marker='.')
x,y = ax[1,0].get_xlim(),ax[1,0].get_ylim()
sns.regplot('MPI Score','MPI Natl', data=LT, marker = '.', scatter=False, ax=ax[1,0]).set_title("National vs. Regional MPI")
ax[1,0].set_xlim(x);ax[1,0].set_ylim(y)
ax[1,0].plot(ax[1,0].get_xlim(),ax[1,0].get_ylim(),'g--',label="Regional==National"); ax[1,0].legend()
#~ Compare to Rural % by field partner
sns.regplot('Rural','MPI Score', data=LT, order=2, marker = '.', ax=ax[1,1]).set_title("Rural % of Borrowers vs. Regional MPI")
plt.legend(); plt.tight_layout()