# Pandas is for creating df - data loaded directly in to the DF
import pandas as pd


# Preparing the variables
ipldata_deliveries = pd.read_csv('../input/ipldata/deliveries.csv', index_col=0)
ipldata_matches = pd.read_csv('../input/ipldata/matches.csv', index_col=0)
ipldata_deliveries.shape
ipldata_matches.shape
ipldata_deliveries.head()
ipldata_matches.head()
# Understanding teh data types 
ipldata_matches.dtypes
ipldata_matches.index
ipldata_matches.index
ipldata_matches.describe
ipldata_matches['city']


ipl_del = pd.read_csv('../input/ipl/deliveries.csv', index_col=0)
ipl_mat = pd.read_csv('../input/ipl/matches.csv', index_col=0)
ipl_del.shape
ipl_mat.shape
ipl_del.head()
ipl_mat.head()
ipl_mat.tail()
ipl_mat.head()

ipl_mat.season
#Selection via loc and iloc - But this method I havent seen much being used in the notebooks 

ipl_mat.iloc[0:4,1:5]
# Using the loc command 

ipl_mat.loc[:,['season', 'player_of_match', 'win_by_runs', 'win_by_wickets']]
# This is the main method which you will use 
ipl_mat[['season','city']]
ipl_mat.info
ipl_mat.isna
ipl_mat.head()
ipl_mat_xna = ipl_mat.dropna()

ipl_mat_xna.shape

ipl_mat_xna = ipl_mat.fillna(0)
ipl_mat_xna.shape
ipl_mat_xna.head(20)
ipl_mat.head(20)
ipl_mat.loc[3]
ipl_mat_xna.head()


ipl_mat_xna.shape
ipl_mat_xna.columns
ipl_mat_xna.dtypes
# Importing ths SNS libray which has the pairplot function
import seaborn as sns
import matplotlib.pyplot as plt
ipl_mat_xna.corr()
ipl_mat_xna_corr = ipl_mat_xna.corr
type(ipl_mat_xna_corr)
cmap = sns.diverging_palette(230, 20, as_cmap=True) # Color map 
sns.heatmap(ipl_mat_xna_corr, cmap=cmap, annot=True)

sns.pairplot(ipl_mat_xna, height=1.5)

in1 = ipl_mat_xna.nunique()
in1
ipl_mat_xna_Unique_Values = ipl_mat_xna.nunique()

ipl_mat_xna_Unique_Values = ipl_mat_xna_Unique_Values.to_frame()

in1d





in1p.iloc[:2]