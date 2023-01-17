import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns

sns.set()

from matplotlib import style

from matplotlib.backends.backend_pdf import PdfPages

%matplotlib inline

plt.style.use(['seaborn-dark'])

%config IPCompleter.greedy=True

from IPython.display import display

from IPython.display import Image
df = pd.read_csv('../input/vHoneyNeonic_v03.csv')

# Show data frame

df.head()
#Checking for null values

df.isnull().sum()
#Fill all NaN with 0

df = df.fillna(0)
df.shape
df.dtypes
#Convert state, StateName, Region to category

#Select all columns of type 'object'

object_columns = df.select_dtypes(['object']).columns

object_columns
#Convert selected columns to type 'category'

for column in object_columns: 

    df[column] = df[column].astype('category')

df.dtypes
df.describe().T
df.corr()
#print unique features for each row

print("Feature, UniqueValues") 

for column in df:

    print(column + "," + str(len(df[column].unique())))
#Add new column determined by pre- and post-neonics (2003)

df['post-neonics(2003)'] = np.where(df['year']>=2003, 1, 0)
# Correlation matrix using code found on https://stanford.edu/~mwaskom/software/seaborn/examples/many_pairwise_correlations.html

#USA_youtube_df = pd.read_csv("USvideos.csv")

sns.set(style="white")



# Select columns containing continuous data

continuous_columns = df[['numcol','yieldpercol','totalprod','stocks','priceperlb','prodvalue','year','nCLOTHIANIDIN','nIMIDACLOPRID','nTHIAMETHOXAM','nACETAMIPRID','nTHIACLOPRID','nAllNeonic','post-neonics(2003)']].columns



# Calculate correlation of all pairs of continuous features

corr = df[continuous_columns].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom colormap - blue and red

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=1, vmin=-1,

            square=True, xticklabels=True, yticklabels=True,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

plt.yticks(rotation = 0)

plt.xticks(rotation = 45)
#Making a new dataframe containing all data before 2003

df_pre_2003 = df[(df['year']<2003)]
#Making a new dataframe containing all data including and after 2003

df_2003 = df[(df['year']>=2003)]
#Units of neonic used each year 

df.groupby(['year'])['nAllNeonic'].sum()
import seaborn as sns

plt.style.use(['seaborn-dark'])
df_pre_2003.groupby(['year'])['nAllNeonic'].sum().plot(color='green')

plt.title("Neonic usage prior to 2003")
df_2003.groupby(['year'])['nAllNeonic'].sum().plot(color='green')

plt.title("Neonic usage after 2003")
#Timeline of neonic usage

df.groupby(['year'])['nAllNeonic'].sum().plot(color='green')

plt.title("Complete timeline of neonic usage")
#bivariate distribution of all neonics vs. year

sns.jointplot(data=df, x='year', y='nAllNeonic', kind='reg', color='g')
#Resizing plots

plt.rcParams["figure.figsize"] = (10, 5)

plt.rcParams["xtick.labelsize"] = 10

plt.rcParams["ytick.labelsize"] = 10
df_pre_2003.groupby(['StateName'])['nAllNeonic'].sum().plot(kind='bar')

plt.title("Neonic usage by state prior to 2003")

plt.rcParams["figure.figsize"] = (10, 5)

plt.rcParams["xtick.labelsize"] = 10

plt.rcParams["ytick.labelsize"] = 10
df_2003.groupby(['StateName'])['nAllNeonic'].sum().plot(kind='bar')

plt.title("Neonic usage by state post 2003")
df_pre_2003.groupby(['Region'])['nAllNeonic'].sum().plot(kind='bar')

plt.title("Neonic use prior to 2003")
df_2003.groupby(['Region'])['nAllNeonic'].sum().plot(kind='bar')

plt.title("Neonic usage after 2003")
g = sns.FacetGrid(df, col="Region") 

g.map(sns.regplot, "year", "numcol", line_kws={"color": "red"})
df_pre_2003.groupby(['StateName'])['numcol'].sum().plot(kind='bar')

plt.title('No. of colonies per state pre-2003')
df_2003.groupby(['StateName'])['numcol'].sum().plot(kind='bar')

plt.title('Number of colonies per state from 2003 to present')
df_pre_2003.groupby(['StateName'])['nIMIDACLOPRID'].sum().plot(kind='bar')

plt.title('nIMIDACLOPRID usage pre-2003')
df_pre_2003.groupby(['StateName'])['nTHIAMETHOXAM'].sum().plot(kind='bar')

plt.title('nTHIAMETHOXAM usage pre-2003')
df_pre_2003.groupby(['StateName'])['nACETAMIPRID'].sum().plot(kind='bar')

plt.title('nACETAMIPRID usage pre-2003')
df_pre_2003.groupby(['StateName'])['nAllNeonic'].sum().plot(kind='bar')

plt.title('All Neonic usage pre-2003')
df.groupby(['StateName'])['nAllNeonic'].sum().plot(kind='bar')

plt.title("All neonic usage per state")
#Yield per colony over time in each region

g = sns.FacetGrid(df, col="Region") 

g.map(sns.regplot, "year", "yieldpercol", line_kws={"color": "red"})
#Total production per region over time

g = sns.FacetGrid(df, col="Region") 

g.map(sns.regplot, "year", "totalprod", line_kws={"color": "red"})
plt.plot( 'year', 'nCLOTHIANIDIN', data=df.sort_values('year'), marker='', color='blue', linewidth=2)

plt.plot( 'year', 'nIMIDACLOPRID', data=df.sort_values('year'), marker='', color='olive', linewidth=2)

plt.plot( 'year', 'nTHIAMETHOXAM', data=df.sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")

plt.plot( 'year', 'nACETAMIPRID', data=df.sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")

plt.plot( 'year', 'nTHIACLOPRID', data=df.sort_values('year'), marker='', color='#cd71ff', linewidth=2, label="nTHIACLOPRID")

plt.legend()

plt.title("Neonic usage over time")
g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='California'])

g.set_axis_labels("Year", "Total Production")

plt.title('California')



g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='California'])

g.set_axis_labels("Year", "Yield per col")

plt.title('California')



g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='California'])

g.set_axis_labels("Year", "No. of colonies")

plt.title('California')
plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="California"].sort_values('year'), marker='', color='blue', linewidth=2)

plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="California"].sort_values('year'), marker='', color='olive', linewidth=2)

plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="California"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")

plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="California"].sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")

plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="California"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")

plt.legend()
g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Illinois'])

g.set_axis_labels("Year", "Total Production")

plt.title('Illinois')



g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Illinois'])

g.set_axis_labels("Year", "Yield per col")

plt.title('Illinois')



g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Illinois'])

g.set_axis_labels("Year", "No. of colonies")

plt.title('Illinois')
plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="Illinois"].sort_values('year'), marker='', color='blue', linewidth=2)

plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="Illinois"].sort_values('year'), marker='', color='olive', linewidth=2)

plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="Illinois"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")

plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="Illinois"].sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")

plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="Illinois"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")

plt.title('Illinois')

plt.legend()
g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Iowa'])

g.set_axis_labels("Year", "Total Production")

plt.title('Iowa')



g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Iowa'])

g.set_axis_labels("Year", "Yield per col")

plt.title('Iowa')



g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Iowa'])

g.set_axis_labels("Year", "No. of colonies")

plt.title('Iowa')
plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="Iowa"].sort_values('year'), marker='', color='blue', linewidth=2)

plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="Iowa"].sort_values('year'), marker='', color='olive', linewidth=2)

plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="Iowa"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")

plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="Iowa"].sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")

plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="Iowa"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")

plt.title("Iowa")

plt.legend()
g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='North Dakota'])

g.set_axis_labels("Year", "Total Production")

plt.title('North Dakota')



g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='North Dakota'])

g.set_axis_labels("Year", "Yield per col")

plt.title('North Dakota')



g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='North Dakota'])

g.set_axis_labels("Year", "No. of colonies")

plt.title('North Dakota')
plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="North Dakota"].sort_values('year'), marker='', color='blue', linewidth=2)

plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="North Dakota"].sort_values('year'), marker='', color='olive', linewidth=2)

plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="North Dakota"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")

plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="North Dakota"].sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")

plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="North Dakota"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")

plt.legend()
g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Idaho'])

g.set_axis_labels("Year", "Total Production")

plt.title('Idaho')



g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Idaho'])

g.set_axis_labels("Year", "Yield per col")

plt.title('Idaho')



g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Idaho'])

g.set_axis_labels("Year", "No. of colonies")

plt.title('Idaho')
plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="Idaho"].sort_values('year'), marker='', color='blue', linewidth=2, label="nCLOTHIANIDIN")

plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="Idaho"].sort_values('year'), marker='', color='olive', linewidth=2, label="nIMIDACLOPRID")

plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="Idaho"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")

plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="Idaho"].sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")

plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="Idaho"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")

plt.legend()
g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Washington'])

g.set_axis_labels("Year", "Total Production")

plt.title('Washington')



g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Washington'])

g.set_axis_labels("Year", "Yield per col")

plt.title('Washington')



g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Washington'])

g.set_axis_labels("Year", "No. of colonies")

plt.title('Washington')
plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="Washington"].sort_values('year'), marker='', color='blue', linewidth=2, label="nCLOTHIANIDIN")

plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="Washington"].sort_values('year'), marker='', color='olive', linewidth=2, label="nIMIDACLOPRID")

plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="Washington"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")

plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="Washington"].sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")

plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="Washington"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")

plt.legend()
g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Florida'])

g.set_axis_labels("Year", "Total Production")

plt.title('Florida')



g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Florida'])

g.set_axis_labels("Year", "Yield per col")

plt.title('Florida')



g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Florida'])

g.set_axis_labels("Year", "No. of colonies")

plt.title('Florida')
plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="Florida"].sort_values('year'), marker='', color='blue', linewidth=2)

plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="Florida"].sort_values('year'), marker='', color='olive', linewidth=2)

plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="Florida"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")

plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="Florida"].sort_values('year'), marker='', color='orange', linewidth=2,linestyle='dashed', label="nACETAMIPRID")

plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="Florida"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")

plt.legend()
g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Texas'])

g.set_axis_labels("Year", "Total Production")

plt.title('Texas')



g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Texas'])

g.set_axis_labels("Year", "Yield per col")

plt.title('Texas')



g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",

               truncate=True, size=5, data=df[df.StateName =='Texas'])

g.set_axis_labels("Year", "No. of colonies")

plt.title('Texas')
plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="Texas"].sort_values('year'), color='blue', linewidth=2)

plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="Texas"].sort_values('year'), color='olive', linewidth=2)

plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="Texas"].sort_values('year'), color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")

plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="Texas"].sort_values('year'), color='orange', linewidth=2, label="nACETAMIPRID")

plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="Texas"].sort_values('year'), color='pink', linewidth=2, label="nTHIACLOPRID")

plt.legend()

plt.title('Texas')
sns.jointplot(data=df, x='year', y='priceperlb', kind='reg', color='g')
sns.jointplot(data=df, x='year', y='stocks', kind='reg', color='g')
sns.jointplot(data=df, x='year', y='prodvalue', kind='reg', color='g')