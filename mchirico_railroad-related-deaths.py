# You might need to uncomment, if not working on Kaggle, to
# display the graphs
#%matplotlib inline

# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the DeathRecords dataset, which is in the "../input/" directory
death = pd.read_csv("../input/DeathRecords.csv") # the DeathRecords dataset is now a Pandas DataFrame
icd = pd.read_csv("../input/Icd10Code.csv") # the Icd10Code, which gives the description
mann = pd.read_csv("../input/MannerOfDeath.csv") # Manner of Death

# Let's see what's in the death data - Jupyter notebooks print the result of the last thing you do
death.head()

# Reference:
# https://www.kaggle.com/benhamner/d/uciml/iris/python-data-visualizations
# Join 
icd.head()
# Quick look
mann.head()
# Join the tables
# 
r = pd.merge(death, mann, how='inner',left_on='MannerOfDeath', right_on='Code')


# Let's get a quick look at this
r["Description"].value_counts()
 
# Rename "Description to mDescription 
r=r.rename(columns = {'Description':'mDescription'})
# Now join with Icd10Code
r = pd.merge(r, icd, how='inner',left_on='Icd10Code', right_on='Code')
r = r.rename(columns = {'Description':'iDescription'})


# i.Description like '% railway train %'
# But store as separate variable, in case you want to
# come back to this...
t=r[r.iDescription.str.match(r'.* railway train .*')]


# Some quick checks...just curious.
# What's the manner of death?
t["mDescription"].value_counts()
# Repeated assignemnt here...not needed, but if
# manipulated the code above it can be nice.
t=r[r.iDescription.str.match(r'.* railway train .*')]
p=t.groupby(['iDescription','mDescription'])['mDescription'].agg(['count']).reset_index()
# Sort and check
p.sort_values(['count'], ascending=[False],inplace=True)
p.head()
# 

g=p.pivot('iDescription', 'mDescription')['count'].fillna(0)
g.sort_values(['Accident'], ascending=[False],inplace=True)
g.head()
# Reset index example
# See, it's cleaner
g=g.reset_index()
g.head()
# Plot Accident vs. Suicide 
# 

sns.FacetGrid(g, hue="iDescription", size=8) \
   .map(plt.scatter, "Accident", "Suicide") \
   .add_legend()
# 
data=g[(g['Could not determine'] > 0)]
data
# Plot Accident vs. Could not determine

sns.FacetGrid(data, hue="iDescription", size=8) \
   .map(plt.scatter, "Accident", "Could not determine") \
   .add_legend()