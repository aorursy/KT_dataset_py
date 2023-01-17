import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
bli = pd.read_csv('../input/OECDBLI2017cleanedcsv.csv', index_col='Country')
bli.head()
bli.describe()
#Comparing life metrics to water quality
g = sns.pairplot(bli, vars=["Life expectancy in yrs","Dwellings without basic facilities as pct","Labour market insecurity as pct","Feeling safe walking alone at night as pct","Educational attainment as pct"]
                 , hue='Water quality as pct', size=3, diag_kind="kde", palette="Blues")
g.map_diag(plt.hist, edgecolor="w")
g = g.map_offdiag(plt.scatter, edgecolor="w", s=40)
# check highest corr vals
bli.corr().abs().stack().drop_duplicates().sort_values(ascending=False).head(n=20)
# graph linear reg. between Water quality as percentage and Life expectancy in years
plt.figure(figsize=(20,10))
sns.regplot(x=bli['Water quality as pct'], y=bli['Life expectancy in yrs'], scatter_kws={'s':50})
# graph linear reg. between Dwellings withOUT basic facilites as percentage and Life expectancy in years
plt.figure(figsize=(20,10))
sns.regplot(x=bli['Dwellings without basic facilities as pct'], y=bli['Life expectancy in yrs'], scatter_kws={'s':50})
# graph linear reg. between personal earnings in USD and Feeling safe walking alone at night as percentage
plt.figure(figsize=(20,10))
sns.regplot(x=bli['Personal earnings in usd'], y=bli['Feeling safe walking alone at night as pct'], scatter_kws={'s':100})
# check life satisfaction corr vals
bli.corr()['Life satisfaction as avg score'].abs().drop_duplicates().sort_values(ascending=False)
# graph life satisfaction with OECD-total threshhold marker
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.bar(bli.index.values, bli['Life satisfaction as avg score'], width=.3)
ax.set_ylabel('Life satisfaction as avg score', fontsize=16)
ax.set_xlabel('Country', fontsize=16)
ax.set_title('Life satisfaction as average score by country')
plt.xticks(rotation='vertical')
ax.axhline(bli.loc['OECD - Total', 'Life satisfaction as avg score'], color='r', label='OECD-Total')
plt.legend()
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(8, 15))
data = bli.sort_values("Water quality as pct", ascending=False)
# Plot the water quality
sns.set_color_codes("pastel")
sns.barplot(x="Water quality as pct", y=bli.index, data=data,
            label="Water quality", color="b")

# Plot the Feeling safe walking alone at night
sns.set_color_codes("muted")
sns.barplot(x="Feeling safe walking alone at night as pct", y=bli.index, data=data,label="Feeling safe walking alone at night", color="b")

# Add a legend and informative axis label
ax.legend(ncol=1, loc="lower right", frameon=True)
ax.set(xlim=(0, 100), ylabel="",
       xlabel="Water quality vs Feeling safe walking alone at night")
sns.despine(left=True, bottom=True)
