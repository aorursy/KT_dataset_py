import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import missingno as msno

import matplotlib as mpl
%matplotlib inline

import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Customize Matplotlib
mpl.rcParams['font.size'] = 14
mpl.rcParams["figure.figsize"] = (10,8)
df = pd.read_csv('../input/winemag-data_first150k.csv',index_col=0)
df.head()
msno.bar(df)
sns.countplot(y="country", data=df, order=df.country.value_counts().iloc[:10].index)
plt.show()
sns.countplot(y="variety", data=df, order=df.variety.value_counts().iloc[:10].index)
plt.show()
groupby_variety = df.groupby(["country", "variety"])["variety"].count()

top_variety = groupby_variety.groupby(level=0, group_keys=False)
top3_variety = top_variety.nlargest(3)

df_top3_variety = top3_variety.loc[["US", "Italy", "France", "Spain","Chile"],:].to_frame()

df_top3_variety.columns = ["Frequency"]
df_top3_variety.index.names = ['Countries','Variety']
df_top3_variety.reset_index(inplace=True)

sns.catplot(x="Countries", y="Frequency", hue="Variety", data=df_top3_variety,
                height=7, kind="bar", palette="muted")
percentage_NaN_price = round(df.price.isnull().sum() / df.shape[0], 2)
print("Percentage of NaN Price: {}".format(percentage_NaN_price))

df.price.fillna(0, inplace=True)

ax = sns.distplot(np.log10(df.price + 1))
plt.title("Price Log distribution")
plt.show()
df.points.describe()
ax = sns.distplot(df.points)
plt.show()
points_percentage = df.points.value_counts(normalize=True) * 100
great_wines_percentage = round(points_percentage.loc[95:100].sum(), 2)
print("Proportion of great Wines>= 95: {}".format(great_wines_percentage))
ax = sns.boxplot(x="points", y="price", data=df)
plt.title("Wine Reviews vs Price")
plt.show()

subset_ratings_step5 = df.loc[df.points.isin([80, 85, 90, 95, 100]), ["price","points"]]
ax = sns.boxplot(x="points", y="price", data=subset_ratings_step5)
plt.ylim([0,1000])
plt.show()

subset_ratings_step2 = df.loc[df.points.isin([80, 82, 84, 86, 88, 90]), ["price","points"]]
ax = sns.boxplot(x="points", y="price", data=subset_ratings_step2)
plt.ylim([0,200])
plt.show()
#ANOVA
model = smf.ols(formula='points ~ price', data=df)
results = model.fit()
print("The ANOVA Test find a significant difference between these 2 variables. \n"
      "F-statistic: {} F-pvalue: {}".format(results.fvalue, results.f_pvalue))
median_top20_variety = df.loc[:, ["variety", "price"]] \
         .groupby("variety") \
         .median() \
         .sort_values(by="price", ascending= False)  \
         .index[:20] \
         .values
plt.figure(figsize=(10,8))
ax = sns.boxplot(x="price", y="variety",
                 data=df.loc[df["variety"].isin(median_top20_variety),:],
                 order=median_top20_variety
                )
ax = sns.swarmplot(x="price", y="variety", 
                   data=df.loc[df["variety"].isin(median_top20_variety),:],
                   color=".25")
plt.title("Price Distribution by Variety")
plt.xlim([0,800])
plt.show()
top20_wineries = df.winery.value_counts(normalize=True).nlargest(20) \
.sort_values(ascending= False).index

plt.figure(figsize=(12,8))
sns.countplot(y="winery", data=df[df.winery.isin(top20_wineries)],
              order=top20_wineries, hue="country")
plt.show()
french_df = df.loc[df["country"] == "France",:]
french_df.shape
median_price_variety_fr = french_df.loc[:, ["variety", "price"]] \
         .groupby("variety") \
         .median() \
         .sort_values(by="price", ascending= False)      
median_price_top10_variety_fr = median_price_variety_fr.index[:10]

plt.figure(figsize=(10,8))
ax = sns.boxplot(x="price", y="variety",
                 data=french_df.loc[french_df["variety"].isin(median_price_top10_variety_fr),:],
                 order=median_price_top10_variety_fr
                )
plt.title("Price Distribution by Variety")
plt.xlim([0,800])
plt.show()
top10_variety_fr = french_df.variety.value_counts(normalize=True).nlargest(10) \
.sort_values(ascending= False).index
median_ratings_top10_variety_fr = french_df.loc[french_df["variety"]
         .isin(top10_variety_fr), ["variety", "points"]] \
         .groupby("variety") \
         .median() \
         .sort_values(by="points", ascending= False) \
         .index

plt.figure(figsize=(10,8))
ax = sns.boxplot(x="points", y="variety",
                 data=french_df.loc[french_df["variety"].isin(top10_variety_fr),:],
                 order=median_ratings_top10_variety_fr
                )
plt.title("Ratings by Variety")
plt.show()
top20_wineries_fr = french_df.winery.value_counts(normalize=True).nlargest(20) \
.sort_values(ascending= False).index

plt.figure(figsize=(12,8))
sns.countplot(y="winery", data=french_df[french_df.winery.isin(top20_wineries_fr)],
              order=top20_wineries_fr, hue="province")
plt.show()