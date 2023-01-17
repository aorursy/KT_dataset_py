

import pandas as pd

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn.cluster import KMeans

color = sns.color_palette()

%matplotlib inline
attendance_df = pd.read_csv("../input/nba_2017_attendance.csv");attendance_df.head()
endorsement_df = pd.read_csv("../input/nba_2017_endorsements.csv");endorsement_df.head()

valuations_df = pd.read_csv("../input/nba_2017_team_valuations.csv");valuations_df.head()

salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()
pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()

br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()

elo_df = pd.read_csv("../input/nba_2017_elo.csv");elo_df.head()

attendance_df.describe()
endorsement_df.describe()
valuations_df.describe()
salary_df.describe()
pie_df.describe()
plus_minus_df.describe()
br_stats_df.describe()
elo_df.describe()
valuations_df.hist()
salary_df.hist()
attendance_valuation_df = attendance_df.merge(valuations_df, how="inner", on="TEAM")
attendance_valuation_df.head()

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"));sns.pairplot(attendance_valuation_df, hue="TEAM")
import numpy as np

corr = attendance_valuation_df.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
valuations = attendance_valuation_df.pivot("TEAM", "AVG", "VALUE_MILLIONS")



plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Team AVG Attendance vs Valuation in Millions:  2016-2017 Season")

sns.heatmap(valuations,linewidths=.5, annot=True, fmt='g')
results = smf.ols('VALUE_MILLIONS ~AVG', data=attendance_valuation_df).fit()

print(results.summary())

sns.residplot(y="VALUE_MILLIONS", x="AVG", data=attendance_valuation_df)

attendance_valuation_elo_df = attendance_valuation_df.merge(elo_df, how="inner", on="TEAM")

attendance_valuation_elo_df.head()

corr_elo = attendance_valuation_elo_df.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr_elo, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

#ax.set_title("NBA Team Correlation Heatmap:  2016-2017 Season (ELO, AVG Attendance, VALUATION IN MILLIONS)")

f, ax = plt.subplots(figsize=(11, 9))

ax.set_title("NBA Team Correlation Heatmap:  2016-2017 Season (ELO, AVG Attendance, VALUATION IN MILLIONS)")

sns.heatmap(corr_elo, mask=mask, cmap=cmap, vmax=1, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr_elo



ax = sns.lmplot(x="ELO", y="AVG", data=attendance_valuation_elo_df, hue="CONF", size=12)

ax.set(xlabel='ELO Score', ylabel='Average Attendence Per Game', title="NBA Team AVG Attendance vs ELO Ranking:  2016-2017 Season")

attendance_valuation_elo_df.groupby("CONF")["ELO"].median()

attendance_valuation_elo_df.groupby("CONF")["AVG"].median()

results = smf.ols('AVG ~ELO', data=attendance_valuation_elo_df).fit()

print(results.summary())

from sklearn.cluster import KMeans

k_means = KMeans(n_clusters=3)

cluster_source = attendance_valuation_elo_df.loc[:,["AVG", "ELO", "VALUE_MILLIONS"]]

kmeans = k_means.fit(cluster_source)

attendance_valuation_elo_df['cluster'] = kmeans.labels_

ax = sns.lmplot(x="ELO", y="AVG", data=attendance_valuation_elo_df,hue="cluster", size=12, fit_reg=False)

ax.set(xlabel='ELO Score', ylabel='Average Attendence Per Game', title="NBA Team AVG Attendance vs ELO Ranking Clustered on ELO, AVG, VALUE_MILLIONS:  2016-2017 Season")
kmeans.__dict__

kmeans.cluster_centers_

cluster_1 = attendance_valuation_elo_df["cluster"] == 1

attendance_valuation_elo_df[cluster_1]
