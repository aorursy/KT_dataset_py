# Import all of the usual libraries for analysis...
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ...set appropriate chart type for Jupyter notebook formatting
%matplotlib inline
sns.set(font_scale=1.3)
plt.rcParams['figure.figsize'] = [14, 7]
# Load data
penData = pd.read_csv('../input/penalties_stats_aa.csv')
penData.head()
# Calculate penalties per match metric
penData['PenaltiesPerMatch'] = penData['PenaltiesAwarded']/penData['MatchesPlayed']
sns.barplot(x='Year', y='PenaltiesPerMatch', data=penData, ci=None, palette=['skyblue']*(penData.shape[0]-1) + ['steelblue'])
plt.ylabel("Penalties Awarded per World Cup Match");
# Look at outliers (roughly)
sns.boxplot(penData.PenaltiesPerMatch)
# Add penalties to VAR to the dataset
penData['VARScore'] = [0]*(penData.shape[0]-1) + [6]
penData['VARMiss'] = [0]*(penData.shape[0]-1) + [3]
penData['VARSave'] = [0]*(penData.shape[0]-1) + [1]
# Suppose that none of the VAR penalties were awarded
woVARPenPerMatch = (penData.loc[20, 'PenaltiesAwarded'] - penData.iloc[20, 8:11].sum()) / penData.loc[20, 'MatchesPlayed']
print("2018 Penalty Rate without VAR penalties: %.3f" % woVARPenPerMatch)
# Compare to previous years
penData[woVARPenPerMatch < penData.PenaltiesPerMatch].loc[[10, 12, 13], ['Year', 'Host', 'PenaltiesPerMatch']]
# Scneario Analysis
VARScenarios = penData.append([penData], ignore_index=True)
VARScenarios['VARScenario'] = ['Excluding all VAR penalties']*(penData.shape[0]) + ['Include VAR-overturned penalties']*(penData.shape[0])
VARScenarios.loc[20,'PenaltiesPerMatch'] = woVARPenPerMatch
VARScenarios.loc[41,'PenaltiesPerMatch'] = (penData.loc[20, 'PenaltiesAwarded'] - penData.iloc[20, 8:11].sum() + 3) / penData.loc[20, 'MatchesPlayed']
# Scneario Analysis visual
sns.set(font_scale=1.1)
g = sns.FacetGrid(VARScenarios, row="VARScenario", aspect=3)
g = g.map(sns.barplot, 'Year', 'PenaltiesPerMatch')
# Converted penalties calculation
penData['ScoreRate'] = penData['PenaltiesScored'] / penData['PenaltiesAwarded']
penData['MissRate'] = (penData['PenaltiesAwarded'] - penData['PenaltiesScored']) / penData['PenaltiesAwarded']
# Plot converted penalties rate over time
plt.stackplot(penData.Year, [penData.ScoreRate, penData.MissRate], labels=['Miss', 'Score'])
plt.xlim(min(penData.Year),max(penData.Year))
plt.ylim(0, 1);
print("Average in-game penalty conversion rate: %.1f%%" % (100*penData.ScoreRate.sum()/21))