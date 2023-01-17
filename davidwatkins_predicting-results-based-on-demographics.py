import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

%matplotlib inline 

'''
To add:
  - Modular code
  - Write data files w/ statistics
  - Try/Except for read_csv
  - Correlation matrix, w/ p-values
  - Final list of input variables
  
GH: Multiple regression
'''
#add try/except:
demographicData = pd.read_csv("../input/county_facts.csv")
results = pd.read_csv("../input/primary_results.csv")

#Preview the data
results.head()
resultsParty = results[results.party == "Democrat"].reset_index()
#resultsParty = results[results.candidate == "Bernie Sanders"].reset_index()
resultsGrouped = resultsParty.groupby(["state_abbreviation", "county"])
#winner = resultsParty.loc[resultsGrouped['fraction_votes'].transform('idxmax'),'candidate'].reset_index()
#resultsParty["winner"] = winner['candidate']
resultsParty["totalVotes"] = resultsParty["votes"]
votes = resultsGrouped.agg({"votes": max, "fraction_votes": max, "candidate": "first", "totalVotes": sum, "state_abbreviation": "first"})
availableStates = results.state_abbreviation.unique()
#availableStates = ['OK']#, 'OH', 'NH']
#availableStates = ['SC', 'NH']
availableStatesDemoData = demographicData[demographicData.state_abbreviation.isin(availableStates)]\
                                [['state_abbreviation', 'area_name', 'RTN131207', 'INC110213', 'RHI725214', 'RHI825214', 'RHI225214', 'EDU685213',\
                                  'SEX255214','SBO015207','PST045214','POP645213','POP815213', 'POP060210', 'PVY020213']].reset_index()
availableStatesDemoData.rename(columns={'area_name':'county', 'HSD310213': 'persons_per_household', 'HSG495213': 'median_home_value',
                                'LFE305213': 'mean_commute', 
                                'INC910213': 'income_per_capita', 'INC110213':'income', 'RTN131207': 'retail_per_capita', 'RHI725214':'hispanic', 
                                'RHI825214':'white', 'EDU685213':'bachelors', 'EDU635213': 'highschool', 'SEX255214':'females',\
                                'SBO015207':'femaleFirmOwner', 'POP060210': 'density', 'PST045214':'population','POP815213':'nonEn_language',\
                                'POP645213':'notBornInUS', 'RHI225214':'black', 'PVY020213':'poverty'}, inplace=True)
availableStatesDemoData['county'] = availableStatesDemoData['county'].str.replace(' County', '')
del availableStatesDemoData['index']
availableStatesDemoData["income"] = availableStatesDemoData["income"]/1000
availableStatesDemoData = availableStatesDemoData.set_index(["state_abbreviation", "county"])
allData = pd.merge(votes, availableStatesDemoData, how="inner", left_index=True, right_index=True)
allData["turnout"] = allData.totalVotes/allData.population

#log-transform turnout due to compressed distribution
allData["turnout_log"] = np.log(allData.turnout)

print(pearsonr(allData.fraction_votes, allData.income))
sns.pairplot(allData, hue="candidate", kind="reg", aspect=.85,
             x_vars = ["income", "hispanic", "white", "poverty", "black", "density", "turnout_log"], 
             y_vars = ["fraction_votes"])
markerSize = (0.01+(allData.fraction_votes - allData.fraction_votes.min())/\
              (allData.fraction_votes.max() - allData.fraction_votes.min()))*300
g = sns.lmplot(x="income", y="fraction_votes", data=allData, hue="candidate", ci=False, scatter_kws={'s':markerSize}, size=6,
              legend_out=False)
for ax in g.axes.flat:
    ax.tick_params(labelsize=20,labelcolor="black")
g.set_xlabels("income (k$)", size = 30,color="r",alpha=0.5)
g.set_ylabels("fraction_votes", size = 30,color="r",alpha=0.5)
g.set(xlim=(0, 100), ylim=(0, 1))
g = sns.lmplot(x="white", y="fraction_votes", data=allData, hue="winner", ci=False, scatter_kws={'s':markerSize}, size=6,
              legend_out=False)
for ax in g.axes.flat:
    ax.tick_params(labelsize=20,labelcolor="black")
g.set_xlabels("white percentage", size = 30,color="r",alpha=0.5)
g.set_ylabels("fraction_votes", size = 30,color="r",alpha=0.5)
g.set(xlim=(20, 100), ylim=(0, 1))
g = sns.lmplot(x="females", y="femaleFirmOwner", data=allData, hue="winner", ci=False, scatter_kws={'s':markerSize}, 
                   size=6, legend_out=True)
for ax in g.axes.flat:
    ax.tick_params(labelsize=20,labelcolor="black")
g.set_xlabels("female (%)", size = 30,color="r",alpha=0.5)
g.set_ylabels("female firm owner (%)", size = 30,color="r",alpha=0.5)
g.set(xlim=(40, 55), ylim=(-2, 37))
g = sns.lmplot(x="white", y="turnout", data=allData, hue="winner", ci=False, scatter_kws={'s':markerSize}, 
                   size=6, legend_out=True, fit_reg=False)
for ax in g.axes.flat:
    ax.tick_params(labelsize=20,labelcolor="black")
g.set_xlabels("white (%)", size = 30,color="r",alpha=0.5)
g.set_ylabels("turnout (%)", size = 30,color="r",alpha=0.5)
g.set(xlim=(20, 100), ylim=(-0.01, 0.27))
g = sns.lmplot(x="income", y="turnout", data=allData, hue="winner", ci=False, scatter_kws={'s':markerSize}, 
                   size=6, legend_out=True, fit_reg=False)
for ax in g.axes.flat:
    ax.tick_params(labelsize=20,labelcolor="black")
g.set_xlabels("income (k$)", size = 30,color="r",alpha=0.5)
g.set_ylabels("turnout (%)", size = 30,color="r",alpha=0.5)
g.set(xlim=(20, 80), ylim=(-0.01, 0.27))
g = sns.lmplot(x="notBornInUS", y="nonEn_language", data=allData, hue="winner", ci=False, scatter_kws={'s':markerSize}, 
                   size=6, legend_out=True, fit_reg=False)
for ax in g.axes.flat:
    ax.tick_params(labelsize=20,labelcolor="black")
g.set_xlabels("not Born In US (%)", size = 30,color="r",alpha=0.5)
g.set_ylabels("persons not speaking English (%)", size = 30,color="r",alpha=0.5)
g.set(xlim=(0, 25), ylim=(0, 35))
