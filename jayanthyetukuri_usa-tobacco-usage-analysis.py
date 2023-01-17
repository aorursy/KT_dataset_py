import numpy as np

import matplotlib.pyplot as plt



import pandas as pd

tobaccoUseData = pd.read_csv('../input/tobacco-use/tobacco.csv', usecols=[0, 1, 2, 3, 4, 5])
tobaccoUseData = tobaccoUseData.rename(columns={'Year':'year',

                                            'State':'state',

                                            'Smoke everyday':'dailySmoker', 

                                            'Smoke some days':'nonDailySmoker', 

                                            'Former smoker':'formerSmoker', 

                                            'Never smoked':'nonSmoker'})

for col in tobaccoUseData.columns[2:]:

    tobaccoUseData[col] = tobaccoUseData[col].str.rstrip('%')

    tobaccoUseData[col] = pd.to_numeric(tobaccoUseData[col])

tobaccoUseData.info()
plt.figure(figsize=(9, 4))

ax = plt.subplot(111)

ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)

ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left()

plt.ylim(0, 90)    

plt.xlim(1995, 2010)



plt.title("Change in smokers [in %]", fontsize=16)

plt.plot(tobaccoUseData.sort_values(['year']).groupby(['year']).mean()['dailySmoker'])

plt.plot(tobaccoUseData.sort_values(['year']).groupby(['year']).mean()['nonDailySmoker'])

plt.plot(tobaccoUseData.sort_values(['year']).groupby(['year']).mean()['formerSmoker'])

plt.plot(tobaccoUseData.sort_values(['year']).groupby(['year']).mean()['nonSmoker'])

legend = ax.legend(loc='upper left')
plt.figure(figsize=(6, 15))

ax = plt.subplot(111)

ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False)

ax.get_xaxis().tick_bottom()    

ax.get_yaxis().tick_left()



plt.title("State wise percentage change in smokers", fontsize=16)

tobaccoUse1995 = tobaccoUseData.sort_values(['state']).loc[tobaccoUseData['year'] == 2010].groupby(['state']).mean()

tobaccoUse2010 = tobaccoUseData.sort_values(['state']).loc[tobaccoUseData['year'] == 1995].groupby(['state']).mean()

tobaccoUseChangeDF = tobaccoUse1995 - tobaccoUse2010

tobaccoUseChangeDF = tobaccoUseChangeDF[pd.notnull(tobaccoUseChangeDF['dailySmoker'])]



import seaborn as sns

ax = sns.heatmap(tobaccoUseChangeDF.drop(['year'],axis=1))
import pandas as pd

tobaccoBanDetails = pd.read_csv('../input/tobacco-ban-details-in-usa-states/TobaccoBanUSAstates.csv')

banGroups = tobaccoBanDetails.groupby(tobaccoBanDetails.banDetails)

#banInPublic

#banInRestu

#banInRestuBar

#banInRestuWork

#banInWork

#noBan

fig = plt.figure()

ax = fig.add_subplot(111)

ax.spines["top"].set_visible(False)

ax.spines["bottom"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_visible(False)

plt.ylim(0, 70)

#df2.plot(kind='bar', color='mediumturquoise', ax=ax, position=1, width=0.25)

tobaccoUseData[''] = 0

colors = ['lightcoral', 'lightseagreen', 'sandybrown', 'mediumseagreen', 'salmon', 'mediumturquoise']

i = -2

for gr in banGroups.groups:

    df = tobaccoUseData[tobaccoUseData['state'].isin(

                        banGroups.get_group(gr)['state'])].drop(['year'],axis=1).groupby(['state']).mean().mean()

    df.plot(kind='bar', color=colors[i], ax=ax, position=i, width=0.1)

    plt.text(-0.5, 60-((i+2)*4), gr, color=colors[i], fontweight='bold', fontsize=14)

    i+=1



plt.title("Percentage of smokers arranged by ban imposed by State", fontsize=16)
