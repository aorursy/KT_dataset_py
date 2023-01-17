import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
datatips = pd.read_csv('../input/tips.csv')
datatips.head()
len(datatips)
sns.regplot(x='tip', y='total_bill', data=datatips, ci=None)
plt.show()
plt.hist(datatips['total_bill'], color ="red")
plt.show()
sns.distplot(datatips['total_bill'], kde=True, rug=True)
plt.show()
sns.set_style('darkgrid')
sns.boxplot(x='total_bill', y='day', data=datatips, palette='Blues', orient="h")
plt.show()
sns.boxplot(x='sex', y='tip', data=datatips, palette='Blues')
plt.show()
fig, ax= plt.subplots()
ax.scatter(datatips[datatips['sex']=='Male']['tip'], datatips[datatips['sex']=='Male']['total_bill'], color='blue', label='Муж')
ax.scatter(datatips[datatips['sex']=='Female']['tip'], datatips[datatips['sex']=='Female']['total_bill'], color='green', label='Жен')
ax.set_title('Общий счет и чаевые с распределением по полу', fontdict={'fontsize': 14, 'fontweight': 'bold'})
ax.set_ylabel('Общий счет')
ax.set_xlabel('Чаевые')
ax.legend()
plt.show()
sns.violinplot(x = "total_bill", y = "sex", data=datatips)
sns.catplot(x="day", y="total_bill", hue="smoker",
            kind="bar", data=datatips);
datapl =  pd.read_csv('../input/premier_league.csv')
datapl.head()
len(datapl)
all_goals = datapl.home_goals + datapl.away_goals
datapl['all_goals'] = pd.Series(all_goals, index = datapl.index)
datapl.head()
goals_of_season = datapl.season.value_counts().sort_values(ascending = False).head(12).index.values
sns.boxplot(y = "season", x = "all_goals", data =datapl[datapl.season.isin(goals_of_season)])
datapl.groupby('season').sum().plot(kind = 'bar', rot = 45)

plt.hist(datapl['all_goals'], color ="red")
plt.show()
result_all_goals_season = datapl.pivot_table(
                            index = 'season',
                            columns = 'result',
                            values = 'all_goals',
                            aggfunc = sum).fillna(0).applymap(float)
sns.heatmap(result_all_goals_season, annot=True, fmt='.1f', linewidths=.5)
sns.distplot(datapl['all_goals'], kde=True, rug=True)
plt.show()

