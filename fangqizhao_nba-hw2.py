import pandas as pd
data_df = pd.read_csv("../input/nba_2017_nba_players_with_salary.csv")
data_df.head()
data_df.drop(["Rk"], inplace=True, axis=1)
data_df.head()
import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")
corr = data_df.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap = "YlGnBu",
            vmin = 0, vmax = 1,
            annot = True )
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=data_df)
import statsmodels.formula.api as smf
results = smf.ols('SALARY_MILLIONS ~ WINS_RPM', data = data_df).fit()
print(results.summary())
sns.lmplot(x="SALARY_MILLIONS", y="POINTS", data=data_df)
import statsmodels.formula.api as smf
results = smf.ols('SALARY_MILLIONS ~ POINTS', data = data_df).fit()
print(results.summary())
sns.lmplot(x="SALARY_MILLIONS", y="PIE", data=data_df)
results = smf.ols('SALARY_MILLIONS ~ PIE', data = data_df).fit()
print(results.summary())
salary_df = pd.read_csv("../input/nba_2017_salary.csv")
team_salary = salary_df.groupby(['TEAM']).mean().reset_index()
team_salary.sort_values(by = 'SALARY',ascending = False,inplace = True)
team_salary.head(10)
plt.figure(figsize = (18,4))
sns.barplot(x=team_salary.head(10)['TEAM'], y=team_salary.head(10)['SALARY'],palette="YlGnBu", data=team_salary).set_title("Top 10 Team with highest average salary amount the league")

plt.show()
position_salary = salary_df.groupby(['POSITION']).mean().reset_index()
position_salary.sort_values(by = 'SALARY',ascending = False,inplace = True)
position_salary.head(10)
plt.figure(figsize = (12,4))
sns.barplot(x=position_salary['POSITION'], y=position_salary['SALARY'],palette="YlGnBu", data=salary_df).set_title("Position Vs Salary")


plt.show()