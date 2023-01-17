import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../input/totals-stats/totals_statsv4.csv")
df.head(500)
dfv2=df.loc[df['RSorPO']=='Regular Season']
g = sns.FacetGrid(dfv2, col="Player", hue="Team Outcome")
g.map(plt.scatter, "Year", "PTS", alpha=1)
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Regular Season Points Scored')
g.add_legend();
dfv3=df.loc[df['RSorPO']=='Playoffs']
g = sns.FacetGrid(dfv3, col="Player", hue="Team Outcome")
g.map(plt.scatter, "Year", "PTS", alpha=1)
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Playoffs Points Scored')
g.add_legend();
dfv2=df.loc[df['RSorPO']=='Regular Season']
g = sns.FacetGrid(dfv2, col="Player", hue="Team Outcome", palette="dark")
g.map(plt.scatter, "Year", "eFG%", alpha=1)
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Regular Season Field Goal Efficiency')
g.add_legend();
dfv3=df.loc[df['RSorPO']=='Playoffs']
g = sns.FacetGrid(dfv3, col="Player", hue="Team Outcome", palette="dark")
g.map(plt.scatter, "Year", "eFG%", alpha=1)
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Playoffs Field Goal Efficiency')
g.add_legend();
df2=df.loc[df['Player'] == 'Kobe Bryant']
df3=df2.loc[df2['RSorPO']=='Regular Season']
plt.plot( 'Year', '3P%', data=df3, marker='', color='black', linewidth=2)
plt.plot( 'Year', '2P%', data=df3, marker='', color='olive', linewidth=2, linestyle='dashed')
plt.title("Kobe Bryant - Regular Season Field Goal Efficiencies")
plt.legend()
df2=df.loc[df['Player'] == 'Kobe Bryant']
df3=df2.loc[df2['RSorPO']=='Regular Season']
df4=df2.loc[df2['RSorPO']=='Playoffs']

plt.plot( 'Year', '3P%', data=df4, marker='', color='black', linewidth=2)
plt.plot( 'Year', '2P%', data=df4, marker='', color='olive', linewidth=2, linestyle='dashed')
plt.title("Kobe Bryant - Playoffs Field Goal Efficiencies")
plt.legend()
df4=df.loc[df['Player'] == 'Lebron James']
df5=df4.loc[df4['RSorPO']=='Regular Season']
plt.plot( 'Year', '3P%', data=df5, marker='', color='black', linewidth=2)
plt.plot( 'Year', '2P%', data=df5, marker='', color='olive', linewidth=2, linestyle='dashed')
plt.title("Lebron James - Regular Season Field Goal Efficiencies")
plt.legend()
df4=df.loc[df['Player'] == 'Lebron James']
df6=df4.loc[df4['RSorPO']=='Playoffs']
plt.plot( 'Year', '3P%', data=df6, marker='', color='black', linewidth=2)
plt.plot( 'Year', '2P%', data=df6, marker='', color='olive', linewidth=2, linestyle='dashed')
plt.title("Lebron James - Playoffs Field Goal Efficiencies")
plt.legend()
df7=df.loc[df['Player'] == 'Michael Jordan']
df8=df7.loc[df7['RSorPO']=='Regular Season']
plt.plot( 'Year', '3P%', data=df8, marker='', color='black', linewidth=2)
plt.plot( 'Year', '2P%', data=df8, marker='', color='olive', linewidth=2, linestyle='dashed')
plt.title("Michael Jordan - Regular Season Field Goal Efficiencies")
plt.legend()
df7=df.loc[df['Player'] == 'Michael Jordan']
df8=df7.loc[df7['RSorPO']=='Regular Season']
df9=df7.loc[df7['RSorPO']=='Playoffs']
plt.plot( 'Year', '3P%', data=df9, marker='', color='black', linewidth=2)
plt.plot( 'Year', '2P%', data=df9, marker='', color='olive', linewidth=2, linestyle='dashed')
plt.title("Michael Jordan - Playoffs Field Goal Efficiencies")
plt.legend()
df2=df.loc[df['Player'] == 'Kobe Bryant']
df2['MP'].sum()
df3=df.loc[df['Player'] == 'Lebron James']
df3['MP'].sum()
df4=df.loc[df['Player'] == 'Michael Jordan']
df4['MP'].sum()
df2['PTS'].sum()
df3['PTS'].sum()
df4['PTS'].sum()
df2['AST'].sum()
df3['AST'].sum()
df4['AST'].sum()
df2['FG'].sum()+df2['FT'].sum()+df2['AST'].sum()+df2['STL'].sum()+df2['TRB'].sum()+df2['BLK'].sum()
df3['FG'].sum()+df3['FT'].sum()+df3['AST'].sum()+df3['STL'].sum()+df3['TRB'].sum()+df3['BLK'].sum()
df4['FG'].sum()+df4['FT'].sum()+df4['AST'].sum()+df4['STL'].sum()+df4['TRB'].sum()+df4['BLK'].sum()
df2['PTS'].sum()+(1.5*df2['AST'].sum())+df2['STL'].sum()+df2['TRB'].sum()+(1.5*df2['BLK'].sum())
df3['PTS'].sum()+(1.5*df3['AST'].sum())+df3['STL'].sum()+df3['TRB'].sum()+(1.5*df3['BLK'].sum())
df4['PTS'].sum()+(1.5*df4['AST'].sum())+df4['STL'].sum()+df4['TRB'].sum()+(1.5*df4['BLK'].sum())

from IPython.display import Image
Image("../input/numericalanalysis/GOATv3.jpg")