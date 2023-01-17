# Импортируем библиотеки



import pandas as pd

import numpy as np

import seaborn as sns

import warnings

warnings.simplefilter('ignore')

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
games = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')

games.info()
games = games.dropna()



del games['Year_of_Release']

del games['Developer']

del games['Genre']

del games['Platform']

del games['Critic_Count']

del games['User_Count']

del games['Rating']



games['User_Score'] = games['User_Score'].astype('float64')
games.head()
games.info()
games.isnull().sum()
print ('Число уникальных издателей равно %i' % len(games.Publisher.unique()))

games.groupby('Publisher').size().sort(inplace=False, ascending=False)[:30]
games.describe()
print ('Мода для оценок критиков равна %f, мода для оценок игроков равна %f.' % \

       (games.Critic_Score.mode(), games.User_Score.mode()))
# дисперсия 



games.var()
features = list(set(games.columns) - set(['Name', 'Publisher']))



games[features].hist(figsize=(20,12))
fig = plt.subplots(figsize=(12, 5))



sns.distplot(np.log(games.User_Score + 1), label = 'User Score Log Normalized')

sns.distplot(games.User_Score, label = 'User Score Not Normalized')



plt.legend()
sns.distplot(np.log(games.Critic_Score + 1), label = 'Critic Score Log Normalized')

plt.legend()
corr_matrix = games.drop(['Publisher', 'Name'], axis=1).corr()



sns.heatmap(corr_matrix)
corr_matrix
g = sns.jointplot(x = 'Critic_Score', 

              y = 'User_Score',

              data = games, 

              kind = 'hex', 

              cmap= 'hot', 

              size=6)



sns.regplot(games.Critic_Score, games.User_Score, ax=g.ax_joint, scatter=False, color='grey');
regres_analysis = games[['Publisher', 'NA_Sales', 'Critic_Score']] # выделенная для анализа таблица с данными

print ('Число наблюдений - %i'  % len(regres_analysis))



regres_analysis.head()
n = len(regres_analysis)  # число наблюдений



var_score = regres_analysis.Critic_Score.var() # дисперсия оценок

var_sales = regres_analysis.NA_Sales.var() # дисперсия продаж 



sum_sales_by_scores = sum(regres_analysis.NA_Sales * regres_analysis.Critic_Score) # сумма попарных произведений показателей



mean_score = regres_analysis.Critic_Score.mean() # среднее значение для признака-фактора - оценки критика

mean_sales = regres_analysis.NA_Sales.mean() # среднее значение для признака-результата - объема продаж



# w1 и w2 - параметры уравнения линейной регрессии



w2 = ((sum_sales_by_scores / n) - (mean_sales * mean_score)) / var_score

w1 = mean_sales - (w2 * mean_score)



print (w1)

print (w2)
([regres_analysis.Critic_Score.tolist(), [(4.469281537212315 + 0.0162907217089 * x) for x in regres_analysis.Critic_Score.tolist()]])
regres_analysis.NA_Sales[0]