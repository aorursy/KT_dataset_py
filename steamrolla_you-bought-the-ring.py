import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



salary = pd.read_csv("../input/salary.csv")

teams = pd.read_csv("../input/team.csv")



groupedSalary = salary[salary['year'] >= 1985]['salary'].groupby([salary['year'], salary['team_id']]).sum()

wsWinners = teams[(teams['year'] >= 1985) & (teams['ws_win'] == 'Y')][['year', 'team_id', 'name']]



for index, row in wsWinners.iterrows():

    yearlySalaries = groupedSalary[row['year']]

    wsWinners.set_value(index, 'payroll', yearlySalaries[row['team_id']])

    wsWinners.set_value(index, 'payroll_rank', yearlySalaries.rank(ascending=False)[row['team_id']])

    

wsWinners[['year', 'name', 'payroll', 'payroll_rank']]
winData = wsWinners['payroll_rank'].value_counts().reset_index()

winData.columns = ['payroll_rank', 'win_count']



plt.bar(winData['payroll_rank'], winData['win_count'], color = 'red')

plt.axis([0, 25, 0, 5])

plt.ylabel('World Series Wins')

plt.xlabel('Payroll Rank')

plt.title('World Series Wins by Payroll')