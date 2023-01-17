# The Usual Imports



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use("fivethirtyeight")
# Reading in the data and converting date column to datetime

df = pd.read_csv('../input/2013-to-2019-World-Class-DCI-scores.csv')

df.Date = pd.to_datetime(df.Date)
# Bar Chart of the Number of First Place Finishes throughout the 2019 Summer Tour

ax = df.query("Date > '2019-06-01' and Rank == '1st'").Corps.value_counts().plot(kind = 'barh', figsize = (15,8))

ax.set_title('Number of 1st Place Finishes by Corps in 2019')

ax.set_ylabel('Corps')

ax.set_xlabel('1st Place Finishes');

ax.set_xticks(range(0,df.query("Date > '2019-06-01' and Rank == '1st'").Corps.value_counts().max()+1));
# New Dataframe for just 2019 scores

df_2019 = df.query("Date >= '2019-06-01'")
# Creating a dataframe to use to show the different Corps scores throughout the summer.

# Used ffill method to indicate when Corps were off and not performing



corps_of_interest = ['Bluecoats', 'Blue Devils', 'Santa Clara Vanguard', 'Carolina Crown', 'Boston Crusaders',

                    'The Cavaliers']

df_2019.pivot_table(index='Date',columns='Corps',values='Score')[corps_of_interest].fillna(method='ffill')
df_2019.pivot_table(index='Date',columns='Corps',values='Score')[corps_of_interest].fillna(method='ffill').plot.line(style='o-', figsize = (15, 8))



plt.title('2019 Score Comparision of Top 6 Corps Throughout the Summer Tour')

plt.ylabel('Score')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))



plt.tight_layout()

plt.gcf().autofmt_xdate()

plt.show();
corps_of_interest_2 = ['Blue Knights', 'Blue Stars', 'The Cadets', 'Mandarins', 'Crossmen', 'Phantom Regiment']

df_2019.pivot_table(index='Date',columns='Corps',values='Score')[corps_of_interest_2].fillna(method='ffill')
df_2019.pivot_table(index='Date',columns='Corps',values='Score')[corps_of_interest_2].fillna(method='ffill').plot.line(style='o-', figsize = (15, 8))



plt.title('2019 Score Comparision of Top 7-12 Corps Throughout the Summer Tour')

plt.ylabel('Score')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))



plt.tight_layout()

plt.gcf().autofmt_xdate()

plt.show();