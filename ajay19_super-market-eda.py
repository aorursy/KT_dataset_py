# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
sm_branches = pd.read_csv('/kaggle/input/marketing-data-for-a-supermarket-in-united-states/supermarket_marketing/50_SupermarketBranches.csv')

sm_members = pd.read_csv('/kaggle/input/marketing-data-for-a-supermarket-in-united-states/supermarket_marketing/Supermarket_CustomerMembers.csv')



sm_branches.shape,sm_members.shape
sm_branches.head()
sm_members.head()
countsT = sm_members['Genre'].value_counts()

labels = 'Female' ,'Male'

sizes = countsT.values

explode = (0.1, 0.1) 

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)

ax1.axis('equal')  

plt.show()
age_values = sm_members.Age.value_counts()

age_labels = age_values.index

plt.figure(figsize=(15,8))

ax = sns.barplot(x=age_labels,y=age_values)

ax.set_title('Age distribution')

ax.set_ylabel('Count')

ax.set_xlabel('Age')
#Average spending by a customer

fm_spending_avg = sm_members.groupby('Genre')['Spending Score (1-100)'].sum()['Female']/112

print('Average spending by a female customer is {}'.format(fm_spending_avg))



m_spending_avg = sm_members.groupby('Genre')['Spending Score (1-100)'].sum()['Male']/88

print('Average spending by a Male customer is {}'.format(m_spending_avg))



avg_spending = sm_members['Spending Score (1-100)'].sum()/200

print('Average spending by a customer is {}'.format(avg_spending))
#Average income of a customer

fm_income_avg = sm_members.groupby('Genre')['Annual Income (k$)'].sum()['Female']/112

print('Average income of a female customer is {}k'.format(fm_income_avg))



m_income_avg = sm_members.groupby('Genre')['Annual Income (k$)'].sum()['Male']/88

print('Average income of a Male customer is {}k'.format(m_income_avg))



avg_income = sm_members['Annual Income (k$)'].sum()/200

print('Average income of a customer is {}k'.format(avg_income))
xn = sm_members.groupby('Age')['Annual Income (k$)'].sum()

xn.reset_index(drop=False)

values = xn.values

labels = xn.index

plt.figure(figsize=(15,8))

ax=sns.barplot(x=labels,y=values)

ax.set_title('Age vs Annual Income plot')

ax.set_ylabel('Annual Income Sum in K')

ax.set_xlabel('Age')

plt.figure(figsize=(15,8))
xn = sm_members.groupby('Age')['Spending Score (1-100)'].sum()

xn.reset_index(drop=False)

values = xn.values

labels = xn.index

plt.figure(figsize=(15,8))

ax=sns.barplot(x=labels,y=values)

ax.set_title('Age vs Spending plot')

ax.set_ylabel('Sum of spendings')

ax.set_xlabel('Age')
sm_branches.head(5)
plt.figure(figsize=(8,5))

plt.xticks(rotation=45)

sns.set()

sns.set(style="darkgrid")

ax = sns.countplot(x=sm_branches['State'], data=sm_branches)

ax.set_title('Count of branches in each state')

ax.set_ylabel('No. of Branches')

ax.set_xlabel('State')
coun = 0

coun1 = 0

for i in sm_branches['State']:

    if i == 'Florida':

        coun = coun+1

    if i == 'New York':

        coun1 = coun1+1

print(coun,coun1)
#Florida

florida_profit = sm_branches.groupby('State')['Profit'].sum()['Florida']

print('Total profit at all the branches in florida is {}'.format(florida_profit))

print('Avg profit at branches in florida is {}'.format(florida_profit/16))

print('\n')

#New York

new_york_profit = sm_branches.groupby('State')['Profit'].sum()['New York']

print('Total profit at all the branches in New York is {}'.format(new_york_profit))

print('Avg profit at branches in New York is {}'.format(new_york_profit/17))

print('\n')



#California

california_profit = sm_branches.groupby('State')['Profit'].sum()['California']

print('Total profit at all the branches in California is {}'.format(california_profit))

print('Avg profit at branches in California is {}'.format(california_profit/17))
#Florida

florida_ad_spent = sm_branches.groupby('State')['Advertisement Spend'].sum()['Florida']

print('Total Advertisement Spendings at all the branches in florida is {}'.format(florida_ad_spent))

print('Avg Advertisement Spendings at branches in florida is {}'.format(florida_ad_spent/16))

print('\n')

#New York

new_york_ad_spent = sm_branches.groupby('State')['Advertisement Spend'].sum()['New York']

print('Total Advertisement Spendings at all the branches in New York is {}'.format(new_york_ad_spent))

print('Avg Advertisement Spendings at branches in New York is {}'.format(new_york_ad_spent/17))

print('\n')



#California

california_ad_spent = sm_branches.groupby('State')['Advertisement Spend'].sum()['California']

print('Total Advertisement Spendings at all the branches in California is {}'.format(california_ad_spent))

print('Avg Advertisement Spendings at branches in California is {}'.format(california_ad_spent/17))
#Florida

florida_prm_spent = sm_branches.groupby('State')['Promotion Spend'].sum()['Florida']

print('Total Promotion Spendings at all the branches in florida is {}'.format(florida_prm_spent))

print('Avg Promotion Spendings at branches in florida is {}'.format(florida_prm_spent/16))

print('\n')

#New York

new_york_prm_spent = sm_branches.groupby('State')['Promotion Spend'].sum()['New York']

print('Total Promotion Spendings at all the branches in New York is {}'.format(new_york_prm_spent))

print('Avg Promotion Spendings at branches in New York is {}'.format(new_york_prm_spent/17))

print('\n')



#California

california_prm_spent = sm_branches.groupby('State')['Promotion Spend'].sum()['California']

print('Total Promotion Spendings at all the branches in California is {}'.format(california_prm_spent))

print('Avg Promotion Spendings at branches in California is {}'.format(california_prm_spent/17))
ad_spend_list = [florida_ad_spent,new_york_ad_spent,california_ad_spent]

cities = ['Florida','New York','California']

pr_spend_list = [florida_prm_spent,new_york_prm_spent,california_prm_spent]

plt.figure(figsize=(15,8))

ax=sns.barplot(y=ad_spend_list[:],x=cities[:])

ax.set_title('Branch vs Money spent on Advertisement plot')

ax.set_ylabel('Money spent on Advertisement')

ax.set_xlabel('Branch')
plt.figure(figsize=(15,8))

ax=sns.barplot(y=pr_spend_list[:],x=cities[:])

ax.set_title('Branch vs Money spent on Promotion plot')

ax.set_ylabel('Money spent on Promotion')

ax.set_xlabel('Branch')
basket = pd.read_csv('/kaggle/input/marketing-data-for-a-supermarket-in-united-states/supermarket_marketing/Market_Basket_Optimisation.csv')

basket.head(5)
basket.info()
names = ' '

for name in basket.shrimp:

    name = str(name)

    names = names + name + ' '

from wordcloud import WordCloud, STOPWORDS 

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='black',  

                min_font_size = 10).generate(names) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
basket.shrimp.nunique()
ads = pd.read_csv('/kaggle/input/marketing-data-for-a-supermarket-in-united-states/supermarket_marketing/Ads_CTR_Optimisation.csv')

ads.head(5)
ads.info()
xn = ads.sum(axis = 0, skipna = True) 

xn.reset_index(drop=False)

values = xn.values

labels = xn.index

plt.figure(figsize=(15,8))

ax=sns.barplot(x=labels,y=values)

ax.set_title('Ads vs Count plot')

ax.set_ylabel('Count')

ax.set_xlabel('Ad Type')
def plotCorrelationMatrix(df, graphWidth):

    #filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    #plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()



plotCorrelationMatrix(sm_branches, 8)
plotCorrelationMatrix(sm_members, 8)