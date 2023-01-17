import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings 

warnings.filterwarnings('ignore' )
# Read the dataset

df = pd.read_csv('../input/ls-results-constituency-wise-2014/Data_LS.csv')
# Displaying first 5 enteries

df.head()
# Displaying last 5 enteries

df.tail()
# To find number of rows and columns



print("There are {} rows and {} columns in the dataframe.".format(df.shape[0],df.shape[1]))
# To know datatypes and missing values if any



df.info()
# Total number of votes polled



total_votes = df.groupby(['Name of State/ UT']).agg({'Total Votes Polled':sum}).sum()

print("Total number of votes polled during 2014 Lak sabha election was : {}".format(total_votes[0]))
# Statewise total number of voters



total_votes_statewise = df.groupby(['Name of State/ UT']).agg({'Total Votes Polled':sum}).sort_values(

    'Total Votes Polled', ascending=False).reset_index()



plt.figure(figsize=(15,5))

plt.title('Total number of people casted vote in perticular state',fontweight="bold")

sns.barplot('Name of State/ UT','Total Votes Polled', data=total_votes_statewise)

plt.xticks(rotation=90)

plt.show()
# Total votes partywise



partywise_total_vate = df.groupby(['Party Name']).agg({'Total Votes Polled':sum}).sort_values(

    'Total Votes Polled',ascending=False).reset_index().head(25)



plt.figure(figsize=(15,5))

plt.title('Total number of people casted vote to perticular political party',fontweight="bold")

sns.barplot('Party Name','Total Votes Polled',data=partywise_total_vate)

plt.xticks(rotation=90)

plt.show()
# Statewise top party that recieved total number of votes



swtpv =df.groupby(['Name of State/ UT','Party Name']).agg({'Total Votes Polled':sum}).sort_values([

    'Name of State/ UT','Total Votes Polled'],ascending=False).reset_index().drop_duplicates(

    'Name of State/ UT').reset_index().drop('index',axis=1)

swtpv
# pie chart of percentage of candidates won out of total candidate participated



plt.figure(figsize=(10,7))

plt.title('Pie Chart for percentage of candidates won',fontweight="bold")

plt.pie(df['Winner or Not?'].value_counts(),autopct='%1.1f%%', explode=(0,0.1),labels=df['Winner or Not?'

                                                                                        ].value_counts().index.tolist())

plt.show()
# statewise top 5 party with highest total number of votes 



top_5 = df.groupby(['Name of State/ UT','Party Name']).agg({'Total Votes Polled':sum}).sort_values([

    'Name of State/ UT','Total Votes Polled'],ascending=False).groupby(['Name of State/ UT']).head(5)

top_5
# statewise number of seats a party won



df[df['Winner or Not?']=='yes'].groupby(['Name of State/ UT','Party Name']).agg({'Winner or Not?':'count'})
# Top 15 party with most number of seats won



Top_15 = df[df['Winner or Not?']=='yes'].groupby('Party Name').agg({"Winner or Not?": 'count'}).sort_values(

    "Winner or Not?",ascending=False).rename(columns={"Winner or Not?": 'Seats_won'}).head(15)



Top_15
a=Top_15.reset_index()

plt.figure(figsize=(15,3))

plt.title('Top 15 political party in terms of seats won',fontweight="bold")

sns.barplot(x=a['Party Name'],y=a['Seats_won'])

plt.xticks(rotation=90)

plt.show()
# Grouping

a=df[df['Winner or Not?']=='yes'].groupby('Party Name').agg({"Winner or Not?": 'count'}).sort_values(

    "Winner or Not?",ascending=False).rename(columns={"Winner or Not?": 'Seats_won'})



# Taking top 15 party by number of seats and remaining put in to other

other_value = a.sum()[0]-a.head(15).sum()[0]



a=a.head(15).reset_index()



# DataFrame for other

b=pd.DataFrame({'Party Name':['Other'],

             'Seats_won':[other_value]})



# Joing the other with the top 15

a=a.append(b,ignore_index=True)



data=a.Seats_won.tolist()

label=a['Party Name'].tolist()

plt.figure(figsize=(20,8))

plt.title('Percentage of seats won per political party',fontweight="bold")

plt.pie(data,labels=label, autopct='%1.1f%%')

plt.show()
# Taking out all the candiates that has df_won

df_won = df[df['Winner or Not?']=='yes'].reset_index().drop('index',axis=1)



# Taking column name is list

constituency_name = pd.unique(df_won['Parliamentary Constituency'])



# Creating new column as assigning with 'NaN'

df_won['Vote_margin'] = np.nan
# For loop to calculate vote margin and filling it to its respective place

for i in constituency_name:

    row = df[df['Parliamentary Constituency']== i ].sort_values('Total Votes Polled', ascending=False)

    first = row.iloc[0][3]

    second = row.iloc[1][3]

    difference = first - second

    df_won[df_won['Parliamentary Constituency']== i] = df_won[df_won['Parliamentary Constituency']== i].fillna(difference)

    

# Since the column 'Vote_margin' is in float, changing it to int

df_won['Vote_margin']=df_won['Vote_margin'].astype(int)
# Bottom 10 candidates in turms of vote vargin

buttom_10_margin = df_won.sort_values('Vote_margin',ascending=False).tail(10)
# Plotting for Bottom 10 candidates in turms of vote vargin

plt.figure(figsize=(15,5))

plt.title('Bottom 10 candiadtes with lowest win margin',fontweight="bold")

sns.barplot('Candidate Name','Vote_margin',hue='Party Abbreviation',data=buttom_10_margin, dodge=False)

plt.xticks(rotation=90)

plt.show()
# Top 10 candidates in turms of vote vargin

top_10_margin = df_won.sort_values('Vote_margin',ascending=False).head(10)
# Plotting for Top 10 candidates in turms of vote vargin

plt.figure(figsize=(15,5))

plt.title('Top 10 candiadtes with highest win margin',fontweight="bold")

sns.barplot('Candidate Name','Vote_margin',hue='Party Abbreviation',data=top_10_margin, dodge=False)

plt.xticks(rotation=90)

plt.show()
# Maximum vote margin

df_won[df_won['Vote_margin']==max(df_won['Vote_margin'])]
# Minimum vote margin

df_won[df_won['Vote_margin']==min(df_won['Vote_margin'])]