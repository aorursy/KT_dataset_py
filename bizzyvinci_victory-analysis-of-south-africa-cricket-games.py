import pandas as pd

import matplotlib.pyplot as plt

import sqlite3



%matplotlib inline
conn = sqlite3.connect('../input/south-africa-cricket-data-from-2000-to-2020/sa_cricket.db')

cursor = conn.cursor()
query = '''select

            strftime("%Y", m.match_date) as Year, 

            strftime("%m", m.match_date) as Month,

            m.match_date as Date,

            o.opp_name as Opposition,

            g.ground_name as Ground,

            g.country as Host,

            m.result as Result,

            m.toss as Toss

            from mat as m join

            opposition as o join

            ground as g

            where m.opposition=o.opp_id and m.ground=g.ground_id;

        '''

df = pd.read_sql_query(query, conn)

df.head()
def winner(result):

    '''

    Input game result e.g 'India led the 5-match series 1-0'

    Return Winner e.g 'India'

    '''

    if result == '':

        # This are non-tour games such as world cup and series

        winner = 'No result'

    else:

        if 'led' in result:

            winner = result.split(' led ')[0]

        elif 'won' in result:

            winner = result.split(' won ')[0]

        elif 'advanced' in result:

            winner = result.split(' advanced')[0]

        # There would be a need to handle cases like:

        # 3-match series level 1-1

        # 3-match series drawn 1-1

        elif 'level' in result:

            winner = 'Unknown-level'

        elif 'drawn' in result:

            winner = 'Unknown-draw'

        else:

            winner = 'Unknown'

    return winner

    '''

    The options from above are:

        [Country]

        No result

        Unknown*

    I believe it would better to somehow find winners among Unknown*s

    '''    
def point(result):

    '''

    To make things simpler so that outcome of winner is either:

        1 for Win

        0 for Unknown or No result

        -1 for Losses

    '''

    w = winner(result)

    if ('Unknown' in w) or w=='No result':

        return 0

    elif w=='South Africa':

        return 1

    else:

        return -1
'''

Get tosser and their choice from toss e.g 'South Africa , elected to bat first'

'''

def tosser(t):

    t_split = t.split(' , elected to ')

    tosser = t_split[0]

    return tosser



def toss(t):

    # Return 1 if South Africa tossed

    toss = tosser(t)

    if toss=='South Africa':

        return 1

    else:

        return 0



def choice(t):

    t_split = t.split(' , elected to ')

    choice = t_split[1]

    return choice
df['Point'] = df['Result'].apply(point)

df['Tossed'] = df['Toss'].apply(toss)

df['Toss Choice'] = df['Toss'].apply(choice)



# Drop Result and Toss after getting what's needed

df.drop(['Result', 'Toss'], axis=1, inplace=True)

df.head()
# Number of matches played each month since 2000

match_by_month = df.groupby(by='Month').count()['Point']   #/20 to get the average

match_by_month.plot(kind='bar', figsize=(14,6))

plt.title('Matches played in 2000s')

plt.ylabel('Matches')

plt.xlabel('Month')

plt.xticks(rotation=0)

plt.show()
# Number of matches played each month since 2000



'''

# This commented section does the same thing as win_by_month.

# Kept for reference purpose

match_by_month = df[['Month', 'Point']].groupby(by=['Month','Point'], as_index=False)

match_by_month.size().unstack(fill_value=0)

'''



win_by_month = df.pivot_table(index='Month', columns='Point', aggfunc='size', fill_value=0)

win_by_month.rename(columns={0: 'Unknown', -1: 'Lost', 1: 'Won'}, inplace=True)

# Total games that leads to either a win or loss

win_by_month['Total Games'] = win_by_month['Lost'] + win_by_month['Won']



# Percent of Total games won

win_by_month['Percent Won'] = (win_by_month['Won'] * 100) / win_by_month['Total Games']



# Absolute total of games played including '-' result

win_by_month['Absolute Total'] = win_by_month['Total Games'] + win_by_month['Unknown']



# Drop column name

win_by_month.columns.name=''



win_by_month
win_by_month['Percent Won'].plot(kind='bar', figsize=(14,6))

plt.xticks(rotation=0)

plt.ylabel('Victory (%)')

plt.title('Percentage of games won')

plt.show()
win_by_month.corr()['Percent Won']
win_by_year = df.pivot_table(index='Year', columns='Point', aggfunc='size', fill_value=0)

win_by_year.rename(columns={0: 'Unknown', -1: 'Lost', 1: 'Won'}, inplace=True)
# Total games that leads to either a win or loss

win_by_year['Total Games'] = win_by_year['Lost'] + win_by_year['Won']



# Percent of Total games won

win_by_year['Percent Won'] = (win_by_year['Won'] * 100) / win_by_year['Total Games']



# Absolute total of games played including '-' result

win_by_year['Absolute Total'] = win_by_year['Total Games'] + win_by_year['Unknown']



# Drop column name

win_by_year.columns.name=''



win_by_year
plt.figure(figsize=(14,6))

plt.plot(win_by_year.index, win_by_year['Percent Won'], '--bo')

plt.xticks(rotation=0)

plt.ylabel('Victory')

plt.xlabel('Year')

plt.title('Percentage of games won')

plt.show()
win_by_tosser = df.pivot_table(index='Tossed', columns='Point', aggfunc='size', fill_value=0)

win_by_tosser.rename(columns={0: 'Unknown', -1: 'Lost', 1: 'Won'}, index={0: 'Opposition', 1: 'South Africa'}, inplace=True)



# Percent of games won

win_by_tosser['Percent'] = (win_by_tosser['Won'] * 100) / (win_by_tosser['Won'] + win_by_tosser['Lost'])



# Drop column name

win_by_tosser.columns.name=''

win_by_tosser.head()
# Choice is a little more complicated. The choice is for the tosser (not 'South Africa')

win_by_toss = df.pivot_table(index=['Tossed', 'Toss Choice'], columns='Point', aggfunc='size', fill_value=0)

win_by_toss.rename(columns={0: 'Unknown', -1: 'Lost', 1: 'Won'}, index={0: 'Opposition', 1: 'South Africa'}, inplace=True)



# Percent of games won

win_by_toss['Percent'] = (win_by_toss['Won'] * 100) / (win_by_toss['Won'] + win_by_toss['Lost'])



win_by_toss.head()
cursor.close()

conn.close()