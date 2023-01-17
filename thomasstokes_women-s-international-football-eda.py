# This Python 3 environment comes with many helpful analytics libraries installed.

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



# Packages

import numpy as np # Number crunching

import pandas as pd # Data processing

import matplotlib.pyplot as plt #Visualisation

import seaborn as sns #Visualisation

from scipy.stats import ttest_1samp # 1 Sample T-test

from scipy.stats import ttest_ind # 2 Sample T-test

from sklearn.linear_model import LinearRegression # Linear Regression

# File location

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# I found this data by browsing the datasets on Kaggle.

# Load the football data

data = pd.read_csv('/kaggle/input/womens-international-football-results/results.csv')
# Table

data.head()
# Info

data.info()
# Columns

data.columns
# Rename Columns

data.rename(columns={'home_score':'home_goals','away_score':'away_goals', 'country':'host'}, inplace=True)
# I want to create two results columns which tells us whether a team won (W), drew (D) or lost (L)

# First I will create a results function

def result(a,b):

    if a > b:

        r = 'W'

    elif a < b:

        r = 'L'

    else:

        r = 'D'

    return r

# Then create the results columns

data['home_result'] = [result(data['home_goals'][i], data['away_goals'][i]) for i in range(len(data))]

data['away_result'] = [result(data['away_goals'][i], data['home_goals'][i]) for i in range(len(data))]

                
# Check the updated dataframe

data.head()
# I want to change the neutral column to give a numeric value of either 0 (False) or 1 (True)

data['neutral'] = data['neutral'].apply(lambda x: 1 if x == True else 0)

data.head()
# Check for null values

data.isnull().any()
# Check for duplicates

data[data.duplicated()]
# Let's see if there's a correlation between the results and whether the game was on neutral ground

sns.heatmap(data.corr(),annot=True, linewidths=0.5)
# Let's visualise the home goals

plt.hist(data['home_goals'], bins=max(data['home_goals']))

plt.axvline(np.median(data['home_goals']), color='Black')

plt.axvline(np.mean(data['home_goals']), color='Yellow')

plt.xlabel('Goals')

plt.ylabel('Frequency')

plt.title('Home Goals Histogram')

plt.legend(['Median','Mean'])

plt.show()
# Let's visualise the away goals

plt.hist(data['away_goals'], bins=max(data['away_goals']), color='Red')

plt.axvline(np.median(data['away_goals']), color='Black')

plt.axvline(np.mean(data['away_goals']), color='Yellow')

plt.xlabel('Goals')

plt.ylabel('Frequency')

plt.title('Away Goals Histogram')

plt.legend(['Median','Mean'])

plt.show()
# Let's create a separate dataframe with all the games that had a team playing at home

homeadv_data = data[data['neutral']==0]

homeadv_data = homeadv_data.reset_index()

homeadv_data.head()
# Let's create another dataframe for the games where neither team was playing at home

neut_data = data[data['neutral']==1].reset_index()

neut_data.head()
# Let's see the match results of the home team when they were playing at home

plt.pie(homeadv_data['home_result'].value_counts(),autopct='%1.1f%%',colors=['lightblue', 'lightcoral', 'yellowgreen'])

plt.title('Match Results With Home Advantage')

plt.legend(['W', 'L', 'D'])

plt.show()
# Let's compare this to the match results of games where neither team was playing at home

plt.pie(neut_data['home_result'].value_counts(),autopct='%1.1f%%',colors=['lightblue', 'lightcoral', 'yellowgreen'])

plt.title('Match Results Without Home Advantage')

plt.legend(['W', 'L', 'D'])

plt.show()
# Let's see a comparison on a bar chart

sns.countplot(x=data['neutral'], hue=data['home_result'])

plt.title('Home Team Performance Comparison')
# To test this I'm going to randomly re-order most of the matches in the neutral data set to see if the proportions turn out as expected

length = neut_data['home_result'].count()

length = int(length)

random_swaps = np.random.randint(length, size=round(length*0.7))



# for loop to swap the rows

rand_neut_data = neut_data[['index','date', 'tournament', 'city', 'host', 'neutral']]

home_team = []

away_team = []

home_goals = []

away_goals = []

for i in range(length):

    if i in random_swaps:

        row = [neut_data['away_team'][i],neut_data['home_team'][i],neut_data['away_goals'][i],neut_data['home_goals'][i]]

    else:

        row = [neut_data['home_team'][i],neut_data['away_team'][i],neut_data['home_goals'][i],neut_data['away_goals'][i]]

    #Create the values in the columns

    home_team.append(row[0])

    away_team.append(row[1])

    home_goals.append(row[2])

    away_goals.append(row[3])

                           

# Create the new columns

rand_neut_data['home_team'] = home_team 

rand_neut_data['away_team'] = away_team

rand_neut_data['home_goals'] = home_goals

rand_neut_data['away_goals'] = away_goals

rand_neut_data.head(10)
# Let's add the home and away result columns again

rand_neut_data['home_result'] = [result(rand_neut_data['home_goals'][i], rand_neut_data['away_goals'][i]) for i in range(len(rand_neut_data))]

rand_neut_data['away_result'] = [result(rand_neut_data['away_goals'][i], rand_neut_data['home_goals'][i]) for i in range(len(rand_neut_data))]

# Let's see the new dataframe

rand_neut_data.head(10)
# Let's see the proportions of wins and losses

rand_neut_data['home_result'].value_counts() / len(rand_neut_data['home_result'])
# Create the cleaned dataframe

clean_data = pd.merge(rand_neut_data, homeadv_data, how = 'outer')

clean_data.head()
# Check the cleaned data

clean_data.info()
sns.heatmap(clean_data[['neutral', 'home_goals', 'away_goals']].corr(),annot=True, linewidths=0.5)

plt.title('Correlations')
# Let's visualise the home score

plt.hist(clean_data['home_goals'], bins=max(clean_data['home_goals']))

plt.axvline(np.median(clean_data['home_goals']), color='Black')

plt.axvline(np.mean(clean_data['home_goals']), color='Yellow')

plt.xlabel('Goals')

plt.ylabel('Frequency')

plt.title('Home Goals Histogram')

plt.legend(['Median','Mean'])

plt.show()
# Let's visualise the away score

plt.hist(clean_data['away_goals'], bins=max(clean_data['away_goals']), color='Red')

plt.axvline(np.median(clean_data['away_goals']), color='Black')

plt.axvline(np.mean(clean_data['away_goals']), color='Yellow')

plt.xlabel('Goals')

plt.ylabel('Frequency')

plt.title('Away Goals Histogram')

plt.legend(['Median','Mean'])

plt.show()
# Comparing the results of matches 

sns.countplot(x=clean_data['home_result'], hue=clean_data['neutral'])
# Let's now run some 1 sample T-tests to see if our results are statistically significant

# Null hypothesis: A team's performance is not affected by where they are playing



# Win

t, p_win = ttest_1samp(homeadv_data['home_result'].apply(lambda x: 1 if x == 'W' else 0), 0.4333875)

print('p_win is '+str(p_win))



# Draw

t, p_draw = ttest_1samp(homeadv_data['home_result'].apply(lambda x: 1 if x == 'D' else 0), 0.133225)

print('p_draw is '+str(p_draw))



# Loss

t, p_loss = ttest_1samp(homeadv_data['home_result'].apply(lambda x: 1 if x == 'L' else 0), 0.4333875)

print('p_loss is '+str(p_loss))
# Let's do a 1 sample t-test on the number of home goals to see if there a significant increase in the number of goals scored

# Null hypothesis is that there is no significant change in the number of goals scored

t, p_val = ttest_1samp(homeadv_data['home_goals'], np.mean(clean_data['home_goals']))

print('P value is '+str(p_val))


# List of countries

countries_home = clean_data['home_team'].unique()

countries_away = clean_data['away_team'].unique()

dont_add = []

add = []

for country in countries_away:

    if country in countries_home:

        dont_add.append(country)

    else:

        add.append(country)

unsorted_countries = np.concatenate([countries_home,add])

#Sort the countries alphabetically

countries = np.sort(unsorted_countries)

# Create a separate dataframe for each country

country_data = []

for i in range(len(countries)):

    country = countries[i]

    home_matches = clean_data[clean_data['home_team'] == country]

    away_matches = clean_data[clean_data['away_team'] == country]

    matches = pd.merge(home_matches, away_matches, how='outer')

    country_data.append(matches)

# United States dataframe

US_data = country_data[180]

US_data.head()
# Create the individual dataframes

clean_country = []

for i in range(len(country_data)):

    country_name = countries[i]

    country = country_data[i]

    home_country = country[country['home_team'] == country_name][['index', 'date', 'tournament', 'neutral', 'home_team', 'away_team', 'home_goals', 'away_goals','home_result']]

    home_country.rename(columns = {'home_team':'country', 'away_team':'opponent', 'home_goals':'goals_for', 'away_goals':'goals_against', 'home_result':'result'}, inplace=True)

    away_country =  country[country['away_team'] == country_name][['index', 'date', 'tournament', 'neutral', 'away_team', 'home_team','away_goals', 'home_goals','away_result']]

    away_country.rename(columns = {'away_team':'country', 'home_team':'opponent', 'away_goals':'goals_for', 'home_goals':'goals_against', 'away_result':'result'}, inplace=True)

    new_country = pd.merge(home_country, away_country, how = 'outer')

    clean_country.append(new_country)



# US cleaned data

US = clean_country[180]

US.head(10)
# Now that the country data has been cleaned lets do some EDA

# Let's create a list with the mean goals scored for and against each country



country_goals_for = [clean_country[i]['goals_for'].mean() for i in range(len(clean_country))]

country_goals_against = [clean_country[i]['goals_against'].mean() for i in range(len(clean_country))]



# Now let's visualise this with a bar chart

plt.bar(list(range(len(country_data))),country_goals_for)

plt.axhline(np.mean(country_goals_for), color='Yellow')

plt.xlabel('Country')

plt.ylabel('Mean Goals')

plt.title('Mean Goals Scored')

plt.legend(['Mean'])

plt.show()
# First let's get the dataframes for the USA, England and Brazil

USA = clean_country[180] 

Eng = clean_country[54]

Bra = clean_country[25]



#Let's visualise their match records

# USA

plt.pie(USA['result'].value_counts(),autopct='%1.1f%%',colors=['lightblue', 'lightcoral', 'yellowgreen'])

plt.legend(['W', 'L', 'D'])

plt.title('USA match record')

plt.show()
USA.head(10)
# England

plt.pie(Eng['result'].value_counts(),autopct='%1.1f%%',colors=['lightblue', 'lightcoral', 'yellowgreen'])

plt.legend(['W', 'L', 'D'])

plt.title('England match record')

plt.show()
Eng.head(10)
# Brazil

plt.pie(Bra['result'].value_counts(),autopct='%1.1f%%',colors=['lightblue', 'lightcoral', 'yellowgreen'])

plt.legend(['W', 'L', 'D'])

plt.title('Brazil match record')

plt.show()
Bra.head(10)
# I now want to make a histogram for the frequency of goals scored for/against

# USA

plt.hist(USA['goals_for'], bins=max(USA['goals_for']),color='blue')

plt.hist(USA['goals_against'], bins=max(USA['goals_against']),color='lightblue', alpha=0.6)

plt.xlabel('Number of Goals')

plt.ylabel('Frequency')

plt.title('USA Goals Histogram')

plt.legend(['Goals for','Goals against'])

plt.show()
# England

plt.hist(Eng['goals_for'], bins=max(Eng['goals_for']),color='red')

plt.hist(Eng['goals_against'], bins=max(Eng['goals_against']),color='lightcoral', alpha=0.6)

plt.xlabel('Number of Goals')

plt.ylabel('Frequency')

plt.title('England Goals Histogram')

plt.legend(['Goals for','Goals against'])

plt.show()
# Brazil

plt.hist(Bra['goals_for'], bins=max(Bra['goals_for']),color='green')

plt.hist(Bra['goals_against'], bins=max(Bra['goals_against']),color='yellowgreen', alpha=0.6)

plt.xlabel('Number of Goals')

plt.ylabel('Frequency')

plt.title('Brazil Goals Histogram')

plt.legend(['Goals for','Goals against'])

plt.show()
# If the goals_against histogram is distributed like: Γ ~ (μ = 1/a, σ = 1/a), then the std = mean

print('mean: '+str(US['goals_against'].mean()))

print('std: '+str(US['goals_against'].std()))
# Now I want to look at the distribution of goals scored for each type of result

#USA

plt.hist(USA[USA['result']=='W']['goals_for'], bins=max(USA[USA['result']=='W']['goals_for']), color='darkblue')

plt.hist(USA[USA['result']=='D']['goals_for'], bins=max(USA[USA['result']=='D']['goals_for']), color='yellow', alpha=0.8)

plt.hist(USA[USA['result']=='L']['goals_for'], bins=max(USA[USA['result']=='L']['goals_for']), color='blue', alpha=0.6)

plt.xlabel('Number of Goals')

plt.ylabel('Frequency')

plt.title('USA Goals Histogram')

plt.legend(['Win', 'Draw', 'Loss'])

plt.show()
# England

plt.hist(Eng[Eng['result']=='W']['goals_for'], bins=max(Eng[Eng['result']=='W']['goals_for']), color='darkred')

plt.hist(Eng[Eng['result']=='D']['goals_for'], bins=max(Eng[Eng['result']=='D']['goals_for']), color='yellow', alpha=0.8)

plt.hist(Eng[Eng['result']=='L']['goals_for'], bins=max(Eng[Eng['result']=='L']['goals_for']), color='lightcoral', alpha=0.6)

plt.xlabel('Number of Goals')

plt.ylabel('Frequency')

plt.title('England Goals Histogram')

plt.legend(['Win', 'Draw', 'Loss'])

plt.show()
# Brazil

plt.hist(Bra[Bra['result']=='W']['goals_for'], bins=max(Bra[Bra['result']=='W']['goals_for']), color='darkgreen')

plt.hist(Bra[Bra['result']=='D']['goals_for'], bins=max(Bra[Bra['result']=='D']['goals_for']), color='yellow', alpha=0.8)

plt.hist(Bra[Bra['result']=='L']['goals_for'], bins=max(Bra[Bra['result']=='L']['goals_for']), color='yellowgreen', alpha=0.6)

plt.xlabel('Number of Goals')

plt.ylabel('Frequency')

plt.title('Brazil Goals Histogram')

plt.legend(['Win', 'Draw', 'Loss'])

plt.show()
# I want to compare how well these teams perform in qualifying compared to the actual World Cup

# USA

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.pie(USA[USA['tournament'] == 'FIFA World Cup']['result'].value_counts(),autopct='%1.1f%%',colors=['lightblue', 'lightcoral', 'yellowgreen'])

ax1.set_title('World Cup')

ax1.legend(['W', 'L', 'D'])



ax2.pie(USA[USA['tournament'] != 'FIFA World Cup']['result'].value_counts(),autopct='%1.1f%%',colors=['lightblue', 'lightcoral', 'yellowgreen'])

ax2.set_title('Not World Cup')

ax2.legend(['W', 'L', 'D'])

plt.show()
# England

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.pie(Eng[Eng['tournament'] == 'FIFA World Cup']['result'].value_counts(),autopct='%1.1f%%',colors=['lightblue', 'lightcoral', 'yellowgreen'])

ax1.set_title('World Cup')

ax1.legend(['W', 'L', 'D'])



ax2.pie(Eng[Eng['tournament'] != 'FIFA World Cup']['result'].value_counts(),autopct='%1.1f%%',colors=['lightblue', 'lightcoral', 'yellowgreen'])

ax2.set_title('Not World Cup')

ax2.legend(['W', 'L', 'D'])

plt.show()
# Brazil

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.pie(Bra[Bra['tournament'] == 'FIFA World Cup']['result'].value_counts(),autopct='%1.1f%%',colors=['lightblue', 'lightcoral', 'yellowgreen'])

ax1.set_title('World Cup')

ax1.legend(['W', 'L', 'D'])



ax2.pie(Bra[Bra['tournament'] != 'FIFA World Cup']['result'].value_counts(),autopct='%1.1f%%',colors=['lightblue', 'lightcoral', 'yellowgreen'])

ax2.set_title('Not World Cup')

ax2.legend(['W', 'L', 'D'])

plt.show()
 # For loop to create the columns

country_games = []

country_wins = []

country_draws = []

country_losses=[]

for i in range(len(countries)):

    nation = clean_country[i]

    nation_games= nation['result'].count()

    country_games.append(nation_games)

    nation_wins = nation[nation['result']=='W']['result'].count()

    country_wins.append(nation_wins)

    nation_draws = nation[nation['result']=='D']['result'].count()

    country_draws.append(nation_draws)

    nation_losses = nation[nation['result']=='L']['result'].count()

    country_losses.append(nation_losses)



# Create the table

results_table = pd.DataFrame({'Country': countries,'Games': country_games,'Wins': country_wins, 'Draws': country_draws, 'Losses': country_losses})



# Score system

results_table['Score'] = round((results_table['Wins']*3 + results_table['Draws'])/results_table['Games'],2)



results_table.head(10)
# Histogram of scores

plt.hist(results_table['Score'],bins=20,color='blue')

plt.axvline(x=np.mean(results_table['Score']),color='yellow')

plt.title('Score Histogram')

plt.xlabel('Score')

plt.ylabel('Frequency')

plt.legend(['Mean'])

plt.show()
# Looks like there are a lot of countries with a score of 0, let's see which countries these are

no_score = results_table[results_table['Score']==0]['Country']

no_score
#Let's see the results table for our countries

results_table.loc[[180,25,54]]
# Sort the scores so they are in order

sorted_scores = np.sort(results_table['Score'])

# Create the columns

sorted_games = []

sorted_country = []

sorted_wins = []

sorted_draws = []

sorted_losses = []

# Find the country with these scores

for i in range(len(sorted_scores)):

    score = sorted_scores[-1*(i+1)]

    for j in range(len(sorted_scores)):

        if results_table['Score'][j] == score and results_table['Games'][j] > 10 and results_table['Country'][j] not in sorted_country:

            # Add the countries results in order to create the rows

            sorted_country.append(results_table['Country'][j])

            sorted_games.append(results_table['Games'][j])

            sorted_wins.append(results_table['Wins'][j])

            sorted_draws.append(results_table['Draws'][j])

            sorted_losses.append(results_table['Losses'][j])



# Create the table            

ranking_table = pd.DataFrame({'Country': sorted_country,'Games': sorted_games,'Wins': sorted_wins, 'Draws': sorted_draws, 'Losses': sorted_losses})

ranking_table['Score'] = round((ranking_table['Wins']*3 + ranking_table['Draws'])/ranking_table['Games'],2)



ranking_table.head(10)
# Histogram of scores

plt.hist(ranking_table['Score'],bins=20,color='blue')

plt.axvline(x=np.mean(ranking_table['Score']),color='yellow')

plt.title('Score Histogram')

plt.xlabel('Score')

plt.ylabel('Frequency')

plt.legend(['Mean'])

plt.show()
# Heatmap to check for correlations

sns.heatmap(ranking_table.corr(),annot=True,linewidths=0.5)

plt.title('Rankings Heatmap')

plt.show()
# Games

plt.scatter(y=ranking_table['Score'], x=ranking_table['Games'], color='black', alpha = 0.6)

plt.xlabel('Games Played')

plt.ylabel('Score')

plt.title('Games Scatter Plot')

plt.show()
# Wins

plt.scatter(y=ranking_table['Wins']/ranking_table['Games'], x=ranking_table['Games'], color='blue', alpha = 0.6)

plt.xlabel('Games Played')

plt.ylabel('Win Proportion')

plt.title('Win Proportion vs. Games Played Scatter Plot')

plt.show()
# Portugal and Finland have played more than 150 games played but have a score less than 1

table = ranking_table[ranking_table['Games'] > 150]

table[table['Score'] < 1]



# Norway, Sweden, China PR and Denmark have played more than 250 games but have a score less than 2

# All the Scandanvian countries (apart from Iceland) are outliers

table = ranking_table[ranking_table['Games'] > 250]

table[table['Score'] < 2]



# Northen Ireland have played more than 50 games but have a score less than 0.5

table = ranking_table[ranking_table['Games'] > 50]

table[table['Score'] < 0.5]



#Estonia have played nearly 50 games and have a really low score of 0.10

table = ranking_table[ranking_table['Games'] < 50]

table[table['Score'] < 0.3]



# Dominican Republic and Jordan have played less than 50 games but have a score of 1.86 and 1.85 respectively

table = ranking_table[ranking_table['Games'] < 50]

table[table['Score'] > 1.5]
# Create a new table with the outliers removed

# There's probably a much more efficent way of doing this but this method will suffice

table_a = ranking_table[ranking_table['Country'] != 'Portugal'] 

table_b = table_a[table_a['Country'] != 'Finland'] 

table_c = table_b[table_b['Country'] != 'Norway']

table_d = table_c[table_c['Country'] != 'Sweden']

table_e = table_d[table_d['Country'] != 'China PR']

table_f = table_e[table_e['Country'] != 'Denmark']

table_g = table_f[table_f['Country'] != 'Northen Ireland']

table_h = table_g[table_g['Country'] != 'Estonia']

table_i = table_h[table_h['Country'] != 'Dominican Republic']

table_j = table_i[table_i['Country'] != 'Jordan']
# Linear Regression

line = LinearRegression()

Games = table_j['Games']

Games = Games.values.reshape(-1,1)

line.fit(Games,-np.log1p(-table_j['Wins']/table_j['Games']))

line_y = line.predict(ranking_table['Games'].values.reshape(-1,1))

# Gradient

k= line.coef_[0]

# Intercept

c= line.predict(np.array([0]).reshape(-1,1))[0]



# Scatter Plot with re-arranged variables

plt.scatter(x=ranking_table['Games'], y=-np.log1p(-ranking_table['Wins']/ranking_table['Games']), color='blue', alpha = 0.6)

plt.plot(ranking_table['Games'],line_y, color = 'black')

plt.xlabel('Games Played')

plt.ylabel('Modified Win Proportion')

plt.title('Win Proportion Scatter Plot with Line of Best Fit')

plt.legend(['Line of Best Fit', 'Gradient='+str(round(k,5))])

plt.show()

# Here's the original plot with the exponential curve

y_val = np.arange(0,301,1)

plt.scatter(y=ranking_table['Wins']/ranking_table['Games'], x=ranking_table['Games'], color='blue', alpha = 0.6)

plt.plot(1+(c-1)*np.exp(-k*y_val),color='Black')

plt.xlabel('Games Played')

plt.ylabel('Win Proportion')

plt.title('Win Proportion vs. Games Played Scatter Plot')

plt.legend(['Exponential Curve'])

plt.show()
# The exponential curve above doesn't start close to (0,0), (which would be ideal)

# A gradient of 0.006 fits the plot and goes through (0,0)

plt.scatter(y=ranking_table['Wins']/ranking_table['Games'], x=ranking_table['Games'], color='blue', alpha = 0.6)

plt.plot(1-np.exp(-0.006*y_val),color='Black')

plt.xlabel('Games Played')

plt.ylabel('Win Proportion')

plt.title('Win Proportion vs. Games Played Scatter Plot')

plt.legend(['Exponential Curve'])

plt.show()
# Draws

plt.scatter(y=ranking_table['Draws']/ranking_table['Games'], x=ranking_table['Games'], color='green', alpha = 0.6)

plt.axhline(y=np.mean(ranking_table['Draws']/ranking_table['Games']))

plt.xlabel('Games Played')

plt.ylabel('Draw Proportion')

plt.title('Draw Proportion vs. Games Played Scatter Plot')

plt.legend(['Mean'])

plt.show()
# Losses

plt.scatter(y=ranking_table['Losses']/ranking_table['Games'], x=ranking_table['Games'], color='red', alpha = 0.6)

plt.xlabel('Games Played')

plt.ylabel('Loss Proportion')

plt.title('Loss Proportion vs. Games Played Scatter Plot')

plt.show()