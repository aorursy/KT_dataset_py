import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
url = "https://www.basketball-reference.com/players/"

# to access alphabetical ordered websites by Last Name
alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
            'w','x','y','z']

count = 0

for alphabet in alphabets:
    # get link of each alphabet page
    players_page = url + alphabet
    
    page = requests.get(players_page)
    soup = BeautifulSoup(page.content, 'html.parser')
    print(alphabet) # track progress
    # to find out each hall of fame player name *
    for i in soup.find_all('th'):
        if '*' in i.text:
            name = i.text[:len(i.text)-1]
            count+=1

            for r in soup.find_all('th'):
                # need to re-loop and find the url when it matches the hall of famer name as above
                if r.text == i.text:
                    startpos = str(r).find('href="') +6
                    endpos = str(r).find('.html')
                    short_link = str(r)[startpos:endpos]
                    full_link = url[:len(url)-9] + short_link +'.html' # this is the link to each hall of famer

                    page = requests.get(full_link)
                    soup2 = BeautifulSoup(page.content, 'html.parser')

                    headers = []
                    table = soup2.find('table', attrs={'id':'per_game'})

                    # get Table Header
                    h_rows = table.find_all('tr')
                    for row in h_rows:
                        cols = row.find_all('th')
                        cols = [ele.text.strip() for ele in cols]
                        headers.append([ele for ele in cols if ele]) # Get rid of empty values

                    columns = headers[0]
                    seasons = []

                    season_count = 0
                    for season in headers[1:]:
                        if season_count == 0 and 'Career' not in season:
                            seasons.append(season)
                        if 'Career' in season:
                            season_count += 1

                    # get Table Body data
                    data = []
                    rows = table.find_all('tr')
                    for row in rows:
                        cols = row.find_all('td')
                        cols = [ele.text for ele in cols]
                        data.append([ele for ele in cols])
                    
                    # Create each sub dataframe to append later
                    df = pd.DataFrame(data, columns = columns[1:]).dropna()

                    # filter away not 2 digit values
                    df = df[df.iloc[:,0].map(len) == 2]
                    # Add filtered seasons (remove empty value list)
                    df['Seasons'] = [season[0] for season in seasons if season != []]
                    df['Player'] = name
                    df['Index'] = count
            
            # Append to a master dataframe
            if count == 1:
                full_df = df
            else:
                full_df = pd.DataFrame.merge(full_df, df, how ='outer')

# Re-order columns
new_cols = [
    'Index',
    'Player',
    'Age',
    'Seasons',
    'Tm',
    'Lg',
    'Pos',
    'G',
    'GS',
    'MP',
    'FG',
    'FGA',
    'FG%',
    '3P',
    '3PA',
    '3P%',
    '2P',
    '2PA',
    '2P%',
    'eFG%',
    'FT',
    'FTA',
    'FT%',
    'ORB',
    'DRB',
    'TRB',
    'AST',
    'STL',
    'BLK',
    'TOV',
    'PF',
    'PTS'    
]

full_df = full_df[new_cols]
full_df2 = full_df.copy()

# columns chosen for analysis
required_cols = full_df2[['FG%', 'eFG%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']]

cols_to_fill = [
    'FG%',
    'eFG%',
    'FT%',
    'TRB',
    'AST',
    'STL',
    'BLK',
    'TOV',
    'PF',
    'PTS'    
]

# to replace empty value with nan
for col in cols_to_fill:
    full_df2[col].replace(r'^\s*$', np.nan, regex=True, inplace = True)
    full_df2[col] = full_df2[col].astype(float)
    required_cols[col].replace(r'^\s*$', np.nan, regex=True, inplace = True)

# get the count of non-empty columns (for dividing later)
full_df2['full_count'] = required_cols.apply(lambda x: x.count(), axis=1)

# Calculate 'performance' by normalisation to get the weighted value and divided by total non-empty count
# Ignore blanks / no data
performance = 0
performance_list = []
for idx, row in full_df2.iterrows():
    performance = 0
    for col in cols_to_fill: 
        if (not pd.isnull(row[col])) :
            if col != 'TOV' and col != 'PF':
                performance += (row[col]-full_df2[col].min())/(full_df2[col].max()- full_df2[col].min())
            elif col == 'TOV' or col == 'PF':
                performance -= (row[col]-full_df2[col].min())/(full_df2[col].max()- full_df2[col].min())  
    
    if row['full_count'] != 0:
        performance = performance/row['full_count']
        performance_list.append(performance)
    else:
        performance_list.append(None)

full_df2['Performance']  = performance_list

# get the max of performance by each Player
full_df2['max'] = full_df2.groupby('Player')['Performance'].transform(max)

# Loop to get the peak age
players_list = []
players_peak = []
players_decade = []
count = 0

for idx, row in full_df2.iterrows():
    if row['Performance'] == row['max']:
        players_list.append(row['Player'])
        players_peak.append(row['Age'])
        players_decade.append(row['Seasons'])
        
subset_df = pd.DataFrame(players_list, columns = ['Player'])
subset_df['Peak_Age'] = players_peak
subset_df['Peak_Season'] = players_decade
subset_df['BirthYear'] = subset_df['Peak_Season'].str[:4].astype(int) - subset_df['Peak_Age'].astype(int)
subset_df['Decade'] = subset_df['Peak_Season'].str[:4].astype(int)//10*10

# Merge dataframe
full_df2 = full_df2.merge(subset_df, on = 'Player', how = 'left')

# get min age when they debut in NBA & years needed to reach prime
full_df2['Debut_Age'] = full_df2.groupby('Player')['Age'].transform(min)
full_df2['Years_To_Prime'] = full_df2['Peak_Age'].astype(int) - full_df2['Debut_Age'].astype(int)

# To get normalized variables
full_df3 = full_df2.copy()

full_df3['nFG%'] = (full_df3['FG%'] - full_df3['FG%'].min())/ (full_df3['FG%'].max()-full_df3['FG%'].min())
full_df3['neFG%'] = (full_df3['eFG%'] - full_df3['eFG%'].min())/ (full_df3['eFG%'].max() - full_df3['eFG%'].min())
full_df3['nFT%'] = (full_df3['FT%'] - full_df3['FT%'].min())/ (full_df3['FT%'].max() - full_df3['FT%'].min())
full_df3['nTRB'] = (full_df3['TRB'] - full_df3['TRB'].min())/ (full_df3['TRB'].max() - full_df3['TRB'].min())
full_df3['nAST'] = (full_df3['AST'] - full_df3['AST'].min())/ (full_df3['AST'].max() - full_df3['AST'].min())
full_df3['nSTL'] = (full_df3['STL'] - full_df3['STL'].min())/ (full_df3['STL'].max() - full_df3['STL'].min())
full_df3['nBLK'] = (full_df3['BLK'] - full_df3['BLK'].min())/ (full_df3['BLK'].max() - full_df3['BLK'].min())
full_df3['nTOV'] = (full_df3['TOV'] - full_df3['TOV'].min())/ (full_df3['TOV'].max() - full_df3['TOV'].min())
full_df3['nPF'] = (full_df3['PF'] - full_df3['PF'].min())/ (full_df3['PF'].max() - full_df3['PF'].min())
full_df3['nPTS'] = (full_df3['PTS'] - full_df3['PTS'].min())/ (full_df3['PTS'].max() - full_df3['PTS'].min())


# Write to csv
full_df3.reset_index(drop=True, inplace=True)
full_df3.to_csv('nba_peak_age.csv')
full_df3