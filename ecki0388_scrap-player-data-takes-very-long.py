import numpy as np

import pandas as pd

import requests

from bs4 import BeautifulSoup

import re
seasons=range(2010,2018) #these are the only seasons they have at the moment

statistics=[]

roster=[]

headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:51.0) Gecko/20100101 Firefox/51.0'}

#the website won't take requests, if you don't provide a User-Agent (in firefox this can be found at Tools>Web Developer>Network, but this one is also fine)

for s in seasons:

    #get the links to all teams for the specified season

    base_url='http://stats.ncaa.org/team/inst_team_list?academic_year='+str(s)+'&conf_id=-1&division=1&sport_code=MBB'

    page=requests.get(base_url,headers=headers)

    soup = BeautifulSoup(page.content,'lxml')

    links=[]

    for link in soup.find_all('a', href=True):

        pattern = re.compile(r"/team/[0-9]+/[0-9]+")

        if pattern.match(link['href']):

            links.append(link['href'])

    yearSuffix=links[0].split('/')[-1]

    teams=[link.split('/')[2] for link in links]

    #get the urls for the statistics and roster page of each team

    stats_url_base='http://stats.ncaa.org/team/'

    stats_urls=[stats_url_base+team+'/stats/'+yearSuffix for team in teams]

    roster_urls=[stats_url_base+team+'/roster/'+yearSuffix for team in teams]

    #download the statistics for each team

    stats_frames=[]

    pattern = re.compile('([^, ][^,]*), ([^,]+)')

    for url in stats_urls:    

        site=requests.get(url,headers=headers)

        stats_soup = BeautifulSoup(site.content,'lxml')



        columns=[col.text.strip() for col in stats_soup.find_all('th')]

        cells=[cell.text.strip() for cell in stats_soup.find_all('td')]

        #since there is no good structure we have to count the names to get a player count

        player_count=0

        for c in cells[1:]:

            if pattern.match(c):

                player_count+=1

        player_dicts=[]

        #now we can use this information to get all cells, which belong to each row

        for i in xrange(player_count):

            player_dicts.append(dict(zip(columns, cells[i*len(columns)+1:(i+1)*len(columns)+1])))

            player_dicts[i].update({'Team':stats_soup.find('span',{'class':'org_heading'}).text.strip()})

            player_dicts[i].update({'Season':s})

        stats_frames.append(pd.DataFrame(player_dicts))

    statistics.append(pd.concat(stats_frames))

    #download the roster for each team

    roster_frames=[]

    pattern = re.compile('([^, ][^,]*), ([^,]+)')

    for url in roster_urls:    

        site=requests.get(url,headers=headers)

        roster_soup = BeautifulSoup(site.content,'lxml')



        columns=[col.text.strip() for col in roster_soup.find_all('th',colspan=False)]

        cells=[cell.text.strip() for cell in roster_soup.find_all('td')]

        player_count=0

        for c in cells[1:]:

            if pattern.match(c):

                player_count+=1

        player_dicts=[]

        for i in xrange(player_count):

            player_dicts.append(dict(zip(columns, cells[i*len(columns):(i+1)*len(columns)+1])))

            player_dicts[i].update({'Team':roster_soup.find('span',{'class':'org_heading'}).text.strip()})

            player_dicts[i].update({'Season':s})

        roster_frames.append(pd.DataFrame(player_dicts))

    roster.append(pd.concat(roster_frames))

#now put the frames together

roster1=pd.concat(roster)

statistics1=pd.concat(statistics)

#clean up the team names

roster1['Team']=roster1['Team'].str.split('(').str[0].str.strip()

statistics1['Team']=statistics1['Team'].str.split('(').str[0].str.strip()
#save as csv and ignore the index since it has no meaning

statistics1.to_csv('playerdata_statistics.csv')

roster1.to_csv('playerdata_roster.csv')