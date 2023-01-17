import pandas as pd 

import urllib.request

from bs4 import BeautifulSoup

import numpy as np



import datetime

import matplotlib.lines as mlines

import os

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm



# Roman numeral finder ========================================================

def int_to_Roman(num):

    val = [1000, 900, 500, 400,100, 90, 50, 40,10,9,5,4,1]

    syb = ["M", "CM", "D", "CD","C", "XC", "L", "XL","X", "IX", "V", "IV","I"]

    roman_num = '';i = 0

    while  num > 0:

        for _ in range(num // val[i]):

            roman_num += syb[i]

            num -= val[i]

        i += 1

    return roman_num

# =============================================================================



# Build up dictionary of the statistics based on year

superbowl_stats = {}

# Generate the roman numeral set

roman_numerals = [int_to_Roman(x) for x in range(1,54)]

# First thing to do is add the scores via scraping the web:

numerical_scraping_stats_indicators = ['First downs rushing','First downs passing','First downs penalty','Net yards rushing', \

                            'Rushing attempts','Yards per rush','Interceptions thrown','Net yards passing','Total net yards','Turnovers']

split_scraping_stats_indicators = ['Times sacked-total yards','Punt returns-total yards','Kickoff returns-total yards','Interceptions-total return yards','Punts-average yardage','Fumbles-lost','Penalties-total yards']

time_scraping_stats_indicators = ['Time of possession']

good_rn = []

for rn in roman_numerals:

    try:

        print(rn)

        superbowl_stats[rn] = {}

        wiki_url = 'https://en.wikipedia.org/wiki/Super_Bowl_'+rn



        # Scrape the scores for each team as well as team name

        with urllib.request.urlopen(wiki_url) as response:

            html = response.read()

            soup = BeautifulSoup(html)

            summary = soup.findAll("table",{'class','nowraplinks'})



            # Team name ========================================

            r = summary[0].findAll("tr");tds = r[0].findAll("th")

            A = tds[0].text;B = tds[1].text

            superbowl_stats[rn]['Teams'] = [A,B]



            # Scores ===========================================

            r = summary[0].findAll("tr");tds = r[1].findAll("td")

            A = int(tds[0].text);B = int(tds[1].text)

            superbowl_stats[rn]['Score'] = [A,B]



        # Get the detailed stats:

        with urllib.request.urlopen(wiki_url) as response:

            html = response.read()

            soup = BeautifulSoup(html)

            data = soup.findAll("td")

            for stat in numerical_scraping_stats_indicators:

                A = [td.findNext('td').text.strip() for td in data if td.text.strip() in [stat]]

                B = [td.findNext('td').findNext('td').text.strip() for td in data if td.text.strip() in [stat]]

                superbowl_stats[rn][stat] = [float(A[0]),float(B[0])]



            for stat in split_scraping_stats_indicators:

                A = [td.findNext('td').text.strip() for td in data if td.text.strip() in [stat]]

                B = [td.findNext('td').findNext('td').text.strip() for td in data if td.text.strip() in [stat]]

                print([float(a.strip('(').strip(')').replace('–','-').replace('−','-')) for a in A[0].split('–',1)])

                superbowl_stats[rn][stat] = [[float(a.strip('(').strip(')').replace('–','-').replace('−','-')) for a in A[0].split('–',1)],[float(b.strip('(').strip(')').replace('–','-').replace('−','-')) for b in B[0].split('–',1)]]



            for stat in time_scraping_stats_indicators:

                A = [td.findNext('td').text.strip() for td in data if td.text.strip() in [stat]]

                B = [td.findNext('td').findNext('td').text.strip() for td in data if td.text.strip() in [stat]]

                superbowl_stats[rn][stat] = [[float(a) for a in A[0].split(':')],[float(b) for b in B[0].split(':')]]

        good_rn.append(rn)

    except:

        del superbowl_stats[rn] 



# Now we can look at a scatter of stat vs stat





x = []

y = []

color = []

for rn in good_rn:

    # Get the score

    A_score = superbowl_stats[rn]['Score'][0]

    B_score = superbowl_stats[rn]['Score'][1]

    if A_score>B_score:

        color.append('#00FF00');color.append('#FF0000')

    else:

        color.append('#FF0000');color.append('#00FF00')

    

    metric = 'Score'

    x.append(superbowl_stats[rn][metric][0])

    x.append(superbowl_stats[rn][metric][1])



    y.append(superbowl_stats[rn]['Score'][0])

    y.append(superbowl_stats[rn]['Score'][1])



plt.scatter(x, y,c=color)

plt.xlabel('Total Yards Sacked', fontsize=18)

plt.ylabel('Score', fontsize=16)

plt.show()