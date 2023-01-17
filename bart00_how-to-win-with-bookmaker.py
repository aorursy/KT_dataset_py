from shutil import copyfile #I've faced a problem with loading database.sqlite file, with database.db it works well

copyfile('../input/database.sqlite', 'database.db')
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np # linear algebra
import pandas as pd

import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

import math

import os
print(os.listdir())
def load_database():
    #with sqlite3.connect('../input/database.sqlite') as con:
    with sqlite3.connect('database.db') as con:
        all_data = pd.read_sql_query("SELECT * from football_data", con)
        
    return all_data
  
data = load_database()
BOOKIE = {
    'BET365': ['B365H', 'B365D', 'B365A'],
    'PS': ['PSH', 'PSD', 'PSA'],
    'WH': ['WHH', 'WHD', 'WHA'],
    'BWIN': ['BWH', 'BWD', 'BWA'],
    'LB': ['LBH', 'LBD', 'LBA'] 
}

def load_test_data(division, season: list, data, bookie): 

    data = data[data['Div'] == division] #filter by division

    MATCH_INFO = ['Date', 'Season', 'League', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'] + BOOKIE[bookie]
    data = data[MATCH_INFO] #take only specific columns
    data = data.loc[data['Season'].isin(season)] #filter by season

    return data
#two options - flat bid, e.g 50$ for every match, or bid depends on probability (kelly_crit)
#kelly_crit returns percentage information what part of our budget should we use, it is even 50% when it is high win probability
#bid range is between ~10% to ~60%, to achieve comparable results for kelly_crit and flat bid I will use kelly_crit in reference to 500 units (USD, EUR, whatever)
#mean bid in kelly_crit then is about 100 units, so I will use flat bid equal to 100 units
def calculate_bid(kind, prob_success=0, odds_success=0):
    if kind == 'flat':
        return 100
    elif kind == 'kelly_crit':
        num = (prob_success+0.1)*odds_success - 1
        denom = odds_success - 1
        return num/denom*500
def check_prob_range(prob_down: list, prob_up: list, data, bid):
    """
    Find the best pair of prob_up and prob_down in specific dataset.
    It draws heatmap with account balance after analysing of dataset for every probs pair
    @param prob_down: Probability difference limits when bet on draw
    @param prob_up: Probability difference limits when bet on home_team/away_team
    @param data: Pandas DataFrame with matches analysis, usually narrowed limited to match from one league
    @param bid: Kind of bids, flat or kelly_crit
    @return: x[ind], y[ind], max(cash_tab), min(cash_tab) - prob_down and prob_up for it achieved the highest account balance, highest account balance, lowest account balance
    """
    #max(prob_down) must be smaller than min(prob_up)
    x = []
    y = []
    cash_tab = []
    league = data['League'].iloc[0] #take league's name
    for _prob_down in prob_down: 
        for _prob_up in prob_up:
            cash = 0 #we start every probs' pair with 0, it will be visible, if it is gain or loss
            for i, row in data.iterrows(): 
                p1 = 1 / (1 + row['B365H']/row['B365A'] + row['B365H']/row['B365D'])
                p2 = 1 / (1 + row['B365A']/row['B365H'] + row['B365A']/row['B365D'])
                pX = 1 / (1 + row['B365D']/row['B365A'] + row['B365D']/row['B365H'])

                if math.fabs(p1 - p2) <= _prob_down: #if algorithm claim to bet on draw
                    _bid = calculate_bid(bid, pX, row['B365D'])
                    if row['FTR'] == 'D':
                        cash += _bid * row['B365D'] - _bid #if it was draw really, update account balance
                    else:
                        cash -= _bid #if it was not draw really, update account_balance
                elif (p1 - p2) >= _prob_up: #if algorithm claim to bet on home_team
                    _bid = calculate_bid(bid, p1, row['B365H'])
                    if row['FTR'] == 'H':
                        cash += _bid * row['B365H'] - _bid
                    else:
                        cash -= _bid
                    pass
                elif (p2 - p1) >= _prob_up: #if algorithm claim to bet on away_team
                    _bid = calculate_bid(bid, p2, row['B365A'])
                    if row['FTR'] == 'A':
                        cash += _bid * row['B365A'] - _bid
                    else:
                        cash -= _bid
                    pass
                elif math.fabs(p1 - p2) > _prob_down and math.fabs(p1 - p2) < _prob_up: #if probabilities difference is between prob_down and prob_up algorithm skip that match, doesn't bet
                    pass

            x.append(_prob_down)
            y.append(_prob_up)
            cash_tab.append(cash)
    
    #create DataFrame from 3 tables
    results = pd.DataFrame({"prob_down": np.around(x, decimals=2),
                            "prob_up": np.around(y, decimals=2),
                            "cash": cash_tab}, 
                            columns=["prob_down", "prob_up", "cash"])
    
    results = results.pivot(index="prob_down", columns="prob_up", values="cash")

    f, ax = plt.subplots()
    sns.set()
    ax = sns.heatmap(results, linewidths=.5)
    plt.title(f"Probabilities settings and profit/loss heatmap: {league}")
    plt.show()
    plt.clf()
    
    ind = cash_tab.index(max(cash_tab))
    return x[ind], y[ind], max(cash_tab), min(cash_tab)
def check_prob_range_draw(prob_down: list, data, bid):
    """
    Find the best prob_down in specific dataset.
    It draws heatmap with account balance after analysing of dataset for every prob_down value
    @param prob_down: Probability difference limits when bet on draw
    @param data: Pandas DataFrame with matches analysis, usually narrowed limited to match from one league
    @param bid: Kind of bids, flat or kelly_crit
    @return: x[ind], y[ind], max(cash_tab), min(cash_tab) - prob_down and prob_up for it achieved the highest account balance, highest account balance, lowest account balance
    """
    #max(prob_down) must be smaller than min(prob_up)
    x = []
    y = []
    cash_tab = []
    league = data['League'].iloc[0] #take league's name
    for _prob_down in prob_down: 
        cash = 0 #we start every probs' pair with 0, it will be visible, if it is gain or loss
        for i, row in data.iterrows(): 
            p1 = 1 / (1 + row['B365H']/row['B365A'] + row['B365H']/row['B365D'])
            p2 = 1 / (1 + row['B365A']/row['B365H'] + row['B365A']/row['B365D'])
            pX = 1 / (1 + row['B365D']/row['B365A'] + row['B365D']/row['B365H'])

            if math.fabs(p1 - p2) <= _prob_down: #if algorithm claim to bet on draw
                _bid = calculate_bid(bid, pX, row['B365D'])
                if row['FTR'] == 'D':
                    cash += _bid * row['B365D'] - _bid #if it was draw really, update account balance
                else:
                    cash -= _bid #if it was not draw really, update account_balance
            else: 
                pass

        x.append(_prob_down)
        y.append(0)
        cash_tab.append(cash)
    
    #create DataFrame from 3 tables
    results = pd.DataFrame({"prob_down": np.around(x, decimals=2),
                            "prob_up": np.around(y, decimals=2),
                            "cash": cash_tab}, 
                            columns=["prob_down", "prob_up", "cash"])
    
    results = results.pivot(index="prob_down", columns="prob_up", values="cash")

    f, ax = plt.subplots()
    sns.set()
    ax = sns.heatmap(results, linewidths=.5)
    plt.title(f"Probabilities settings and profit/loss heatmap: {league}")
    plt.show()
    plt.clf()
    
    ind = cash_tab.index(max(cash_tab))
    return x[ind], y[ind], max(cash_tab), min(cash_tab)
def check_loss_profit(data, bid, prob_down, prob_up):
    """
    Draws scatter plots with actual account balance for every season and at the end barplot with account balance after every season, and number of correct/incorrect bet or skipped matches
    @params data: Pandas DataFrame with matches analysis, usually narrowed limited to match from one league
    @param bid: Kind of bids, flat or kelly_crit
    @param prob_down: Probability difference limits when bet on draw
    @param prob_up: Probability difference limits when bet on home_team/away_team
    """
    season = data.Season.unique()
    season.sort() #sort table of seasons, for hierarchically plot at the end
    cash_tab = []
    season_tab = []
    right_tab = []
    wrong_tab = []
    skip_tab = []
    league = data['League'].iloc[0]
    if os.path.exists(f"matches_{league}_{prob_down}_{prob_up}_{bid}"):
        os.remove(f"matches_{league}_{prob_down}_{prob_up}_{bid}")
    f = open(f"matches_{league}_{prob_down}_{prob_up}_{bid}", "a") #I would save match's details to file
    for k, _season in enumerate(season):
        cash = 0
        data_loc = data[data['Season'] == _season]
        f.write(f"{_season}\n")
        right=0
        wrong=0
        skip=0
        bid_avg = 0
        fig = plt.figure(figsize=(16,10)) #fig for profitability plot
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        fig.tight_layout()
        ax = fig.add_subplot(len(season),1,k+1)
        ax.margins(x=0, y=0)
        j = 0 #bet counter
        
        for i, row in data_loc.iterrows(): 
            p1 = 1 / (1 + row['B365H']/row['B365A'] + row['B365H']/row['B365D'])
            p2 = 1 / (1 + row['B365A']/row['B365H'] + row['B365A']/row['B365D'])
            pX = 1 / (1 + row['B365D']/row['B365A'] + row['B365D']/row['B365H'])
        
            if math.fabs(p1 - p2) <= prob_down:
                _bid = calculate_bid(bid, pX, row['B365D'])
                bid_avg += _bid #for calculating average bid
                if row['FTR'] == 'D':
                    cash += _bid * row['B365D'] - _bid
                    #print("OK! Score DRAW", row['HomeTeam'], row['FTHG']," - ", row['FTAG'], row['AwayTeam'], row['FTR'], "Bid:", _bid, "Gain:", _bid * row['B365D'] - _bid)
                    f.write(f"OK! Score DRAW {row['HomeTeam']} {row['FTHG']} - {row['FTAG']} {row['AwayTeam']} {row['FTR']} Bid: {_bid}, Gain: {_bid * row['B365D'] - _bid} \n")
                    right += 1
                else:
                    cash -= _bid
                    #print("BAD! Score DRAW", row['HomeTeam'], row['FTHG']," - ", row['FTAG'], row['AwayTeam'], row['FTR'], "Bid:", _bid)
                    f.write(f"BAD! Score DRAW {row['HomeTeam']} {row['FTHG']} - {row['FTAG']} {row['AwayTeam']} {row['FTR']} Bid: {_bid} \n")
                    wrong += 1
            elif (p1 - p2) >= prob_up:
                _bid = calculate_bid(bid, p1, row['B365H'])
                bid_avg += _bid
                if row['FTR'] == 'H':
                    cash += _bid * row['B365H'] - _bid
                    #print("OK! Score HOME", row['HomeTeam'], row['FTHG']," - ", row['FTAG'], row['AwayTeam'], row['FTR'], "Bid:", _bid,  "Gain:", _bid * row['B365H'] - _bid)
                    f.write(f"OK! Score HOME {row['HomeTeam']} {row['FTHG']} - {row['FTAG']} {row['AwayTeam']} {row['FTR']} Bid: {_bid}, Gain: {_bid * row['B365H'] - _bid} \n")
                    right += 1
                else:
                    cash -= _bid
                    #print("BAD! Score HOME", row['HomeTeam'], row['FTHG']," - ", row['FTAG'], row['AwayTeam'], row['FTR'], "Bid:", _bid)
                    f.write(f"BAD! Score HOME {row['HomeTeam']} {row['FTHG']} - {row['FTAG']} {row['AwayTeam']} {row['FTR']} Bid: {_bid} \n")
                    wrong += 1
                #skip += 1
            elif (p2 - p1) >= prob_up:
                _bid = calculate_bid(bid, p2, row['B365A'])
                bid_avg += _bid
                if row['FTR'] == 'A':
                    cash += _bid * row['B365A'] - _bid
                    #print("OK! Score AWAY", row['HomeTeam'], row['FTHG']," - ", row['FTAG'], row['AwayTeam'], row['FTR'], "Bid:", _bid,  "Gain:", _bid * row['B365A'] - _bid)
                    f.write(f"OK! Score AWAY {row['HomeTeam']} {row['FTHG']} - {row['FTAG']} {row['AwayTeam']} {row['FTR']} Bid: {_bid}, Gain: {_bid * row['B365A'] - _bid} \n")
                    right += 1
                else:
                    cash -= _bid
                    #print("BAD! Score AWAY", row['HomeTeam'], row['FTHG']," - ", row['FTAG'], row['AwayTeam'], row['FTR'], "Bid:", _bid)
                    f.write(f"BAD! Score AWAY {row['HomeTeam']} {row['FTHG']} - {row['FTAG']} {row['AwayTeam']} {row['FTR']} Bid: {_bid} \n")
                    wrong += 1
                #skip += 1
            elif math.fabs(p1 - p2) > prob_down and math.fabs(p1 - p2) < prob_up:
                f.write(f"No bet! {row['HomeTeam']} {row['FTHG']} - {row['FTAG']} {row['AwayTeam']} {row['FTR']} {row['B365H']} {row['B365D']} {row['B365A']} Diff: {math.fabs(p1 - p2)} \n")
                #print("No bet!", row['HomeTeam'], row['FTHG']," - ", row['FTAG'], row['AwayTeam'], row['FTR'], row['B365H'], row['B365D'], row['B365A'], "Diff:", math.fabs(p1 - p2))
                skip += 1
                
            ax.scatter(j, cash, marker='.')
            j += 1 
        ax.grid(linestyle='--')
        ax.set(title=f"Profitability scatter plot - League: {league} - Season: {_season} - Initial cash=0 - Mean bid={np.around(bid_avg/(wrong+right), decimals=2)} - Total bid={np.around(bid_avg, decimals=2)}", xlabel="Match #", ylabel="Cash profit/loss")
             
        cash_tab.append(cash)
        season_tab.append(_season)
        right_tab.append(right)
        wrong_tab.append(wrong)
        skip_tab.append(skip)
    f.close()   
    plt.show()
    
    results = pd.DataFrame({'Season': season_tab,
                            'Cash': cash_tab,
                            'Right': right_tab,
                            'Wrong': wrong_tab,
                            'Skip': skip_tab})
    
    p1 = sns.barplot(x="Season", y='Cash', data=results)
    for line in range(0, results.shape[0]): # G - Good/Correct , B - Bad/Incorrect , S - Skipped
        p1.text(line, 0+0.20*max(results['Cash']), str('G: %.0f' % results.Right[line]), horizontalalignment='center')
        p1.text(line, 0+0.11*max(results['Cash']), str('B: %.0f' % results.Wrong[line]), horizontalalignment='center')
        p1.text(line, 0+0.02*max(results['Cash']), str('S: %.0f' % results.Skip[line]), horizontalalignment='center')
    plt.title(f"Profit/loss after season: {league} - G - Good, B - Bad, S - Skip")
    plt.ylabel("Profit/loss after season")
    plt.grid(linestyle='--')
    plt.show()
def check_loss_profit_draw(data, bid, prob_down):
    """
    Draws scatter plots with actual account balance for every season and at the end barplot with account balance after every season, and number of correct/incorrect bet or skipped matches
    @params data: Pandas DataFrame with matches analysis, usually narrowed limited to match from one league
    @param bid: Kind of bids, flat or kelly_crit
    @param prob_down: Probability difference limits when bet on draw
    """
    season = data.Season.unique()
    season.sort() #sort table of seasons, for hierarchically plot at the end
    cash_tab = []
    season_tab = []
    right_tab = []
    wrong_tab = []
    skip_tab = []
    league = data['League'].iloc[0]
    if os.path.exists(f"matches_{league}_{prob_down}_{bid}"):
        os.remove(f"matches_{league}_{prob_down}_{bid}")
    f = open(f"matches_{league}_{prob_down}_{bid}", "a") #I would save match's details to file
    for k, _season in enumerate(season):
        cash = 0
        data_loc = data[data['Season'] == _season]
        f.write(f"{_season}\n")
        right=0
        wrong=0
        skip=0
        bid_avg = 0
        fig = plt.figure(figsize=(16,10)) #fig for profitability plot
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        fig.tight_layout()
        ax = fig.add_subplot(len(season),1,k+1)
        ax.margins(x=0, y=0)
        j = 0 #bet counter
        for i, row in data_loc.iterrows(): 
            p1 = 1 / (1 + row['B365H']/row['B365A'] + row['B365H']/row['B365D'])
            p2 = 1 / (1 + row['B365A']/row['B365H'] + row['B365A']/row['B365D'])
            pX = 1 / (1 + row['B365D']/row['B365A'] + row['B365D']/row['B365H'])
        
            if math.fabs(p1 - p2) <= prob_down:
                _bid = calculate_bid(bid, pX, row['B365D'])
                bid_avg += _bid 
                if row['FTR'] == 'D':
                    cash += _bid * row['B365D'] - _bid
                    #print("OK! Score DRAW", row['HomeTeam'], row['FTHG']," - ", row['FTAG'], row['AwayTeam'], row['FTR'], "Bid:", _bid, "Gain:", _bid * row['B365D'] - _bid)
                    f.write(f"OK! Score DRAW {row['HomeTeam']} {row['FTHG']} - {row['FTAG']} {row['AwayTeam']} {row['FTR']} Bid: {_bid}, Gain: {_bid * row['B365D'] - _bid} \n")
                    right += 1
                else:
                    cash -= _bid
                    #print("BAD! Score DRAW", row['HomeTeam'], row['FTHG']," - ", row['FTAG'], row['AwayTeam'], row['FTR'], "Bid:", _bid)
                    f.write(f"BAD! Score DRAW {row['HomeTeam']} {row['FTHG']} - {row['FTAG']} {row['AwayTeam']} {row['FTR']} Bid: {_bid} \n")
                    wrong += 1
            else:
                f.write(f"No bet! {row['HomeTeam']} {row['FTHG']} - {row['FTAG']} {row['AwayTeam']} {row['FTR']} {row['B365H']} {row['B365D']} {row['B365A']} Diff: {math.fabs(p1 - p2)} \n")
                #print("No bet!", row['HomeTeam'], row['FTHG']," - ", row['FTAG'], row['AwayTeam'], row['FTR'], row['B365H'], row['B365D'], row['B365A'], "Diff:", math.fabs(p1 - p2))
                skip += 1
                
            ax.scatter(j, cash, marker='.')
            j += 1 
        ax.grid(linestyle='--')
        ax.set(title=f"Profitability scatter plot - League: {league} - Season: {_season} - Initial cash=0 - Mean bid={np.around(bid_avg/(right+wrong), decimals=2)} - Total bid={np.around(bid_avg, decimals=2)}", xlabel="Match #", ylabel="Cash profit/loss")
            
        cash_tab.append(cash)
        season_tab.append(_season)
        right_tab.append(right)
        wrong_tab.append(wrong)
        skip_tab.append(skip)
    f.close()
    plt.show()
    
    results = pd.DataFrame({'Season': season_tab,
                            'Cash': cash_tab,
                            'Right': right_tab,
                            'Wrong': wrong_tab,
                            'Skip': skip_tab})
    
    p1 = sns.barplot(x="Season", y='Cash', data=results)
    for line in range(0, results.shape[0]): # G - Good/Correct , B - Bad/Incorrect , S - Skipped
        p1.text(line, 0+0.20*max(results['Cash']), str('G: %.0f' % results.Right[line]), horizontalalignment='center')
        p1.text(line, 0+0.11*max(results['Cash']), str('B: %.0f' % results.Wrong[line]), horizontalalignment='center')
        p1.text(line, 0+0.02*max(results['Cash']), str('S: %.0f' % results.Skip[line]), horizontalalignment='center')
    plt.title(f"Profit/loss after season: {league} - G - Good, B - Bad, S - Skip")
    plt.ylabel("Profit/loss after season")
    plt.grid(linestyle='--')
    plt.show()
div_tab = data.Div.unique()
#I have to subtract E2 - League One because of lack of some data
e2 = np.array('E2')
div_tab = np.setdiff1d(div_tab,e2)
data[['Div', 'League']].groupby('Div').head(1)
#kelly_crit

#omit League1 (England) E2, because of lack of some data
#I will consider prob_down in range from 0.08 to 0.27 and prob_up from 0.35 to 0.75 with steps equal to 0.01 and 0.02, ~320 pairs analysis for every heatmap
for div in div_tab:
    data1 = load_test_data(div, ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
    prob_down = np.arange(0.08, 0.27, 0.01)
    prob_up = np.arange(0.35, 0.75, 0.02)
    
    prob_down, prob_up, cash_max, cash_min = check_prob_range(prob_down, prob_up, data1, 'kelly_crit')
    print(div, ": Best prob down:", prob_down, "Best prob up:", prob_up, "Cash with best probs:", cash_max, "Cash with worst probs:", cash_min, "\n\n")
#flat

#omit League1 (England) E2, because of lack of some data
#I will consider prob_down in range from 0.08 to 0.27 and prob_up from 0.35 to 0.75 with steps equal to 0.01 and 0.02, ~320 pairs analysis for every heatmap
for div in div_tab:
    data1 = load_test_data(div, ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
    prob_down = np.arange(0.08, 0.27, 0.01)
    prob_up = np.arange(0.35, 0.75, 0.02)
    
    prob_down, prob_up, cash_max, cash_min = check_prob_range(prob_down, prob_up, data1, 'flat')
    print(div, ": Best prob down:", prob_down, "Best prob up:", prob_up, "Cash with best probs:", cash_max, "Cash with worst probs:", cash_min, "\n\n")
#kelly_crit
data1 = load_test_data('E0', ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
check_loss_profit(data1, 'kelly_crit', 0.17, 0.37)
#flat
data1 = load_test_data('E0', ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
check_loss_profit(data1, 'flat', 0.17, 0.37)
#kelly_crit
data1 = load_test_data('E1', ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
check_loss_profit(data1, 'kelly_crit', 0.11, 0.37)
#flat
data1 = load_test_data('E1', ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
check_loss_profit(data1, 'flat', 0.11, 0.37)
#kelly_crit
data1 = load_test_data('G1', ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
check_loss_profit(data1, 'kelly_crit', 0.11, 0.49)
#flat
data1 = load_test_data('G1', ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
check_loss_profit(data1, 'flat', 0.11, 0.49)
#kelly_crit
data1 = load_test_data('I2', ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
check_loss_profit(data1, 'kelly_crit', 0.22, 0.49)
#flat
data1 = load_test_data('I2', ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
check_loss_profit(data1, 'flat', 0.22, 0.49)
#E2 #omit League1 (England) because of lack of some data
#I will consider prob_down in range from 0.06 to 0.31 with step equal to 0.01, ~25 analysis
for div in div_tab:
    data1 = load_test_data(div, ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
    prob_down = np.arange(0.06, 0.31, 0.01)
    
    prob_down, prob_up, cash_max, cash_min = check_prob_range_draw(prob_down, data1, 'flat')
    print(div, ": Best prob down:", prob_down, "Cash with best probs:", cash_max, "Cash with worst probs:", cash_min, "\n\n")
#kelly_crit
data1 = load_test_data('E0', ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
check_loss_profit_draw(data1, 'flat', 0.17)
data1 = load_test_data('G1', ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
check_loss_profit_draw(data1, 'flat', 0.11)
data1 = load_test_data('I2', ['2013/2014', '2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019'], data, 'BET365')
check_loss_profit_draw(data1, 'flat', 0.22)