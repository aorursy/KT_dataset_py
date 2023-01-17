import numpy as np

import cvxpy as cp

import pandas as pd



import pdb

MIN_DECADE = 1920

DATA_DIR = "/kaggle/input/baseball-ip/"

pd.options.display.width = 120

 

batter_header = ['Name', 'Dec', 'Pos', 'WARP', 'BA', 'OBP', 'SLG', 'H', 'HR', 'RBI', 'R', 'AB']

pitcher_header = ['Name', 'Dec', 'Pos', 'WARP', 'Record', 'ERA', 'WHIP', 'K', 'IP']



def print_team(team, is_pitcher):

    team_table = []

    for iter, player in team.iterrows():

        player_row = [player.NAME, round(player.YEAR), player.POS, "{:.1f}".format(player.WARP)]

        if (is_pitcher):

            record =str(round(player.W)) + "-" + str(round(player.L))

            era = "{:.2f}".format(round(player.ER/player.IP * 9, 2))

            whip = "{:.3f}".format(round((player.BB + player.H)/player.IP, 3))

            player_row.extend([record, era, whip, round(player.SO), round(player.IP)])

        else:

            ba = "{:.3f}".format(round(player.H/player.AB,3))

            obp = "{:.3f}".format(round((player.H+player.BB+player.HBP)/(player.AB+player.BB+player.HBP+0),3))

            slg = "{:.3f}".format(round(player.TB/player.AB,3))

            player_row.extend([ba, obp, slg,  round(player.H), round(player.HR), round(player.RBI), round(player.R), round(player.AB)])

        team_table.append(player_row)



    if (is_pitcher):

        team_table = pd.DataFrame(team_table, columns = pitcher_header)

    else:

        team_table = pd.DataFrame(team_table, columns = batter_header)

    print(team_table)

    print("\n")



field_pos = ['C', '1B', '2B', '3B', 'SS', 'OF']

pitcher_pos = ['RHP', 'LHP']



def print_solutions(ar, df):

    count = -1 

    picked_pitchers = pd.DataFrame()

    picked_batters = pd.DataFrame() 

    for x in np.nditer(ar):

        count = count + 1

        if (np.abs(x)  < 0.01):

            continue

        player = df.iloc[count]

        if (player.POS in pitcher_pos):

            picked_pitchers = picked_pitchers.append(player)

        else:

            picked_batters = picked_batters.append(player)

    picked_pitchers = picked_pitchers.sort_values(by = 'YEAR', ascending = True)

    picked_batters = picked_batters.sort_values(by = 'YEAR', ascending = True)

    print_team(picked_batters, False) 

    print_team(picked_pitchers, True) 





def print_top_players(positions, min_year = 1920, count = 10):

    for pos in positions:

        pos_df = pd.read_csv(DATA_DIR + pos + ".csv")

        pos_df['POS'] = pos

        pos_df = pos_df.loc[pos_df.YEAR >= min_year]

        pos_df = pos_df.sort_values(by="WARP", ascending = False)

        is_pitcher = pos in pitcher_pos

        print_team(pos_df.head(count), is_pitcher)
#explore the data    

print_top_players(field_pos + pitcher_pos)







def solve_ip():

    #initialize a dictionary of vectors to hold the position constraints

    pos_list = {pos: [] for pos in field_pos + pitcher_pos}



    #initialize a dictionary of vectors to hold the decade constraints

    decade_vec = {decade: []  for decade in range(MIN_DECADE, 2021, 10)}

    WAR_vec = []

    all_df = pd.DataFrame()



    #loop through each position

    for pos in field_pos + pitcher_pos:

        df = pd.read_csv(DATA_DIR + pos + ".csv")



        #remove players from decades too earlier than our first

        df = df.loc[df['YEAR'] >= MIN_DECADE]



        #create pos column on dataframe for easy printing

        df['POS'] = pos



        #create an overall dataframe

        all_df = all_df.append(df, sort=False)

        zeroes = np.zeros(len(df))

        ones = np.ones(len(df))



        #set the position vector

        for pos_list_item in field_pos + pitcher_pos:

            if (pos == pos_list_item):

                pos_list[pos_list_item].extend(ones)

            else:

                pos_list[pos_list_item].extend(zeroes)



        #set the decade vectors

        for decade in range(MIN_DECADE, 2021, 10):

            this_decade = np.where(df['YEAR'] == decade, 1, 0)

            decade_vec[decade].extend(this_decade)



        #create a vector WAR values

        WAR_vec.extend(df['WARP'].to_list())

            





    selection = cp.Variable(len(WAR_vec), boolean = True)

    constraints = [(decade_vec[i] * selection <= 1) for i in range(MIN_DECADE, 2021, 10)]

    for pos in field_pos + pitcher_pos:

        max_players = 1

        if (pos == 'OF'):

            max_players = 3

        constraints.append(pos_list[pos] * selection <= max_players)

    

    WAR = (WAR_vec * selection)

    problem = cp.Problem(cp.Maximize(WAR), constraints)

    problem.solve()



    all_df = all_df.reset_index(drop=True)

    if (problem.status not in ["infeasible", "infeasible_inaccurate", "unbounded"]):

        print("MAX WAR is {}".format(problem.value))

        print_solutions(selection.value, all_df)

    else:

        print("MAX WAR is Infeasible")

solve_ip()
