!pip install --upgrade -q gspread
from google.colab import auth

auth.authenticate_user()



import gspread

from oauth2client.client import GoogleCredentials



gc = gspread.authorize(GoogleCredentials.get_application_default())
import pandas as pd



############ SAMPLE SUBMISSION #############

worksheet = gc.open('SampleSubmissionStage2').sheet1



# get_all_values gives a list of seeds.

result = worksheet.get_all_values()



# Render as DataFrame.

pd.DataFrame.from_records(result).head()
############ TEAMS #############

worksheet = gc.open('Teams').sheet1



# get_all_values gives a list of seeds.

teams = worksheet.get_all_values()



# Render as DataFrame.

pd.DataFrame.from_records(teams).head()
# Function to get team name from team id

def get_team_name(id):

  id = str(id)

  for team in teams:

    if str(team[0]) == str(id):

      return team[1]

  return 'team id not found'



print(get_team_name(1438)) ## go hoos
############ NCAA TOURNEY SEEDS #############

worksheet = gc.open('NCAATourneySeeds').sheet1



# get_all_values gives a list of seeds.

ncaa_tourney_seeds = worksheet.get_all_values()



# Render as DataFrame.

pd.DataFrame.from_records(ncaa_tourney_seeds).head()
team_rankings = {}



years = ['2019']

for entry in ncaa_tourney_seeds:

  if entry[0] in years:

    rank = ''

    if len(entry[1]) == 3:

      rank = int((entry[1])[1:])

    elif len(entry[1]) == 4:

      rank = int((entry[1])[1:3])

    team_rankings[entry[2]] = rank

    

print(team_rankings)
# h_n > 32

h_n = 50

h_b = (h_n - 32) / (2 * h_n)



for i in range(1, len(result)):

  year = result[i][0][:4]

  first_team = result[i][0][5:9]

  second_team = result[i][0][10:14]

  for seed in ncaa_tourney_seeds:

    if seed[0] == year:

      if seed[2] == first_team:

        first_team_seed = int(seed[1][1:3])

      elif seed[2] == second_team:

        second_team_seed = int(seed[1][1:3])

  

  result[i][1] = (1/h_n)*(second_team_seed - first_team_seed + 16) + h_b

  

#   if team_rankings[first_team] > team_rankings[second_team]:

#     result[i][1] = .2

#   elif team_rankings[first_team] < team_rankings[second_team]:

#     result[i][1] = .8

#   else:

#     result[i][1] = .5
print(result)

df = pd.DataFrame.from_records(result)
############ REGULAR SEASON RESULTS #############



worksheet = gc.open('RegularSeasonDetailedResults').sheet1



# get_all_values gives a list of seeds.

reg_season = worksheet.get_all_values()



# Render as DataFrame.

pd.DataFrame.from_records(reg_season).head()
############ REGULAR SEASON RESULTS #############



worksheetConf = gc.open('TeamConferences').sheet1



# get_all_values gives a list of seeds.

wsConf = worksheetConf.get_all_values()[1:]



years  =  ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

teamConfDict = {}



for row in wsConf:

  if row[0] in years:

    if row[1] not in teamConfDict.keys():

      teamConfDict[row[1]] = row[2]



teamConfDict
def giveD(sos_year):

  ret = {}

  for x in sos_year:

    ret[x[0]] = float(x[1])

  

  return ret





worksheet2019 = gc.open('conference_rankings').worksheet("2019")

sos_2019 = giveD(worksheet2019.get_all_values())



worksheet2018 = gc.open('conference_rankings').worksheet("2018")

sos_2018 = giveD(worksheet2018.get_all_values())



worksheet2017 = gc.open('conference_rankings').worksheet("2017")

sos_2017 = giveD(worksheet2017.get_all_values())



worksheet2016 = gc.open('conference_rankings').worksheet("2016")

sos_2016 = giveD(worksheet2016.get_all_values())



worksheet2015 = gc.open('conference_rankings').worksheet("2015")

sos_2015 = giveD(worksheet2015.get_all_values())



worksheet2014 = gc.open('conference_rankings').worksheet("2014")

sos_2014 = giveD(worksheet2014.get_all_values())



worksheet2013 = gc.open('conference_rankings').worksheet("2013")

sos_2013 = giveD(worksheet2013.get_all_values())



worksheet2012= gc.open('conference_rankings').worksheet("2012")

sos_2012 = giveD(worksheet2012.get_all_values())



worksheet2011 = gc.open('conference_rankings').worksheet("2011")

sos_2011 = giveD(worksheet2011.get_all_values())



worksheet2010 = gc.open('conference_rankings').worksheet("2010")

sos_2010 = giveD(worksheet2010.get_all_values())



sos_2016
def get_regular_season(year, sos_year):



  teams_2010 = {}

  

  for game in reg_season:

    if game[0] == str(year):



      # winning team

      

      w_team = game[2]

      if w_team not in teams_2010.keys():

        teams_2010[w_team] = {}

        teams_2010[w_team]['games_played'] = 0

        teams_2010[w_team]['points_scored'] = 0

        teams_2010[w_team]['points_allowed'] = 0

        teams_2010[w_team]['fgm'] = 0

        teams_2010[w_team]['fgm_allowed'] = 0

        teams_2010[w_team]['fga'] = 0

        teams_2010[w_team]['fga_allowed'] = 0

        teams_2010[w_team]['fgm3'] = 0

        teams_2010[w_team]['fgm3_allowed'] = 0

        teams_2010[w_team]['fga3'] = 0

        teams_2010[w_team]['fga3_allowed'] = 0

        teams_2010[w_team]['ftm'] = 0

        teams_2010[w_team]['ftm_allowed'] = 0

        teams_2010[w_team]['fta'] = 0

        teams_2010[w_team]['fta_allowed'] = 0

        teams_2010[w_team]['or'] = 0

        teams_2010[w_team]['or_allowed'] = 0

        teams_2010[w_team]['dr'] = 0

        teams_2010[w_team]['dr_allowed'] = 0

        teams_2010[w_team]['ast'] = 0

        teams_2010[w_team]['ast_allowed'] = 0

        teams_2010[w_team]['to'] = 0

        teams_2010[w_team]['to_allowed'] = 0

        teams_2010[w_team]['stl'] = 0

        teams_2010[w_team]['stl_allowed'] = 0

        teams_2010[w_team]['blk'] = 0

        teams_2010[w_team]['blk_allowed'] = 0

        teams_2010[w_team]['pf'] = 0

        teams_2010[w_team]['pf_allowed'] = 0

        



      else:

        teams_2010[w_team]['games_played'] = teams_2010[w_team]['games_played'] + 1 

        teams_2010[w_team]['points_scored'] = teams_2010[w_team]['points_scored'] + int(game[3])

        teams_2010[w_team]['points_allowed'] = teams_2010[w_team]['points_allowed'] + int(game[5])

        teams_2010[w_team]['fgm'] = teams_2010[w_team]['fgm'] + int(game[8])

        teams_2010[w_team]['fga'] = teams_2010[w_team]['fga'] + int(game[9])

        teams_2010[w_team]['fgm3'] = teams_2010[w_team]['fgm3'] + int(game[10])

        teams_2010[w_team]['fga3'] = teams_2010[w_team]['fga3'] + int(game[11])

        teams_2010[w_team]['ftm'] = teams_2010[w_team]['ftm'] + int(game[12])

        teams_2010[w_team]['fta'] = teams_2010[w_team]['fta'] + int(game[13])

        teams_2010[w_team]['or'] = teams_2010[w_team]['or'] + int(game[14])

        teams_2010[w_team]['dr'] = teams_2010[w_team]['dr'] + int(game[15])

        teams_2010[w_team]['ast'] = teams_2010[w_team]['ast'] + int(game[16])

        teams_2010[w_team]['to'] = teams_2010[w_team]['to'] + int(game[17])

        teams_2010[w_team]['stl'] = teams_2010[w_team]['stl'] + int(game[18])

        teams_2010[w_team]['blk'] = teams_2010[w_team]['blk'] + int(game[19])

        teams_2010[w_team]['pf'] = teams_2010[w_team]['pf'] + int(game[20])



        teams_2010[w_team]['fgm_allowed'] = teams_2010[w_team]['fgm_allowed'] + int(game[21])

        teams_2010[w_team]['fga_allowed'] = teams_2010[w_team]['fga_allowed'] + int(game[22])

        teams_2010[w_team]['fgm3_allowed'] = teams_2010[w_team]['fgm3_allowed'] + int(game[23])

        teams_2010[w_team]['fga3_allowed'] = teams_2010[w_team]['fga3_allowed'] + int(game[24])

        teams_2010[w_team]['ftm_allowed'] = teams_2010[w_team]['ftm_allowed'] + int(game[25])

        teams_2010[w_team]['fta_allowed'] = teams_2010[w_team]['fta_allowed'] + int(game[26])

        teams_2010[w_team]['or_allowed'] = teams_2010[w_team]['or_allowed'] + int(game[27])

        teams_2010[w_team]['dr_allowed'] = teams_2010[w_team]['dr_allowed'] + int(game[28])

        teams_2010[w_team]['ast_allowed'] = teams_2010[w_team]['ast_allowed'] + int(game[29])

        teams_2010[w_team]['to_allowed'] = teams_2010[w_team]['to_allowed'] + int(game[30])

        teams_2010[w_team]['stl_allowed'] = teams_2010[w_team]['stl_allowed'] + int(game[31])

        teams_2010[w_team]['blk_allowed'] = teams_2010[w_team]['blk_allowed'] + int(game[32])

        teams_2010[w_team]['pf_allowed'] = teams_2010[w_team]['pf_allowed'] + int(game[33])



      # losing team

      l_team = game[4]

      if l_team not in teams_2010.keys():

        teams_2010[l_team] = {}

        teams_2010[l_team]['games_played'] = 0

        teams_2010[l_team]['games_played'] = 0

        teams_2010[l_team]['points_scored'] = 0

        teams_2010[l_team]['points_allowed'] = 0

        teams_2010[l_team]['fgm'] = 0

        teams_2010[l_team]['fgm_allowed'] = 0

        teams_2010[l_team]['fga'] = 0

        teams_2010[l_team]['fga_allowed'] = 0

        teams_2010[l_team]['fgm3'] = 0

        teams_2010[l_team]['fgm3_allowed'] = 0

        teams_2010[l_team]['fga3'] = 0

        teams_2010[l_team]['fga3_allowed'] = 0

        teams_2010[l_team]['ftm'] = 0

        teams_2010[l_team]['ftm_allowed'] = 0

        teams_2010[l_team]['fta'] = 0

        teams_2010[l_team]['fta_allowed'] = 0

        teams_2010[l_team]['or'] = 0

        teams_2010[l_team]['or_allowed'] = 0

        teams_2010[l_team]['dr'] = 0

        teams_2010[l_team]['dr_allowed'] = 0

        teams_2010[l_team]['ast'] = 0

        teams_2010[l_team]['ast_allowed'] = 0

        teams_2010[l_team]['to'] = 0

        teams_2010[l_team]['to_allowed'] = 0

        teams_2010[l_team]['stl'] = 0

        teams_2010[l_team]['stl_allowed'] = 0

        teams_2010[l_team]['blk'] = 0

        teams_2010[l_team]['blk_allowed'] = 0

        teams_2010[l_team]['pf'] = 0

        teams_2010[l_team]['pf_allowed'] = 0

      else:

        teams_2010[l_team]['games_played'] = teams_2010[l_team]['games_played'] + 1 

        teams_2010[l_team]['points_scored'] = teams_2010[l_team]['points_scored'] + int(game[5])

        teams_2010[l_team]['points_allowed'] = teams_2010[l_team]['points_allowed'] + int(game[3])

        teams_2010[l_team]['fgm'] = teams_2010[l_team]['fgm'] + int(game[21])

        teams_2010[l_team]['fga'] = teams_2010[l_team]['fga'] + int(game[22])

        teams_2010[l_team]['fgm3'] = teams_2010[l_team]['fgm3'] + int(game[23])

        teams_2010[l_team]['fga3'] = teams_2010[l_team]['fga3'] + int(game[24])

        teams_2010[l_team]['ftm'] = teams_2010[l_team]['ftm'] + int(game[25])

        teams_2010[l_team]['fta'] = teams_2010[l_team]['fta'] + int(game[26])

        teams_2010[l_team]['or'] = teams_2010[l_team]['or'] + int(game[27])

        teams_2010[l_team]['dr'] = teams_2010[l_team]['dr'] + int(game[28])

        teams_2010[l_team]['ast'] = teams_2010[l_team]['ast'] + int(game[29])

        teams_2010[l_team]['to'] = teams_2010[l_team]['to'] + int(game[30])

        teams_2010[l_team]['stl'] = teams_2010[l_team]['stl'] + int(game[31])

        teams_2010[l_team]['blk'] = teams_2010[l_team]['blk'] + int(game[32])

        teams_2010[l_team]['pf'] = teams_2010[l_team]['pf'] + int(game[33])



        teams_2010[w_team]['fgm_allowed'] = teams_2010[l_team]['fgm_allowed'] + int(game[8])

        teams_2010[w_team]['fga_allowed'] = teams_2010[l_team]['fga_allowed'] + int(game[9])

        teams_2010[w_team]['fgm3_allowed'] = teams_2010[l_team]['fgm3_allowed'] + int(game[10])

        teams_2010[w_team]['fga3_allowed'] = teams_2010[l_team]['fga3_allowed'] + int(game[11])

        teams_2010[w_team]['ftm_allowed'] = teams_2010[l_team]['ftm_allowed'] + int(game[12])

        teams_2010[w_team]['fta_allowed'] = teams_2010[l_team]['fta_allowed'] + int(game[13])

        teams_2010[w_team]['or_allowed'] = teams_2010[l_team]['or_allowed'] + int(game[14])

        teams_2010[w_team]['dr_allowed'] = teams_2010[l_team]['dr_allowed'] + int(game[15])

        teams_2010[w_team]['ast_allowed'] = teams_2010[l_team]['ast_allowed'] + int(game[16])

        teams_2010[w_team]['to_allowed'] = teams_2010[l_team]['to_allowed'] + int(game[17])

        teams_2010[w_team]['stl_allowed'] = teams_2010[l_team]['stl_allowed'] + int(game[18])

        teams_2010[w_team]['blk_allowed'] = teams_2010[l_team]['blk_allowed'] + int(game[19])

        teams_2010[w_team]['pf_allowed'] = teams_2010[l_team]['pf_allowed'] + int(game[20])



  

  for team in teams_2010.keys():

    

    conference = teamConfDict[str(team)]



    if conference == 'pac_ten':

      conference = 'pac_twelve'

      

    

    try:

      sos_TEMP = sos_year[conference]

    except:

      sos_TEMP = sos_year['meac']

      

    teams_2010[team]['sos'] = sos_TEMP

    

    

    if teams_2010[team]['fga'] != 0:

      teams_2010[team]['fgpercent'] = teams_2010[team]['fgm'] / teams_2010[team]['fga']

    else:

      teams_2010[team]['fgpercent'] = 0

    if teams_2010[team]['fga_allowed'] != 0:

      teams_2010[team]['fgpercent_allowed'] = teams_2010[team]['fgm_allowed'] / teams_2010[team]['fga_allowed']

    else:

      teams_2010[team]['fgpercent_allowed'] = 0

    for key in teams_2010[team].keys():



      if key != 'games_played' and key != 'fgpercent' and key != 'fgpercent_allowed' and teams_2010[team]['games_played'] != 0 and key != 'sos':

        teams_2010[team][key] = teams_2010[team][key] / teams_2010[team]['games_played']

  

  

  teams_2010 = pd.DataFrame.from_dict(teams_2010).T

  

  return teams_2010





teams_2010 = get_regular_season(2010, sos_2010)



teams_2011 = get_regular_season(2011, sos_2011)

teams_2012 = get_regular_season(2012, sos_2012)

teams_2013 = get_regular_season(2013, sos_2013)

teams_2014 = get_regular_season(2014, sos_2014)

teams_2015 = get_regular_season(2015, sos_2015)

teams_2016 = get_regular_season(2016, sos_2016)

teams_2017 = get_regular_season(2017, sos_2017)

teams_2018 = get_regular_season(2018, sos_2018)

teams_2019 = get_regular_season(2019, sos_2019)



teams_2019.head()
############ Player rankings #############



def get_top_five(file):

  worksheet = gc.open(file).sheet1



  # get_all_values gives a list of seeds.

  top_five = worksheet.get_all_values()



  # Render as DataFrame.

  top_five = pd.DataFrame.from_records(top_five).set_index(0)

  new_header = top_five.iloc[0]

  top_five = top_five[1:]

  top_five.columns = new_header

  return top_five



top_five_2010 = get_top_five('topFive2010').astype('float')

top_five_2011 = get_top_five('topFive2011').astype('float')

top_five_2012 = get_top_five('topFive2012').astype('float')

top_five_2013 = get_top_five('topFive2013').astype('float')

top_five_2014 = get_top_five('topFive2014').astype('float')

top_five_2015 = get_top_five('topFive2015').astype('float')

top_five_2016 = get_top_five('topFive2016').astype('float')

top_five_2017 = get_top_five('topFive2017').astype('float')

top_five_2018 = get_top_five('topFive2018').astype('float')

top_five_2019 = get_top_five('topFive2019').astype('float')

type(top_five_2019.iloc[0][0])
teams_2010 = pd.merge(top_five_2010, teams_2010, left_index=True, right_index=True)

teams_2011 = pd.merge(top_five_2011, teams_2011, left_index=True, right_index=True)

teams_2012 = pd.merge(top_five_2012, teams_2012, left_index=True, right_index=True)

teams_2013 = pd.merge(top_five_2013, teams_2013, left_index=True, right_index=True)

teams_2014 = pd.merge(top_five_2014, teams_2014, left_index=True, right_index=True)

teams_2015 = pd.merge(top_five_2015, teams_2015, left_index=True, right_index=True)

teams_2016 = pd.merge(top_five_2016, teams_2016, left_index=True, right_index=True)

teams_2017 = pd.merge(top_five_2017, teams_2017, left_index=True, right_index=True)

teams_2018 = pd.merge(top_five_2018, teams_2018, left_index=True, right_index=True)

teams_2019 = pd.merge(top_five_2019, teams_2019, left_index=True, right_index=True)



teams_2010
############ TOURNEY RESULTS #############



worksheet = gc.open('NCAATourneyCompactResults').sheet1



# get_all_values gives a list of seeds.

tourney_results = worksheet.get_all_values()



# Render as DataFrame.

pd.DataFrame.from_records(tourney_results).head()
print(teams_2010.columns)
import random

import numpy as np





def combine_reg_tourney(teams1, year):

  df = pd.DataFrame(columns=['1stPlayer', '2ndPlayer', '3rdPlayer', '4thPlayer', '5thPlayer', 'ast',

         'ast_allowed', 'blk', 'blk_allowed', 'dr', 'dr_allowed', 'fga', 'fga3',

         'fga3_allowed', 'fga_allowed', 'fgm', 'fgm3', 'fgm3_allowed',

         'fgm_allowed', 'fgpercent', 'fgpercent_allowed', 'fta', 'fta_allowed', 'ftm', 'ftm_allowed',

         'games_played', 'or', 'or_allowed', 'pf', 'pf_allowed',

         'points_allowed', 'points_scored', 'sos', 'stl', 'stl_allowed', 'to',

         'to_allowed', '1stPlayer_two', '2ndPlayer_two', '3rdPlayer_two', '4thPlayer_two', '5thPlayer_two', 'ast_two', 'ast_allowed_two', 'blk_two', 'blk_allowed_two', 

                             'dr_two', 'dr_allowed_two', 'fga_two', 'fga3_two', 'fga3_allowed_two', 'fga_allowed_two', 'fgm_two', 'fgm3_two', 

                             'fgm3_allowed_two', 'fgm_allowed_two', 'fgpercent_two', 'fgpercent_allowed_two', 'fta_two', 'fta_allowed_two', 'ftm_two', 'ftm_allowed_two', 'games_played_two', 

                             'or_two', 'or_allowed_two', 'pf_two', 'pf_allowed_two', 'points_allowed_two', 'points_scored_two', 'sos_two', 'stl_two', 

                             'stl_allowed_two', 'to_two', 'to_allowed_two', 'team_1', 'team_2', 'winner'

  ])

  

  for game in tourney_results:

    if game[0] == year:

      coin = random.randint(0,1)

      if coin == 0:

        team_1 = game[2]

        team_2 = game[4]

      else:

        team_1 = game[4]

        team_2 = game[2]



      teams = np.array([team_1, team_2])





      if str(game[2]) == str(team_1):

        array = np.array([1])



      else:

        array = np.array([0])



      data = np.append(teams1.loc[team_1].values, teams1.loc[team_2].values)

      data = np.append(data, teams)

      data = np.append(data, array)

      df.loc[tourney_results.index(game)] = data

  return df



df_all = []



df_all.append(combine_reg_tourney(teams_2010, "2010"))

df_all.append(combine_reg_tourney(teams_2011, "2011"))

df_all.append(combine_reg_tourney(teams_2012, "2012"))

df_all.append(combine_reg_tourney(teams_2013, "2013"))

df_all.append(combine_reg_tourney(teams_2014, "2014"))

df_all.append(combine_reg_tourney(teams_2015, "2015"))

df_all.append(combine_reg_tourney(teams_2016, "2016"))

df_all.append(combine_reg_tourney(teams_2017, "2017"))

df_all.append(combine_reg_tourney(teams_2018, "2018"))

df2019 = combine_reg_tourney(teams_2019, "2019")
df = pd.concat(df_all)

df.info()
############ SAMPLE SUBMISSION #############

worksheet = gc.open('SampleSubmissionStage2').sheet1



# get_all_values gives a list of seeds.

sample = worksheet.get_all_values()



# Render as DataFrame.

pd.DataFrame.from_records(sample).head()
df_2019 = pd.DataFrame(columns=['1stPlayer', '2ndPlayer', '3rdPlayer', '4thPlayer', '5thPlayer', 'ast',

         'ast_allowed', 'blk', 'blk_allowed', 'dr', 'dr_allowed', 'fga', 'fga3',

         'fga3_allowed', 'fga_allowed', 'fgm', 'fgm3', 'fgm3_allowed',

         'fgm_allowed', 'fgpercent', 'fgpercent_allowed', 'fta', 'fta_allowed', 'ftm', 'ftm_allowed',

         'games_played', 'or', 'or_allowed', 'pf', 'pf_allowed',

         'points_allowed', 'points_scored', 'sos', 'stl', 'stl_allowed', 'to',

         'to_allowed', '1stPlayer_two', '2ndPlayer_two', '3rdPlayer_two', '4thPlayer_two', '5thPlayer_two', 'ast_two', 'ast_allowed_two', 'blk_two', 'blk_allowed_two', 

                             'dr_two', 'dr_allowed_two', 'fga_two', 'fga3_two', 'fga3_allowed_two', 'fga_allowed_two', 'fgm_two', 'fgm3_two', 

                             'fgm3_allowed_two', 'fgm_allowed_two', 'fgpercent_two', 'fgpercent_allowed_two', 'fta_two', 'fta_allowed_two', 'ftm_two', 'ftm_allowed_two', 'games_played_two', 

                             'or_two', 'or_allowed_two', 'pf_two', 'pf_allowed_two', 'points_allowed_two', 'points_scored_two', 'sos_two', 'stl_two', 

                             'stl_allowed_two', 'to_two', 'to_allowed_two', 'team_1', 'team_2'

  ])





#print(len(df_2019.columns))

for row in sample[1:]:

  team_1 = row[0].split('_')[1]

  team_2 = row[0].split('_')[2]



  

  coin = random.randint(0,1)

  if coin == 1:

    temp = team_1

    team_1 = team_2

    team_2 = temp



  teams = np.array([team_1, team_2])

  



  data = np.append(teams_2019.loc[team_1].values, teams_2019.loc[team_2].values)

  data = np.append(data, teams)

  df_2019.loc[sample[1:].index(row)] = data



df_2019
from sklearn.model_selection import train_test_split



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

        ('std_scaler', StandardScaler())

    ])



ncaa_temp = df.drop("winner", axis=1)

ncaa_temp = ncaa_temp.drop("team_1", axis=1)

ncaa_temp = ncaa_temp.drop("team_2", axis=1)

ncaa_labels = df["winner"].copy()



ncaa_test_temp = df_2019.drop("team_1", axis=1)

ncaa_test_temp = ncaa_test_temp.drop("team_2", axis=1)

ncaa_test_labels = [] #df_2019["winner"].copy()



ncaa_prepared = (num_pipeline.fit_transform(ncaa_temp))

ncaa_test_prepared = (num_pipeline.fit_transform(ncaa_test_temp))



X_train = ncaa_prepared;

y_train = ncaa_labels;

y_train=y_train.astype('int')

X_test = ncaa_test_prepared;

y_test = ncaa_test_labels;

#y_test=y_test.astype('int')
###### LINEAR REGRESSION ######



from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)



from sklearn.metrics import mean_squared_error



ncaa_predictions = lin_reg.predict(X_train)

lin_mse = mean_squared_error(y_train, ncaa_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')



def testBestC(c):

  scaler = StandardScaler()



  # Training your svm here

  svm_clf = LinearSVC(C=c, loss="hinge", random_state=42)



  scaled_svm_clf = Pipeline([

          ("scaler", scaler),

          ("linear_svc", svm_clf),

      ])



  scaled_svm_clf.fit(X_train, y_train)



  y_predictions = scaled_svm_clf.predict(X_test)

  y_confidence = scaled_svm_clf.decision_function(X_test)



  return y_predictions, y_confidence

  #return cross_val_score(scaled_svm_clf, X_train, y_train, cv=2, scoring="precision")

  #return f1_score(y_test, y_predictions)



# bestC = 0

# bestAcc = 0

# cList = []

# fscoreList = []

# for x in range(1, 10):

#     temp = testBestC(x)

#     cList.append(x)

#     fscoreList.append(temp) 

#     if bestAcc < temp:

#         bestC = x

#         bestAcc = temp

        

# print(bestC)

# print(bestAcc)



# print(test_set)



import scipy.stats as st



lmao, lol = testBestC(6)



prob = []

for x in range(0, len(lmao)):

  y = st.norm.cdf(lol[x])

  p = 0.5 + (y/2)

  if lmao[x] == 1:

    prob.append(p)

  elif lmao[x] == 0:

    prob.append(1-p)

    
allPredictions = {}



for i in range(0, len(df_2019)):

  teamOne = get_team_name(int(df_2019.iloc[i]['team_1']))

  teamTwo = get_team_name(int(df_2019.iloc[i]['team_2']))

  #stringFormatted = "2019_" + str(df_2019.iloc[i]['team_1']) + "_" + str(df_2019.iloc[i]['team_2'])

  stringFormatted = "2019_" + teamOne + "_" + teamTwo

  

  allPredictions[stringFormatted] = prob[i]

  

allPredictions
from sklearn.calibration import CalibratedClassifierCV



svm = LinearSVC()

clf = CalibratedClassifierCV(svm) 

clf.fit(X_train, y_train)

y_proba = clf.predict_proba(X_test)

y_proba

allPredictions = {}



############ TEAMS #############

worksheet = gc.open('Teams').sheet1



# get_all_values gives a list of seeds.

teams = worksheet.get_all_values()



# Render as DataFrame.

pd.DataFrame.from_records(teams).head()







# Function to get team name from team id

def get_team_name(id):

  id = str(id)

  for team in teams:

    if str(team[0]) == str(id):

      return team[1]

  return 'team id not found'



for i in range(0, len(df_2019)):

  teamOne = get_team_name(int(df_2019.iloc[i]['team_1']))

  teamTwo = get_team_name(int(df_2019.iloc[i]['team_2']))

  #stringFormatted = "2019_" + str(df_2019.iloc[i]['team_1']) + "_" + str(df_2019.iloc[i]['team_2'])

  stringFormatted = "2019_" + teamOne + "_" + teamTwo

  

  allPredictions[stringFormatted] = y_proba[i][1]

  

allPredictions
from google.colab import files



with open('resultLMAO.csv', 'w') as f:

  f.write('ID,Pred\n')

  for key in allPredictions.keys():

    f.write("%s,%s\n"%(key,allPredictions[key]))







files.download('resultLMAO.csv')

# !pip install binarytree

# !pip install matplotlib

# !pip install numpy

# !pip install pandas

# !pip install PIL

# !pip install bracketeer

# from google.colab import drive

# drive.mount('/content/gdrive')



# !pwd; cd gdrive/'My Drive'/empty_string/mens-machine-learning-competition-2019; ls; pwd



# from bracketeer import build_bracket



# b = build_bracket(

#         outputPath='output.png',

#         teamsPath='/content/gdrive/My Drive/empty_string/mens-machine-learning-competition-2019/Stage2DataFiles/Teams.csv',

#         seedsPath='/content/gdrive/My Drive/empty_string/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySeeds.csv',

#         submissionPath='/content/gdrive/My Drive/empty_string/mens-machine-learning-competition-2019/SampleSubmissionStage2.csv',

#         slotsPath='/content/gdrive/My Drive/empty_string/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySlots.csv',

#         year=2018

# )