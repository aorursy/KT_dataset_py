import json
import numpy as np
from regression import *
from shotEffectiveness import estimateProbabilities
from Index import *
from collections import OrderedDict
import operator
import unicodedata
file_pointer = open("/Users/shubhampawar/epl_data/xml_stats/season16-17/season_stats.json", "rb")

data = json.load(file_pointer, object_pairs_hook=OrderedDict)

# previous two seasons data and current season data
file_pointer = open("/Users/shubhampawar/epl_data/xml_stats/season17-18/season_stats.json", "rb")

data.update(json.load(file_pointer, object_pairs_hook=OrderedDict))

# file_pointer = open("/Users/shubhampawar/epl_data/xml_stats/season15-16/season_stats.json", "rb")

# data.update(json.load(file_pointer, object_pairs_hook=OrderedDict))

# file_pointer = open("/Users/shubhampawar/epl_data/xml_stats/season14-15/season_stats.json", "rb")

# data.update(json.load(file_pointer, object_pairs_hook=OrderedDict))
aggregate_stats_dict = dict()

total_shots = []
won_contest = []
yellow_card_opp = []
red_card_opp = []
accurate_pass = []
total_tackle_opp= []
touches = []
aerial_won = []

shots = 0
goals = 0
onTarget = 0
saves = 0
blocks = 0

for matchId, value1 in data.iteritems():
    for teamId, value2 in data[matchId].iteritems():
        for key in data[matchId].keys():
            if key != teamId:
                opponent_teamId = key 
        for key in data[matchId][teamId]["aggregate_stats"].keys():
            if key in aggregate_stats_dict.keys():
                aggregate_stats_dict[key] +=1
            else:
                aggregate_stats_dict.update({key: 1})
        total_shots.append(int(data[matchId][teamId]["aggregate_stats"]["total_scoring_att"]))
        
        if "total_scoring_att" in data[matchId][teamId]["aggregate_stats"].keys():
            shots += float(data[matchId][teamId]["aggregate_stats"]["total_scoring_att"])
        if "goals" in data[matchId][teamId]["aggregate_stats"].keys():
            goals += float(data[matchId][teamId]["aggregate_stats"]["goals"])
        if ("ontarget_scoring_att" in data[matchId][teamId]["aggregate_stats"].keys()) and ("blocked_scoring_att" in data[matchId][teamId]["aggregate_stats"].keys()):
            onTarget += float(data[matchId][teamId]["aggregate_stats"]["ontarget_scoring_att"]) + float(data[matchId][teamId]["aggregate_stats"]["blocked_scoring_att"])
        if ("ontarget_scoring_att" in data[matchId][teamId]["aggregate_stats"].keys()) and not("blocked_scoring_att" in data[matchId][teamId]["aggregate_stats"].keys()):
            onTarget += float(data[matchId][teamId]["aggregate_stats"]["ontarget_scoring_att"])
        if not("ontarget_scoring_att" in data[matchId][teamId]["aggregate_stats"].keys()) and ("blocked_scoring_att" in data[matchId][teamId]["aggregate_stats"].keys()):
            onTarget += float(data[matchId][teamId]["aggregate_stats"]["blocked_scoring_att"])
        if ("ontarget_scoring_att" in data[matchId][teamId]["aggregate_stats"].keys()) and ("goals" in data[matchId][teamId]["aggregate_stats"].keys()):
            saves += float(data[matchId][teamId]["aggregate_stats"]["ontarget_scoring_att"]) - float(data[matchId][teamId]["aggregate_stats"]["goals"])
        if ("ontarget_scoring_att" in data[matchId][teamId]["aggregate_stats"].keys()) and not("goals" in data[matchId][teamId]["aggregate_stats"].keys()):
            saves += float(data[matchId][teamId]["aggregate_stats"]["ontarget_scoring_att"])
        if "blocked_scoring_att" in data[matchId][teamId]["aggregate_stats"].keys():
            blocks += float(data[matchId][teamId]["aggregate_stats"]["blocked_scoring_att"])

        won_contest_sum = 0
        yellow_card_opp_sum = 0
        red_card_opp_sum = 0
        accurate_pass_sum = 0
        total_tackle_opp_sum = 0
        touches_sum = 0
        aerial_won_sum = 0

        for player_name, value3 in data[matchId][teamId]["Player_stats"].iteritems():
            for key in data[matchId][teamId]["Player_stats"][player_name]["Match_stats"].keys():    
                if key == "won_contest":
                        won_contest_sum += float(data[matchId][teamId]["Player_stats"][player_name]["Match_stats"]["won_contest"])
                if key == "accurate_pass":
                    accurate_pass_sum += float(data[matchId][teamId]["Player_stats"][player_name]["Match_stats"]["accurate_pass"])
                if key == "touches":
                    touches_sum += float(data[matchId][teamId]["Player_stats"][player_name]["Match_stats"]["touches"])
                if key == "aerial_won": 
                    aerial_won_sum += float(data[matchId][teamId]["Player_stats"][player_name]["Match_stats"]["aerial_won"])   
            
        won_contest.append(won_contest_sum)
        accurate_pass.append(accurate_pass_sum)
        touches.append(touches_sum)
        aerial_won.append(aerial_won_sum)

        for player_name, value3 in data[matchId][opponent_teamId]["Player_stats"].iteritems():
            for key in data[matchId][opponent_teamId]["Player_stats"][player_name]["Match_stats"].keys():
                if key == "total_tackle":
                    total_tackle_opp_sum += float(data[matchId][opponent_teamId]["Player_stats"][player_name]["Match_stats"]["total_tackle"])
                if key == "yellow_card":
                    yellow_card_opp_sum += float(data[matchId][opponent_teamId]["Player_stats"][player_name]["Match_stats"]["yellow_card"])
                if key == "red_card":
                    red_card_opp_sum += float(data[matchId][opponent_teamId]["Player_stats"][player_name]["Match_stats"]["red_card"])
                if key == "second_yellow":
                    red_card_opp_sum += float(data[matchId][opponent_teamId]["Player_stats"][player_name]["Match_stats"]["second_yellow"])
            
        total_tackle_opp.append(total_tackle_opp_sum)   
        yellow_card_opp.append(yellow_card_opp_sum)
        red_card_opp.append(red_card_opp_sum)
        
print(aggregate_stats_dict.keys())
shots_data = np.transpose(np.array([total_shots, won_contest, accurate_pass, total_tackle_opp, aerial_won, yellow_card_opp, red_card_opp]))
#print shots_data.shape
coefficients = shotRegression(shots_data)
#print total_shots[0], won_contest[0], accurate_pass[0], total_tackle[0], touches[0], aerial_won[0], yellow_card_opp[0], red_card_opp[0]
probabilities = estimateProbabilities(shots, goals, onTarget, saves, blocks)
playerRatings = dict()
playerPosition = dict()
playerIdNameData = dict()
playerMatchSummary = dict()
playerMeanContributions = dict()
teamMeanContribution = dict()
teamRoster = dict()
for matchId, value in data.iteritems():
    estimates = matchEstimates(data[matchId], coefficients, probabilities) 
    #lambda_ = estimates[0]
    #goalProbabilities(lambda_)
    playerData = playerPoints(data[matchId], estimates, coefficients, probabilities)
    position_update(playerPosition, playerData)
    summary_update(playerMatchSummary, playerData)
    player_name_update(playerIdNameData, playerData)
    roster_update(teamRoster, playerData)
    aggregate_to_mean_contribution(playerMeanContributions, playerMatchSummary)
    dict_update(playerRatings, playerData)

match_contributions(teamMeanContribution, playerMeanContributions, teamRoster)
#result = game_result("13", "16", teamMeanContribution, coefficients, probabilities)
#print result
 
#playerRatings = sorted(playerRatings.items(), key=operator.itemgetter(1))
#print playerRatings
ratings_file = "playerRatings.csv"
positions_file = "playerPositions.csv"
contributions_file = "playerContributions.csv"
player_name_file = "playerNameId.csv"
team_roster = "teamRoster.csv"
mean_starts_contributions_file = "playerStartsMeanContributions.csv"
mean_subs_contributions_file = "playerSubsMeanContributions.csv"
team_mean_contributions_file = "teamMeanContributions.csv"


csvfiles = [ratings_file, positions_file, contributions_file, player_name_file]

fp = open(ratings_file, 'wb')
fp.write("Player_id, Ratings\n")
for player, ratings in playerRatings.iteritems():
    ratings_string = ""
    for rating in ratings:
        ratings_string += "," + str(rating)
    fp.write(str(player) + ratings_string +  "\n")
fp.close()

fp = open(positions_file, 'wb')
fp.write("Player_id, Others, Goalkeeper, Defender, Midfielder, Forward, Substitute\n")
for player, positions in playerPosition.iteritems():
    positions_string = ""
    for position in positions:
        positions_string += "," + str(position)
    fp.write(str(player) + positions_string + "\n")
fp.close()

''' fp = open(contributions_file, 'wb')
# contributions vector is [won_contest, accurate_pass, touch_factor, aerial_won, saves, total_tackle_opp, yellow_card_opp, red_card_opp, goals, assists]
fp.write("Player_id, won_contest, accurate_pass, touch_factor, aerial_won, saves, total_tackle_opp, yellow_card_opp, red_card_opp, goals, assists\n")
for player, contributions in playerContributions.iteritems():
    contributions_string = ""
    for contribution in contributions:
        contributions_string += "," + str(contribution)
    fp.write(str(player) + contributions_string + "\n")
fp.close() '''

fp = open(player_name_file, 'wb')
fp.write("Player_id, Player_name, Player_team1, Appearances1, Player_team2, Appearances2\n")
for player, playerInfo in playerIdNameData.iteritems():
    player_name = unicodedata.normalize("NFKD", playerInfo[0]).encode('ascii', 'ignore')
    player_team_info_string = ""
    for team, appearances in playerInfo[1].iteritems():
        player_team_info_string += "," + str(team) + "," + str(appearances)
    fp.write(str(player) + "," + player_name + player_team_info_string + "\n")
fp.close()


fp = open(contributions_file, 'wb')
fp.write("Player_id, Date, won_contest, accurate_pass, touch_factor, aerial_won, saves, tackles, yellow_card, red_card, goal_factor, assist_factor, shot_factor, cross_factor, tackle_factor, Position, Rating\n")
for player, matchData in playerMatchSummary.iteritems():
    for date, contributions in matchData.iteritems():
        contributions_string_ = ""
        for contribution in contributions:
            contributions_string_ += "," + str(contribution)
        fp.write(str(player) + "," + str(date) + contributions_string_ + "\n") 
fp.close()

fp = open(team_roster, 'wb')
fp.write("Team_id, Player_id, Start_prob, Sub_prob\n")
for team, roster in teamRoster.iteritems():
    for player_id, data in teamRoster[team].iteritems():
        fp.write(str(team) + "," + str(player_id) + "," + str(data[0]) + "," + str(data[1]) + "\n")
fp.close()

fstart = open(mean_starts_contributions_file, 'wb')
fsub = open(mean_subs_contributions_file, 'wb')

fstart.write("Player_id, won_contest, accurate_pass, touch_factor, aerial_won, saves, tackles, yellow_card, red_card, goal_factor, assist_factor, shot_factor, cross_factor, tackle_factor\n")
fsub.write("Player_id, won_contest, accurate_pass, touch_factor, aerial_won, saves, tackles, yellow_card, red_card, goal_factor, assist_factor, shot_factor, cross_factor, tackle_factor\n")
for player, mean_data in playerMeanContributions.iteritems():
    mean_start_contribution_string = ""
    mean_sub_contribution_string = ""
    for contribution in mean_data[0]:
        mean_start_contribution_string += "," + str(contribution)
    for contribution in mean_data[1]:
        mean_sub_contribution_string += "," + str(contribution)

    fstart.write(str(player) + mean_start_contribution_string + "\n")
    fsub.write(str(player)  + mean_sub_contribution_string + "\n")

fstart.close()
fsub.close()

fp = open(team_mean_contributions_file, 'wb')
fp.write("Team_id, won_contest, accurate_pass, touch_factor, aerial_won, saves, total_tackle_opp, yellow_card_opp, red_card_opp\n")
for team, contributions in teamMeanContribution.iteritems():
    contributions_string = ""
    for contribution in contributions:
        contributions_string += "," + str(contribution)

    fp.write(str(team) + contributions_string + "\n")

fp.close()

fp = open("info_file.csv", "wb")
fp.write("Constant, Won_contest, accurate_pass, tackle_opp, aerial_won, yellow_card, red_card\n")
count = 0
for coefs in coefficients:
    if count == 0:
        fp.write(str(coefs))
        count = 1
        continue
    fp.write("," + str(coefs))
fp.write("\n")

fp.write("P(G|S), P(onTarget|shot), P(saveOrBlock|onTarget)\n")
count = 0
for prob in probabilities:
    if count == 0:
        fp.write(str(prob))
        count = 1
        continue
    fp.write("," + str(prob))
fp.write("\n")

fp.close()
