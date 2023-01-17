import os
from random import randint
import numpy as np 
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import normalize

massey_path = "../input/mens-machine-learning-competition-2018/"
old_data_path = "../input/supplemental-march-madness-data/"
tourney_games = pd.read_csv(massey_path + "NCAATourneyCompactResults.csv")
tourney_seeds = pd.read_csv(massey_path + "NCAATourneySeeds.csv")
rankings = pd.read_csv(massey_path + "MasseyOrdinals.csv") 
teams = pd.read_csv(massey_path + "Teams.csv")
team_names = dict(zip(list(teams["TeamID"]), list(teams["TeamName"])))
team_conf = pd.read_csv(massey_path + "TeamConferences.csv")
team_coaches = pd.read_csv(massey_path + "TeamCoaches.csv")
team_cities = pd.read_csv(massey_path + "Cities.csv")
reg_season_stats = pd.read_csv(massey_path + "RegularSeasonDetailedResults.csv")
reg_season_wins = pd.read_csv(massey_path + "RegularSeasonCompactResults.csv")
#AP POLLS
ap_polls = pd.read_csv(old_data_path + "ap_polls.csv")
team_spellings = pd.read_csv(massey_path + "TeamSpellings.csv", encoding='latin1')

#Add team ID to ap_polls
team_ids = []
for i, row in ap_polls.iterrows():
    school_name = row["School"].lower()
    school_id = team_spellings.loc[team_spellings["TeamNameSpelling"] == school_name]
    school_id = int(school_id.iloc[0]["TeamID"])
    team_ids.append(school_id)
ap_polls["TeamID"] = team_ids

#COACH TOURNAMENT APPEARANCES
coach_tourney_appearances = pd.read_csv(old_data_path + "coach_tourney_appearances.csv")
coach_appearances = {row["CoachName"]: row["TourneyAppearances"] for _, row in coach_tourney_appearances.iterrows()}

#GAME LOCATION DISTANCES
game_distances = pd.read_csv(old_data_path + "GameDistances.csv")
tournament_distance_matrix = pd.read_csv(old_data_path + "TournamentDistanceMatrix.csv")
#Feature Config
features = {
    'round': 1,
    'seed': 1,
    'coach': 1,
    'distance': 1,
    'last': 1,
    'rpi': 1,
    'pom': 1,
    'reb': 1,
    'fga': 1,
    'fgpct': 1,
    'ftpct': 1
}
num_features = 2 * len([1 for (k, v) in features.items() if (k != 'round' and v)]) + features['round']

#Filename for saving
def gen_unique_name():
    name = ''
    if features['round']: name += 'Rnd'
    if features['seed']: name += 'S'
    if features['coach']: name += 'C'
    if features['distance']: name += 'D'
    if features['last']: name += 'L'
    if features['rpi']: name += 'Rpi'
    if features['pom']: name += 'P'
    if features['reb']: name += 'Reb'
    if features['fga']: name += 'Fga'
    if features['fgpct']: name += 'Fgp'
    if features['ftpct']: name += 'Ft'
    return name
feature_filename = gen_unique_name()
#Returns the tournament round of the game in question
def find_round(day_num):
    #Return -1 if feature is not used
    if not features['round']:
        return -1
    
    #Play in games
    if day_num in [134, 135]:
        return 0
    #First Round
    elif day_num in [136, 137]:
        return 1
    #Round of 32
    elif day_num in [138, 139]:
        return 2
    #Sweet Sixteen
    elif day_num in [143, 144]:
        return 3
    #Elite Eight
    elif day_num in [145, 146]:
        return 4
    #Final Four
    elif day_num == 152:
        return 5
    #Champtionship
    elif day_num == 154:
        return 6

#Seed comes in the form W00a
#Where W is the region, 00 is the seed, and a indicates if it was play in game
def find_seed(year, team_id):
    #Return -1 if feature is not used
    if not features['seed']:
        return -1
    
    seed = tourney_seeds.loc[(tourney_seeds["Season"] == year) & (tourney_seeds["TeamID"] == team_id)]
    seed = seed.iloc[0]["Seed"]
    if len(seed) > 3: #Play-in
        seed = seed[:-1]
    seed = seed[1:]
    return int(seed)
 
#Return the RPI and POM ranking for the team if the data is available
#If RPI and POM are not available, return the AP Polls rank
#If AP Poll rank is not available either, return the default value
def get_rankings(year, team_id):
    #Change rank to -1 if feature is not used
    def override(ranks):
        if not features['rpi']:
            ranks[0] = -1
        if not features['pom']:
            ranks[1] = -1
        return ranks
    
    default_rank = 55 #Average RPI/POM rank for teams not in top 25 AP polls
    
    #Extract rank from Massey dataset or my personal dataset depending on rank type
    def rank_by_type (name):
        #Massey
        if name != "AP":
            rank = rankings.loc[(rankings["SystemName"] == name) & (rankings["TeamID"] == team_id) 
                                & (rankings["Season"] == year) & (rankings["RankingDayNum"] == 133)]
            return rank.iloc[0]["OrdinalRank"]
        #AP Polls
        else:
            rank = ap_polls.loc[(ap_polls["TeamID"] == team_id) & (ap_polls["Year"] == year)]
            if not rank.empty:
                return rank.iloc[0]["Rank"]
            return None
        
    #If data is available, get RPI/POM
    if year > 2002:
        rpi_rank = rank_by_type("RPI")
        pom_rank = rank_by_type("POM")
        return override((rpi_rank, pom_rank))
    #Otherwise check AP
    else:
        ap_rank = rank_by_type("AP")
        if ap_rank:
            return override((ap_rank, ap_rank))
        return override((default_rank, default_rank))

#Return the number of years a coach has been in the tournament
#Uses global variable "coach_appearances"
def coach_experience(year, team_id):
    #Return -1 if feature is not used
    if not features['coach']:
        return -1
    
    if year == 2018:
        days = 77
    else:
        days = 154
        
    coach = team_coaches.loc[(team_coaches["TeamID"] == team_id) & (team_coaches["Season"] == year)
                                & (team_coaches["LastDayNum"] == days)]
    coach = coach.iloc[0]["CoachName"]
    
    if coach not in coach_appearances:
        coach_appearances[coach] = 1
        return 0
    
    coach_appearances[coach] += 1
    return (coach_appearances[coach] - 1)

#Regular season statistics leading up to the tournament
#Returns average rebounds, field goal attempts, field goal percentage, and free throw percentage
#If the data is unavailable, return average
default_win_values = (36.8, 57.0, 46.6, 70.5)
default_lose_values = (36.0, 56.2, 45.9, 70.2)
def get_stats(year, team_id, win=None):
    #Change rank to -1 if feature is not used
    def override(ranks):
        if not features['reb']:
            ranks[0] = -1
        if not features['fga']:
            ranks[1] = -1
        if not features['fgpct']:
            ranks[2] = -1
        if not features['ftpct']:
            ranks[3] = -1
        return ranks
    
    #Data unavailable, return averages
    if year < 2003:
        if win == "w":
            return override(default_win_values)
        else:
            return override(default_lose_values)
    
    #Regular season games
    team_stats = reg_season_stats.loc[((reg_season_stats["Season"] == year) 
                         & ((reg_season_stats["WTeamID"] == team_id) | (reg_season_stats["LTeamID"] == team_id)))]
    
    games = len(team_stats)
    rebounds = np.zeros(games)
    fga = np.zeros(games)
    fgm = np.zeros(games)
    fta = np.zeros(games)
    ftm = np.zeros(games)
    for i, (_, row) in enumerate(team_stats.iterrows()):
        if row["WTeamID"] == team_id:
            t = "W"
        else:
            t = "L"
        rebounds[i] = row[t+"OR"] + row[t+"DR"]
        fga[i] = row[t+"FGA"]
        fgm[i] = row[t+"FGM"]
        fta[i] = row[t+"FTA"]
        ftm[i] = row[t+"FTM"]
    fg_pct = np.array([100*(fgm[i]/float(fga[i])) if fga[i] != 0 else 0 for i in range(games)])
    ft_pct = np.array([100*(ftm[i]/float(fta[i])) if fta[i] != 0 else 0 for i in range(games)])
    
    rebounds = np.mean(rebounds)
    fga = np.mean(fga)
    fg_pct = np.mean(fg_pct)
    ft_pct = np.mean(ft_pct)
    
    return override((rebounds, fga, fg_pct, ft_pct))

#Returns the distance from the team's hometown to the location of play
default_win_distance = 1050
default_lose_distance = 1150
def find_distance (year, day_num, team_id, win=None):
    #Return -1 if feature is not used
    if not features['distance']:
        return -1
    
    if year < 2010:
        if win == "w":
            return default_win_distance
        return default_lose_distance
    
    distance = game_distances.loc[(game_distances["Season"] == year) & (game_distances["DayNum"] == day_num)
                                 & (game_distances[win.upper()+"TeamID"] == team_id)]
    return distance.iloc[0][win.upper()+"TeamDistance"]

#Returns the number of the last 10 games the team has won
def last_ten (year, team_id):
    #Return -1 if feature is not used
    if not features['last']:
        return -1
    
    games = reg_season_stats.loc[((reg_season_stats["Season"] == year) 
                         & ((reg_season_stats["WTeamID"] == team_id) | (reg_season_stats["LTeamID"] == team_id)))]
    last = games[-10:]
    wins = len([0 for _, row in last.iterrows() if row["WTeamID"] == team_id])
    return wins


#Normalizes scores as a probablility distribution for the labels
differentials = {
    11: 7,
    10: 7,
    9: 6.5,
    8: 6,
    7: 5.5,
    6: 5,
    5: 4.2,
    4: 3.4,
    3: 2.6,
    2: 1.8,
    1: 0.8
}
def normalize_scores (score1, score2):
    if (score1 - score2) >= 12:
        return np.array([1.0, 0.0])
    diff = score1 - score2
    score2 += differentials[diff]
    scores = np.array([score1, score2])
    return np.exp(scores) / sum(np.exp(scores))

sizeA = 1136
sizeB = 448
sizeC = 533
num_features = 21
num_outputs = 2

training1985 = None
training2003 = None
training2010 = None

def shuffle_data (x, y):
    group = np.array(list(zip(x, y)))
    np.random.shuffle(group)
    x = np.array([i[0] for i in group])
    y = np.array([i[1] for i in group])
    return (x, y)
    
def gather_data():
    training1985x = np.zeros((sizeA, num_features))
    training2003x = np.zeros((sizeB, num_features))
    training2010x = np.zeros((sizeC, num_features))
    training1985y = np.zeros((sizeA, num_outputs))
    training2003y = np.zeros((sizeB, num_outputs))
    training2010y = np.zeros((sizeC, num_outputs))
    global training1985, training2003, training2010

    ai, bi, ci = 0, 0, 0
    for i, row in tourney_games.iterrows():
        year = row["Season"]
        day_num = row["DayNum"]
        w_team = row["WTeamID"]
        l_team = row["LTeamID"]
        w_score = row["WScore"]
        l_score = row["LScore"]
            
        rnd = find_round(day_num)

        w_seed = find_seed(year, w_team)
        l_seed = find_seed(year, l_team)

        w_coach = coach_experience(year, w_team)
        l_coach = coach_experience(year, l_team)

        w_rpi, w_pom = get_rankings(year, w_team)
        l_rpi, l_pom = get_rankings(year, l_team)

        w_distance = find_distance(year, day_num, w_team, "w")
        l_distance = find_distance(year, day_num, l_team, "l")

        w_last = last_ten(year, w_team)
        l_last = last_ten(year, l_team)

        w_reb, w_fga, w_fgpct, w_ftpct = get_stats(year, w_team, "w")
        l_reb, l_fga, l_fgpct, l_ftpct = get_stats(year, l_team, "l")

        win_prob = normalize_scores(w_score, l_score)

        if randint(0, 1) == 0:
            examp = np.array([i for i in [rnd, w_seed, w_coach, w_distance, w_last, w_rpi, w_pom, w_reb, w_fga, w_fgpct, w_ftpct,
                                  l_seed, l_coach, l_distance, l_last, l_rpi, l_pom, l_reb, l_fga, l_fgpct, l_ftpct]
                             if i != -1])
            label = win_prob
        else:
            examp = np.array([i for i in [rnd, l_seed, l_coach, l_distance, l_last, l_rpi, l_pom, l_reb, l_fga, l_fgpct, l_ftpct,
                                 w_seed, w_coach, w_distance, w_last, w_rpi, w_pom, w_reb, w_fga, w_fgpct, w_ftpct]
                              if i != -1])
            label = np.array([win_prob[1], win_prob[0]])
            
        if year < 2003:
            training1985x[ai] = examp
            training1985y[ai] = label
            ai += 1
        elif year < 2010:
            training2003x[bi] = examp
            training2003y[bi] = label
            bi += 1
        else:
            training2010x[ci] = examp
            training2010y[ci] = label
            ci += 1

    training1985x = normalize(training1985x)
    training2003x = normalize(training2003x)
    training2010x = normalize(training2010x)
    
    x1985, y1985 = shuffle_data(training1985x, training1985y)
    x2003, y2003 = shuffle_data(training2003x, training2003y)
    x2010, y2010 = shuffle_data(training2010x, training2010y)
    
    np.save('x1985_{}.npy'.format(feature_filename), x1985)
    np.save('y1985_{}.npy'.format(feature_filename), y1985)
    np.save('x2003_{}.npy'.format(feature_filename), x2003)
    np.save('y2003_{}.npy'.format(feature_filename), y2003)
    np.save('x2010_{}.npy'.format(feature_filename), x2010)
    np.save('y2010_{}.npy'.format(feature_filename), y2010)
gather_data()
x1985 = np.load('x1985_{}.npy'.format(feature_filename))
y1985 = np.load('y1985_{}.npy'.format(feature_filename))
x2003 = np.load('x2003_{}.npy'.format(feature_filename))
y2003 = np.load('y2003_{}.npy'.format(feature_filename))
x2010 = np.load('x2010_{}.npy'.format(feature_filename))
y2010 = np.load('y2010_{}.npy'.format(feature_filename))

training_data = {1985: (x1985[:-15], y1985[:-15]), 2003: (x2003[:-15], y2003[:-15]), 2010: (x2010[:-120], y2010[:-120])}
training_data[2000] = (np.concatenate((x2003[:-15], x2010[:-120])), np.concatenate((y2003[:-15], y2010[:-120])))
x_test = np.concatenate((x1985[-15:], x2003[-15:], x2010[-120:]))
y_test = np.concatenate((y1985[-15:], y2003[-15:], y2010[-120:]))
test = (x_test, y_test)
n_in = num_features
n_out = num_outputs
n_hid = 64
batch_size = 16

def get_batch(year):
    training = training_data[year]
    size = training[0].shape[0]
        
    x, y = training
    seed = randint(0, size - batch_size)
    
    return x[seed : seed+batch_size], y[seed : seed + batch_size]
    

x = tf.placeholder(tf.float32, shape=[None, n_in])
y_ = tf.placeholder(tf.float32, shape=[None, n_out])
lr = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.truncated_normal(shape=[n_in, n_hid]))
w2 = tf.Variable(tf.truncated_normal(shape=[n_hid, n_out]))
b1 = tf.Variable(tf.ones(n_hid))
b2 = tf.Variable(tf.ones(n_out))

h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
y = tf.matmul(h1, w2) + b2

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


params = {
    1985: {
        "epoch_len": 1000, 
        "num_epochs": 10, 
        "learning_rate": 0.001
    },
    2003: {
        "epoch_len": 500, 
        "num_epochs": 5, 
        "learning_rate": 0.001
    },
    2010: {
        "epoch_len": 500, 
        "num_epochs": 10, 
        "learning_rate": 0.001
    },
    2000: {
        "epoch_len": 1000,
        "num_epochs": 10,
        "learning_rate": 0.0005
    }
}


sess = tf.Session()
sess.run(tf.global_variables_initializer())
    
for year in [1985, 2003, 2010]:
    for i in range(params[year]["num_epochs"]):
        for _ in range(params[year]["epoch_len"]):
            learning_rate = params[year]["learning_rate"]
            examp, label = get_batch(year)
                
            _ = sess.run([train_step], feed_dict={ x: examp, y_: label, lr: learning_rate })
        loss_val, acc = sess.run([loss, accuracy], feed_dict={x: training_data[year][0],
                                                              y_: training_data[year][1], lr: learning_rate})
        print ("Epoch: {}, Loss: {}, Accuracy: {}".format(i, loss_val, acc))
            
    loss_val, acc = sess.run([loss, accuracy], feed_dict={x: test[0], y_: test[1], lr: learning_rate})
    print("Test Results:\t Loss: {}, Accuracy:{}".format(loss_val, acc))

tournament_cities = ["", "", "", "", "Pittsburgh PN", "Detroit MI", "Dallas TX", "San Diego CA", "San Diego CA",
    "Dallas TX", "Detroit MI", "Pittsburgh PN", "Wichita KS", "Pittsburgh PN", "Detroit MI", "San Diego CA",
    "San Diego CA", "Detroit MI", "Pittsburgh PN", "Wichita KS", "Charlotte NC", "Nashville TN", "Dallas TX", 
    "Boise iD", "Boise iD", "Dallas TX", "Nashville TN", "Charlotte NC", "Nashville TN", "Charlotte NC", "Wichita KS", 
    "Boise iD", "Boise iD", "Wichita KS", "Charlotte NC", "Nashville TN", "Pittsburgh PN", "Detroit MI", "Dallas TX",
    "San Diego CA", "Wichita KS", "Pittsburgh PN", "Detroit MI", "San Diego CA", "Charlotte NC", "Nashville TN",
    "Dallas TX", "Boise iD", "Nashville TN", "Charlotte NC", "Wichita KS", "Boise iD", "Boston MA", "Boston MA", "Omaha NE",
    "Omaha NE", "Atlanta GA", "Atlanta GA", "Los Angelos CA", "Los Angelos CA", "Boston MA", "Omaha NE", "Atlanta GA",
    "Los Angelos CA", "San Antonio TX", "San Antonio TX", "San Antonio TX"]
tourney_slots = pd.read_csv(massey_path + "NCAATourneySlots.csv")
seeds = pd.read_csv(massey_path + "NCAATourneySeeds.csv")

year = 2018

tourney_slots = tourney_slots.loc[tourney_slots["Season"] == year]
tourney_slots["City"] = tournament_cities
seeds = seeds.loc[seeds["Season"] == year]

#Play in games
game1 = seeds.loc[seeds["Seed"] == "W11a"]
seeds.at[game1.index[0], "Seed"] = "W11"
game2 = seeds.loc[seeds["Seed"] == "W16b"]
seeds.at[game2.index[0], "Seed"] = "W16"
game3 = seeds.loc[seeds["Seed"] == "X11b"]
seeds.at[game3.index[0], "Seed"] = "X11"
game4 = seeds.loc[seeds["Seed"] == "Z16b"]
seeds.at[game4.index[0], "Seed"] = "Z16"

rounds = [1, 2, 3, 4, 5, 6]
slots = None
winners = {}
for r in rounds:
    slots = tourney_slots.loc[tourney_slots["Slot"].str.startswith("R{}".format(r))]
    print("ROUND {}".format(r))
    for _, row in slots.iterrows():
        slot = row["Slot"]
        s_seed = row["StrongSeed"]
        w_seed = row["WeakSeed"]
        city = row["City"]
        
        if slot not in winners:
            winners[slot] = {}
        
        if r == 1:
            s_team = seeds.loc[seeds["Seed"] == s_seed]
            w_team = seeds.loc[seeds["Seed"] == w_seed]
            s_team = s_team.iloc[0]["TeamID"]
            w_team = w_team.iloc[0]["TeamID"]

            s_name = team_names[s_team]
            w_name = team_names[w_team]

            s_seed_num = find_seed(year, s_team)
            w_seed_num = find_seed(year, w_team)

            s_coach = coach_experience(year, s_team)
            w_coach = coach_experience(year, w_team)

            s_rpi, s_pom = get_rankings(year, s_team)
            w_rpi, w_pom = get_rankings(year, w_team)

            s_distance = tournament_distance_matrix.loc[(tournament_distance_matrix["TeamID"] == s_team)]
            w_distance = tournament_distance_matrix.loc[(tournament_distance_matrix["TeamID"] == w_team)]
            s_distance = s_distance.iloc[0][city]
            w_distance = w_distance.iloc[0][city]

            s_last = last_ten(year, s_team)
            w_last = last_ten(year, w_team)

            s_reb, s_fga, s_fgpct, s_ftpct = get_stats(year, s_team)
            w_reb, w_fga, w_fgpct, w_ftpct = get_stats(year, w_team)

            game = np.array([[i for i in [r, s_seed_num, s_coach, s_distance, s_last, s_rpi, s_pom, s_reb, s_fga, s_fgpct, s_ftpct,
                                     w_seed_num, w_coach, w_distance, w_last, w_rpi, w_pom, w_reb, w_fga, w_fgpct, w_ftpct]
                             if i != -1]])
        
        else:
            s_name = winners[s_seed][0]
            w_name = winners[w_seed][0]
            s_stuff = winners[s_seed][1]
            w_stuff = winners[w_seed][1]
            
            game = np.array([[r] + s_stuff + w_stuff])

        print(s_name)
        print(w_name)
        
        result = sess.run([y], feed_dict={x: game})
        result = result[0][0]
        
        if result[0] > result[1]:
            print("\t", s_name)
            winners[slot] = (s_name, [s_seed_num, s_coach, s_distance, s_last, s_rpi, s_pom, s_reb, s_fga, s_fgpct, s_ftpct])
        else:
            winners[slot] = (w_name, [w_seed_num, w_coach, w_distance, w_last, w_rpi, w_pom, w_reb, w_fga, w_fgpct, w_ftpct])
            print("\t", w_name)
        print()
        

