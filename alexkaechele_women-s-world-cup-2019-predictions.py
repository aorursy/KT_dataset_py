# Load packages

import pandas as pd

import numpy as np

import pystan

import matplotlib.pyplot as plt

import random





# Load wc qualifying data

df = pd.read_csv("../input/womens_world_cup_data.csv")



# Load fifa rankings

dat = pd.read_csv("../input/womens_world_cup_rankings.csv")







## Formatting qualifying data

df['date'] = df['date'].str.replace('ÿ', ' ')

df['date'] = pd.to_datetime(df['date'])



# Filter to needed columns and rename

df = df.rename(columns={'Team_i'  : 'teami',

                        'score_i' : 'scorei',

                        'Team_j'  : 'teamj', 

                        'score_j' : 'scorej',

                        'home_i'  : 'homei',

                        'home_j'  : 'homej'})



# Create margin

df['margin'] = df.scorei - df.scorej







## Formatting ranking data

dat["team"] = dat["team"].str.lower()

dat = dat.loc[dat.team.isin(df.teami.append(df.teamj))]



# Making fifa ranking mean 0 sd .5

dat["ranking"] = (dat.rating - dat.rating.mean())/np.std(dat.rating)/2

# merge with df (need to add to final dataset)



# List of womens world cup teams and rankings

fifarnk = pd.DataFrame({"team" : ["france", "norway", "south korea", "nigeria", "germany", "spain", "china", "south africa", "australia", "brazil", "italy", "jamaica",

                                  "england", "japan", "scotland", "argentina", "canada", "netherlands", "new zealand", "cameroon", "united states", "sweden", "thailand", "chile"],

                         "ranking" : [3, 13, 14, 39, 2, 12, 15, 48, 6, 10, 16, 53, 4, 8, 20, 36, 5, 7, 19, 46, 1, 9, 29, 38] })









## Set up team id mapping

team_key = pd.DataFrame({"teamname" : df.teami.append(df.teamj).unique(),

                         "teamid"   : range(1, 1+len(df.teami.append(df.teamj).unique()))})



# Recoding ids in qualifying data

df = df.merge(team_key, left_on="teami" , right_on="teamname")

df = df.drop(columns=["teamname", "teami"])

df = df.rename(index = str, columns = {"teamid" : "teami"})



df = df.merge(team_key, left_on="teamj" , right_on="teamname")

df = df.drop(columns=["teamname", "teamj"])

df = df.rename(index = str, columns = {"teamid" : "teamj"})



# Recoding ids in ranking data

dat = dat.merge(team_key, left_on="team" , right_on="teamname")

dat = dat.drop(columns=["teamname"])







# Final dataset for modeling

names = ["N", "y", "h_i", "h_j", "team_i", "team_j", "N_g"]

values = [len(df.index), df.margin, df.homei, df.homej, df.teami, df.teamj, 79]



train = dict(zip(names, values))


model = """

data {

    int N;

    vector[N] y;

    int team_i[N];

    int team_j[N];

    int h_i[N];

    int h_j[N];

    int N_g;

}

parameters {

    vector[N_g] alpha_raw;

    vector[N_g] theta_raw;

    real eta;

    real<lower=0> tau_theta;

    real<lower=0> tau_alpha;

    real<lower=0> sigma;

}

transformed parameters {

    vector[N_g] alpha;

    vector[N_g] theta;

    alpha = eta + alpha_raw*tau_alpha;

    theta = theta_raw*tau_theta;

}

model {

    // vector for conditional mean storage

    vector[N] mu;



    // priors

    tau_theta ~ cauchy(0,1)T[0,];

    tau_alpha ~ cauchy(0,.25)T[0,];

    sigma ~ cauchy(0,1)T[0,];

    eta ~ normal(.5,.25);

    theta_raw ~ normal(0,1);

    alpha_raw ~ normal(0,1);



    // define mu for the Gaussian

    for( t in 1:N ) {

    mu[t] = (theta[team_i[t]] + alpha[team_i[t]]*h_i[t]) - 

    (theta[team_j[t]] + alpha[team_j[t]]*h_j[t]);

}



    // the likelihood

    y ~ normal(mu,sigma);

}

"""



sm = pystan.StanModel(model_code = model)

fit = sm.sampling(data = train, 

                  iter = 1500, 

                  warmup = 750,

                  refresh = 100,

                  control = dict(adapt_delta = 0.9))
# Extracting team skill levels

th = pd.DataFrame(fit.extract()["theta"])

a = pd.DataFrame(fit.extract()["alpha"])

sig = fit.extract()["sigma"]

a.columns = team_key.teamname

th.columns = team_key.teamname



# Filtering to top 25 teams

theta25 = th[th.median().nlargest(25).index]

theta25 = theta25[theta25.columns[::-1]]



# Creating the plot

theta25.boxplot(grid = False, vert = False, showfliers = False, figsize=(12, 8))

plt.title('Team Power Rankings')

plt.xlabel('Skill Level')

plt.ylabel('Teams')
def compare(i, j, homei = 0, homej = 0, th= th, a = a, sig = sig, reps = 1000, allowdraw = False):

    if allowdraw:

        win_prob = []

        draw_prob = []

        

        # Simulating specified number of games

        for r in range(1, reps):

            win_prob.append(

                np.mean(

                    

                    # Ability difference

                    th[i].sample(1).values - th[j].sample(1).values + 

                    

                    # Home field advantage

                    a[i].sample(1).values*homei - a[j].sample(1).values*homej

                    ) > 0

                )

         

        # Averaging game results

        win_prob = np.mean(win_prob)

        

        # Displaying results

        print(i + " has a " + str(round(win_prob*100, 2)) + "% chance of beating " + j)

    

    else:

        win_prob = []

        

        # Simulating specified number of games

        for r in range(1, reps):

            win_prob.append(

                np.mean(

                    

                    # Ability difference

                    th[i].sample(1).values - th[j].sample(1).values + 

                    

                    # Home field advantage

                    a[i].sample(1).values*homei - a[j].sample(1).values*homej

                    ) > 0

                )

         

        # Averaging game results

        win_prob = np.mean(win_prob)

        

        # Displaying results

        print(i + " has a " + str(round(win_prob*100, 2)) + "% chance of beating " + j)

    

    

def short_compare(i, j, homei, homej, th= th, a = a, sig = sig, allowdraw = True):

    

    gamescore = th[i].sample(1).values - th[j].sample(1).values + a[i].sample(1).values*homei - a[j].sample(1).values*homej

    

    if(allowdraw):

        if(abs(gamescore) < 1):

            return([1, 1])

        if(gamescore > 0):

            return([3, 0])        

        else:

            return([0, 3])

    

    else:

        if(gamescore > 0):

            return([3, 0])

        else:

            return([0, 3])



def group_sim_short(i, j, k, l, homei, homej, homek, homel, th= th, a = a, sig = sig):



    # Initial values

    score_i, score_j, score_k, score_l = 0, 0, 0, 0

    

    # Round robin games

    score_i, score_j = [score_i, score_j] + np.array(short_compare(i, j, homei, homej))

    score_i, score_k = [score_i, score_k] + np.array(short_compare(i, k, homei, homek))

    score_i, score_l = [score_i, score_l] + np.array(short_compare(i, l, homei, homel))

    score_j, score_k = [score_j, score_k] + np.array(short_compare(j, k, homej, homek))

    score_j, score_l = [score_j, score_l] + np.array(short_compare(j, l, homej, homel))

    score_k, score_l = [score_k, score_l] + np.array(short_compare(k, l, homek, homel))

    

    # Determining winners

    d = {'teams': [i, j, k, l], 'points': [score_i, score_j, score_k, score_l]}

    df = pd.DataFrame(data=d)

    df.sort_values(by=["points"], inplace=True, ascending=False)

    df.index = np.arange(1, len(df) + 1)

    df["ranking"] = df.index    

    

    return(df)



def group_sim(i, j, k, l, homei, homej, homek, homel, th= th, a = a, sig = sig, reps = 1000):



    df = pd.DataFrame()

    

    for r in range(1, reps):

        

        # Simulate games

        df_temp = group_sim_short(i, j, k, l, homei, homej, homek, homel)

        df = df.append(df_temp)

        

    # Average points in season

    dat = pd.DataFrame(df.groupby("teams").points.mean()).reset_index()

    

    # Count of Rankings

    df["first"] = (df.ranking == 1)

    df["second"] = (df.ranking == 2)

    df["third"] = (df.ranking == 3)

    df["fourth"] = (df.ranking == 4)

    

    # Percents

    dat = dat.merge(pd.DataFrame(df.groupby('teams')["first"].sum()/reps*100).reset_index(), on="teams")

    dat = dat.merge(pd.DataFrame(df.groupby('teams')["second"].sum()/reps*100).reset_index(), on="teams")

    dat = dat.merge(pd.DataFrame(df.groupby('teams')["third"].sum()/reps*100).reset_index(), on="teams")

    dat = dat.merge(pd.DataFrame(df.groupby('teams')["fourth"].sum()/reps*100).reset_index(), on="teams")

    

    # Sort by points

    dat = dat.sort_values(by=["points"], ascending=False)

    dat.index = np.arange(1, len(dat)+1)



    return(dat)



def choose_matchup(inputs):

    

    inputs1 = inputs.group.sort_values().tolist()

    

    if inputs1 == ["A", "B", "C", "D"]:

        output = [inputs[inputs.group == "B"]["team"].values[0], inputs[inputs.group == "C"]["team"].values[0], inputs[inputs.group == "A"]["team"].values[0], inputs[inputs.group == "D"]["team"].values[0]]

    elif inputs1 == ["A", "B", "C", "E"]:

        output = [inputs[inputs.group == "E"]["team"].values[0], inputs[inputs.group == "C"]["team"].values[0], inputs[inputs.group == "B"]["team"].values[0], inputs[inputs.group == "A"]["team"].values[0]]

    elif inputs1 == ["A", "B", "C", "F"]:

        output = [inputs[inputs.group == "F"]["team"].values[0], inputs[inputs.group == "C"]["team"].values[0], inputs[inputs.group == "B"]["team"].values[0], inputs[inputs.group == "A"]["team"].values[0]]

    elif inputs1 == ["A", "B", "D", "E"]:

        output = [inputs[inputs.group == "E"]["team"].values[0], inputs[inputs.group == "D"]["team"].values[0], inputs[inputs.group == "B"]["team"].values[0], inputs[inputs.group == "A"]["team"].values[0]]

    elif inputs1 == ["A", "B", "D", "F"]:

        output = [inputs[inputs.group == "F"]["team"].values[0], inputs[inputs.group == "D"]["team"].values[0], inputs[inputs.group == "B"]["team"].values[0], inputs[inputs.group == "A"]["team"].values[0]]

    elif inputs1 == ["A", "B", "E", "F"]:

        output = [inputs[inputs.group == "F"]["team"].values[0], inputs[inputs.group == "E"]["team"].values[0], inputs[inputs.group == "B"]["team"].values[0], inputs[inputs.group == "A"]["team"].values[0]]

    elif inputs1 == ["A", "C", "D", "E"]:

        output = [inputs[inputs.group == "E"]["team"].values[0], inputs[inputs.group == "C"]["team"].values[0], inputs[inputs.group == "A"]["team"].values[0], inputs[inputs.group == "D"]["team"].values[0]]

    elif inputs1 == ["A", "C", "D", "F"]:

        output = [inputs[inputs.group == "F"]["team"].values[0], inputs[inputs.group == "C"]["team"].values[0], inputs[inputs.group == "A"]["team"].values[0], inputs[inputs.group == "D"]["team"].values[0]]

    elif inputs1 == ["A", "C", "E", "F"]:

        output = [inputs[inputs.group == "E"]["team"].values[0], inputs[inputs.group == "C"]["team"].values[0], inputs[inputs.group == "F"]["team"].values[0], inputs[inputs.group == "A"]["team"].values[0]]

    elif inputs1 == ["A", "D", "E", "F"]:

        output = [inputs[inputs.group == "E"]["team"].values[0], inputs[inputs.group == "D"]["team"].values[0], inputs[inputs.group == "F"]["team"].values[0], inputs[inputs.group == "A"]["team"].values[0]]

    elif inputs1 == ["B", "C", "D", "E"]:

        output = [inputs[inputs.group == "E"]["team"].values[0], inputs[inputs.group == "C"]["team"].values[0], inputs[inputs.group == "B"]["team"].values[0], inputs[inputs.group == "D"]["team"].values[0]]

    elif inputs1 == ["B", "C", "D", "F"]:

        output = [inputs[inputs.group == "F"]["team"].values[0], inputs[inputs.group == "C"]["team"].values[0], inputs[inputs.group == "B"]["team"].values[0], inputs[inputs.group == "D"]["team"].values[0]]

    elif inputs1 == ["B", "C", "E", "F"]:

        output = [inputs[inputs.group == "F"]["team"].values[0], inputs[inputs.group == "E"]["team"].values[0], inputs[inputs.group == "B"]["team"].values[0], inputs[inputs.group == "C"]["team"].values[0]]

    elif inputs1 == ["B", "D", "E", "F"]:

        output = [inputs[inputs.group == "F"]["team"].values[0], inputs[inputs.group == "E"]["team"].values[0], inputs[inputs.group == "B"]["team"].values[0], inputs[inputs.group == "D"]["team"].values[0]]

    else:

        output = [inputs[inputs.group == "E"]["team"].values[0], inputs[inputs.group == "C"]["team"].values[0], inputs[inputs.group == "D"]["team"].values[0], inputs[inputs.group == "F"]["team"].values[0]]

    

    return(output)
def tourn_sim(group, fifarnk = fifarnk, th= th, a = a, sig = sig, reps = 1000):

    # setting up final results

    results = pd.DataFrame({"team" : fifarnk.team, 

                            "grp4" : [0]*24, 

                            "grp3" : [0]*24, 

                            "rd16" : [0]*24, 

                            "quarters" : [0]*24, 

                            "semis" : [0]*24, 

                            "fourth" : [0]*24, 

                            "third" : [0]*24, 

                            "second" : [0]*24, 

                            "first" : [0]*24})

    

    for reps in range(0, reps):

                

        # Setting up group results

        grp = pd.DataFrame({"ranking" : [1,2,3,4]})

            

            

        # Simulate groups

        for ind in range(0, len(group.index)):

            grp_temp = group_sim_short(group.team1[ind],   group.team2[ind],   group.team3[ind],   group.team4[ind], 

                                       group.team1_h[ind], group.team2_h[ind], group.team3_h[ind], group.team4_h[ind])

            

            grp = grp.merge(grp_temp, on = "ranking", suffixes = ["", group.group[ind]])

        grp = pd.DataFrame(grp)

        

            

        # Recording 4th place teams

        done_teams = [grp.teams[3], grp.teamsB[3], grp.teamsC[3], grp.teamsD[3], grp.teamsE[3], grp.teamsF[3]]

        results.loc[results.team.isin(done_teams), "grp4"] += 1

        

        

        # Determing 3rd place advancers

        tie = grp[grp.ranking == 3]

        tie = pd.DataFrame({"team"   : [tie.teams[2], tie.teamsB[2], tie.teamsC[2], tie.teamsD[2], tie.teamsE[2], tie.teamsF[2]],

                            "group"  : ["A", "B", "C", "D", "E", "F"],

                            "points" : [tie.points[2], tie.pointsB[2], tie.pointsC[2], tie.pointsD[2], tie.pointsE[2], tie.pointsF[2]],

                            "fifa"   : [-fifarnk[fifarnk.team.isin(tie.teams)].ranking.values[0],

                                        -fifarnk[fifarnk.team.isin(tie.teamsB)].ranking.values[0],

                                        -fifarnk[fifarnk.team.isin(tie.teamsC)].ranking.values[0],

                                        -fifarnk[fifarnk.team.isin(tie.teamsD)].ranking.values[0],

                                        -fifarnk[fifarnk.team.isin(tie.teamsE)].ranking.values[0],

                                        -fifarnk[fifarnk.team.isin(tie.teamsF)].ranking.values[0]]})

    

        # Selecting 3rd place teams 

        tie = tie.sort_values(by=["points", "fifa"], ascending=False)

        tie.index = range(1,7)

        

        # Recording 3rd place, non-advancing teams

        done_teams = [tie.team[5], tie.team[6]]

        results.loc[results.team.isin(done_teams), "grp3"] += 1

        

        # Determining matchups of advanceing, 3rd place teams

        thirdplace = tie.iloc[0:4][["team", "group"]]

        thirdplace = choose_matchup(inputs=thirdplace)

    

        # Setting up round of 16

        rd16 = pd.DataFrame({"team1" : [grp.teams[1],  grp.teamsD[0], grp.teams[0], grp.teamsB[1], grp.teamsC[0], grp.teamsE[0], grp.teamsB[0], grp.teamsF[1]],

                             "team2" : [grp.teamsC[1], thirdplace[0], thirdplace[1], grp.teamsF[0], thirdplace[2], grp.teamsD[1], thirdplace[3], grp.teamsE[1]]})

        

        # Recording those who made rd of 16

        results.loc[results.team.isin(rd16.team1.append(rd16.team2)), "rd16"] += 1

        

        # Preparing round 16 playing

        winners = []

        losers = []

        

        # Playing rd 16

        for ind in range(0, 8):

            i  = rd16.team1[ind]

            j  = rd16.team2[ind]

            homei = (i=="france")

            homej = (j=="france")

            team1, team2 = short_compare(i, j, homei, homej, th= th, a = a, sig = sig, allowdraw = False)

            if (team1 == 3):

                winners.append(i)

            elif (team2 == 3):    

                winners.append(j)

        

        # Recording those who made quarters

        results.loc[results.team.isin(winners), "quarters"] += 1

        

        

        # Setting up quarters

        quarters = pd.DataFrame({"team1" : [winners[0], winners[2], winners[4], winners[6]],

                                 "team2" : [winners[1], winners[3], winners[5], winners[7]]})

        

        # Preparing quarters playing

        winners = []

        

        # Playing quarters

        for ind in range(0, 4):

            i  = quarters.team1[ind]

            j  = quarters.team2[ind]

            homei = (i=="france")

            homej = (j=="france")

            team1, team2 = short_compare(i, j, homei, homej, th= th, a = a, sig = sig, allowdraw = False)

            if (team1 == 3):

                winners.append(i)

            elif (team2 == 3):    

                winners.append(j)

        

        # Recording those who made semis

        results.loc[results.team.isin(winners), "semis"] += 1

    

        # Setting up semis

        semis = pd.DataFrame({"team1" : [winners[0], winners[2]],

                              "team2" : [winners[1], winners[3]]})

        

        # Preparing semis playing

        winners = []

        losers = []

        

        # Playing semis

        for ind in range(0, 2):

            i  = semis.team1[ind]

            j  = semis.team2[ind]

            homei = (i=="france")

            homej = (j=="france")

            team1, team2 = short_compare(i, j, homei, homej, th= th, a = a, sig = sig, allowdraw = False)

            if (team1 == 3):

                winners.append(i)

                losers.append(j)

            elif (team2 == 3):    

                winners.append(j)

                losers.append(i)

        

        # Setting up finals

        finals = pd.DataFrame({"team1" : [winners[0], losers[0]],

                               "team2" : [winners[1], losers[1]]})

        

        # Preparing finals playing

        winners = []

        losers = []

        

        # Playing finals

        for ind in range(0, 2):

            i  = finals.team1[ind]

            j  = finals.team2[ind]

            homei = (i=="france")

            homej = (j=="france")

            team1, team2 = short_compare(i, j, homei, homej, th= th, a = a, sig = sig, allowdraw = False)

            if (team1 == 3):

                winners.append(i)

                losers.append(j)

            elif (team2 == 3):    

                winners.append(j)

                losers.append(i)

                

        # Recording finals results

        results.loc[results.team == winners[0], "first"] += 1

        results.loc[results.team == losers[0], "second"] += 1

        results.loc[results.team == winners[1], "third"] += 1

        results.loc[results.team == losers[1], "fourth"] += 1

    

    results[results.select_dtypes(include=['number']).columns] /= (reps+1)

    results[results.select_dtypes(include=['number']).columns] *= 100

    

    results.index = ["A"]*4 + ["B"]*4 + ["C"]*4 + ["D"]*4 + ["E"]*4 + ["F"]*4

    results = results.sort_values(by=["first", "second", "third", "fourth", "semis", "quarters", "rd16"], ascending=False)

    

    results = results.round(2)

    

    return(results)
groupA = group_sim("france", "norway", "south korea", "nigeria", homei=1, homej=0, homek=0, homel=0)

groupA
groupB = group_sim("germany", "spain", "china", "south africa", homei=0, homej=0, homek=0, homel=0)

groupB
groupC = group_sim("australia", "brazil", "italy", "jamaica", homei=0, homej=0, homek=0, homel=0)

groupC
groupD = group_sim("england", "japan", "scotland", "argentina", homei=0, homej=0, homek=0, homel=0)

groupD
groupE = group_sim("canada", "netherlands", "new zealand", "cameroon", homei=0, homej=0, homek=0, homel=0)

groupE
groupF = group_sim("united states", "sweden", "thailand", "chile", homei=0, homej=0, homek=0, homel=0)

groupF
# Group input

group = pd.DataFrame({'group': ["A", "B", "C", "D", "E", "F"], 

                      'team1': ["france", "germany", "australia", "england", "canada", "united states"],

                      'team2': ["norway", "spain", "brazil", "japan", "netherlands", "sweden"],

                      'team3': ["south korea", "china", "italy", "scotland", "new zealand", "thailand"],

                      'team4': ["nigeria", "south africa", "jamaica", "argentina", "cameroon", "chile"],

                      'team1_h' : [1, 0, 0, 0, 0, 0],

                      'team2_h' : [0, 0, 0, 0, 0, 0],

                      'team3_h' : [0, 0, 0, 0, 0, 0],

                      'team4_h' : [0, 0, 0, 0, 0, 0],

                      'team1_r' : [3, 2, 6, 4, 5, 1],

                      'team2_r' : [13, 12, 10, 8, 7, 9],

                      'team3_r' : [14, 15, 16, 20, 19, 29],

                      'team4_r' : [39, 48, 53, 36, 46, 38]

                    })



# Simulating tournament

tourn_results = tourn_sim(group)

tourn_results["finals"] = tourn_results["first"] + tourn_results["second"]

tourn_results
nikki = pd.DataFrame({"team"          : ["norway", "france", "nigeria", "south korea", "spain", "germany", "south africa", "china", "brazil", "italy", "australia", "jamaica", "england", "argentina", "scotland", "japan", "netherlands", "cameroon", "canada", "new zealand", "united states", "sweden", "chile", "thailand"],

                      "group_place"   : [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],

                      "advance"       : [True, True, False, False, True, True, True, False, True, True, True, False, True, True, True, False, True, True, False, False, True, True, True, False],

                      "make_quarters" : [True, True, False, False, True, False, False, False, True, False, False, False, True, True, False, False, False, False, False, False, True, True, False, False],

                      "make_semis"    : [False, False, False, False, True, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, False, False, False],

                      "make_finals"   : [False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False],

                      "win_it"        : [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False],

                      "third"         : [False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]})



nikki.index = ["A"]*4 + ["B"]*4 + ["C"]*4 + ["D"]*4 + ["E"]*4 + ["F"]*4







nikki["odds_of_place"] = [float(groupA.loc[groupA.teams == "norway",        "first"]) /100,

                          float(groupA.loc[groupA.teams == "france",        "second"])/100,

                          float(groupA.loc[groupA.teams == "nigeria",       "third"]) /100,

                          float(groupA.loc[groupA.teams == "south korea",   "fourth"])/100,

                          float(groupB.loc[groupB.teams == "spain",         "first"]) /100,

                          float(groupB.loc[groupB.teams == "germany",       "second"])/100,

                          float(groupB.loc[groupB.teams == "south africa",  "third"]) /100,

                          float(groupB.loc[groupB.teams == "china",         "fourth"])/100,

                          float(groupC.loc[groupC.teams == "brazil",        "first"]) /100,

                          float(groupC.loc[groupC.teams == "italy",         "second"])/100,

                          float(groupC.loc[groupC.teams == "australia",     "third"]) /100,

                          float(groupC.loc[groupC.teams == "jamaica",       "fourth"])/100,

                          float(groupD.loc[groupD.teams == "england",       "first"]) /100,

                          float(groupD.loc[groupD.teams == "argentina",     "second"])/100,

                          float(groupD.loc[groupD.teams == "scotland",      "third"]) /100,

                          float(groupD.loc[groupD.teams == "japan",         "fourth"])/100,

                          float(groupE.loc[groupE.teams == "netherlands",   "first"]) /100,

                          float(groupE.loc[groupE.teams == "cameroon",      "second"])/100,

                          float(groupE.loc[groupE.teams == "canada",        "third"]) /100,

                          float(groupE.loc[groupE.teams == "new zealand",   "fourth"])/100,

                          float(groupF.loc[groupF.teams == "united states", "first"]) /100,

                          float(groupF.loc[groupF.teams == "sweden",        "second"])/100,

                          float(groupF.loc[groupF.teams == "chile",         "third"]) /100,

                          float(groupF.loc[groupF.teams == "thailand",      "fourth"])/100]
# Perfect Group Prediction

group_perfect = pd.DataFrame(nikki["odds_of_place"].groupby(level=0).prod()*100).rename({"odds_of_place" : "chance of perfect group"}, axis='columns')

n_perf_group = 3*group_perfect["chance of perfect group"].sum()/100

print("Expected perfect group points: " + str(n_perf_group))

group_perfect
# Predict First Place

group_first = pd.DataFrame(nikki[nikki.group_place == 1]["odds_of_place"]).rename({"odds_of_place" : "chance of first"}, axis='columns')

n_first_place = 2*group_first["chance of first"].sum()

print("Expected first place points: " + str(n_first_place))

group_first*100
# Odds of Advancing

advance_group = tourn_results[tourn_results.team.isin(nikki[nikki.advance == True].team)][["team", "rd16"]]

n_advance_group = advance_group.rd16.sum()/100

print("Expected points from picking advancing team: " + str(n_advance_group))

advance_group
# Odds of Quarters

quarters_odds = tourn_results[tourn_results.team.isin(nikki[nikki.make_quarters == True].team)][["team", "quarters"]]

n_quarters = 2*quarters_odds.quarters.sum()/100

print("Expected points from picking quarters team: " + str(n_quarters))

quarters_odds
# Odds of Semis

semis_odds = tourn_results[tourn_results.team.isin(nikki[nikki.make_semis == True].team)][["team", "semis"]]

n_semis = 4*semis_odds.semis.sum()/100

print("Expected points from picking semis team: " + str(n_semis))

semis_odds
# Odds of Finals

finals_odds = tourn_results[tourn_results.team.isin(nikki[nikki.make_finals == True].team)][["team", "finals"]]

n_finals = 8*finals_odds.finals.sum()/100

print("Expected points from picking finals team: " + str(n_finals))

finals_odds
# Odds of Third

third_odds = tourn_results[tourn_results.team.isin(nikki[nikki.third == True].team)][["team", "third"]]

n_thirds = 8*third_odds.third.sum()/100

print("Expected points from picking third place team: " + str(n_thirds))

third_odds
# Odds of Winning

winner_odds = tourn_results[tourn_results.team.isin(nikki[nikki.win_it == True].team)][["team", "first"]]

n_winners_odds = 16*float(winner_odds["first"])/100

print("Expected points from picking winner: " + str(n_winners_odds))

winner_odds
jerry = pd.DataFrame({"team"          : ["france", "south korea", "nigeria", "norway", "germany", "spain", "china", "south africa", "brazil", "italy", "jamaica", "australia", "japan", "england", "scotland", "argentina", "netherlands", "canada", "new zealand", "cameroon", "united states", "sweden", "chile", "thailand"],

                      "group_place"   : [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],

                      "advance"       : [True, True, True, False, True, True, True, False, True, True, True, False, True, True, False, False, True, True, True, False, True, True, False, False],

                      "make_quarters" : [True, True, False, False, True, False, False, False, True, False, False, False, True, True, False, False, False, False, False, False, True, True, False, False],

                      "make_semis"    : [False, False, False, False, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, True, True, False, False],

                      "make_finals"   : [False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False],

                      "win_it"        : [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False],

                      "third"         : [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False]})



jerry.index = ["A"]*4 + ["B"]*4 + ["C"]*4 + ["D"]*4 + ["E"]*4 + ["F"]*4



jerry["odds_of_place"] = [float(groupA.loc[groupA.teams == "france",        "first"]) /100,

                          float(groupA.loc[groupA.teams == "south korea",   "second"])/100,

                          float(groupA.loc[groupA.teams == "nigeria",       "third"]) /100,

                          float(groupA.loc[groupA.teams == "norway",        "fourth"])/100,

                          float(groupB.loc[groupB.teams == "germany",       "first"]) /100,

                          float(groupB.loc[groupB.teams == "spain",         "second"])/100,

                          float(groupB.loc[groupB.teams == "china",         "third"]) /100,

                          float(groupB.loc[groupB.teams == "south africa",  "fourth"])/100,

                          float(groupC.loc[groupC.teams == "brazil",        "first"]) /100,

                          float(groupC.loc[groupC.teams == "italy",         "second"])/100,

                          float(groupC.loc[groupC.teams == "jamaica",       "third"]) /100,

                          float(groupC.loc[groupC.teams == "australia",     "fourth"])/100,

                          float(groupD.loc[groupD.teams == "japan",         "first"]) /100,

                          float(groupD.loc[groupD.teams == "england",       "second"])/100,

                          float(groupD.loc[groupD.teams == "scotland",      "third"]) /100,

                          float(groupD.loc[groupD.teams == "argentina",     "fourth"])/100,

                          float(groupE.loc[groupE.teams == "netherlands",   "first"]) /100,

                          float(groupE.loc[groupE.teams == "canada",        "second"])/100,

                          float(groupE.loc[groupE.teams == "new zealand",   "third"]) /100,

                          float(groupE.loc[groupE.teams == "cameroon",      "fourth"])/100,

                          float(groupF.loc[groupF.teams == "united states", "first"]) /100,

                          float(groupF.loc[groupF.teams == "sweden",        "second"])/100,

                          float(groupF.loc[groupF.teams == "chile",         "third"]) /100,

                          float(groupF.loc[groupF.teams == "thailand",      "fourth"])/100]
# Perfect Group Prediction

group_perfect = pd.DataFrame(jerry["odds_of_place"].groupby(level=0).prod()*100).rename({"odds_of_place" : "chance of perfect group"}, axis='columns')

ja_group_perfect = 3*group_perfect["chance of perfect group"].sum()/100

print("Expected perfect group points: " + str(ja_group_perfect))

group_perfect
# Predict First Place

group_first = pd.DataFrame(jerry[jerry.group_place == 1]["odds_of_place"]).rename({"odds_of_place" : "chance of first"}, axis='columns')

ja_first = 2*group_first["chance of first"].sum()

print("Expected first place points: " + str(ja_first))

group_first*100
# Odds of Advancing

advance_group = tourn_results[tourn_results.team.isin(jerry[jerry.advance == True].team)][["team", "rd16"]]

ja_advance_16 = advance_group.rd16.sum()/100

print("Expected points from picking advancing team: " + str(ja_advance_16))

advance_group
# Odds of Quarters

quarters_odds = tourn_results[tourn_results.team.isin(jerry[jerry.make_quarters == True].team)][["team", "quarters"]]

ja_quarters = 2*quarters_odds.quarters.sum()/100

print("Expected points from picking quarters team: " + str(ja_quarters))

quarters_odds
# Odds of Semis

semis_odds = tourn_results[tourn_results.team.isin(jerry[jerry.make_semis == True].team)][["team", "semis"]]

ja_semis = 4*semis_odds.semis.sum()/100

print("Expected points from picking semis team: " + str(ja_semis))

semis_odds
# Odds of Finals

finals_odds = tourn_results[tourn_results.team.isin(jerry[jerry.make_finals == True].team)][["team", "finals"]]

ja_finals = 8*finals_odds.finals.sum()/100

print("Expected points from picking finals team: " + str(ja_finals))

finals_odds
# Odds of Third

third_odds = tourn_results[tourn_results.team.isin(jerry[jerry.third == True].team)][["team", "third"]]

ja_thirds = 8*third_odds.third.sum()/100

print("Expected points from picking third place team: " + str(ja_thirds))

third_odds
# Odds of Winning

winner_odds = tourn_results[tourn_results.team.isin(jerry[jerry.win_it == True].team)][["team", "first"]]

ja_champs = 16*float(winner_odds["first"])/100

print("Expected points from picking winner: " + str())

winner_odds
jerrod = pd.DataFrame({"team"          : ["france", "norway", "nigeria", "south korea", "germany", "spain", "south africa", "china", "brazil", "italy", "jamaica", "australia", "argentina", "scotland", "japan", "england", "netherlands", "new zealand", "cameroon", "canada", "chile", "united states", "sweden", "thailand"],

                      "group_place"   : [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],

                      "advance"       : [True, True, True, False, True, True, True, False, True, True, False, False,True, True, True, False, True, True, False, False, True, True, True, False],

                      "make_quarters" : [False, False, False, False, True, True, False, False,True, True, False, False, True, False, True, False, True, False, False, False, False, True, False, False],

                      "make_semis"    : [False, False, False, False, False, False, False, False,True, False, False, False, True, False, True, False, False, False, False, False, False, True, False, False],

                      "make_finals"   : [False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, True, False, False],

                      "win_it"        : [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False],

                      "third"         : [False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False]})



jerrod.index = ["A"]*4 + ["B"]*4 + ["C"]*4 + ["D"]*4 + ["E"]*4 + ["F"]*4



jerrod["odds_of_place"] = [float(groupA.loc[groupA.teams == "france",       "first"]) /100,

                          float(groupA.loc[groupA.teams == "norway",        "second"])/100,

                          float(groupA.loc[groupA.teams == "nigeria",       "third"]) /100,

                          float(groupA.loc[groupA.teams == "south korea",   "fourth"])/100,

                          float(groupB.loc[groupB.teams == "germany",       "first"]) /100,

                          float(groupB.loc[groupB.teams == "spain",         "second"])/100,

                          float(groupB.loc[groupB.teams == "south africa",  "third"]) /100,

                          float(groupB.loc[groupB.teams == "china",         "fourth"])/100,

                          float(groupC.loc[groupC.teams == "brazil",        "first"]) /100,

                          float(groupC.loc[groupC.teams == "italy",         "second"])/100,

                          float(groupC.loc[groupC.teams == "jamaica",       "third"]) /100,

                          float(groupC.loc[groupC.teams == "australia",     "fourth"])/100,

                          float(groupD.loc[groupD.teams == "argentina",     "first"]) /100,

                          float(groupD.loc[groupD.teams == "scotland",      "second"])/100,

                          float(groupD.loc[groupD.teams == "japan",         "third"]) /100,

                          float(groupD.loc[groupD.teams == "england",       "fourth"])/100,

                          float(groupE.loc[groupE.teams == "netherlands",   "first"]) /100,

                          float(groupE.loc[groupE.teams == "new zealand",   "second"])/100,

                          float(groupE.loc[groupE.teams == "cameroon",      "third"]) /100,

                          float(groupE.loc[groupE.teams == "canada",        "fourth"])/100,

                          float(groupF.loc[groupF.teams == "chile",         "first"]) /100,

                          float(groupF.loc[groupF.teams == "united states", "second"])/100,

                          float(groupF.loc[groupF.teams == "sweden",        "third"]) /100,

                          float(groupF.loc[groupF.teams == "thailand",      "fourth"])/100]
# Perfect Group Prediction

group_perfect = pd.DataFrame(jerrod["odds_of_place"].groupby(level=0).prod()*100).rename({"odds_of_place" : "chance of perfect group"}, axis='columns')

jc_group_perf = 3*group_perfect["chance of perfect group"].sum()/100

print("Expected perfect group points: " + str(jc_group_perf))

group_perfect
# Predict First Place

group_first = pd.DataFrame(jerrod[jerrod.group_place == 1]["odds_of_place"]).rename({"odds_of_place" : "chance of first"}, axis='columns')

jc_first = 2*group_first["chance of first"].sum()

print("Expected first place points: " + str(jc_first))

group_first*100
# Odds of Advancing

advance_group = tourn_results[tourn_results.team.isin(jerrod[jerrod.advance == True].team)][["team", "rd16"]]

jc_advance_16 = advance_group.rd16.sum()/100

print("Expected points from picking advancing team: " + str(jc_advance_16))

advance_group
# Odds of Quarters

quarters_odds = tourn_results[tourn_results.team.isin(jerrod[jerrod.make_quarters == True].team)][["team", "quarters"]]

jc_quarters = 2*quarters_odds.quarters.sum()/100

print("Expected points from picking quarters team: " + str(jc_quarters))

quarters_odds
# Odds of Semis

semis_odds = tourn_results[tourn_results.team.isin(jerrod[jerrod.make_semis == True].team)][["team", "semis"]]

jc_semis = 4*semis_odds.semis.sum()/100

print("Expected points from picking semis team: " + str(jc_semis))

semis_odds
# Odds of Finals

finals_odds = tourn_results[tourn_results.team.isin(jerrod[jerrod.make_finals == True].team)][["team", "finals"]]

jc_finals = 8*finals_odds.finals.sum()/100

print("Expected points from picking finals team: " + str(jc_finals))

finals_odds
# Odds of Third

third_odds = tourn_results[tourn_results.team.isin(jerrod[jerrod.third == True].team)][["team", "third"]]

jc_third = 8*third_odds.third.sum()/100

print("Expected points from picking third place team: " + str(jc_third))

third_odds
# Odds of Winning

winner_odds = tourn_results[tourn_results.team.isin(jerrod[jerrod.win_it == True].team)][["team", "first"]]

jc_champs = 16*float(winner_odds["first"])/100

print("Expected points from picking winner: " + str(jc_champs))

winner_odds
alex = pd.DataFrame({"team"          : ["france", "norway", "south korea", "nigeria", 

                                        "germany", "spain", "china", "south africa", 

                                        "australia", "brazil", "italy", "jamaica", 

                                        "england", "japan", "scotland", "argentina", 

                                        "canada", "netherlands", "new zealand", "cameroon", 

                                        "united states", "sweden", "chile", "thailand"],

                      "group_place"   : [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],

                      "advance"       : [True, True, True, False,True, True, True, False,True, True, True, False,True, True, False, False,True, True, True, False,True, True, False, False],

                      "make_quarters" : [True, False, False, False,True, False, False, False,True, True, False, False,True, False, False, False,True, True, False, False,True, False, False, False],

                      "make_semis"    : [False, False, False, False,True, False, False, False,False, False, False, False,True, False, False, False,True, False, False, False,True, False, False, False],

                      "make_finals"   : [False, False, False, False,True, False, False, False,False, False, False, False,False, False, False, False,False, False, False, False,True, False, False, False],

                      "win_it"        : [False, False, False, False,True, False, False, False,False, False, False, False,False, False, False, False,False, False, False, False,False, False, False, False],

                      "third"         : [False, False, False, False,False, False, False, False,False, False, False, False,True, False, False, False,False, False, False, False,False, False, False, False]})



alex.index = ["A"]*4 + ["B"]*4 + ["C"]*4 + ["D"]*4 + ["E"]*4 + ["F"]*4



alex["odds_of_place"] = [float(groupA.loc[groupA.teams == "france",         "first"]) /100,

                          float(groupA.loc[groupA.teams == "norway",        "second"])/100,

                          float(groupA.loc[groupA.teams == "south korea",   "third"]) /100,

                          float(groupA.loc[groupA.teams == "nigeria",       "fourth"])/100,

                          float(groupB.loc[groupB.teams == "germany",       "first"]) /100,

                          float(groupB.loc[groupB.teams == "spain",         "second"])/100,

                          float(groupB.loc[groupB.teams == "china",         "third"]) /100,

                          float(groupB.loc[groupB.teams == "south africa",  "fourth"])/100,

                          float(groupC.loc[groupC.teams == "australia",     "first"]) /100,

                          float(groupC.loc[groupC.teams == "brazil",        "second"])/100,

                          float(groupC.loc[groupC.teams == "italy",         "third"]) /100,

                          float(groupC.loc[groupC.teams == "jamaica",       "fourth"])/100,

                          float(groupD.loc[groupD.teams == "england",       "first"]) /100,

                          float(groupD.loc[groupD.teams == "japan",         "second"])/100,

                          float(groupD.loc[groupD.teams == "scotland",      "third"]) /100,

                          float(groupD.loc[groupD.teams == "argentina",     "fourth"])/100,

                          float(groupE.loc[groupE.teams == "canada",        "first"]) /100,

                          float(groupE.loc[groupE.teams == "netherlands",   "second"])/100,

                          float(groupE.loc[groupE.teams == "new zealand",   "third"]) /100,

                          float(groupE.loc[groupE.teams == "cameroon",      "fourth"])/100,

                          float(groupF.loc[groupF.teams == "united states", "first"]) /100,

                          float(groupF.loc[groupF.teams == "sweden",        "second"])/100,

                          float(groupF.loc[groupF.teams == "chile",         "third"]) /100,

                          float(groupF.loc[groupF.teams == "thailand",      "fourth"])/100]
# Perfect Group Prediction

group_perfect = pd.DataFrame(alex["odds_of_place"].groupby(level=0).prod()*100).rename({"odds_of_place" : "chance of perfect group"}, axis='columns')

a_group_perf = 3*group_perfect["chance of perfect group"].sum()/100

print("Expected perfect group points: " + str(a_group_perf))

group_perfect
# Predict First Place

group_first = pd.DataFrame(alex[alex.group_place == 1]["odds_of_place"]).rename({"odds_of_place" : "chance of first"}, axis='columns')

a_first = 2*group_first["chance of first"].sum()

print("Expected first place points: " + str(a_first))

group_first*100
# Odds of Advancing

advance_group = tourn_results[tourn_results.team.isin(alex[alex.advance == True].team)][["team", "rd16"]]

a_advance_16 = advance_group.rd16.sum()/100

print("Expected points from picking advancing team: " + str(a_advance_16))

advance_group
# Odds of Quarters

quarters_odds = tourn_results[tourn_results.team.isin(alex[alex.make_quarters == True].team)][["team", "quarters"]]

a_quarters = 2*quarters_odds.quarters.sum()/100

print("Expected points from picking quarters team: " + str(a_quarters))

quarters_odds
# Odds of Semis

semis_odds = tourn_results[tourn_results.team.isin(alex[alex.make_semis == True].team)][["team", "semis"]]

a_semis = 4*semis_odds.semis.sum()/100

print("Expected points from picking semis team: " + str(a_semis))

semis_odds
# Odds of Finals

finals_odds = tourn_results[tourn_results.team.isin(alex[alex.make_finals == True].team)][["team", "finals"]]

a_finals = 8*finals_odds.finals.sum()/100

print("Expected points from picking finals team: " + str(a_finals))

finals_odds
# Odds of Third

third_odds = tourn_results[tourn_results.team.isin(alex[alex.third == True].team)][["team", "third"]]

a_third = 8*third_odds.third.sum()/100

print("Expected points from picking third place team: " + str(a_third))

third_odds
# Odds of Winning

winner_odds = tourn_results[tourn_results.team.isin(alex[alex.win_it == True].team)][["team", "first"]]

a_champs = 16*float(winner_odds["first"])/100

print("Expected points from picking winner: " + str(a_champs))

winner_odds
summary = pd.DataFrame({"competitor"      : ["Alex",       "Jerrod",      "Jerry",          "Nikki"],

                        "EX pefect group" : [a_group_perf, jc_group_perf, ja_group_perfect, n_perf_group],

                        "EX 1st in group" : [a_first,      jc_first,      ja_first,         n_first_place],

                        "EX round of 16"  : [a_advance_16, jc_advance_16, ja_advance_16,    n_advance_group],

                        "EX quarters"     : [a_quarters,   jc_quarters,   ja_quarters,      n_quarters],

                        "EX semis"        : [a_semis,      jc_semis,      ja_semis,         n_semis],

                        "EX finals"       : [a_finals,     jc_finals,     ja_finals,        n_finals],

                        "EX 3rd place"    : [a_third,      jc_third,      ja_thirds,        n_thirds],

                        "EX champion"     : [a_champs,     jc_champs,     ja_champs,        n_winners_odds]})

summary["EX total"] = summary["EX pefect group"] + summary["EX 1st in group"] + summary["EX round of 16"] + summary["EX quarters"] + summary["EX semis"] + summary["EX finals"] + summary["EX 3rd place"] + summary["EX champion"]



summary
compare("germany", "china", )
compare("spain", "south africa")
compare("norway", "nigeria")