import random, collections
def game(team1,team2,verbose=False):

    score1,score2,inning=0,0,0

    stillplaying=True

    while stillplaying:

        #play an inning

        inning+=1

        #team1 at bat

        if verbose:print(f"{team1[0]} at bat")

        outs,onbase=0,0

        while outs<3:

            r=random.random()

            if r<team1[1]:

                if onbase==team1[2]:score1+=1

                else: onbase+=1

            else:outs+=1

            if verbose:print(f"{onbase} on base, {outs} outs, {score1} runs {r}")

        #team2 at bat

        if verbose:print(f"{team2[0]} at bat")

        outs,onbase=0,0

        while outs<3:

            r=random.random()

            if r<team2[1]:

                if onbase==team2[2]:score2+=1

                else: onbase+=1

            else:outs+=1

            if verbose:print(f"{onbase} on base, {outs} outs, {score2} runs {r}")

        if inning>=9:

            if score1!=score2:stillplaying=False

        if verbose:print(f"After inning {inning}, score is {score1} to {score2}")

    if score1>score2:return(team1[0])

    else:return(team2[0])

        
# Name, chance of a hit, how many hits to exceed before scoring

teams=[["Mississippi Moonwalkers",0.4,3],["Delaware Doubloons",0.2,1],["Tennessee Taters",0.1,0]]

winners=[]

games_in_the_season=100000

for _ in range(games_in_the_season//2):

    winners.append(game(teams[0],teams[1]))

    winners.append(game(teams[0],teams[2]))

    winners.append(game(teams[2],teams[1]))
collections.Counter(winners)

pascal_coefficients=[]

for x in range(25):

    pascal_coefficients.append((x+1)*(x+2)/2)

inning_hit_chances=[]

inning_expected_runs=[]

for team in teams:

    name=team[0]

    p=team[1]

    q=1-p

    exceed=team[2]

    team_hit_chances=[pascal_coefficients[x]*p**x*q**3 for x in range(25)]

    inning_hit_chances.append(team_hit_chances)

    team_runs_by_hits=[max(0,x-exceed) for x in range(25)]

    expected_hit_values=[team_hit_chances[x]*team_runs_by_hits[x] for x in range(25)]

    inning_expected_runs.append(sum(expected_hit_values))

    print(f"{name}: {sum(expected_hit_values)} runs per inning")

        

    
game2(teams[0],teams[2],True)
def game2(team1,team2,verbose=False):

    score1,score2,inning=0,0,0

    inning_wins=[]

    stillplaying=True

    while stillplaying:

        #play an inning

        inning+=1

        #team1 at bat

        if verbose:print(f"{team1[0]} at bat")

        outs,onbase,iscore1=0,0,0

        while outs<3:

            r=random.random()

            if r<team1[1]:

                if onbase==team1[2]:iscore1+=1

                else: onbase+=1

            else:outs+=1

            score1+=iscore1

            if verbose:print(f"{onbase} on base, {outs} outs, {score1} runs {r}")

        #team2 at bat

        if verbose:print(f"{team2[0]} at bat")

        outs,onbase,iscore2=0,0,0

        while outs<3:

            r=random.random()

            if r<team2[1]:

                if onbase==team2[2]:iscore2+=1

                else: onbase+=1

            else:outs+=1

            score2+=iscore2

            if verbose:print(f"{onbase} on base, {outs} outs, {score2} runs {r}")

        if iscore1>iscore2:inning_wins.append([team1[0],iscore1,iscore2])

        elif iscore2>iscore1:inning_wins.append([team2[0],iscore2,iscore1])

        else:inning_wins.append(['tie',iscore1])

        if inning>=9:

            if score1!=score2:stillplaying=False

        if verbose:print(f"After inning {inning}, score is {score1} to {score2}")

    print(inning_wins)

    if score1>score2:return(team1[0],team2[0],score1,score2,inning_wins)

    else:return(team2[0],team1[0],score2,score1,inning_wins)

        
# Name, chance of a hit, how many hits to exceed before scoring

one_inning_winners=[]

games_in_the_season=100000

for _ in range(games_in_the_season//2):

    one_inning_winners.append(game2(teams[0],teams[1]))

    one_inning_winners.append(game2(teams[0],teams[2]))

    one_inning_winners.append(game2(teams[2],teams[1]))

collections.Counter(one_inning_winners)