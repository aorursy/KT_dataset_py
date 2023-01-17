import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import re

import sys



YEAR = 2017



# lists of knocked out teams

playins = ["New Orleans", "Providence", "Wake Forest", "NC Central"]

round1 = ["Mt St Mary's", 'Virginia Tech', 'UNC Wilmington', 'ETSU', 'SMU', 'New Mexico St', 'Marquette', 'Troy', 'S Dakota St', 'Vanderbilt', 'Princeton', 'Bucknell', 'Maryland', 'FL Gulf Coast', 'VA Commonwealth', 'North Dakota', 'TX Southern', 'Seton Hall', 'Minnesota', 'Winthrop', 'Kansas St', 'Kent', 'Dayton', 'N Kentucky', 'UC Davis', 'Miami FL', 'Nevada', 'Vermont', 'Creighton', 'Iona', 'Oklahoma St', 'Jacksonville St']

round2 = ['Arkansas', 'Cincinnati', 'Duke', 'Florida St', 'Iowa St', 'Louisville', 'Michigan St', 'MTSU', 'Northwestern', 'Notre Dame', 'Rhode Island', "St Mary's CA", 'USC', 'Villanova', 'Virginia', 'Wichita St']



teamsToRemove = playins + round1 + round2



teams = pd.read_csv('../input/Teams.csv')

teamId2Name = dict(teams[['Team_Id','Team_Name']].values)

teamName2Id = dict(teams[['Team_Name','Team_Id']].values)



seeds = pd.read_csv('../input/TourneySeeds.csv')

seeds = seeds.loc[seeds.Season==YEAR]

teamIds = list(seeds.Team.values) # team ids conveniently ordered by region, seed



for team in teamsToRemove:

    teamIds.remove(teamName2Id[team])



# maps global team ids to 0..n ordered by 2017 tournament region & seed

teamId2Index = {t:i for i,t in enumerate(teamIds)}

heatmapLabels = [teamId2Name[teamIds[i]] for i in range(len(teamIds))]

nteams = len(heatmapLabels)





# TODO a few submissions are zips within zips!

def extractSubmission(id):

    os.system("unzip -d %d ../input/predictions/%d.zip" % (id, id))    

    return os.path.join(str(id), os.listdir("%d" % id)[0])





def loadSubmission(filename):

    df = pd.read_csv(filename)

    df.columns = df.columns.str.lower()

    # t1 is always the lower team id

    df['t1'] = df.apply(lambda x: int(x.id.split('_')[1]), axis=1)

    df['t2'] = df.apply(lambda x: int(x.id.split('_')[2]), axis=1)

    # filter out teams here

    df = df.ix[np.logical_and(df.t1.isin(teamIds), df.t2.isin(teamIds))]

    df[['t1','t2']] = df[['t1','t2']].applymap(teamId2Index.get)

    # better to leave unassigned diagonal as zero, stands out more

    m = np.zeros((nteams, nteams))

    m[df.t1, df.t2] = df.pred

    m[df.t2, df.t1] = 1 - df.pred

    return m





def showHeatmap(probs, filename):

    plt.clf()

    fig, ax = plt.subplots()

    fig.set_size_inches(10, 10)

    # try different colormaps:

    # http://matplotlib.org/users/colormaps.html

    heatmap = ax.pcolormesh(probs, vmin=0, vmax=1, cmap=plt.cm.seismic)



    # put the major ticks at the middle of each cell

    ax.set_xticks(np.arange(nteams)+0.5, minor=False)

    ax.set_yticks(np.arange(nteams)+0.5, minor=False)



    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.spines['left'].set_visible(False)



    # want a more natural, table-like display

    ax.invert_yaxis()

    ax.tick_params(direction='out')

    ax.xaxis.tick_top()

    ax.yaxis.tick_left()

    plt.xticks(rotation=90)



    fontsize = 8 if nteams >= 32 else 12

    ax.set_xticklabels(heatmapLabels, fontsize=fontsize)

    ax.set_yticklabels(heatmapLabels, fontsize=fontsize)

    plt.show()

    plt.savefig(filename, bbox_inches='tight')
KAGGLERS = [ 'James Trotman' ]



submissions = pd.read_csv('../input/team_submission_key.csv')



for i, row in submissions.ix[submissions.TeamName.isin(KAGGLERS)].iterrows():

    csv = extractSubmission(row.Id)

    probs = loadSubmission(csv)

    print(row.TeamName, '-', csv)

    showHeatmap(probs, '%d.png' % row.Id)