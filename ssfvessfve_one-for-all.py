%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

competitions = (pd.read_csv('../input/Competitions.csv')
                .rename(columns={'Id':'CompetitionId'}))
competitions = competitions[(competitions.UserRankMultiplier > 0)]

evals = (pd.read_csv('../input/EvaluationAlgorithms.csv')
           .rename(columns={'Id':'EvaluationAlgorithmId'}))
competitions = competitions.merge(evals[['EvaluationAlgorithmId','IsMax']], 
                                  how='left',on='EvaluationAlgorithmId')
# Fill missing values for two competitions
competitions.loc[competitions.CompetitionId==4488,'IsMax'] = True # Flavours of physics
competitions.loc[competitions.CompetitionId==4704,'IsMax'] = False # Santa's Stolen Sleigh

scriptprojects = pd.read_csv('../input/ScriptProjects.csv')
competitions = competitions[competitions.CompetitionId.isin(scriptprojects.CompetitionId)]
print("Found {} competitions with scripts enabled.".format(competitions.shape[0]))
if competitions.IsMax.isnull().any():
    # in case this is rerun after more competitions are added
    print("Please fill IsMax value for:")
    print(competitions.loc[competitions.IsMax.isnull(),['CompetitionId','Title']])
competitions.shape[0]
teams = (pd.read_csv('../input/Teams.csv')
         .rename(columns={'Id':'TeamId'}))
teams = teams[teams.CompetitionId.isin(competitions.CompetitionId)]
teams['Score'] = teams.Score.astype(float)

submissions = pd.read_csv('../input/Submissions.csv')
submissions = submissions[(submissions.TeamId.isin(teams.TeamId))
                         &(submissions.IsAfterDeadline==False)
                         &(~(submissions.PublicScore.isnull()))]
submissions = submissions.merge(teams[['TeamId','CompetitionId']],
                                how='left',on='TeamId')
submissions = submissions.merge(competitions[['CompetitionId','IsMax']],
                                how='left',on='CompetitionId')

competitions.set_index("CompetitionId",inplace=True)
# How many teams participated in a competition?
competitions['Nteams'] = (submissions.groupby('CompetitionId')
                          ['TeamId'].nunique())
# How many teams used at least one script submission?
competitions['TeamsSubmittedScripts'] = (submissions
                                         [~(submissions.SourceScriptVersionId.isnull())]
                                         .groupby('CompetitionId')['TeamId'].nunique())
submissions
submissions.groupby('TeamId')['IsSelected'].sum().value_counts()

def isscored(group):
    # if two or less submissions select all
    if group.shape[0] <= 2:
        pd.Series(np.ones(group.shape[0],dtype=np.bool),index=group.index)
    nsel = group.IsSelected.sum()
    # if two selected return them
    if nsel == 2:
        return group.IsSelected
    # if need to select more - choose by highest public score
    toselect = list(group.IsSelected.values.nonzero()[0])
    ismax = group['IsMax'].iloc[0]
    ind = np.argsort(group['PublicScore'].values)
    scored = group.IsSelected.copy()
    if ismax:
        ind = ind[::-1]
    for i in ind:
        if i not in toselect:
            toselect.append(i)
        if len(toselect)==2:
            break
    scored.iloc[toselect] = True
    return scored
submissions['PublicScore'] = submissions['PublicScore'].astype(float)
submissions['PrivateScore'] = submissions['PrivateScore'].astype(float)
scored = submissions.groupby('TeamId',sort=False).apply(isscored)
scored.index = scored.index.droplevel()
submissions['IsScored'] = scored
# How many teams selected a script submission for private LB scoring?
competitions['TeamsSelectedScripts'] = (submissions
                                        [~(submissions.SourceScriptVersionId.isnull())&
                                          (submissions.IsScored)]
                                        .groupby('CompetitionId')['TeamId'].nunique())
competitions.sort_values(by='Nteams',inplace=True)
fig, ax = plt.subplots(figsize=(10,8))
h = np.arange(len(competitions))
colors = cm.Blues(np.linspace(0.5, 1, 3))
ax.barh(h, competitions.Nteams,color=colors[0])
ax.barh(h, competitions.TeamsSubmittedScripts,color=colors[1])
ax.barh(h, competitions.TeamsSelectedScripts,color=colors[2])
ax.set_yticks(h+0.4)
ax.set_yticklabels(competitions.Title.values);
ax.set_ylabel('');
ax.legend(['Total teams',
           'Submitted from a script',
           'Selected a script submission'],loc=4,fontsize='large');
ax.set_title('Usage of script submissions by teams');
ax.set_ylim(0,h.max()+1);
h

competitions["NScriptSubs"] = (submissions
                               [~(submissions.SourceScriptVersionId.isnull())]
                               .groupby('CompetitionId')['Id'].count())
scriptycomps = competitions[competitions.NScriptSubs > 0].copy()
scriptycomps.shape
def find_private_score(df):
    if df.SourceScriptVersionId.isnull().all():
        # no scripts
        return
    ismax = df.IsMax.iloc[0]
    submit = (df.loc[~(df.SourceScriptVersionId.isnull())]
                .groupby('SourceScriptVersionId')
                [['PublicScore','PrivateScore']]
                .agg('first')
                .sort_values(by='PublicScore',ascending = not ismax)
                .iloc[:2])
    score = submit.PrivateScore.max() if ismax else submit.PrivateScore.min()
    # Find scores from all teams
    results = (df.loc[df.IsScored]
                 .groupby('TeamId')
                 ['PrivateScore']
                 .agg('max' if ismax else 'min')
                 .sort_values(ascending = not ismax)
                 .values)
    if ismax:
        ranktail = (results <  score).nonzero()[0][0] + 1
        rankhead = (results <= score).nonzero()[0][0] + 1
    else:
        ranktail = (results >  score).nonzero()[0][0] + 1
        rankhead = (results >= score).nonzero()[0][0] + 1
    rank = int(0.5*(ranktail+rankhead))
    return pd.Series({'Rank':rank,'Score':score})

scriptycomps[['Rank','Score']] = (submissions.groupby('CompetitionId')
                                             .apply(find_private_score))
scriptycomps['TopPerc'] = np.ceil(100*scriptycomps['Rank']
                                  /scriptycomps['Nteams'])
scriptycomps['Points'] = (1.0e5*((scriptycomps.Rank)**(-0.75))
                          *np.log10(1+np.log10(scriptycomps.Nteams))
                          *scriptycomps.UserRankMultiplier)
scriptycomps[['Title','Score','Nteams',
              'Rank','TopPerc','Points']].sort_values(by='Rank')

top10p = (scriptycomps.TopPerc <= 10).sum()
top25p = ((scriptycomps.TopPerc > 10)&(scriptycomps.TopPerc <= 25)).sum()
print("{} Top10% badges and {} Top25% badges".format(top10p, top25p))

lastdeadline = pd.to_datetime(competitions.Deadline.max())
decay = np.exp((pd.to_datetime(scriptycomps.Deadline) - lastdeadline).dt.days/500)
totalpoints = (decay*scriptycomps.Points).sum()
totalpoints
users = pd.read_csv('../input/Users.csv').sort_values(by='Points',ascending=False)
rank = (users.Points < totalpoints).nonzero()[0][0] + 1
print("Number {} in the global ranking".format(rank))
scriptycomps.loc[4471,'Rank'] = 150
scriptycomps['TopPerc'] = np.ceil(100*scriptycomps['Rank']/scriptycomps['Nteams'])
scriptycomps['Points'] = 1.0e5*((scriptycomps.Rank)**(-0.75))*np.log10(1+np.log10(scriptycomps.Nteams))*scriptycomps.UserRankMultiplier
totalpoints1 = (decay*scriptycomps.Points).sum()
totalpoints1
rank1 = (users.Points < totalpoints1).nonzero()[0][0] + 1
rank1
def find_private_score(df):
    if df.SourceScriptVersionId.isnull().all():
        # no scripts
        return
    ismax = df.IsMax.iloc[0]
    competition = df.name
    submit = (df.loc[~(df.SourceScriptVersionId.isnull())
                     &(df.IsScored)]
                .groupby('SourceScriptVersionId')
                .agg({'PublicScore':'first','PrivateScore':'first','Id':'size'})
                .rename(columns={'Id':'Nteams'})
                .sort_values(by='Nteams',ascending = False)
                .iloc[:2])
    score = submit.PrivateScore.max() if ismax else submit.PrivateScore.min()
    # Find scores from all teams
    results = (df.loc[df.IsScored]
                 .groupby('TeamId')
                 ['PrivateScore']
                 .agg('max' if ismax else 'min')
                 .sort_values(ascending = not ismax)
                 .values)
    rank = int(np.median((results==score).nonzero()[0])) + 1
    return pd.Series({'Rank':rank,'Score':score})

scriptycomps[['Rank','Score']] = (submissions.groupby('CompetitionId')
                                             .apply(find_private_score))
scriptycomps['TopPerc'] = np.ceil(100*scriptycomps['Rank']
                                  /scriptycomps['Nteams'])
scriptycomps['Points'] = (1.0e5*((scriptycomps.Rank)**(-0.75))
                          *np.log10(1+np.log10(scriptycomps.Nteams))
                          *scriptycomps.UserRankMultiplier)
scriptycomps[['Title','Score','Nteams',
              'Rank','TopPerc','Points']].sort_values(by='Rank')
top10p = (scriptycomps.TopPerc <= 10).sum()
top25p = ((scriptycomps.TopPerc > 10)&(scriptycomps.TopPerc <= 25)).sum()
print("{} Top10% badges and {} Top25% badges".format(top10p, top25p))
totalpoints2 = (decay*scriptycomps.Points).sum()
rank2 = (users.Points < totalpoints2).nonzero()[0][0] + 1
print("Ranked {} with {:.1f} points.".format(rank2,totalpoints2))