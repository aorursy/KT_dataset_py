%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Competitions - use only those that award points
competitions = (pd.read_csv('../input/Competitions.csv')
                .rename(columns={'Id':'CompetitionId'}))
competitions = competitions[(competitions.UserRankMultiplier > 0)]
# Scriptprojects to link scripts to competitions
scriptprojects = (pd.read_csv('../input/ScriptProjects.csv')
                    .rename(columns={'Id':'ScriptProjectId'}))
# Evaluation algorithms
evaluationalgorithms = (pd.read_csv('../input/EvaluationAlgorithms.csv')
                          .rename(columns={'Id':'EvaluationAlgorithmId'}))
competitions = (competitions.merge(scriptprojects[['ScriptProjectId','CompetitionId']],
                                   on='CompetitionId',how='left')
                            .merge(evaluationalgorithms[['IsMax','EvaluationAlgorithmId']],
                                   on='EvaluationAlgorithmId',how='left')
                            .dropna(subset = ['ScriptProjectId'])
                            .set_index('CompetitionId'))
competitions['ScriptProjectId'] = competitions['ScriptProjectId'].astype(int)
# Fill missing values for two competitions
competitions.loc[4488,'IsMax'] = True # Flavours of physics
competitions.loc[4704,'IsMax'] = False # Santa's Stolen Sleigh
# Scripts
scripts = pd.read_csv('../input/Scripts.csv')
# Script versions
# List necessary columns to avoid reading script versions content
svcols = ['Id','Title','DateCreated','ScriptId',
          'LinesInsertedFromPrevious','LinesDeletedFromPrevious', 
          'LinesChangedFromPrevious','LinesInsertedFromFork', 
          'LinesDeletedFromFork', 'LinesChangedFromFork']
scriptversions = pd.read_csv('../input/ScriptVersions.csv', 
                             usecols=svcols)
scriptversions['DateCreated'] = pd.to_datetime(scriptversions['DateCreated'])
# Determine if a script version contains changes 
#(either from fork parent or from previous version)
isfirst = scriptversions.Id.isin(scripts.FirstScriptVersionId)
scriptversions.loc[isfirst, 'IsChanged'] = scriptversions.loc[isfirst, 
            ['LinesInsertedFromFork', 
             'LinesDeletedFromFork', 
             'LinesChangedFromFork']].any(axis=1)
scriptversions.loc[~(isfirst), 'IsChanged'] = scriptversions.loc[~(isfirst), 
            ['LinesInsertedFromPrevious', 
             'LinesDeletedFromPrevious', 
             'LinesChangedFromPrevious']].any(axis=1)
# Submissions
submissions = pd.read_csv('../input/Submissions.csv')
submissions = submissions.dropna(subset=['Id','DateSubmitted','PublicScore'])
submissions.DateSubmitted = pd.to_datetime(submissions.DateSubmitted)
def report_script_activity(scriptversions, submissions, ismax):
    scores = pd.DataFrame()
    scores['BestPublic'] = submissions.PublicScore.cummax() if ismax else submissions.PublicScore.cummin()
    scores.loc[scores.BestPublic == submissions.PublicScore, 'BestPrivate'] = submissions.PrivateScore
    scores.BestPrivate = scores.BestPrivate.fillna(method='ffill')
    scores['DateSubmitted'] = submissions['DateSubmitted']
    activity = pd.DataFrame()
    activity['Submissions'] = submissions.groupby(submissions.DateSubmitted.dt.date)['Id'].size()
    activity['SubmissionsBest'] = ((submissions['PublicScore']==scores['BestPublic'])
                                   .groupby(submissions.DateSubmitted.dt.date).sum())
    activity['Versions'] = scriptversions.groupby(scriptversions.DateCreated.dt.date)['Id'].size()
    activity['VersionsChanged'] = scriptversions.groupby(scriptversions.DateCreated.dt.date)['IsChanged'].sum()
    return scores, activity

def plot_script_activity(scores, activity):
    fig, ax = plt.subplots(3,1, figsize=(10,8), sharex=True, gridspec_kw = {'height_ratios':[1,3,1]})
    colors = cm.Blues(np.linspace(0.5, 0.8, 2))
    ax[0].bar(activity.index, activity.Versions,color=colors[0])
    ax[0].bar(activity.index, activity.VersionsChanged,color=colors[1])
    ax[0].set_title('Daily new versions')
    ax[0].legend(['all','with changes'])
    ax[1].plot(scores.DateSubmitted, scores.BestPublic, '-', 
               scores.DateSubmitted, scores.BestPrivate, '-')
    ax[1].set_title('Best public submission scores')
    ax[1].legend(['Public','Private'],loc=4)
    ax[2].bar(activity.index, activity.Submissions,color=colors[0]);
    ax[2].bar(activity.index, activity.SubmissionsBest,color=colors[1]);
    ax[2].set_title('Daily submissions');
    ax[2].legend(['all','best public']);
    return fig, ax
def report_competition(competitionId):
    ismax = competitions.loc[competitionId,'IsMax']
    scriptprojectid = competitions.loc[competitionId, 'ScriptProjectId']
    s = scripts.loc[scripts.ScriptProjectId==scriptprojectid,'Id'].values
    v = scriptversions.loc[scriptversions.ScriptId.isin(s)]
    sub = (submissions.loc[submissions.SourceScriptVersionId.isin(v.Id)]
                      .sort_values(by='DateSubmitted'))
    scores, activity = report_script_activity(v,sub,ismax)
    fig, ax = plot_script_activity(scores, activity)
    plt.suptitle(competitions.loc[competitionId,'Title'],fontsize='large')
# Recent competitions
competitions.sort_values(by='Deadline',ascending=False)[['Title']].head()
report_competition(5056)
# find scriptIds of scripts forked from ancestors, and of their forks, and of their forks...
def find_descendants(ancestors, scripts, scriptversions):
    if len(ancestors) == 0:
        return np.array([],dtype=int)
    ancestors_versions = scriptversions.loc[scriptversions.ScriptId.isin(ancestors),'Id']
    children = scripts.loc[scripts.ForkParentScriptVersionId.isin(ancestors_versions.values),'Id'].values
    return np.concatenate((children, find_descendants(children, scripts, scriptversions)))
# find scripts with most descendants in a competition
def find_most_forked_scripts(competitionId, n = 5):
    print('Most forked scripts in {}'.format(competitions.loc[competitionId,'Title']))
    # Find scripts project id
    projectId = competitions.loc[competitionId,'ScriptProjectId']
    # Read in scripts and scriptversions data
    s = scripts.loc[(scripts.ScriptProjectId==projectId)].copy()
    v = scriptversions.loc[scriptversions.ScriptId.isin(s.Id)]
    origmask = s.ForkParentScriptVersionId.isnull()
    s.loc[origmask,'Nforks'] = s.loc[origmask,'Id'].apply(lambda x,s,v: find_descendants([x],s,v).shape[0],args=(s,v))
    return s[['Id','UrlSlug','Nforks']].sort_values(by='Nforks',ascending=False).head(n)
find_most_forked_scripts(5056)
def report_competition_script(competitionId, scriptId):
    ismax = competitions.loc[competitionId,'IsMax']
    children = find_descendants([scriptId],scripts,scriptversions)
    family = np.append(children,[scriptId])
    v = scriptversions.loc[scriptversions.ScriptId.isin(family)]
    sub = (submissions.loc[submissions.SourceScriptVersionId.isin(v.Id)]
                      .sort_values(by='DateSubmitted'))
    scores, activity = report_script_activity(v,sub,ismax)
    fig, ax = plot_script_activity(scores, activity)
    scriptname = scripts.loc[scripts.Id==scriptId,'UrlSlug'].values[0]
    competitionname = competitions.loc[competitionId,'Title']
    title = '{} script and all its forks\n{}'.format(scriptname, competitionname)
    plt.suptitle(title,fontsize='large')
report_competition_script(5056, 60666)
def plot_script(scriptId, ax, x=0, vmin=0.49, vmax = 0.502):
    ax.set_title('The history of forks of {}'.format(s.loc[scripts.Id==scriptId,'UrlSlug'].values[0]),
                 fontsize='x-large')
    versions = v.loc[v.ScriptId==scriptId].sort_values(by='DateCreated', ascending=False)
    ax.plot(versions.DateCreated.values,
            np.ones(versions.shape[0])*x, 
            'k-',zorder=1, linewidth=0.5)
    ax.scatter(versions.DateCreated.values, 
               np.ones(versions.shape[0])*x, 
               s = 2*versions.Nsubmissions.values,
               c = versions.PublicScore.values,
               cmap = cm.rainbow,marker='o',alpha=0.9,
               vmin = vmin, zorder=2,vmax=vmax)
    n = 1
    for versionId in versions.index:
        versionDate = versions.loc[versionId,'DateCreated']
        desc = s.loc[s.ForkParentScriptVersionId==versionId]
        if desc.shape[0] == 0:
            continue
        desc = desc.sort_values(by='Id',ascending=False)
        for script in desc.Id.values:
            forkversion = desc.loc[desc.Id==script,'FirstScriptVersionId'].values[0]
            forkversionDate = v.loc[forkversion,'DateCreated']
            ax.plot([versionDate, forkversionDate],
                    [x,x+n],
                    'k-',zorder=1, linewidth=0.5,alpha = 0.5)
            nd = plot_script(script, ax, x=x+n)
            n += nd
    return n
scriptId=60666
children = find_descendants([scriptId],scripts,scriptversions)
family = np.append(children,[scriptId])
s = scripts.loc[scripts.Id.isin(family)]
v = scriptversions.loc[scriptversions.ScriptId.isin(family)].set_index('Id')
sub = (submissions.loc[submissions.SourceScriptVersionId.isin(v.index)]
                  .sort_values(by='DateSubmitted'))
v['Nsubmissions'] = sub.groupby('SourceScriptVersionId').size()
v['PublicScore'] = sub.groupby('SourceScriptVersionId')['PublicScore'].agg('first')
fig, ax = plt.subplots(figsize=(12,12))
n = plot_script(60666, ax)
oneday = pd.to_timedelta(1, unit='day')
ax.set_xlim(v.DateCreated.min()-oneday, 
            competitions.loc[5056,'Deadline']);
ax.set_ylim(-10, n+10);
competitionId = 4986
report_competition(competitionId)
find_most_forked_scripts(competitionId)
projectId = competitions.loc[competitionId,'ScriptProjectId']
s = scripts.loc[(scripts.ScriptProjectId==projectId)]
v = scriptversions.loc[scriptversions.ScriptId.isin(s.Id)].set_index('Id')
sub = (submissions.loc[submissions.SourceScriptVersionId.isin(v.index)]
                  .sort_values(by='DateSubmitted'))
v['Nsubmissions'] = sub.groupby('SourceScriptVersionId').size()
v['PublicScore'] = sub.groupby('SourceScriptVersionId')['PublicScore'].agg('first')
fig, ax = plt.subplots(figsize=(12,12))
n = 0
for script in [49934, 43840]:
    n += plot_script(script, ax,x=n,vmin=0.841, vmax=v.PublicScore.max())
ax.set_xlim(competitions.loc[competitionId,'DateEnabled'], 
            competitions.loc[competitionId,'Deadline']);
ax.set_ylim(-10,n+10);
ax.set_title('The history of forks of popular Santander scripts',fontsize='large');