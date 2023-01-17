import numpy as np

import pandas as pd

from IPython.core.display import HTML



import warnings

warnings.filterwarnings("ignore")



sub = pd.read_csv('../input/Submissions.csv')

sub.dropna(subset=['SourceKernelVersionId','PrivateScoreFullPrecision'], inplace=True)

sub = sub[['SubmittedUserId', 'TeamId', 'SourceKernelVersionId', 'PrivateScoreFullPrecision']]



users = pd.read_csv('../input/Users.csv').rename(columns={'Id':'SubmittedUserId'})[['SubmittedUserId', 'UserName', 'DisplayName']]

sub = pd.merge(sub, users, how='inner', on='SubmittedUserId')



teams = pd.read_csv('../input/Teams.csv')[['Id', 'CompetitionId', 'PrivateLeaderboardRank']].rename(columns={'Id':'TeamId'})

sub = pd.merge(sub, teams, how='inner', on='TeamId')



comp = pd.read_csv('../input/Competitions.csv')

comp = comp[((comp['HostSegmentTitle'].isin(['Featured', 'Research', 'Recruitment', 'Playground'])) 

             & (comp['HasKernels']==True) 

             & (comp['HasLeaderboard']==True))]

comp.dropna(subset=['DeadlineDate'], inplace=True)

comp['DeadlineDate'] = pd.to_datetime(comp['DeadlineDate'])

comp = comp.sort_values(by='DeadlineDate', ascending=False).reset_index(drop=True)

comp['Sort'] = comp.index

comp = comp[['Sort', 'Id', 'Slug', 'Title', 'EvaluationAlgorithmIsMax']].rename(columns={'Id':'CompetitionId', 'Title':'CompetitionTitle'})

sub = pd.merge(sub, comp, how='inner', on='CompetitionId')



versions = pd.read_csv('../input/KernelVersions.csv')[['Id', 'KernelId', 'Title']].rename(columns={'Id':'SourceKernelVersionId'})

sub = pd.merge(sub, versions, how='inner', on='SourceKernelVersionId')



kernels = pd.read_csv('../input/Kernels.csv')[['Id', 'CurrentKernelVersionId', 'CurrentUrlSlug', 'TotalViews', 'TotalComments', 'TotalVotes']].rename(columns={'Id':'KernelId'})

sub = pd.merge(sub, kernels, how='inner', on='KernelId')

sub['Latest'] = sub['CurrentKernelVersionId'] == sub['SourceKernelVersionId']



cslug = sub[['Sort', 'Slug']].drop_duplicates()

cslug = cslug.sort_values(by=['Sort'], ascending=[True]).reset_index(drop=True)['Slug'].values



sub.drop(columns=['Sort','SubmittedUserId', 'TeamId', 'KernelId', 'CurrentKernelVersionId'], inplace=True)



#https://www.kaggle.com/shivamb/data-science-glossary-on-kaggle

#https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions

def best_kernels(df, n = 10):

    if df['EvaluationAlgorithmIsMax'].iloc[0] == False:

        df = df.sort_values(by=['PrivateScoreFullPrecision'], ascending=[True])

    else:

        df = df.sort_values(by=['PrivateScoreFullPrecision'], ascending=[False])



    df = df.drop_duplicates(subset=['CurrentUrlSlug'], keep='first')[:n].reset_index(drop=True)

    

    comp_url = "https://www.kaggle.com/c/"+str(df['Slug'].iloc[0])

    comp_img1 = 'https://storage.googleapis.com/kaggle-competitions/kaggle/' + str(df['CompetitionId'].iloc[0]) + '/logos/header.png'

    comp_img2 = 'https://www.kaggle.com/static/images/competition-noimage.png'

    hs = """<div style="border: 2px solid black; padding: 10px; height:100px; width:500; background-image: url('""" + comp_img1 + """'), url('""" + comp_img2 + """'); background-size: cover;">

                <h3><a style='color:#ffffff;  text-shadow: 2px 2px #000000;' href="""+comp_url+""">"""+df['CompetitionTitle'].iloc[0]+"""</a></h3>

            </div>

            <table>

            <th>

                <td><b>Kernel</b></td>

                <td><b>Author</b></td>

                <td><b>Leaderboard Rank</b></td>

                <td><b>Private Score</b></td>

                <td><b>Views</b></td>

                <td><b>Comments</b></td>

                <td><b>Votes</b></td>

            </th>"""

    for i, row in df.iterrows():

        url = "https://www.kaggle.com/"+row['UserName']+"/"+row['CurrentUrlSlug']+"?scriptVersionId=" + str(int(row['SourceKernelVersionId']))

        aurl= "https://www.kaggle.com/"+row['UserName']

        latest = ''

        if row['Latest'] == True:

            latest = ' *'

        hs += """<tr>

                    <td>"""+str(i+1)+"""</td>

                    <td><a href="""+url+""" target="_blank"><b>"""  + row['Title']  + latest + """</b></a></td>

                    <td><a href="""+aurl+""" target="_blank">"""  + row['DisplayName'] + """</a></td>

                    <td>"""+str(row['PrivateLeaderboardRank'])+"""</td>

                    <td>"""+str(round(row['PrivateScoreFullPrecision'],5))+"""</td>

                    <td>"""+str(row['TotalViews'])+"""</td>

                    <td>"""+str(row['TotalComments'])+"""</td>

                    <td>"""+str(row['TotalVotes'])+"""</td>

                    </tr>"""

    hs += "</table><hr/>"

    display(HTML(hs))



for slug in cslug:

    best_kernels(sub[sub['Slug']==slug])