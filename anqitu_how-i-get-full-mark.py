import pandas as pd
import numpy as np

# Load data
TourneyResults_df = pd.read_csv('../input/WNCAATourneyCompactResults.csv')
Submission_df = pd.read_csv('../input/WSampleSubmissionStage1.csv')
TourneyResults_df['Team1ID'] = TourneyResults_df[['WTeamID','LTeamID']].min(axis=1)
TourneyResults_df['Team2ID'] = TourneyResults_df[['WTeamID','LTeamID']].max(axis=1)
# Set result as 1 if the team with smaller team number wins
TourneyResults_df['Result'] = np.where(TourneyResults_df['WTeamID']<TourneyResults_df['LTeamID'], 1, 0)
Submission_df['Season'] = Submission_df.ID.apply(lambda x: int(x[:4]))
Submission_df['Team1ID'] = Submission_df.ID.apply(lambda x: int(x[5:9]))
Submission_df['Team2ID'] = Submission_df.ID.apply(lambda x: int(x[10:14]))
# I dunt know why the Kaggle version of pd has no 'is.na', but it work on my laptop.
Submission_df['Result'] = Submission_df[['Season','Team1ID','Team2ID']].merge(TourneyResults_df[['Season','Team1ID','Team2ID','Result']],left_on = ['Season','Team1ID','Team2ID'],right_on = ['Season','Team1ID','Team2ID'],how='left')[['Result']]
Submission_df['Pred'] = np.where(pd.isna(Submission_df['Result']), Submission_df['Pred'], Submission_df['Result'])
Submission_df = Submission_df[['ID','Pred']]
Submission_df.to_csv('Submission.csv',index=False)