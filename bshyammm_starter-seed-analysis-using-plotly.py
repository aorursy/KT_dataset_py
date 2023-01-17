import numpy as np 

import pandas as pd 



in_path = '../input/datafiles/'



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



#suppress warnings

import warnings

warnings.filterwarnings('ignore')

#import data

NCAATourneyCompactResults = pd.read_csv(in_path + 'NCAATourneyCompactResults.csv')

NCAATourneyCompactResults.head(5)
NCAATourneySeeds = pd.read_csv(in_path + 'NCAATourneySeeds.csv')

#convert seed to int

NCAATourneySeeds.Seed = NCAATourneySeeds.Seed.str.replace('[a-zA-Z]', '')

NCAATourneySeeds.Seed = NCAATourneySeeds.Seed.astype('int64')

NCAATourneySeeds.head(5)
#Join winning team's seed

NCAA = pd.merge(NCAATourneyCompactResults, NCAATourneySeeds, how='inner', 

               left_on=['Season', 'WTeamID'], 

               right_on=['Season', 'TeamID'])

NCAA.rename(columns={"Seed": "W_SEED"}, inplace=True)

#Join losing team's seed

NCAA = pd.merge(NCAA, NCAATourneySeeds, how='inner', 

               left_on=['Season', 'LTeamID'], 

               right_on=['Season', 'TeamID'])

NCAA.rename(columns={"Seed": "L_SEED"}, inplace=True)

NCAA.drop(columns=['TeamID_x', 'TeamID_y'], inplace=True)

NCAA.head(5)
NCAA['OUTCOME'] = 1

NCAA['Seed_diff'] = NCAA.L_SEED - NCAA.W_SEED

NCAA['Lower_Seed_Win'] = np.where(NCAA.Seed_diff>0, 1, 0)

NCAA['Higher_Seed_Win'] = np.where(NCAA.Seed_diff<0, 1, 0)

NCAA.tail(5)
counts = pd.DataFrame(NCAA.Lower_Seed_Win.value_counts()/len(NCAA))

counts = counts.reset_index()

counts.columns = ['Outcome', 'Percent']

counts
data = [

    go.Bar(

        x = counts.Outcome,

        y = counts.Percent,

        #text = (NCAA.Lower_Seed_Win.value_counts()/len(NCAA)), 

        #textposition = 'auto', 

        marker = dict(

          color = ['rgba(50, 171, 96, 0.7)', 'rgba(219, 64, 82, 0.7)']

        ),

        name = 'Seeds'

    )

]

fig = go.Figure(data=data)

iplot(fig, filename='base-bar')
NCAA_counts = NCAA.groupby(['Season'])['Lower_Seed_Win', 'Higher_Seed_Win'].agg('sum').reset_index()

NCAA_counts.tail(5)
data = [

    go.Bar(

        x = NCAA_counts.Season,

        y = NCAA_counts.Higher_Seed_Win,

        marker = dict(

          color = 'rgba(219, 64, 82, 0.7)'

        ),

        name = 'Higher Seed Win'

    ),

    go.Bar(

        x = NCAA_counts.Season,

        y = NCAA_counts.Lower_Seed_Win,

        marker = dict(

          color = 'rgba(55, 128, 191, 0.7)'

        ),

        name = 'Lower Seed Win'

    )

]

layout = go.Layout(

    barmode='group'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='base-bar')
NCAA['Seed_diff_abs'] = abs(NCAA.Seed_diff)

NCAA_counts = NCAA.groupby(['Seed_diff_abs'])['Lower_Seed_Win', 'Higher_Seed_Win'].agg('sum').reset_index()

NCAA_counts
data = [

    go.Bar(

        x = NCAA_counts.Seed_diff_abs,

        y = NCAA_counts.Higher_Seed_Win,

        text = NCAA_counts.Higher_Seed_Win, 

        textposition = 'auto', 

        marker = dict(

          color = 'rgba(219, 64, 82, 0.7)'

        ),

        name = 'Higher Seed Win'

    ),

    go.Bar(

        x = NCAA_counts.Seed_diff_abs,

        y = NCAA_counts.Lower_Seed_Win,

        text = NCAA_counts.Lower_Seed_Win, 

        textposition = 'auto', 

        marker = dict(

          color = 'rgba(55, 128, 191, 0.7)'

        ),

        name = 'Lower Seed Win'

    )

]





fig = go.Figure(data=data)

iplot(fig, filename='base-bar')