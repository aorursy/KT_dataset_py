## IMPORTS 
import glob 
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import norm
from scipy import interpolate
from sklearn.neighbors import KernelDensity

import plotly.io as pio
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
init_notebook_mode(connected=True)
## FUNCTIONS (utility/preprocessing)
DDIR = '../input/NFL-Punt-Analytics-Competition/'

def collect_outcomes(data):
    """
    Extract the punt outcome from the PlayDescription field.

    Parameters:
        data: dict (keys: labels, values: DataFrames)
            Data dictionary - likely the output from load_data().
    """

    play_info =  data['play_info']

    def _process_description(row):
        tmp_desc = row.PlayDescription
        outcome = ''

        if 'touchback' in tmp_desc.lower():
            outcome = 'touchback'
        elif 'fair catch' in tmp_desc.lower():
            outcome = 'fair catch'
        elif 'out of bounds' in tmp_desc.lower():
            outcome = 'out of bounds'
        elif 'muff' in tmp_desc.lower():
            outcome = 'muffed punt'
        elif 'downed' in tmp_desc.lower():
            outcome = 'downed'
        elif 'no play' in tmp_desc.lower():
            outcome = 'no play'
        elif 'blocked' in tmp_desc.lower():
            outcome = 'blocked punt'
        elif 'fumble' in tmp_desc.lower():
            outcome = 'fumble'
        elif 'pass' in tmp_desc.lower():
            outcome = 'pass'
        elif 'declined' in tmp_desc.lower():
            outcome = 'declined penalty'
        elif 'direct snap' in tmp_desc.lower():
            outcome = 'direct snap'
        elif 'safety' in tmp_desc.lower():
            outcome = 'safety'
        else:
            if 'punts' in tmp_desc.lower():
                outcome = 'return'
            else:
                outcome = 'SPECIAL'

        return outcome

    play_info.loc[:, 'Punt_Outcome'] = play_info.apply(_process_description, axis=1)

    def _identify_penalties(row):
        if 'penalty' in row.PlayDescription.lower():
            return 1
        else:
            return 0

    play_info.loc[:, 'Penalty_on_Punt'] = play_info.apply(_identify_penalties, axis=1)

    # Update dictionary to include additional set of features.
    data.update({'play_info': play_info})

    return data

def expand_play_description(data):
    """
    Expand the PlayDescription field in a standardized fashion. This function
    extracts a number of relevant additional features from PlayDescription,
    including punt distance, post-punt field location, and a few other derived
    features.

    Parameters:
        data: dict (keys: labels, values: DataFrames)
            Data dictionary - likely the output from load_data().
    """

    play_info = data['play_info']

    def _split_punt_distance(row):
        try:
            return int(row.PlayDescription.split('punts ')[1].split('yard')[0])
        except IndexError:
            return np.nan

    def _split_field_position(row):
        try:
            return row.PlayDescription.split(',')[0].split('to ')[1]
        except IndexError:
            return ''

    def _post_punt_territory(row):
        if row.Poss_Team == row.Post_Punt_FieldSide:
            return 0
        else:
            return 1

    def _start_punt_field_position(row):
        try:
            field_position = int(row.YardLine.split(' ')[1])
        except:
            print(row.YardLine)

        if row.Poss_Team in row.YardLine:
            return field_position
        else:
            return 100 - field_position

    def _field_position_punt(row):
        if 'end zone' in row.Post_Punt_YardLine:
            return 0
        elif '50' in row.Post_Punt_YardLine:
            return 50
        else:
            try:
                yard_line = int(row.Post_Punt_YardLine.split(' ')[1])
                own_field = int(row.Post_Punt_Own_Territory)

                if not own_field:
                    return 100 - yard_line
                else:
                    return yard_line
            except:
                return -999

    play_info.loc[:, 'Punt_Distance'] = play_info.apply(_split_punt_distance, axis=1)
    play_info.loc[:, 'Post_Punt_YardLine'] = play_info.apply(_split_field_position, axis=1)
    play_info.loc[:, 'Post_Punt_FieldSide'] = play_info.Post_Punt_YardLine.apply(lambda x: x.split(' ')[0])
    play_info.loc[:, 'Post_Punt_Own_Territory'] = play_info.apply(_post_punt_territory, axis=1)
    play_info.loc[:, 'Pre_Punt_RelativeYardLine'] = play_info.apply(_start_punt_field_position, axis=1)
    play_info.loc[:, 'Post_Punt_RelativeYardLine'] = play_info.apply(_field_position_punt, axis=1)

    # Extract additional information from play info (home team, away team, score
    # differential, home/away punt identifier).
    play_info.loc[:, 'Home_Team'] = play_info.Home_Team_Visit_Team.apply(lambda x: x.split('-')[0])
    play_info.loc[:, 'Away_Team'] = play_info.Home_Team_Visit_Team.apply(lambda x: x.split('-')[1])
    play_info.loc[:, 'Home_Points'] = play_info.Score_Home_Visiting.apply(lambda x: x.split('-')[0]).astype(int)
    play_info.loc[:, 'Away_Points'] = play_info.Score_Home_Visiting.apply(lambda x: x.split('-')[1]).astype(int)

    def _home_away_punt_bool(row):
        if row.Home_Team == row.Poss_Team:
            return 1
        else:
            return 0

    play_info.loc[:, 'Home_Visit_Team_Punt'] = play_info.apply(_home_away_punt_bool, axis=1)

    def _get_score_differential(row):
        if not row.Home_Visit_Team_Punt:
            return int(row.Away_Points - row.Home_Points)
        else:
            return int(row.Home_Points - row.Away_Points)

    play_info.loc[:, 'Score_Differential'] = play_info.apply(_get_score_differential, axis=1)

    # Update dictionary to include additional set of features.
    data.update({'play_info': play_info})

    return data

def load_data(raw_bool=False):
    """
    When called, this function will load in all of the data and do the relevant
    preprocessing (mainly just a series of merges to link the injury data with a
    few of the other data sources). The output of this function is a dictionary
    with key/value pairs that are labels/DataFrames, respectively.

    Parameters:
        raw_bool: bool (default False)
            Boolean indicating whether you wish to perform the necessary
            preprocessing steps (False) or not (True).
    """

    # Load data.
    game_data = pd.read_csv(f'{DDIR}game_data.csv')
    play_info = pd.read_csv(f'{DDIR}play_information.csv')
    play_role = pd.read_csv(f'{DDIR}play_player_role_data.csv')
    punt_data = pd.read_csv(f'{DDIR}player_punt_data.csv')

    video_injury = pd.read_csv(f'{DDIR}video_footage-injury.csv')
    video_review = pd.read_csv(f'{DDIR}video_review.csv')
    video_control = pd.read_csv(f'{DDIR}video_footage-control.csv')

    if raw_bool:
        pass
    else:
        # Rename columns to match format (between video_injury/video_control and
        # everything else).
        ren_dict = {
            'season': 'Season_Year',
            'Type': 'Season_Type',
            'Home_team': 'Home_Team',
            'gamekey': 'GameKey',
            'playid': 'PlayId'
        }

        video_injury.rename(index=str, columns=ren_dict, inplace=True)
        video_control.rename(index=str, columns=ren_dict, inplace=True)
        video_review.rename(index=str, columns={'PlayID':'PlayId'}, inplace=True)

        # Join video_review to video_injury.
        video_injury = video_injury.merge(video_review, how='outer',
                                          left_on=['Season_Year', 'GameKey', 'PlayId'],
                                          right_on=['Season_Year', 'GameKey', 'PlayId'])

        # Process punt_data - it's possible to have multiple numbers for the same
        # player, so we'll drop number to get rid of duplicates.
        punt_data.drop('Number', axis=1, inplace=True)
        punt_data.drop_duplicates(inplace=True)

        # Add player primary position to video_injury.
        video_injury = video_injury.merge(punt_data, how='inner', on=['GSISID'])
        video_injury.rename(index=str, columns={'Position':'Player_Position'},
                            inplace=True)

        # Fix a few values in Primary_Partner_GSISID that will cause the next
        # merge to barf (one nan, one 'Unclear').
        video_injury.replace(to_replace={'Primary_Partner_GSISID':'Unclear'},
                             value=99999, inplace=True)
        video_injury.replace(to_replace={'Primary_Partner_GSISID':np.nan},
                             value=99999, inplace=True)
        video_injury.loc[:, 'Primary_Partner_GSISID'] = video_injury.Primary_Partner_GSISID.astype(int)

        # Add primary partner primary position to video_injury.
        video_injury = video_injury.merge(punt_data, how='left',
                                          left_on=['Primary_Partner_GSISID'],
                                          right_on=['GSISID'])
        video_injury.drop('GSISID_y', axis=1, inplace=True)
        video_injury.rename(index=str, columns={'GSISID_x':'GSISID'}, inplace=True)
        video_injury.rename(index=str, columns={'Position':'Primary_Partner_Position'},
                            inplace=True)

        # Add punt specific play role for players to video_injury.
        play_role.rename(index=str, columns={'PlayID':'PlayId'}, inplace=True)
        video_injury = video_injury.merge(play_role, how='left',
                                          left_on=['Season_Year', 'GameKey', 'PlayId', 'GSISID'],
                                          right_on=['Season_Year', 'GameKey', 'PlayId', 'GSISID'])
        video_injury.rename(index=str, columns={'Role':'Player_Punt_Role'}, inplace=True)

        # Add punt specific play role for primary partners to video_injury.
        video_injury = video_injury.merge(play_role, how='left',
                                          left_on=['Season_Year', 'GameKey', 'PlayId', 'Primary_Partner_GSISID'],
                                          right_on=['Season_Year', 'GameKey', 'PlayId', 'GSISID'])
        video_injury.drop('GSISID_y', axis=1, inplace=True)
        video_injury.rename(index=str, columns={'GSISID_x':'GSISID'}, inplace=True)
        video_injury.rename(index=str, columns={'Role':'Primary_Partner_Punt_Role'},
                            inplace=True)

    # Stick everything in a dictionary to return as output.
    out_dict = {
        'game_data': game_data,
        'play_info': play_info,
        'play_role': play_role,
        'punt_data': punt_data,
        'video_injury': video_injury,
        'video_control': video_control,
        'video_review': video_review
    }

    return out_dict

def parse_penalties(play_info_df):
    """
    Extract penalty types for plays on which we had penalties.

    Parameters:
        play_info_df: pd.DataFrame
            DataFrame containing play information.
    """

    pen_df = play_info_df.loc[play_info_df.Penalty_on_Punt == 1].reset_index(drop=True)

    def _extract_penalty_type(row):
        try:
            tmp_desc = row.PlayDescription.lower()
            pen_suff = tmp_desc.split('penalty on ')[1]
            drop_plr = pen_suff.split(', ')[1]

            penalty = drop_plr.split(',')[0]
        except:
            penalty = 'EXCEPTION'

        return penalty

    pen_df.loc[:, 'Penalty_Type'] = pen_df.apply(_extract_penalty_type, axis=1)

    return pen_df

def trim_player_partner_data(ngs_df):
    """
    Given a DataFrame with NGS data for player/partner on punt play, cut out
    the relevant NGS data.

    Parameters:
        ngs_df: pd.DataFrame
            DataFrame containing NGS data.
    """

    # Isolate player/partner data.
    play_df = ngs_df.loc[ngs_df.Identifier == 'PLAYER'].dropna().reset_index(drop=True)
    part_df = ngs_df.loc[ngs_df.Identifier == 'PARTNER'].dropna().reset_index(drop=True)

    # Figure out where the ball snap occurred and get the index so that we can
    # discard all data prior to that instant.
    try:
        play_st = play_df.loc[play_df.Event == 'punt'].index[0]
        part_st = part_df.loc[part_df.Event == 'punt'].index[0]
    except IndexError:
        try:
            play_st = play_df.loc[play_df.Event == 'ball_snap'].index[0]
            part_st = part_df.loc[part_df.Event == 'ball_snap'].index[0]
        except IndexError:
            play_st = play_df.index.min()
            part_st = part_df.index.min()

    # Figure out where the play "ended" so that we can discard all data after
    # that. For simplicity, we assume that any concussion event would have occured
    # prior to a penalty flag being thrown or within five seconds of a tackle.
    try:
        play_ei = play_df.loc[play_df.Event == 'tackle'].index[0] + 50
        part_ei = part_df.loc[part_df.Event == 'tackle'].index[0] + 50
        
        play_ps = play_df.loc[play_df.Event == 'play_submit'].index[0]
        part_ps = part_df.loc[part_df.Event == 'play_submit'].index[0]
        
        while play_ei > play_ps:
            play_ei -= 10

        while part_ei > part_ps:
            part_ei -= 10
    except IndexError:
        try:
            play_ei = play_df.loc[play_df.Event == 'penalty_flag'].index[0]
            part_ei = part_df.loc[part_df.Event == 'penalty_flag'].index[0]
        except IndexError:
            play_ei = play_df.index.max()
            part_ei = part_df.index.max()

    # Slice out the data that we actually need.
    play_df = play_df.iloc[play_st:play_ei]
    part_df = part_df.iloc[part_st:part_ei]

    return play_df, part_df
# Load data from plays with concussions.
WDIR = '../input/ngs-dataset-playerpartnerinjuries/'
ngs_data = pd.read_csv(f'{WDIR}injury_ngs_data.csv')

# Add column for easy indexing.
merge_cols = ['Season_Year', 'GameKey', 'PlayID']
ind_df = ngs_data.drop_duplicates(merge_cols).reset_index(drop=True)
ind_df.loc[:, 'eventIndex'] = ind_df.index.values
play_indexes = ind_df.eventIndex.tolist()

ind_df = ind_df.loc[:, ['eventIndex', 'Season_Year', 'GameKey', 'PlayID']]
ngs_data = ngs_data.merge(ind_df, how='inner', left_on=merge_cols, right_on=merge_cols)

# Run to generate animated figure.
figure = {
    'data': [],
    'layout': {},
    'frames': []
}

## CUSTOM
field_xaxis=dict(
        range=[0,120],
        linecolor='black',
        linewidth=2,
        mirror=True,
        showticklabels=False
)
field_yaxis=dict(
        range=[0,53.3],
        linecolor='black',
        linewidth=2,
        mirror=True,
        showticklabels=False
)
field_annotations=[
        dict(
            x=0,
            y=0.5,
            showarrow=False,
            text='HOME ENDZONE',
            textangle=270,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=24,
                color='white'
            )
        ),
        dict(
            x=1,
            y=0.5,
            showarrow=False,
            text='AWAY ENDZONE',
            textangle=90,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=24,
                color='white'
            )
        ),
        dict(
            x=float(17./120.),
            y=1,
            showarrow=False,
            text='10',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(27./120.),
            y=1,
            showarrow=False,
            text='20',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(37./120.),
            y=1,
            showarrow=False,
            text='30',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(50./120.),
            y=1,
            showarrow=False,
            text='40',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(60./120.),
            y=1,
            showarrow=False,
            text='50',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(70./120.),
            y=1,
            showarrow=False,
            text='40',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(80./120.),
            y=1,
            showarrow=False,
            text='30',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(93./120.),
            y=1,
            showarrow=False,
            text='20',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        ),
        dict(
            x=float(103./120.),
            y=1,
            showarrow=False,
            text='10',
            textangle=180,
            xref='paper',
            yref='paper',
            font=dict(
                family='sans serif',
                size=20,
                color='white'
            )
        )
]
field_shapes=[
        {
            'type': 'line',
            'x0': 10,
            'y0': 0,
            'x1': 10,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 2
            },
        },
        {
            'type': 'line',
            'x0': 110,
            'y0': 0,
            'x1': 110,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 2
            },
        },
        {
            'type': 'line',
            'x0': 20,
            'y0': 0,
            'x1': 20,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 30,
            'y0': 0,
            'x1': 30,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 40,
            'y0': 0,
            'x1': 40,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 50,
            'y0': 0,
            'x1': 50,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 60,
            'y0': 0,
            'x1': 60,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 70,
            'y0': 0,
            'x1': 70,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 80,
            'y0': 0,
            'x1': 80,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 90,
            'y0': 0,
            'x1': 90,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 100,
            'y0': 0,
            'x1': 100,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1
            },
        },
        {
            'type': 'line',
            'x0': 15,
            'y0': 0,
            'x1': 15,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 25,
            'y0': 0,
            'x1': 25,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 35,
            'y0': 0,
            'x1': 35,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 45,
            'y0': 0,
            'x1': 45,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 55,
            'y0': 0,
            'x1': 55,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 65,
            'y0': 0,
            'x1': 65,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 75,
            'y0': 0,
            'x1': 75,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 85,
            'y0': 0,
            'x1': 85,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 95,
            'y0': 0,
            'x1': 95,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        },
        {
            'type': 'line',
            'x0': 105,
            'y0': 0,
            'x1': 105,
            'y1': 53.3,
            'line': {
                'color': 'white',
                'width': 1,
                'dash':'dot'
            },
        }
]


# fill in most of layout
figure['layout']['autosize'] = True
figure['layout']['showlegend'] = False
figure['layout']['plot_bgcolor'] = '#008000'

figure['layout']['xaxis'] = field_xaxis
figure['layout']['yaxis'] = field_yaxis
figure['layout']['annotations'] = field_annotations
figure['layout']['shapes'] = field_shapes

figure['layout']['hovermode'] = 'closest'
figure['layout']['sliders'] = {
    'args': [
        'transition', {
            'duration': 400,
            'easing': 'cubic-in-out'
        }
    ],
    'initialValue': 0,
    'plotlycommand': 'animate',
    'values': play_indexes,
    'visible': True
}
figure['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]

sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Play Index: ',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

# Make data (for single play).
plt_dicts = []
pidx = 0

sp_data = ngs_data.loc[ngs_data.eventIndex == pidx].reset_index(drop=True)

# Grab some stuff for labeling saved figure.
sy = sp_data.Season_Year.values[0]
gk = sp_data.GameKey.values[0]
pi = sp_data.PlayID.values[0]

plt_dict = {}
plt_dict['playIndex'] = pi
plt_dict['seasonYear'] = sy
plt_dict['gameKey'] = gk
plt_dict['playID'] = pi
plt_dicts.append(plt_dict)

# Get player/partner data (reduced).
rd_play_df, rd_part_df = trim_player_partner_data(sp_data)

for i in range(2):
    if not i:
        ngs_dataset = rd_play_df.copy()
        plt_name = 'Player'
        color_scale = 'Reds'
        cb_loc = 1.0
    else:
        ngs_dataset = rd_part_df.copy()
        plt_name = 'Partner'
        color_scale = 'Blues'
        cb_loc = 1.1

    data_dict = {
        'x': list(ngs_dataset['x']),
        'y': list(ngs_dataset['y']),
        'mode': 'markers',
        'marker': {
            'color': list(ngs_dataset['s']),
            'colorbar': {'x':cb_loc},
            'colorscale':color_scale,
            'size':12
        },
        'name':plt_name
    }
    
    if i == 1:
        data_dict['marker']['reversescale'] = True
        
    figure['data'].append(data_dict)

# Add data for yardline trace.
data_dict = {
    'x': [20, 30, 40, 50, 60, 70, 80, 90, 100],
    'y': [1, 1, 1, 1, 1, 1, 1, 1, 1],
    'mode': 'text',
    'text': ['10','20','30','40','50','40','30','20','10'],
    'textposition': 'top center',
    'textfont': {
        'family': 'sans serif',
        'size': 20,
        'color': 'white'
    }
}
figure['data'].append(data_dict)

# Make frames.
for pidx in play_indexes:
    frame = {'data': [], 'name': pidx}
    try:
        sp_data = ngs_data.loc[ngs_data.eventIndex == pidx].reset_index(drop=True)

        # Grab some stuff for labeling saved figure.
        sy = sp_data.Season_Year.values[0]
        gk = sp_data.GameKey.values[0]
        pi = sp_data.PlayID.values[0]

        plt_dict = {}
        plt_dict['playIndex'] = pi
        plt_dict['seasonYear'] = sy
        plt_dict['gameKey'] = gk
        plt_dict['playID'] = pi
        plt_dicts.append(plt_dict)

        # Get player/partner data (reduced).
        rd_play_df, rd_part_df = trim_player_partner_data(sp_data)

        for i in range(2):
            if not i:
                ngs_dataset = rd_play_df.copy()
                plt_name = 'Player'
                color_scale = 'Reds'
                cb_loc = 1.0
            else:
                ngs_dataset = rd_part_df.copy()
                plt_name = 'Partner'
                color_scale = 'Blues'
                cb_loc = 1.1

            data_dict = {
                'x': list(ngs_dataset['x']),
                'y': list(ngs_dataset['y']),
                'mode': 'markers',
                'marker': {
                    'color': list(ngs_dataset['s']),
                    'colorbar': {'x':cb_loc},
                    'colorscale':color_scale,
                    'size':12
                },
                'name':plt_name
            }
            
            if i == 1:
                data_dict['marker']['reversescale'] = True
                
            frame['data'].append(data_dict)

        # Add data for yardline trace.
        data_dict = {
            'x': [20, 30, 40, 50, 60, 70, 80, 90, 100],
            'y': [1, 1, 1, 1, 1, 1, 1, 1, 1],
            'mode': 'text',
            'text': ['10','20','30','40','50','40','30','20','10'],
            'textposition': 'top center',
            'textfont': {
                'family': 'sans serif',
                'size': 20,
                'color': 'white'
            }
        }
        frame['data'].append(data_dict)

        # Add frames.
        figure['frames'].append(frame)
        slider_step = {'args': [
            [pidx],
            {'frame': {'duration': 300, 'redraw': False},
             'mode': 'immediate',
           'transition': {'duration': 300}}
         ],
         'label': pidx,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)
    except TypeError:
        continue

figure['layout']['sliders'] = [sliders_dict]
iplot(figure)
## FUNCTIONS 
def ecdf(data):
    """
    Compute empirical cumulative distribution function for set of 1D data.

    Parameters:
        data: np.array
    Returns:
        x: np.array
            Sorted data.
        y: np.array
            ECDF(x).
    """

    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n

    return x, y

def make_histogram(ngs_df, col_to_plot, title, add_lines=False):
    """
    Generate plotly histogram using maximum accelerations experienced by
    players.

    Parameters:
        ngs_df: pd.DataFrame
            DataFrame containing NGS data.
        col_to_plot: str
            Name of column to plot.
        title: str 
            Title to display over histogram. 
        add_lines: bool (default False)
            Boolean indicating whether we wish to superpose lines for values
            from concussed players.
    """

    trace = (
        go.Histogram(
            x=ngs_df[col_to_plot].values,
            histnorm='probability',
            nbinsx=100
        )
    )

    layout = go.Layout(
                title=title, 
                autosize=True,
                xaxis=dict(
                    range=[0,100],
                    title='Maximum Acceleration (m/s^2)'
                ),
                yaxis=dict(
                    title='Normalized Density'
                )
             )

    if add_lines:
        _ = 'placeholder'
    else:
        data = [trace]

    fig = go.Figure(data=data, layout=layout)

    return fig

def plot_ecdf(int_ecdf, inj_ecdf, title, unit, max_x):
    """
    Plot ECDF from entire set of play data and from the subset where a concussion
    occurred.

    Parameters:
        int_ecdf: np.array (values: [quantity, ecdf])
            Values from interpolated ECDF from entire dataset.
        inj_ecdf: np.array (values: [quantity, ecdf])
            ECDF/values for players in the injury set. 
        title: str 
            Title to display over figure. 
        unit: str 
            Unit to stick on horizontal axis. 
        max_x: int
            Max x-value (used for plot range).
    """

    int_trace = go.Scatter(
                    x = int_ecdf[:,0],
                    y = int_ecdf[:,1],
                    mode = 'markers',
                    marker = dict(opacity=0.3)
                )

    inj_trace = go.Scatter(
                    x = inj_ecdf[:,0],
                    y = inj_ecdf[:,1],
                    mode = 'markers',
                    marker = dict(color='red')
                )

    data = [int_trace, inj_trace]

    layout = go.Layout(
                title=title,
                autosize=True,
                showlegend=False,
                xaxis=dict(
                    range=[0,max_x],
                    title=f'{title} ({unit})'
                ),
                yaxis=dict(
                    title='CDF'
                )
             )

    fig = go.Figure(data=data, layout=layout)

    return fig
# Load data.
WDIR = '../input/ngs-data-summary-statistics/sumdynamics/sumdynamics/'
files = glob.glob(f'{WDIR}*.csv')
flist = []

for f in files:
    tmp_df = pd.read_csv(f)
    flist.append(tmp_df)

sum_df = pd.concat(flist, ignore_index=True)

# Drop unreasonable accelerations.
sum_df = sum_df.loc[sum_df.max_a <= 150.]

# Construct ECDF for players.
srt_spds, spd_ecdf = ecdf(sum_df.max_s.values)
int_s_ecdf = interpolate.interp1d(srt_spds, spd_ecdf)
srt_accs, acc_ecdf = ecdf(sum_df.max_a.values)
int_a_ecdf = interpolate.interp1d(srt_accs, acc_ecdf)

# Load in set of summary statistics for players involved in concussions.
WDIR = '../input/ngs-data-speedacceleration-summary-stats/'
inj_df = pd.read_csv(f'{WDIR}spd_acc_summary.csv')

ren_dict = {'season_year':'Season_Year', 'game_key':'GameKey',
            'play_id': 'PlayID'}
inj_df.rename(index=str, columns=ren_dict, inplace=True)

# Add column for player/partner action.
WDIR = '../input/NFL-Punt-Analytics-Competition/'
ppa_df = pd.read_csv(f'{WDIR}video_review.csv')
ppa_df = ppa_df.loc[:, ['Season_Year', 'GameKey', 'PlayID', 'Player_Activity_Derived', 'Primary_Partner_Activity_Derived']]

mer_cols = ['Season_Year', 'GameKey', 'PlayID']
inj_df = inj_df.merge(ppa_df, how='inner', left_on=mer_cols, right_on=mer_cols)

def _identify_moving_pp(row):
    player_activity = row.Player_Activity_Derived

    if 'ing' in player_activity:
        return 1
    elif 'ed' in player_activity:
        return 0
    else:
        raise ValueError('Check derived activity!')

def _moving_play_s(row, dyn_opt):
    pp_activity = row.PP_Activity

    if dyn_opt == 's':
        return pp_activity * row.max_play_s + abs(pp_activity - 1) * row.max_part_s
    elif dyn_opt == 'a':
        return pp_activity * row.max_play_a + abs(pp_activity - 1) * row.max_part_a
    else:
        raise ValueError('Check derived activity!')

inj_df.loc[:, 'PP_Activity'] = inj_df.apply(_identify_moving_pp, axis=1)
inj_df.loc[:, 'max_move_s'] = inj_df.apply(_moving_play_s, args=('s'), axis=1)
inj_df.loc[:, 'max_move_a'] = inj_df.apply(_moving_play_s, args=('a'), axis=1)

def _get_s_cdf(x):
    return int_s_ecdf(x)

def _get_a_cdf(x):
    return int_a_ecdf(x)

inj_df.loc[:, 's_cumprob'] = inj_df.max_play_s.apply(_get_s_cdf)
inj_df.loc[:, 'a_cumprob'] = inj_df.max_play_a.apply(_get_a_cdf)
inj_df.loc[:, 's_part_cumprob'] = inj_df.max_part_s.apply(_get_s_cdf)
#inj_df.loc[:, 'a_part_cumprob'] = inj_df.max_part_a.apply(_get_a_cdf)
# Make figure (histogram), then plot.
plotcol = 'max_a'

figure = make_histogram(sum_df, plotcol, 'Maximum Acceleration')
iplot(figure, filename='acc-histogram')
# Make figure (acceleration ECDF), then plot.
a_vals = np.linspace(min(srt_accs), max(srt_accs),200)
a_ecdf = np.array([int_a_ecdf(x) for x in a_vals])
a_ecdf = np.vstack([a_vals,a_ecdf]).T

inj_a_data = inj_df.loc[:, ['max_play_a', 'a_cumprob']].values

figure = plot_ecdf(a_ecdf, inj_a_data, 'Maximum Player Acceleration', 'm/s^2', 60)
iplot(figure, filename='acc-ecdf-play')
# Make figure (speed ECDF), then plot.
s_vals = np.linspace(min(srt_spds), max(srt_spds),200)
s_ecdf = np.array([int_s_ecdf(x) for x in s_vals])
s_ecdf = np.vstack([s_vals,s_ecdf]).T

inj_s_data = inj_df.loc[:, ['max_play_s', 's_cumprob']].values

figure = plot_ecdf(s_ecdf, inj_s_data, 'Maximum Player Speed', 'm/s', 10)
iplot(figure, filename='spd-ecdf')
# Make figure (speed ECDF), then plot.
s_vals = np.linspace(min(srt_spds), max(srt_spds),200)
s_ecdf = np.array([int_s_ecdf(x) for x in s_vals])
s_ecdf = np.vstack([s_vals,s_ecdf]).T

inj_s_data = inj_df.loc[:, ['max_part_s', 's_part_cumprob']].values

figure = plot_ecdf(s_ecdf, inj_s_data, 'Maximum Partner Speed', 'm/s', 10)
iplot(figure, filename='spd-ecdf')
## FUNCTIONS
def calculate_pp_distance(ngs_df):
    """
    Given the NGS data for the injury set, calculate player-partner distance.

    Parameters:
        ngs_df: pd.DataFrame
            DataFrame containing NGS data for player/partner on plays with a
            concussion.
    """

    # Split player/partner data.
    play_df = ngs_df.loc[ngs_df.Identifier == 'PLAYER']
    part_df = ngs_df.loc[ngs_df.Identifier == 'PARTNER']
    play_df = play_df.drop(['Identifier'], axis=1)
    part_df = part_df.drop(['Identifier'], axis=1)

    # Rename columns for ease of merge.
    ren_cols = ['GSISID', 'x', 'y', 'dis', 'o', 'dir', 'vx', 'vy', 's', 'ax',
                'ay', 'a', 't']
    play_cols = {x:f'play_{x}' for x in ren_cols}
    part_cols = {x:f'part_{x}' for x in ren_cols}
    play_df.rename(index=str, columns=play_cols, inplace=True)
    part_df.rename(index=str, columns=part_cols, inplace=True)

    # Perform merge.
    mer_cols = ['Season_Year', 'GameKey', 'PlayID', 'Event',
                'eventIndex', 'Time']
    pp_df = play_df.merge(part_df, how='inner', left_on=mer_cols, right_on=mer_cols)

    # Add extra columns that will assist with determining when tackle was made.
    pp_df.loc[:, 'diff_x'] = pp_df.part_x - pp_df.play_x
    pp_df.loc[:, 'diff_y'] = pp_df.part_y - pp_df.play_y
    pp_df.loc[:, 'pp_dis'] = np.sqrt(np.square(pp_df.diff_x.values) + np.square(pp_df.diff_y.values))

    return pp_df

def find_impact(pp_df):
    """
    Given a DataFrame containing player-parter NGS data (for a single play),
    identify the most likely time of impact and return all data from one second
    around that time of impact.

    Parameters:
        pp_df: pd.DataFrame
            NGS data for player/partner pair.
    """

    # Figure out where the ball snap occurred and get the index so that we can
    # discard all data prior to that instant.
    try:
        play_st = pp_df.loc[pp_df.Event == 'punt'].index[0]
    except IndexError:
        try:
            play_st = pp_df.loc[pp_df.Event == 'ball_snap'].index[0]
        except IndexError:
            play_st = pp_df.index.min()

    # Figure out where the play "ended" so that we can discard all data after
    # that. For simplicity, we assume that any concussion event would have occured
    # prior to a penalty flag being thrown or within five seconds of a tackle.
    try:
        play_ei = pp_df.loc[pp_df.Event == 'penalty_flag'].index[0]
    except IndexError:
        try:
            play_ei = pp_df.loc[pp_df.Event == 'tackle'].index[0] + 50

            play_ps = pp_df.loc[pp_df.Event == 'play_submit'].index[0]

            while play_ei > play_ps:
                play_ei -= 10

        except IndexError:
            play_ei = pp_df.index.max()

    # Slice out the data that we actually need.
    pp_df = pp_df.iloc[play_st:play_ei].reset_index(drop=True)

    # Find the DataFrame index corresponding to the minimum player-partner
    # distance.
    min_dis_index = int(pp_df.loc[pp_df.pp_dis == pp_df.pp_dis.min()].index[0])

    # Grab a few rows around that index.
    min_dis_df = pp_df.iloc[(min_dis_index-5):(min_dis_index+5)]
    min_dis_df.loc[min_dis_index,'impact'] = 1
    min_dis_df.fillna(0, inplace=True)
    min_dis_df.reset_index(drop=True, inplace=True)

    return min_dis_df

def make_radial_plot(plot_df, angle_opt, title):
    """
    Make a radial plot using a DataFrame built specifically for this purpose
    (i.e., expected column names are hard-coded).

    Parameters:
        plot_df: pd.DataFrame
            DataFrame containing data to be plotted.
        angle_opt: str
            Angle to plot ('o', 'dir', 'dir_tt').
        title: str 
            Title to display over figure. 
    """

    if angle_opt == 'o':
        plt_col = 'pp_o_diff'
        color = 'orange'
    elif angle_opt == 'dir':
        plt_col = 'pp_dir_diff'
        color = 'green'
    elif angle_opt == 'dir_tt':
        plt_col = 'pp_dir_diff'

        impact_types = plot_df.Primary_Impact_Type.tolist()
        imp_col_dict = {'Helmet-to-helmet':'red', 'Helmet-to-body':'blue'}
        color = [imp_col_dict[x] for x in impact_types]

        #pa_types = plot_df.Player_Activity_Derived.tolist()
        #pa_col_dict = {
        #    'Blocking': 'red', 'Blocked': 'orange', 'Tackling': 'green',
        #    'Tackled': 'blue'
        #}
        #color = [pa_col_dict[x] for x in pa_types]
    else:
        raise ValueError('Not a valid option!')

    angles = plot_df.loc[:,plt_col].values

    def _fix_domain(ang):
        if ang < 0:
            return 360. - abs(ang)
        else:
            return ang

    radii = plot_df.acc_rank.astype(float).tolist()
    fix_angles = [_fix_domain(x) for x in angles]

    data = [
        go.Scatterpolar(
            r = radii,
            theta = fix_angles,
            mode = 'markers',
            marker = dict(
                color = color
            )
        )
    ]

    layout = go.Layout(
        title=title,
        showlegend = False,
        font=dict(
            family='sans serif',
            size=24,
            color='black'
        ),
        polar = dict(
            radialaxis = dict(
                showticklabels = False,
                showline=False,
                ticklen=0
            )
        )
    )

    fig = go.Figure(data=data,layout=layout)

    return fig
# Load data.
WDIR = '../input/ngs-dataset-playerpartnerinjuries/'
inj_df = pd.read_csv(f'{WDIR}injury_ngs_data.csv')

# Add column for easy indexing.
merge_cols = ['Season_Year', 'GameKey', 'PlayID']
ind_df = inj_df.drop_duplicates(merge_cols).reset_index(drop=True)
ind_df.loc[:, 'eventIndex'] = ind_df.index.values

ind_df = ind_df.loc[:, ['eventIndex', 'Season_Year', 'GameKey', 'PlayID']]
inj_df = inj_df.merge(ind_df, how='outer', left_on=merge_cols, right_on=merge_cols)

# Get player-partner processed DataFrame.
play_part_df = calculate_pp_distance(inj_df)

# Step through each play, identifying the most likely point of impact and
# grabbing a few rows around it.
impacts = []

for play_idx in range(len(inj_df.eventIndex.unique())):
    try:
        sp_df = play_part_df.loc[play_part_df.eventIndex == play_idx].reset_index(drop=True)

        # Find most probable time for impact between player/partner.
        impact_df = find_impact(sp_df)
        impacts.append(impact_df)
    except TypeError:
        continue

pp_impact_df = pd.concat(impacts, ignore_index=True)

# Add angle difference columns.
pp_impact_df.loc[:, 'pp_dir_diff'] = pp_impact_df.play_dir - pp_impact_df.part_dir
pp_impact_df.loc[:, 'pp_o_diff'] = pp_impact_df.play_o - pp_impact_df.part_o

# Isolate most likely time step for impact.
just_impact = pp_impact_df.loc[pp_impact_df.impact == 1]

# Bring in summary stats for speed/acceleration (to be used on plot).
WDIR = '../input/ngs-data-speedacceleration-summary-stats/'
spd_acc_ss = pd.read_csv(f'{WDIR}spd_acc_summary.csv')
spd_acc_ss.rename(index=str, columns={'season_year': 'Season_Year', 'game_key': 'GameKey', 'play_id': 'PlayID'}, inplace=True)

just_impact = just_impact.merge(spd_acc_ss, how='inner', left_on=merge_cols,
                                right_on=merge_cols)

# Bring in impact types/player activity.
WDIR = '../input/NFL-Punt-Analytics-Competition/'
impact_type = pd.read_csv(f'{WDIR}video_review.csv')
impact_type = impact_type.loc[:, ['Season_Year', 'GameKey', 'PlayID', 'Player_Activity_Derived', 'Primary_Impact_Type']]

just_impact = just_impact.merge(impact_type, how='inner', left_on=merge_cols,
                                right_on=merge_cols)

# Pluck out the columns relevant for plotting.
#keep_columns = ['max_play_a', 'pp_dir_diff', 'pp_o_diff']
#plot_df = just_impact.loc[:, keep_columns].sort_values(by='max_play_a').reset_index(drop=True)
plot_df = just_impact.sort_values(by='max_play_a').reset_index(drop=True)
plot_df.loc[:, 'acc_rank'] = (plot_df.index.values+1)/(plot_df.index.max())
# Make orientation plot. 
plt_opt = 'o'
figure = make_radial_plot(plot_df, plt_opt, 'Player-Partner Orientation')
iplot(figure, filename='pp-orientation')
# Make direction plot. 
plt_opt = 'dir'
figure = make_radial_plot(plot_df, plt_opt, 'Player-Partner Direction')
iplot(figure, filename='pp-direction')
# Make direction plot that utilizes tackle type. 
plt_opt = 'dir_tt'
figure = make_radial_plot(plot_df, plt_opt, 'Player-Partner Direction')
iplot(figure, filename='pp-direction-tackletype')
## FUNCTIONS
def perform_ks_test(pop_df, sam_df, col_of_interest):
    """
    Perform Kolmogorov-Smirnov test for population (all punts) and sample
    (punts with concussions) distribution of provided quantity (col_of_interest).

    Parameters:
        pop_df: pd.DataFrame
            DataFrame containing data from all selected punts (excluding the
            ones in the concussion set).
        sam_df: pd.DataFrame
            DataFrame containing data from concussion set.
        col_of_interest: str
            Name of column/quantity for which you'd like to conduct the KS test.
    """

    pdata = pop_df.loc[:, col_of_interest].values
    sdata = sam_df.loc[:, col_of_interest].values

    stat, p = stats.ks_2samp(pdata, sdata)

    return (stat, p)

def plot_distribution(pop_df, sam_df, col_of_interest, plot_hp, title):
    """
    Plot distribution of quantity (col_of_interest) for population/sample.

    Parameters:
        pop_df: pd.DataFrame
            DataFrame containing data from all selected punts (excluding the
            ones in the concussion set).
        sam_df: pd.DataFrame
            DataFrame containing data from concussion set.
        col_of_interest: str
            Name of column/quantity that you'd like to plot.
        plot_hp: tuple (ints/floats)
            Hyperparameter for plotting (bandwidth for KDE, number of bins for
            histogram). Index 0 contains population hyperparameter, Index 1
            contains sample hyperparameter. 
        title: str 
            Title to display over figure. 
    """

    pdata = pop_df.loc[:, col_of_interest].values
    sdata = sam_df.loc[:, col_of_interest].values

    # Uncomment for histograms.
    pop_trace = go.Histogram(
                    x=pdata,
                    name='Population',
                    opacity=0.75,
                    marker=dict(color='red'),
                    histnorm='probability',
                    nbinsx=plot_hp[0]
                )
    sam_trace = go.Histogram(
                    x=sdata,
                    name='Sample (concussions)',
                    opacity=0.75,
                    marker=dict(color='blue'),
                    histnorm='probability',
                    nbinsx=plot_hp[1]
                )

    """
    # Uncomment for KDEs.
    # Make KDE for each sample.
    pop_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(pdata[:, np.newaxis])
    pop_plot = np.linspace(np.min(pdata), np.max(pdata), 1000)[:, np.newaxis]
    plog_dens = pop_kde.score_samples(pop_plot)

    sam_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(sdata[:, np.newaxis])
    sam_plot = np.linspace(np.min(sdata), np.max(sdata), 1000)[:, np.newaxis]
    slog_dens = sam_kde.score_samples(sam_plot)

    pop_trace = go.Scatter(
                    x=pop_plot[:, 0],
                    y=np.exp(plog_dens),
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='red', width=2)
                )

    sam_trace = go.Scatter(
                    x=sam_plot[:, 0],
                    y=np.exp(slog_dens),
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='blue', width=2)
                )
    """

    # Make figure.
    fig = tools.make_subplots(rows=2,cols=1,shared_xaxes=True)
    fig.append_trace(pop_trace, 1, 1)
    fig.append_trace(sam_trace, 2, 1)
    fig['layout'].update(
        barmode='overlay', 
        title=title, 
        xaxis=dict(title='Receiving Team Points - Punting Team Points'),
        yaxis1=dict(title='Normalized Density'),
        yaxis2=dict(title='Normalized Density')
    )

    #data = [pop_trace, sam_trace]
    #layout = go.Layout(barmode='overlay')
    #fig = go.Figure(data=data, layout=layout)

    return fig
# Load data.
data_dict = load_data()
data_dict = collect_outcomes(data_dict)
data_dict = expand_play_description(data_dict)

# Focus on play information.
play_info = data_dict['play_info']
outcomes = ['return', 'downed', 'muffed punt']
play_info = play_info.loc[play_info.Punt_Outcome.isin(outcomes)].reset_index(drop=True)
play_info.loc[:, 'playIndex'] = play_info.index.values

# Split out the plays on which we had an identified concussion.
inj_df = data_dict['video_injury']
inj_df.loc[:, 'concussionPlay'] = 1
drop_cols = ['Home_Team', 'Visit_Team', 'Qtr', 'PlayDescription', 'Week']
inj_df.drop(drop_cols, axis=1, inplace=True)
inj_df.rename(index=str, columns={'PlayId':'PlayID'}, inplace=True)

# Join onto play_info.
mer_cols = ['Season_Year', 'Season_Type', 'GameKey', 'PlayID']

inj_play_info = play_info.merge(inj_df, how='inner', left_on=mer_cols,
                                right_on=mer_cols)

# Exclude plays from injury set from population set.
play_info = play_info.loc[~play_info.playIndex.isin(inj_play_info.playIndex.tolist())].reset_index(drop=True)
cols_of_interest = ['Quarter', 'Score_Differential', 'Pre_Punt_RelativeYardLine', 'Post_Punt_RelativeYardLine']
hp_dict = {'Quarter':(5,4), 'Score_Differential':(20,20), 'Pre_Punt_RelativeYardLine': (20,20),
           'Post_Punt_RelativeYardLine': (20,20)}

# Perform KS Test for quantities of interest. 
print('Kolmogorov-Smirnov Test Results:')

for col in cols_of_interest:
    # Drop some values.
    if col == 'Post_Punt_RelativeYardLine':
        pop_info = play_info.loc[play_info.Post_Punt_RelativeYardLine != -999]
    else:
        pop_info = play_info.copy()

    ks_stat, pval = perform_ks_test(pop_info, inj_play_info, col)
    print(f'quantity: {col}, test-statistic: {ks_stat}, p-value:{pval}')
# Generate plot for score difference. 
col = 'Score_Differential'
figure = plot_distribution(pop_info, inj_play_info, col, hp_dict[col], 'Score Difference')
iplot(figure, filename='score-difference-dist')
# Make heatmap for score difference using population set. 
# Split score differential into bins.
def _bin_score_differential(row):
    row_sd = row.Score_Differential

    if row_sd <= -21:
        return '< -21'
    elif (row_sd > -21) & (row_sd <= -14):
        return '-20 to -14'
    elif (row_sd > -14) & (row_sd <= -7):
        return '-13 to -7'
    elif (row_sd > -7) & (row_sd <= -1):
        return '-6 to -1'
    elif row_sd == 0:
        return 'TIE'
    elif (row_sd > 0) & (row_sd < 7):
        return '+1 to +6'
    elif (row_sd >= 7) & (row_sd < 14):
        return '+7 to +13'
    elif (row_sd >= 14) & (row_sd < 21):
        return '+14 to +20'
    elif row_sd >= 21:
        return '> +21'
    else:
        return 'ERROR'

play_info.loc[:, 'binSD'] = play_info.apply(_bin_score_differential, axis=1)

# Tally up outcomes as a function of score differential.
reorder_indices = ['< -21', '-20 to -14', '-13 to -7', '-6 to -1', 'TIE',
                   '+1 to +6', '+7 to +13', '+14 to +20', '> +21']
binned_outcomes = play_info.pivot_table(index='binSD', columns='Punt_Outcome',
                                        aggfunc='size', fill_value=0)
test = binned_outcomes.copy()
binned_outcomes = binned_outcomes.div(binned_outcomes.sum(axis=1), axis=0)
binned_outcomes = binned_outcomes.reindex(reorder_indices)

# Plot!
trace = go.Heatmap(
            z=binned_outcomes.values.T,
            x=binned_outcomes.index.tolist(),
            y=binned_outcomes.columns.tolist())
data=[trace]

layout = go.Layout(
            title='Score Difference (Punt Outcomes)',
            xaxis=dict(title='Receiving Team Points - Punting Team Points')
        )

figure = go.Figure(data=data, layout=layout)

iplot(data, filename='sd-heatmap')
