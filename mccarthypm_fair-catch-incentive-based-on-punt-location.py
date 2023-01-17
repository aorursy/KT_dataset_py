import os
from collections import namedtuple
import pandas as pd
import numpy as np

# data visualization modules
from bokeh.io import output_notebook, show, push_notebook
from bokeh.layouts import layout, column, widgetbox
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter, CustomJS
from bokeh.models.glyphs import ImageURL
from bokeh.models.ranges import Range1d
from bokeh.models.widgets import Slider
from bokeh.palettes import Category20
from bokeh.plotting import figure, Figure
from IPython.core.display import display, HTML
from ipywidgets import interact
output_notebook()
data_files = [file for file in os.listdir('../input/') if file.endswith('.csv') and not file.startswith('NGS')]
data_files
GUNNERS = ['GL', 'GLi', 'GLo', 'GR', 'GRi', 'GRo']
SHIELD = ['PC', 'PPL', 'PPLi', 'PPLo','PPR', 'PPRi', 'PPRo']
PUNT_LINE = ['PLG', 'PLS', 'PLT', 'PRG', 'PRT']
WINGS = ['PLW', 'PRW']
COVERAGE_TEAM = GUNNERS + SHIELD + WINGS + PUNT_LINE + ['P']

PUNT_RUSH = ['PDL1', 'PDL2', 'PDL3', 'PDL4', 'PDL5', 'PDL6', 'PDM',
             'PDR1', 'PDR2', 'PDR3', 'PDR4', 'PDR5', 'PDR6']
JAMMERS = ['VL', 'VLi', 'VLo', 'VR', 'VRi', 'VRo']
LINEBACKERS = ['PLL', 'PLL1', 'PLL2', 'PLL3', 'PLM',
               'PLM1', 'PLR', 'PLR1', 'PLR2', 'PLR3']
RETURNERS = ['PFB', 'PR']
RETURN_TEAM = PUNT_RUSH + JAMMERS + LINEBACKERS + RETURNERS


def team_lookup(role):
    if role in COVERAGE_TEAM:
        return 'COVERAGE'
    return 'RETURN'


def role_group_lookup(role):
    if role in GUNNERS:
        return 'GUNNER'
    elif role in SHIELD:
        return 'SHIELD'
    elif role in PUNT_LINE:
        return 'PUNT_LINE'
    elif role in WINGS:
        return 'WINGS'
    elif role in PUNT_RUSH:
        return 'PUNT_RUSH'
    elif role in JAMMERS:
        return 'JAMMERS'
    elif role in LINEBACKERS:
        return 'LINEBACKERS'
    elif role in RETURNERS:
        return 'RETURNERS'
    elif role == 'P':
        return 'PUNTER'
    else:
        return 'UNKNOWN ROLE'


roles = pd.read_csv('../input/play_player_role_data.csv').drop(columns=['Season_Year'])
roles['Team'] = roles['Role'].apply(team_lookup)
roles['Role_Group'] = roles['Role'].apply(role_group_lookup)
roles.head()
partner_roles = roles.rename(columns={'Role':'Partner_Role',
                                      'Team':'Partner_Team',
                                      'Role_Group':'Partner_Role_Group',
                                      'GSISID':'Primary_Partner_GSISID'})
injury_footage = pd.read_csv('../input/video_footage-injury.csv')[['gamekey','playid','PREVIEW LINK (5000K)']]\
                   .rename(columns={"gamekey": "GameKey",
                                    "playid": "PlayID",
                                    "PREVIEW LINK (5000K)": "Video"})

injuries = pd.read_csv('../input/video_review.csv')
injuries['Primary_Partner_GSISID'] = injuries['Primary_Partner_GSISID'].replace(['Unclear', None], 0).astype(np.int64)
injuries = injuries.drop(['Turnover_Related'], axis=1)\
                   .merge(injury_footage, how='left', on=['GameKey','PlayID'], validate='1:1')\
                   .merge(roles, how='left', on=['GameKey','PlayID','GSISID'], validate='1:1')\
                   .merge(partner_roles, how='left', on=['GameKey','PlayID','Primary_Partner_GSISID'])
injuries.head()
role_group_injuries = roles.merge(injuries[['GameKey','PlayID','GSISID','Player_Activity_Derived']],
                                     how="left", on=['GameKey','PlayID','GSISID'])\
                     .rename(columns={'PlayID': 'Plays Participated',
                                      'Player_Activity_Derived': 'Injuries'})\
                     .groupby('Role_Group')[['Plays Participated','Injuries']]\
                     .count()

role_group_injuries['Injuries per 10k Plays'] = 10000 * role_group_injuries['Injuries'] / role_group_injuries['Plays Participated']
role_group_injuries.sort_values('Injuries per 10k Plays', ascending=False, inplace=True)
display(role_group_injuries)
display(injuries.groupby('Team')['PlayID'].count())
role_injuries = roles.merge(injuries[['GameKey','PlayID','Player_Activity_Derived']],
                                     how="left", on=['GameKey','PlayID'])\
                   .rename(columns={'PlayID': 'Plays Participated',
                                      'Player_Activity_Derived': 'Injuries'})\
                   .groupby('Role')[['Plays Participated','Injuries']]\
                   .count()

role_injuries['Injuries per 10k Plays'] = 10000 * role_injuries['Injuries'] / role_injuries['Plays Participated']
role_injuries.sort_values('Injuries per 10k Plays', ascending=False, inplace=True)
display(role_injuries)
player_data = pd.read_csv('../input/player_punt_data.csv')
player_data.head()
injury = injuries.loc[5]
print(injury)
print('\n')
print(player_data[player_data['GSISID']==injury['GSISID']])
print(player_data[player_data['GSISID']==injury['Primary_Partner_GSISID']])
HTML("""
<video width="640" height="480" controls>
  <source src="{}" type="video/mp4">
</video>
""".format(injury['Video']))
def get_play_summary(ngs_df, role_df):
    """
    Given NGS Data, and roles from each play collect a summary of
    interesting statistics about each play
    """
    Moment = namedtuple('Moment', 'event role')

    INTERESTING_MOMENTS = [
        Moment('ball_snap', 'PLS'),
        Moment('punt_received', 'PR'),
        Moment('fair_catch', 'PR'),
        Moment('tackle', 'PR'),
        Moment('out_of_bounds', 'PR'),
        Moment('punt_downed', 'PR'),
        Moment('touchback', 'PR'),
        Moment('touchdown', 'PR'),
    ]

    # gets the location of a player at the moment of an event
    def get_player_location(ngs_df, role_df, event, role, *args):
        """
        Given NGS Data, and roles from each play, identify the location
        of the player(s) with a given role at a specific event
        """
        event_df = ngs_df[ngs_df['Event'] == event].copy()
        event_df.drop_duplicates(subset=['GameKey','PlayID','GSISID'], keep=False, inplace=True)
        player_df = role_df[role_df['Role'] == role].copy()
        player_df.drop_duplicates(subset=['GameKey','PlayID','Role'], keep=False, inplace=True)
        event_locations = event_df.merge(player_df, how='inner', on=['GameKey','PlayID','GSISID'], validate='1:1')
        return event_locations[['GameKey', 'PlayID', 'Time', 'x', 'y']]\
                              .rename(index=str, columns={"x": event + "_x",
                                                          "y": event + "_y",
                                                          "Time": event + "_time"})

    # get all punts and initialize the punt_plays DataFrame
    punt_plays = df = get_player_location(ngs_df, role_df, *Moment('punt', 'P'))

    # for all additional moments, left join onto punt plays
    for moment in INTERESTING_MOMENTS:
        df = get_player_location(ngs_df, role_df, *moment)
        punt_plays = punt_plays.merge(df, how='left', on=['GameKey','PlayID'])

    def orient_field(row):
        """Orient the field so that the punting team is punting from x=0 to x=120"""
        if row['punt_x'] > row['ball_snap_x']:
            for col in row.index:
                if col.endswith('_x'):
                    row[col] = 120 - row[col]
        return row
    
    def get_final_event(row):
        """Gets the final event of the play from the following list of alternatives"""
        ending_events = ['tackle', 'out_of_bounds', 'punt_downed', 'touchback', 'fair_catch', 'touchdown']
        for event in ending_events:
            if not pd.isna(row[event + '_x']):
                row['final_event'] = event
        return row
    
    punt_plays = punt_plays.apply(orient_field, axis=1)\
                           .apply(get_final_event, axis=1)

    punt_plays['hang_time'] = punt_plays['punt_received_time'] - punt_plays['punt_time']
    punt_plays['hang_time'] = punt_plays['hang_time'].dt.total_seconds()

    return punt_plays

def get_play_summaries(role_df):
    play_summaries = pd.DataFrame()
    for file in os.listdir('../input/'):
        if file.endswith('.csv') and file.startswith('NGS'):
            ngs_data = pd.read_csv('../input/' + file)
            ngs_data['Time'] = pd.to_datetime(ngs_data['Time'])
            play_summary = get_play_summary(ngs_data, role_df)
            play_summaries = pd.concat([play_summaries, play_summary], sort=False)
    play_summaries = play_summaries.merge(injuries[['GameKey','PlayID','Player_Activity_Derived']],
                                      how='left', on=['GameKey','PlayID'])
    play_summaries['punt_length'] = play_summaries['punt_received_x'] - play_summaries['punt_x']
    play_summaries['return_length'] = play_summaries['punt_received_x'] - play_summaries['tackle_x']
    return play_summaries.reset_index(drop=True)
play_summaries = get_play_summaries(roles)
def custom_round(x, base=5):
    if pd.isna(x):
        return None
    return int(base * round(float(x)/base))

play_summaries['punt_x_rounded'] = play_summaries['punt_x'].apply(custom_round)
punt_location_returns = play_summaries.groupby('punt_x_rounded').agg({'return_length': 'mean',
                                                                       'punt_length':'mean',
                                                                       'hang_time': 'mean',
                                                                       'PlayID': 'count',
                                                                       'punt_received_x': 'count',
                                                                       'fair_catch_x': 'count',
                                                                       'out_of_bounds_x':'count',
                                                                       'Player_Activity_Derived': 'count'})\
                                       .rename(columns={'punt_x_rounded': 'Punt Starting Location',
                                                        'punt_length': 'Avg. Punt Length',
                                                        'return_length': 'Avg. Return Length',
                                                        'hang_time': 'Avg. Hang Time',
                                                        'punt_received_x': 'Returns',
                                                        'fair_catch_x': 'Fair Catches',
                                                        'out_of_bounds_x':'Out-of-Bounds',
                                                        'PlayID': 'Punts',
                                                        'Player_Activity_Derived': 'Injuries'})
punt_location_returns['Injuries per 1k Punts'] = 1000* punt_location_returns['Injuries'] / punt_location_returns['Punts']
punt_location_returns['Injuries per 1k Returns'] = 1000* punt_location_returns['Injuries'] / punt_location_returns['Returns']
punt_location_returns
play_summaries['punt_received_x_rounded'] = play_summaries['punt_received_x'].apply(custom_round)
punt_received_location = play_summaries.groupby('punt_received_x_rounded').agg({'return_length': 'mean',
                                                                       'punt_length':'mean',
                                                                       'hang_time': 'mean',
                                                                       'PlayID': 'count',
                                                                       'punt_received_x': 'count',
                                                                       'fair_catch_x': 'count',
                                                                       'out_of_bounds_x':'count',
                                                                       'Player_Activity_Derived': 'count'})\
                                       .rename(columns={'punt_x_rounded': 'Punt Starting Location',
                                                        'punt_length': 'Avg. Punt Length',
                                                        'return_length': 'Avg. Return Length',
                                                        'hang_time': 'Avg. Hang Time',
                                                        'punt_received_x': 'Returns',
                                                        'fair_catch_x': 'Fair Catches',
                                                        'out_of_bounds_x':'Out-of-Bounds',
                                                        'PlayID': 'Punts',
                                                        'Player_Activity_Derived': 'Injuries'})
punt_received_location['Injuries per 1k Punts'] = 1000* punt_received_location['Injuries'] / punt_received_location['Punts']
punt_received_location['Injuries per 1k Returns'] = 1000* punt_received_location['Injuries'] / punt_received_location['Returns']
punt_received_location
returns_qualify_for_new_rule = play_summaries[play_summaries['punt_x']<30][play_summaries['punt_received_x']>80]
fc_qualify_for_new_rule = play_summaries[play_summaries['punt_x']<30][play_summaries['fair_catch_x']>80]
rq = returns_qualify_for_new_rule['PlayID'].count() # number of returns affected by proposed rule
rt = play_summaries['punt_received_x'].count()      # total number of returns

fq = fc_qualify_for_new_rule['PlayID'].count()# number of fair catches affected by proposed rule
ft = play_summaries['fair_catch_x'].count()# total number of fair catches

iq = returns_qualify_for_new_rule['Player_Activity_Derived'].count() # number of injuries affected by proposed rule
it = play_summaries['Player_Activity_Derived'].count()      # total number of injuries


print(str(rq) + " returns qualify, accounting for " + str(100*rq/rt) + "% of all returns" )
print(str(fq) + " fair catches qualify, accounting for " + str(100*fq/ft) + "% of all fair catches" )
print(str(iq) + " injuries qualify, accounting for " + str(100*iq/it) + "% of all injuries" )
class Field(Figure):
    __subtype__ = "Field"
    __view_model__ = "Plot"
    
    def __init__(self, **kwargs):
        Figure.__init__(
            self,
            x_range = Range1d(start=0, end=500),
            y_range = Range1d(start=0, end=241),
            plot_height = 250,
            toolbar_location = None,
            active_drag = None,
            **kwargs
        )
        self.axis.visible = False
        self.image_url(url=["https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/AmFBfield.svg/500px-AmFBfield.svg.png"], x=[0], y=[0], w=500, h=241, anchor="bottom_left")
class Play():
    """
    Create an object that defines a single Play and can be used for visualization
    """
    def __init__(self, game_key, play_id, ngs_df, role_df):
        play_df = ngs_df[(ngs_data['GameKey'] == game_key)\
                       & (ngs_data['PlayID'] == play_id)]
        
        # make sure the play exists in the current ngs_dataframe
        assert not play_df.empty
        
        # remove records after the play ends
        play_df = play_df.sort_values('Time', ascending=False).reset_index(drop=True)
        ending_events = ['tackle', 'out_of_bounds', 'punt_downed', 'touchback', 'fair_catch', 'touchdown']
        play_df = play_df[play_df['Event'].isin(ending_events).idxmax():]

        # sort and remove records prior to snap
        play_df = play_df.sort_values('Time').reset_index(drop=True)
        play_df = play_df[(play_df['Event']=='ball_snap').idxmax():]

        # convert x/y coordinates to fit on the field background
        # buffer on each side of the field is ~ 10 pixels, 1yd = 4px
        play_df['x'] = play_df['x'] * 4 + 10
        play_df['y'] = play_df['y'] * 4 + 10

        # associate role & team with each player
        play_df = play_df.merge(role_df, how='left', on=['GameKey','PlayID','GSISID'])

        # establish a time dimension for the play where ball_snap = 0
        play_df['seconds_since_snap'] = play_df['Time'] - play_df['Time'].min()
        play_df['seconds_since_snap'] = play_df['seconds_since_snap'].dt.total_seconds()

        self.game_key = game_key
        self.play_id = play_id
        self.play_df = play_df.copy()
        self.play_duration = play_df['seconds_since_snap'].max()
        self.field = Field()
        self.handle = None
        
        # Define teams and color of markers
        Team = namedtuple('Team', 'role color')
        self.teams = [Team('RETURN','red'), Team('COVERAGE','blue')]
        
    def add_players_to_field(self, seconds_since_snap):
        snapshot_df = self.play_df[self.play_df['seconds_since_snap'] == seconds_since_snap]
        snapshot_cds = ColumnDataSource(snapshot_df)
        for team in self.teams:
            team_bool = [player_team == team.role for player_team in snapshot_df['Team']]
            team_view = CDSView(source=snapshot_cds, filters=[BooleanFilter(team_bool)])
            self.field.circle(x='x', y='y', source=snapshot_cds, size=5,
                              color=team.color, name=team.role, view=team_view)
#         push_notebook(handle=self.handle)

    def update_field(self, seconds_since_snap):
        self.field.renderers.pop()
        self.field.renderers.pop()
        self.add_players_to_field(seconds_since_snap)
        push_notebook(handle=self.handle)

    def show_field(self, callback_name, seconds_since_snap=0):
        self.add_players_to_field(0)
        callback = CustomJS(code="""
        if (IPython.notebook.kernel !== undefined) {{
            var kernel = IPython.notebook.kernel;
            cmd = "{}.update_field(" + cb_obj.value + ")";
            kernel.execute(cmd, {{}}, {{}});
        }}""".format(callback_name))
        
        slider = Slider(title="Seconds Since Snap", value=0.0, start=0.0,
                        end=self.play_duration, step=0.5, callback=callback)
        field_layout = layout(
            column(
                self.field,
                widgetbox(slider)
            )
        )

        self.handle = show(field_layout, notebook_handle=True)
ngs_files = [file for file in os.listdir('../input/') if file.endswith('.csv') and file.startswith('NGS')]
ngs_files
ngs_data = pd.read_csv('../input/NGS-2016-reg-wk1-6.csv')
ngs_data['Time'] = pd.to_datetime(ngs_data['Time'])
ngs_data
