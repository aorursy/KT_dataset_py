import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy import stats
import datetime as dt
from IPython.core.display import display, HTML

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cmath

plt.style.use('ggplot')
pd.set_option('display.max_columns', 50)
# Read in non-NGS data sources
ppd = pd.read_csv('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')
gd = pd.read_csv('../input/NFL-Punt-Analytics-Competition/game_data.csv')
pprd = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
vr = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
vfi = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_footage-injury.csv')
pi = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')
vfi = vfi.rename(columns={'season' : 'season_year'})

gsisid_numbers = ppd.groupby('GSISID')['Number'].apply(lambda x: "%s" % ', '.join(x))
gsisid_numbers = pd.DataFrame(gsisid_numbers).reset_index()
vr_with_number = pd.merge(vr, gsisid_numbers, how='left', on='GSISID', suffixes=('','_injured'))
vr_with_number['Primary_Partner_GSISID'] = vr_with_number['Primary_Partner_GSISID'].fillna(0).replace('Unclear',0).astype('int64')
vr_with_number = pd.merge(vr_with_number,
                          gsisid_numbers,
                          how='left',
                          left_on='Primary_Partner_GSISID',
                          right_on='GSISID',
                          suffixes=('','_primary_partner'))
vr = vr_with_number
all_dfs = [ppd, gd, pprd, vr, vfi, pi]
# Change column names so they are all lowercase 
# never have to guess about which letters are uppercase
for mydf in all_dfs:
    mydf.columns = [col.lower() for col in mydf.columns]
"""
Create Dataframe with Generalized Punting Roles
include which team they are on (punting/returning)
"""
role_info_dict = {'GL': ['Gunner', 'Punting_Team'],
                  'GLi': ['Gunner', 'Punting_Team'],
                  'GLo': ['Gunner', 'Punting_Team'],
                  'GR': ['Gunner', 'Punting_Team'],
                  'GRi': ['Gunner', 'Punting_Team'],
                  'GRo': ['Gunner', 'Punting_Team'],
                  'P': ['Punter', 'Punting_Team'],
                  'PC': ['Punter_Protector', 'Punting_Team'],
                  'PPR': ['Punter_Protector', 'Punting_Team'],
                  'PPRi': ['Punter_Protector', 'Punting_Team'],
                  'PPRo': ['Punter_Protector', 'Punting_Team'],
                  'PDL1': ['Defensive_Lineman', 'Returning_Team'],
                  'PDL2': ['Defensive_Lineman', 'Returning_Team'],
                  'PDL3': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR1': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR2': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR3': ['Defensive_Lineman', 'Returning_Team'],
                  'PDL5': ['Defensive_Lineman', 'Returning_Team'],
                  'PDL6': ['Defensive_Lineman', 'Returning_Team'],
                  'PFB': ['PuntFullBack', 'Returning_Team'],
                  'PLG': ['Punting_Lineman', 'Punting_Team'],
                  'PLL': ['Defensive_Backer', 'Returning_Team'],
                  'PLL1': ['Defensive_Backer', 'Returning_Team'],
                  'PLL3': ['Defensive_Backer', 'Returning_Team'],
                  'PLS': ['Punting_Longsnapper', 'Punting_Team'],
                  'PLT': ['Punting_Lineman', 'Punting_Team'],
                  'PLW': ['Punting_Wing', 'Punting_Team'],
                  'PRW': ['Punting_Wing', 'Punting_Team'],
                  'PR': ['Punt_Returner', 'Returning_Team'],
                  'PRG': ['Punting_Lineman', 'Punting_Team'],
                  'PRT': ['Punting_Lineman', 'Punting_Team'],
                  'VLo': ['Jammer', 'Returning_Team'],
                  'VR': ['Jammer', 'Returning_Team'],
                  'VL': ['Jammer', 'Returning_Team'],
                  'VRo': ['Jammer', 'Returning_Team'],
                  'VRi': ['Jammer', 'Returning_Team'],
                  'VLi': ['Jammer', 'Returning_Team'],
                  'PPL': ['Punter_Protector', 'Punting_Team'],
                  'PPLo': ['Punter_Protector', 'Punting_Team'],
                  'PPLi': ['Punter_Protector', 'Punting_Team'],
                  'PLR': ['Defensive_Backer', 'Returning_Team'],
                  'PRRo': ['Defensive_Backer', 'Returning_Team'],
                  'PDL4': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR4': ['Defensive_Lineman', 'Returning_Team'],
                  'PLM': ['Defensive_Backer', 'Returning_Team'],
                  'PLM1': ['Defensive_Backer', 'Returning_Team'],
                  'PLR1': ['Defensive_Backer', 'Returning_Team'],
                  'PLR2': ['Defensive_Backer', 'Returning_Team'],
                  'PLR3': ['Defensive_Backer', 'Returning_Team'],
                  'PLL2': ['Defensive_Backer', 'Returning_Team'],
                  'PDM': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR5': ['Defensive_Lineman', 'Returning_Team'],
                  'PDR6': ['Defensive_Lineman', 'Returning_Team'],
                  }
role_info = pd.DataFrame.from_dict(role_info_dict, orient='index',
                                   columns=['generalized_role', 'punting_returning_team']) \
    .reset_index() \
    .rename(columns={'index': 'role'})
# More Data Prep
injury_play_ngs = pd.read_parquet(
    '../input/nfl-punt-data-preprocessing-ngs-injury-plays/NGS-injury-plays.parquet')
gsisid_numbers = ppd.groupby('gsisid')['number'].apply(
    lambda x: "%s" % ', '.join(x))
gsisid_numbers = pd.DataFrame(gsisid_numbers).reset_index()
# Add Player Number and Direction
vr_with_number = pd.merge(
    vr, gsisid_numbers, how='left', suffixes=('', '_injured'))
vr_with_number['primary_partner_gsisid'] = vr_with_number['primary_partner_gsisid'].replace(
    'Unclear', np.nan).fillna(0).astype('int')
vr_with_number = pd.merge(vr_with_number, gsisid_numbers, how='left',
                          left_on='primary_partner_gsisid', right_on='gsisid', suffixes=('', '_primary_partner'))
vr = vr_with_number

vr_merged = pd.merge(vr, pprd)
vr_merged = pd.merge(vr_merged, role_info)


vr_merged = pd.merge(vr_merged, pprd, left_on=['season_year', 'gamekey', 'playid', 'primary_partner_gsisid'],
                     right_on=['season_year', 'gamekey', 'playid', 'gsisid'], how='left',
                     suffixes=('', '_primary_partner'))
vr_merged = pd.merge(vr_merged, role_info, left_on='role_primary_partner',
                     right_on='role', how='left', suffixes=('', '_primary_partner'))

vr_merged = vr_merged.fillna('None')
vr_merged['count'] = 1

vr_merged['generalized_role'] = vr_merged['generalized_role'].str.replace(
    '_', ' ')
vr_merged['generalized_role_primary_partner'] = vr_merged['generalized_role_primary_partner'].str.replace(
    '_', ' ')
def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')

    return fig, ax
"""
This cell block contains functions for interacting with the NGS data.
Plotting compass of player angle and velocity along with the playing field
"""


def compass(angles, radii, arrowprops=None, ax=None):
    """
    Compass draws a graph that displays the vectors with
    components `u` and `v` as arrows from the origin.

    Examples
    --------
    >>> import numpy as np
    >>> u = [+0, +0.5, -0.50, -0.90]
    >>> v = [+1, +0.5, -0.45, +0.85]
    >>> compass(u, v)
    """

    #angles, radii = cart2pol(u, v)
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    kw = dict(arrowstyle="->", color='k')
    if arrowprops:
        kw.update(arrowprops)
    [ax.annotate("", xy=(angle, radius), xytext=(0, 0),
                 arrowprops=kw) for
     angle, radius in zip(angles, radii)]

    ax.set_ylim(0, np.max(radii))

    return ax


def trim_play_action(df):
    """
    Trims a play to only the duration of action
    """
    if len(df.loc[df['event'] == 'ball_snap']['time'].values) == 0:
        print('........No Snap for this play')
        ball_snap_time = df['time'].min()
    else:
        ball_snap_time = df.loc[df['event'] ==
                                'ball_snap']['time'].values.min()

    try:
        end_time = df.loc[(df['event'] == 'out_of_bounds') |
                          (df['event'] == 'downed') |
                          (df['event'] == 'tackle') |
                          (df['event'] == 'punt_downed') |
                          (df['event'] == 'fair_catch') |
                          (df['event'] == 'touchback') |
                          (df['event'] == 'touchdown')]['time'].values.max()
    except ValueError:
        end_time = df['time'].values.max()
    df = df.loc[(df['time'] >= ball_snap_time) & (df['time'] <= end_time)]
    return df

def plot_injury_play(season_year, gamekey, playid,
                     plot_velocity=False, ax3=None, display_url=False,
                     figsize_velocity=(5, 4),
                     **kwargs):
    """
    Plot the injury play with velocity and details
    """
    vr_thisplay = vr.loc[(vr['season_year'] == season_year) &
                         (vr['playid'] == playid) &
                         (vr['gamekey'] == gamekey)]

    play = injury_play_ngs.loc[(injury_play_ngs['season_year'] == season_year) &
                               (injury_play_ngs['playid'] == playid) &
                               (injury_play_ngs['gamekey'] == gamekey)].copy()

    # Calculate velocity in meters per second
    play['dis_meters'] = play['dis'] / 1.0936  # Add distance in meters
    # Speed
    play['dis_meters'] / 0.01
    play['v_mps'] = play['dis_meters'] / 0.1

    # Filter to only duration of play
    play = trim_play_action(play)

    # play = pd.read_csv('../working/playlevel/during_play/{}-{}-{}.csv'.format(season_year, gamekey, playid))
    play['dir_theta'] = play['dir'] * np.pi / 180

    # Video footage link
    url_link = vfi.loc[(vfi['season_year'] == season_year) &
                       (vfi['playid'] == playid) &
                       (vfi['gamekey'] == gamekey)]['preview link (5000k)'].values[0]

    playdescription = vfi.loc[(vfi['season_year'] == season_year) &
                              (vfi['playid'] == playid) &
                              (vfi['gamekey'] == gamekey)]['playdescription'].values[0]

    print('==========================================================')
    print('======= Running for Season {} PlayID {} GameKey {} ==='.format(season_year, playid, gamekey))
    print('==========================================================')
    print("=== PLAY DESCRIPTION: ===")
    print(playdescription)
    
    if display_url:
        print("=== INJURY INFO: ===")

        print('Injured player number {} was injured while {} with primary impact {}'
              .format(vr_thisplay['number'].values[0],
                      vr_thisplay['player_activity_derived'].values[0],
                      vr_thisplay['primary_impact_type'].values[0]))
        
        if vr_thisplay['gsisid_primary_partner'].values[0][0] != np.nan:
            print('Injuring player number was injured the other player while {} with primary impact {}'
                  .format(vr_thisplay['primary_partner_activity_derived'].values[0],
                          vr_thisplay['primary_impact_type'].values[0]))
        display(HTML("""<a href="{}">LINK TO VIDEO FOOTAGE FOR SEASON: {} PLAYID: {} GAMEKEY: {}</a>""" \
                     .format(url_link, season_year, playid, gamekey)))

    # Determine time of injury
    injured = play.loc[play['injured_player']]
    primarypartner = play.loc[play['primary_partner_player']]
    injury_time = None
    if len(primarypartner) != 0:
        inj_and_pp = pd.merge(injured[['time', 'x', 'y']], primarypartner[[
                              'time', 'x', 'y']], on='time', suffixes=('_inj', '_pp'))
        inj_and_pp['dis_from_eachother'] = np.sqrt(np.square(inj_and_pp['x_inj'] -
                                                             inj_and_pp['x_pp']) +
                                                   np.square(inj_and_pp['y_inj'] -
                                                             inj_and_pp['y_pp']))
        injury_time = inj_and_pp.sort_values('dis_from_eachother')[
            'time'].values[0]
    # PLOT
    fig, ax3 = create_football_field(**kwargs)

    # Plot path of injured player
    d = play.loc[play['injured_player']]
    injured_player_role = play.loc[play['injured_player']]['role'].values[0]
    d.plot('x', 'y', kind='scatter', ax=ax3,  zorder=5, color='blue', alpha=0.3,
           xlim=(0, 120), ylim=(0, 53.3),
           label='Injured Player Path - Role: {}'.format(injured_player_role))  # Plot injured player path
    play.loc[(play['punting_returning_team'] == 'Returning_Team') &
             (play['event'] == 'ball_snap')].plot('x', 'y', alpha=1, kind='scatter',
                                                  color='purple', ax=ax3, zorder=5, style='+',
                                                  label='Returning Team Player')
    play.loc[(play['punting_returning_team'] == 'Punting_Team') &
             (play['event'] == 'ball_snap')].plot('x', 'y', alpha=1, kind='scatter',
                                                  color='orange', ax=ax3, zorder=4, style='+',
                                                  label='Punting Team Player')
    start_pos = d.loc[d['time'] == d['time'].min()]
    inj_star_pos = ax3.scatter(start_pos['x'], start_pos['y'], color='red',
                               zorder=5, label='Injured Player Starting Position')
    end_pos = d.loc[d['time'] == d['time'].max()]
    ax3.scatter(end_pos['x'], end_pos['y'], color='black',
                zorder=5, label='Injured Player Ending Position')
    if injury_time:
        inj_pos = d.loc[d['time'] == injury_time]
    pp_player_role = None
    if len(primarypartner) != 0:
        pp_player_role = play.loc[play['primary_partner_player']
                                  ]['role'].values[0]
        play.loc[play['primary_partner_player']].plot('x', 'y', kind='scatter',
                                                      xlim=(0, 120), ylim=(0, 53.3),
                                                      ax=ax3, color='yellow', alpha=0.3, zorder=3,
                                                      label='Primary Partner Path - Role {}'.format(pp_player_role))
        ax3.scatter(inj_pos['x'],
                    inj_pos['y'],
                    color='red',
                    zorder=5,
                    s=50,
                    marker='+',
                    label='Aproximate Location of Injury')
    play_info_string = 'Season {} - Gamekey {} - Playid {}'.format(
        season_year, gamekey, playid)
    injured_player_string = 'Injured Player Number: {} - action {}' \
        .format(vr_thisplay['number'].values[0],
                vr_thisplay['player_activity_derived'].values[0])
    primary_partner_string = 'Primary Partner Player Number: {} - action {}' \
        .format(vr_thisplay['number_primary_partner'].values[0],
                vr_thisplay['primary_partner_activity_derived'].values[0])
    # Plot punt return path if not one of the players.
    if (injured_player_role != 'PR') and (pp_player_role != 'PR'):
        punt_returner = play.loc[play['role'] == 'PR']
        punt_returner.plot('x', 'y', kind='scatter', ax=ax3,  zorder=3, color='white', alpha=0.3,
                           label='Punt Returner Path')

    plt.suptitle(play_info_string, fontsize=15)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if plot_velocity:
        # Plot injured player compass

        fig3, (ax1, ax2) = plt.subplots(
            1, 2, subplot_kw=dict(polar=True), figsize=figsize_velocity)

        d = play.loc[play['injured_player']]
        role = d.role.values[0]

        ax1 = compass(d['dir_theta'], d['v_mps'],
                      arrowprops={'alpha': 0.3}, ax=ax1)
        ax1.set_theta_zero_location("N")
        ax1.set_theta_direction(-1)
        ax1.set_title('Injured Player: {}'.format(role))
        # Color point of time when inujury happened
        if len(primarypartner) != 0:
            theta_at_inj = d.loc[d['time'] ==
                                 injury_time]['dir_theta'].values[0]
            dis_at_inj = d.loc[d['time'] == injury_time]['v_mps'].values[0]
            impact_arrow = ax1.annotate("",
                                        xy=(theta_at_inj, dis_at_inj), xytext=(
                                            0, 0),
                                        arrowprops={'color': 'orange'},
                                        label='Aproximate Point of Impact')  # use cir mean
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.suptitle('Velocity and Direction (Injured Player): {}'.format(role), x=0.52, y=1.01, fontsize=15)

        if len(primarypartner) != 0:
            # Plot primary partner compass
            d = play.loc[play['primary_partner_player']]
            role = d.role.values[0]
            ax2 = compass(d['dir_theta'], d['v_mps'],
                          arrowprops={'alpha': 0.3}, ax=ax2)
            ax2.set_theta_zero_location("N")
            ax2.set_theta_direction(-1)
            ax2.set_title('Primary Partner: {}'.format(role))
            # Color point of time when inujury happened
            theta_at_inj = d.loc[d['time'] ==
                                 injury_time]['dir_theta'].values[0]
            dis_at_inj = d.loc[d['time'] == injury_time]['v_mps'].values[0]
            ax2.annotate("", xy=(theta_at_inj, dis_at_inj), xytext=(
                0, 0), arrowprops={'color': 'orange'})  # use cir mean
            # plt.suptitle('Velocity and Direction (Primary Partner): {}'.format(role), x=0.52, y=1.01, fontsize=15)
            plt.show()
    return ax3
for row in vr.iterrows():
    """
    Loop through each play in the video review dataframe and call the
    plot injury plat to show information about the play.
    """
    season_year = row[1]['season_year']
    gamekey = row[1]['gamekey']
    playid = row[1]['playid']
    
    plot_injury_play(season_year=season_year, 
                     gamekey=gamekey,
                     playid=playid,
                     figsize=(10, 5),
                     plot_velocity=True,
                     figsize_velocity=(15, 5),
                     display_url=True)
    plt.show()
