import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.patches as patches



import seaborn as sns; sns.set()

import itertools

import time
start = time.time()

InjuryRecord = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

PlayList = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")

PlayerTrackData = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")

print("Took {:.4f}s".format(time.time() - start))
"""

    Step 1: perform a left join on PlayList, appending InjuryRecord info to the right.



"""

left_cols=["PlayerKey", "GameID", "PlayKey", "FieldType"]

right_cols=["PlayerKey", "GameID", "PlayKey", "Surface"]

merge = pd.merge(PlayList, InjuryRecord, how='left', left_on=left_cols, right_on=right_cols).drop(columns=["Surface"])



"""

    Step 2: fill nan fields

"""

filter_col = [col for col in merge if col.startswith('DM_M')]

merge[filter_col] = merge[filter_col].fillna(int(0))

merge["BodyPart"] = merge["BodyPart"].fillna("No Injury")





"""

    Step 3: set up replacement and indication variables for standardizing stadium type + detecting

    when a play is in progress (no dead ball time)

"""

replace_dict = {"Bowl":"Dome, unspecified", "Closed Dome":"Indoors", "Dome":"Dome, unspecified", 

                "Domed":"Dome, unspecified", "Domed, Open":"Outdoors", "Domed, closed":"Indoors",

               "Domed, open":"Outdoors", "Heinz Field":"Outdoors", "Indoor":"Indoors", "Dome, closed":"Indoors",

               "Indoor, Open Roof":"Outdoors", "Indoor, Roof Closed":"Indoors", "Open":"Outdoors",

               "Oudoor":"Outdoors", "Ourdoor":"Outdoors", "Outddors":"Outdoors", "Outdoor":"Outdoors",

               "Outdor":"Outdoors", "Outside":"Outdoors", "Retr. Roof - Closed":"Indoors",

                "Retr. Roof - Open":"Outdoors", "Retr. Roof Closed":"Indoors", "Retr. Roof-Closed":"Indoors",

               "Retr. Roof-Open":"Outdoors", "Retractable Roof":"Unknown",

               "Outdoor Retr Roof-Open":"Outdoors", "Cloudy":"Unknown"}

replace_dict_2 = {'Clear and warm':"Clear",  'Clear skies':"Clear", 'Coudy':'Cloudy', 

                'Sun & clouds':"Partly Cloudy", 'Mostly cloudy':"Mostly Cloudy", 

                'Mostly sunny':"Mostly Sunny", 'Cloudy and Cool':"Cloudy", 'Indoors':"Controlled Climate", 

                'Clear Skies':"Clear", 'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.':"Cloudy",  

                'Rain shower':"Rain", 'Cloudy, 50% change of rain':"Cloudy", "Indoor":"Controlled Climate"}



for orig, replacement in replace_dict.items():

    merge["StadiumType"] = merge["StadiumType"].replace(orig, replacement)

for orig, replacement in replace_dict_2.items():

    merge["Weather"] = merge["Weather"].replace(orig, replacement)

"""

    Step 4: discretize temperature

"""

merge["Temperature"] = np.clip(merge["Temperature"], 0, 100)

# merge["TempInterval"] = pd.cut(merge["Temperature"], np.arange(0,100,10))



play_event_counts = PlayerTrackData.groupby(["event"]).count()["PlayKey"]



merge.head()

play_event_counts.head()
def category_to_ints(df, cols):

    for colname in cols:

        df[colname], _ = pd.factorize(df[colname])

    return df

merge_test = merge.copy()

merge_test = category_to_ints(merge, ["RosterPosition", "StadiumType", "FieldType", "Weather",

                                      "PlayType", "Position", "PositionGroup", "BodyPart"])

merge_test = merge_test.drop(columns=["PlayerKey", "GameID", "PlayKey"])

merge_test.head()



correlation = merge_test.corr()



fig = plt.figure(figsize=(18, 18))

sns.heatmap(correlation, vmax=.9, square=True, annot=True)





def plot_diffs(merge, condition, group_by, title="", ylabel="", scale=10000):

    groups = merge[condition].groupby(group_by)

    play_counts = merge.groupby("PlayType").count()

    df = groups.count()

    for play_type in groups.count().index.get_level_values(0):

        df.loc[pd.IndexSlice[play_type, :], :] = df.loc[pd.IndexSlice[play_type, :], :].apply(lambda x: scale * x / play_counts["PlayKey"][play_type]) 

    ax = df["PlayKey"].unstack(level=0).plot(kind='bar', subplots=False)

    ax.set_title(title)

    ax.set_ylabel(ylabel)

    return ax

    

noinjury = (merge['BodyPart'] != "No Injury")

no_punt_no_kickoff = ~(merge["PlayType"].str.startswith(("Punt", "Kickoff"), na=False))

is_punt = (merge["PlayType"].str.startswith(("Punt"), na=False))

is_kickoff = (merge["PlayType"].str.startswith(("Kickoff"), na=False))



plot_diffs(merge, condition=(noinjury & no_punt_no_kickoff), 

           group_by=["PlayType", "BodyPart", "FieldType"], 

           title="Injury Comparison by Play Type, Field Type, and Location", ylabel="Injuries per play")

plot_diffs(merge, condition=(noinjury & is_punt), 

           group_by=["PlayType", "BodyPart", "FieldType"], 

           title="Injuries on Punt Plays based on Field Type, and Location", ylabel="Injuries per play")

plot_diffs(merge, condition=(noinjury & is_kickoff), 

           group_by=["PlayType", "BodyPart", "FieldType"],

           title="Injuries on Kickoffs based on Field Type, and Location", ylabel="Injuries per play")
start = time.time()

np.random.seed(42)

non_injured_plays = 150 # CHANGE THIS AT YOUR OWN RISK

inj_play_list = InjuryRecord['PlayKey'].tolist()

non_inj_play_list = np.random.choice(PlayerTrackData["PlayKey"][~PlayerTrackData["PlayKey"].isin(inj_play_list)].unique().tolist(), size=non_injured_plays)

injury_plays = PlayerTrackData.query("PlayKey in @inj_play_list").groupby("PlayKey", as_index=False)

non_injury_plays = PlayerTrackData.query("PlayKey in @non_inj_play_list").groupby("PlayKey", as_index=False)

print("Took {:.4f}s".format(time.time() - start))
"""

    Column-wise application of functions on GroupBy objects. Useful for quickly aggregating summary stats

    on a per-play basis.

    

    @param classes: a list of the GroupBy objects containing the classes.

    @param fn: a callable representing the function or dictionary of functions (column to function mapping)

                to be applied

                

    @return: aggregate results of applying the function to the GroupBy.

"""



def get_aggregate_data(classes, features, fn=np.mean):

    if type(fn) == tuple:

        col_dict = {x:fn[i] for i, x in enumerate(features) if PlayerTrackData[x].dtype == np.float64}

    elif callable(fn):

        col_dict = {x:fn for x in features if PlayerTrackData[x].dtype == np.float64}

    else:

        raise Exception("col_dict must be a single function or a mapping from columns to functions")

    summaries = [cls.agg(col_dict) for cls in classes]

    return summaries



"""

    Plots classes of data.

    

    @param *classes: an argument list of GroupBy objects containing the data to be plotted

    @param fn: the function/dictionary of functions that will be used to aggregate the columns of the data

    @param features: the features to plot, by their column name in the original DataFrame

    @param title: the title of the plot

    @param class_labels: a string description of each class in the order of the argument list. Used in the legend.

"""

def plot_classes(*classes, fn=np.mean, features=['x', 'y'], title="", class_labels=[], colors=None, 

                 mode='scatter'):

    

    if mode is 'scatter':

        assert len(features) == 2, "Must plot two features"

        feature1, feature2 = features

        assert feature1 in PlayerTrackData.columns, "Not a real column"

        assert feature2 in PlayerTrackData.columns, "Not a real column"

    elif mode is 'box':

        if type(features) is list:

            assert len(features) == 1, "Must plot one features"

            features = features[0]

    else:

        raise ValueError("'{}' is not a valid plot type. Set mode to 'scatter' or 'box' instead.")

        

    plt.clf()     

    summaries = get_aggregate_data(classes, features, fn)



    plt.title(title)

    if mode is 'scatter':

        plt.xlabel(feature1)

        plt.ylabel(feature2)



    # plot, and show stuff

    if mode is 'scatter':

        for i, summary in enumerate(summaries):

            if colors is None:

                plt.scatter(summary[feature1], summary[feature2], label=class_labels[i], alpha=0.5)

            else:

                plt.scatter(summary[feature1], summary[feature2], label=class_labels[i], alpha=0.5, color=next(colors))

    elif mode is 'box':

        plt.boxplot([summary[features] for summary in summaries])

        plt.xticks(range(1, len(summaries)+1), class_labels)

        

    plt.legend()

    plt.show()



"""

    Football field drawing code. Courtesy of Rob Mulla's public kernel, which can be found at

    https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position

"""

def create_football_field(linenumbers=True,

                          endzones=True,

                          highlight_line=False,

                          highlight_line_number=50,

                          highlighted_name='Line of Scrimmage',

                          fifty_is_los=False,

                          figsize=(14, 8)):

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

    Adapted from Rob Mulla's public kernel, which can be found at 

    https://www.kaggle.com/robikscube/nfl-1st-and-future-analytics-intro

"""

def plot_plays(play_ids, title="", path_color='orange', alpha=0.2, prev_ax=None, annotate=False, event_filter=["tackle", "ball_snap", "pass_outcome_incomplete", "out_of_bounds", "first_contact",

           "handoff", "pass_forward", "pass_outcome_caught", "touchdown", "qb_sack","touchback",

           "kickoff", "punt", "pass_outcome_touchdown", "pass_arrived", "extra_point", "field_goal", 

           "play_action", "kick_received","fair_catch", "punt_downed", "run", "punt_received", 

           "qb_kneel", "pass_outcome_interception", "field_goal_missed", "fumble", 

           "fumble_defense_recovered", "qb_spike","extra_point_missed", "fumble_offense_recovered", 

           "pass_tipped", "lateral", "qb_strip_sack", "safety", "kickoff_land", "snap_direct", 

           "kick_recovered","field_goal_blocked", "punt_muffed", "pass_shovel", "extra_point_blocked", 

           "pass_lateral", "punt_blocked", "run_pass_option", "free_kick", "punt_fake","end_path", 

           "drop_kick", "field_goal_fake", "extra_point_fake", "xp_fake"]):

    

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]

        annot.xy = pos

        text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 

                               " ".join([names[n] for n in ind["ind"]]))

        annot.set_text(text)

        annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))

        annot.get_bbox_patch().set_alpha(0.4)

        

    def hover(event):

        fig = ax.get_figure()

        vis = annot.get_visible()

        if event.inaxes == ax:

            cont, ind = sc.contains(event)

            if cont:

                update_annot(ind)

                annot.set_visible(True)

                fig.canvas.draw_idle()

            else:

                if vis:

                    annot.set_visible(False)

                    fig.canvas.draw_idle()



    if prev_ax is None:

        _, ax = create_football_field()

    else:

        ax = prev_ax

    ax.set_title(title)

    for playkey, inj_play in PlayerTrackData.query('PlayKey in @play_ids').groupby('PlayKey'):

        sc = plt.scatter(x=inj_play['x'], y=inj_play['y'], color=path_color, alpha=alpha)

        """

        if annotate:

            for i, point in inj_play.iterrows():

                if i == inj_play.index[0]:

                    player_position = PlayList[PlayList["PlayKey"] == playkey]["Position"].values[0]

                    ax.text(point['x'], point['y'], player_position, color='white', fontsize=15)

                if pd.notna(point['event']) and point['event'] in event_filter:

                    ax.text(point['x'], point['y'], str(point['event']), color='yellow', fontsize=10)

                    

                    """

        annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",

                    bbox=dict(boxstyle="round", fc="w"),

                    arrowprops=dict(arrowstyle="->"))

        annot.set_visible(False)

    ax.get_figure().canvas.mpl_connect("motion_notify_event", hover)

    return ax



"""

    Corrects the angle difference between time-steps, given that it is highly unlikely for a player to be 

    able to make half of a rotation in 0.1s.

"""

def angle_diff(arr, n=1):

    diff = np.diff(arr)

    # correct speeds

    diff[diff > 180] -= 360

    diff[diff < -180] += 360

    for i in range(n-1):

        diff = np.diff(diff)

    return diff

plot_classes(injury_plays, non_injury_plays, features=['dir', 's'], fn=np.mean,

             class_labels=['injury', 'no injury'], title="Plays with and without injury w.r.t. mean speed and direction")

plot_classes(injury_plays, non_injury_plays, features=['dir', 's'], fn=np.max,

             class_labels=['injury', 'no injury'], title="Plays with and without injury w.r.t. max speed and direction")
plot_classes(non_injury_plays, injury_plays,fn=(lambda x: np.max(np.abs(np.diff(x))), lambda x: np.max(np.abs(angle_diff(x, n=2)))), features=[ 's', 'dir'], 

             class_labels=['no injury', 'injury'],

            title="Plays with and without injury w.r.t. max angular and linear acceleration")

plot_classes(non_injury_plays, injury_plays,fn=(lambda x: np.max(np.abs(np.diff(x, n=2))), lambda x: np.max(np.abs(angle_diff(x, n=3)))), features=['s', 'dir'], 

             class_labels=['no injury', 'injury'],

            title="Plays with and without injury w.r.t. max angular and linear jerk")
pass_play_keys = PlayList[PlayList["PlayType"] == "Pass"]["PlayKey"].dropna().tolist()

rush_play_keys = PlayList[PlayList["PlayType"] == "Pass"]["PlayKey"].dropna().tolist()

kickoff_play_keys = PlayList[PlayList["PlayType"].str.startswith("Kickoff", na=False)]["PlayKey"].dropna().tolist()

punt_play_keys = PlayList[PlayList["PlayType"].str.startswith("Punt", na=False)]["PlayKey"].dropna().tolist()



ankle_inj_keys = InjuryRecord[InjuryRecord['BodyPart'] == 'Ankle']["PlayKey"].dropna().tolist()

knee_inj_keys = InjuryRecord[InjuryRecord['BodyPart'] == 'Knee']["PlayKey"].dropna().tolist()

toe_inj_keys = InjuryRecord[InjuryRecord['BodyPart'] == 'Toes']["PlayKey"].dropna().tolist()

foot_inj_keys = InjuryRecord[InjuryRecord['BodyPart'] == 'Foot']["PlayKey"].dropna().tolist()

heel_inj_keys  = InjuryRecord[InjuryRecord['BodyPart'] == 'Heel']["PlayKey"].dropna().tolist()



synth_keys = PlayList[PlayList['FieldType']=='Synthetic']["PlayKey"].dropna().tolist()

nat_keys = PlayList[PlayList['FieldType']=='Natural']["PlayKey"].dropna().tolist()
pass_ankle_inj_keys = set(pass_play_keys) & set(ankle_inj_keys)

pass_ankle_non_inj_keys = set(pass_play_keys) & set(non_inj_play_list)

pass_ankle_injury_plays = PlayerTrackData.query("PlayKey in @pass_ankle_inj_keys").groupby("PlayKey", as_index=False)

pass_ankle_non_injury_plays = PlayerTrackData.query("PlayKey in @pass_ankle_non_inj_keys").groupby("PlayKey", as_index=False)

plot_classes(pass_ankle_non_injury_plays, pass_ankle_injury_plays, features=['s', 'dir'], fn=np.max,

             class_labels=['no injury', 'injury'], title="Plays with and without injury w.r.t. max speed and direction")

plot_classes(pass_ankle_non_injury_plays, pass_ankle_injury_plays,fn=(lambda x: np.max(np.abs(np.diff(x))), lambda x: np.max(np.abs(angle_diff(x, n=2)))), features=[ 's', 'dir'], 

             class_labels=['no injury', 'injury'],

            title="Plays with and without injury w.r.t. max angular and linear acceleration")

plot_classes(pass_ankle_non_injury_plays, pass_ankle_injury_plays,fn=(lambda x: np.max(np.abs(np.diff(x, n=2))), lambda x: np.max(np.abs(angle_diff(x, n=3)))), features=['s', 'dir'], 

             class_labels=['no injury', 'injury'],

            title="Plays with and without injury w.r.t. max angular and linear jerk")
pass_ankle_inj_synth_keys = pass_ankle_inj_keys & set(synth_keys)

pass_ankle_inj_nat_keys = pass_ankle_inj_keys & set(nat_keys)

pass_ankle_non_inj_synth_keys = pass_ankle_non_inj_keys & set(synth_keys)

pass_ankle_non_inj_nat_keys = pass_ankle_non_inj_keys & set(nat_keys)



pass_ankle_inj_synth_plays = PlayerTrackData.query("PlayKey in @pass_ankle_inj_synth_keys").groupby("PlayKey", as_index=False)

pass_ankle_inj_nat_plays = PlayerTrackData.query("PlayKey in @pass_ankle_inj_nat_keys").groupby("PlayKey", as_index=False)

pass_ankle_non_inj_synth_plays = PlayerTrackData.query("PlayKey in @pass_ankle_non_inj_synth_keys").groupby("PlayKey", as_index=False)

pass_ankle_non_inj_nat_plays = PlayerTrackData.query("PlayKey in @pass_ankle_non_inj_nat_keys").groupby("PlayKey", as_index=False)



plot_classes(pass_ankle_inj_synth_plays, pass_ankle_inj_nat_plays, pass_ankle_non_inj_synth_plays,

             pass_ankle_non_inj_nat_plays, features=['s', 'dir'], fn=np.max,

             class_labels=['injury, synthetic', 'injury, natural', 'no injury, synthetic', 

                           'no injury, natural'], title="Pass plays with and without injury w.r.t. max speed and direction",

             colors=itertools.cycle(["r", "orangered", "b", "c"]))

plot_classes(pass_ankle_inj_synth_plays, pass_ankle_inj_nat_plays, pass_ankle_non_inj_synth_plays,

             pass_ankle_non_inj_nat_plays,fn=(lambda x: np.max(np.abs(np.diff(x))), lambda x: np.max(np.abs(angle_diff(x, n=2)))), features=[ 's', 'dir'], 

             class_labels=['injury, synthetic', 'injury, natural', 'no injury, synthetic', 

                           'no injury, natural'],

            title="Pass plays with and without ankle injury w.r.t. max angular and linear acceleration", 

             colors=itertools.cycle(["r", "orangered", "b", "c"]))

plot_classes(pass_ankle_inj_synth_plays, pass_ankle_inj_nat_plays, pass_ankle_non_inj_synth_plays,

             pass_ankle_non_inj_nat_plays, fn=(lambda x: np.max(np.abs(np.diff(x, n=2))), lambda x: np.max(np.abs(angle_diff(x, n=3)))), features=['s', 'dir'], 

             class_labels=['injury, synthetic', 'injury, natural', 'no injury, synthetic', 

                           'no injury, natural'],

            title="Pass plays with and without ankle injury w.r.t. max angular and linear jerk", 

             colors=itertools.cycle(["r", "orangered", "b", "c"]))
plot_classes(pass_ankle_inj_synth_plays, pass_ankle_inj_nat_plays, fn=(lambda x: np.max(np.abs(np.diff(x, n=2))), lambda x: np.max(np.abs(angle_diff(x, n=3)))), features=['s', 'dir'], 

             class_labels=['injury, synthetic', 'injury, natural', 'no injury, synthetic', 

                           'no injury, natural'],

            title="Pass plays with and without ankle injury w.r.t. max angular and linear jerk", 

             colors=itertools.cycle(["r", "orangered"]))

plot_classes(pass_ankle_inj_synth_plays, pass_ankle_inj_nat_plays, fn=(lambda x: np.max(np.abs(np.diff(x, n=2))), lambda x: np.max(np.abs(angle_diff(x, n=3)))), features=['s'],

             class_labels=['injury, synthetic', 'injury, natural'],title="Speed boxplot for pass plays with ankle injury by turf type",

             mode='box')
plot_plays(pass_ankle_inj_synth_keys, path_color='red', 

           event_filter=['tackle', 'first_contact', 'handoff', 'pass_arrived'],

          annotate=True, title="Pass plays on synthetic turf resulting in ankle injury")

plt.show()

plot_plays(pass_ankle_inj_nat_keys, path_color='orange', 

           event_filter=['tackle', 'first_contact', 'handoff', 'pass_arrived'], annotate=True,

          title="Pass plays on natural turf resulting in ankle injury")

plt.show()

grp = PlayerTrackData.groupby("PlayKey")

play_enders = {'end_path','extra_point','extra_point_blocked','extra_point_missed','fair_catch','field_goal','field_goal_blocked',

 'field_goal_missed','fumble_defense_recovered','fumble_offense_recovered','out_of_bounds','pass_outcome_touchdown',

 'play_submit\t','punt_downed','qb_kneel','qb_sack','qb_spike','qb_strip_sack','safety','timeout','timeout_away','timeout_booth_review',

 'timeout_halftime','timeout_home','timeout_injury','timeout_quarter','timeout_tv','touchback','touchdown',

 'two_minute_warning','two_point_conversion','penalty_accepted', 'pass_outcome_incomplete', 'tackle'}

post_snap_mask = grp['event'].apply(lambda x: x.shift().isin(['ball_snap']).cumsum().eq(1))

play_end_mask = grp['event'].apply(lambda x: x.shift().isin(play_enders).cumsum().eq(0))

in_play = PlayerTrackData[post_snap_mask & play_end_mask]
in_play.groupby("PlayKey").count()["time"].hist(bins=100)
in_play.loc[:, 'x'] = in_play['x'] - in_play.groupby("PlayKey")['x'].transform('first')

# in_play.loc[:, 'y'] = in_play['y'] - in_play.groupby("PlayKey")['y'].transform('first')

in_play.loc[:, 'time'] = in_play['time'] - in_play.groupby("PlayKey")['time'].transform('first')

in_play.loc[:, 'x'] = in_play.groupby("PlayKey")['x'].transform(lambda x: x if (x[x<0].__len__() <= x[x>=0].__len__()) else -x)
from scipy.interpolate import splint, splprep, splev, BSpline
plays = np.random.choice(in_play["PlayKey"], 100)



fig, ax = plt.subplots(figsize=(12, 12))

ax.set_xlabel("Yards from starting point")

ax.set_ylabel("Horizontal position")

ax.set_title("Approximated B-spline paths")



splines = []

cache = []

for play_no in plays:

    x = in_play.query("PlayKey == @play_no")['x']

    y = in_play.query("PlayKey == @play_no")['y']

    # t = in_play.query("PlayKey == @play_no")['time']

    

    # perturb x, y a tiny bit because splprep hates duplicate values

    x = x + np.random.normal(0, 1e-10, len(x)) 

    y = y + np.random.normal(0, 1e-10, len(x)) 

    """

    x_poly = np.polyfit(t, x, 10)

    fx = np.poly1d(x_poly)



    y_poly = np.polyfit(t, y, 10)

    fy = np.poly1d(y_poly)

    """

    tck, u = splprep([x, y], s=0.1)

    cache.append(u)

    splines.append(tck)

    x_new, y_new = splev(u, tck)



    """

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))



    ax[0].scatter(t, x, alpha=0.2)

    ax[0].plot(t, x_new, color='red')

    ax[1].scatter(t, y, color='orange', alpha=0.2)

    ax[1].plot(t, y_new, color='red')

    ax[2].scatter(x, y, alpha=0.2, color='cyan')

    ax[2].plot(x_new, y_new, color='green')

    ax[2].axvline(x=0, color='b', linewidth=4)

    """

    #ax.scatter(x, y, alpha=0.1, color='cyan')

    ax.plot(x_new, y_new, alpha=0.2, color='red')



"""

event = in_play.query("PlayKey == @play_no")['event']

for i in range(len(x)):

    if i == 0:

        plt.text(x.iloc[0], y.iloc[0], PlayList.query("PlayKey == @play_no")["Position"].values[0], color='black', fontsize=14)

    if pd.notna(event.iloc[i]):

        plt.text(x.iloc[i], y.iloc[i], str(event.iloc[i]), color='black', fontsize=10)

"""

plt.axvline(x=0, color='b', linewidth=4)

plt.savefig('cool.png')
from functools import reduce

def dist(tck1, tck2):

    return abs(reduce(lambda x, y: x*y, splint(0, 1, tck1)) - reduce(lambda x, y: x*y, splint(0, 1, tck2)))
X = np.zeros((len(splines), len(splines)))

for i in range(len(splines)):

    for j in range(len(splines)):

        X[i, j] = dist(splines[i], splines[j])

        

from sklearn.cluster import OPTICS
y = OPTICS(min_samples=4, metric='precomputed', xi=0.2).fit_predict(X)



fig, ax = plt.subplots(figsize=(12, 12))

ax.set_xlabel("Yards from starting point")

ax.set_ylabel("Horizontal position")

ax.set_title("Approximated B-spline paths")



colors = ['blue', 'darkorange', 'lime', 'red', 'fuchsia', 'yellow', 'crimson', 'chocolate', 'cyan', 'dodgerblue']

for i in range(len(splines)):

    u = cache[i]

    tck = splines[i]

    x_new, y_new = splev(u, tck)

    ax.plot(x_new, y_new, alpha=0.3, color=colors[y[i] % len(colors)] if y[i] != -1 else 'gray')

!pip install dtw
from scipy.cluster.hierarchy import dendrogram, linkage

!pip install fastdtw

from fastdtw import fastdtw

linked = linkage(list(in_play.groupby("PlayKey")["x", "y"]), 'single')



labelList = range(1, 11)



plt.figure(figsize=(10, 7))

dendrogram(linked,

            orientation='top',

            labels=labelList,

            distance_sort='descending',

            show_leaf_counts=True)

plt.show()
np.random.seed(42)

plot_plays(np.random.choice(list(pass_ankle_non_inj_synth_keys), 10), path_color="b", 

           event_filter=['tackle', 'first_contact', 'handoff', 'pass_arrived'], annotate=True,

          title="Pass plays on synthetic turf resulting in no injury")

plt.show()

plot_plays(np.random.choice(list(pass_ankle_non_inj_nat_keys), 10), path_color="c", 

           event_filter=['tackle', 'first_contact', 'handoff', 'pass_arrived'], annotate=True,

          title="Pass plays on natural turf resulting in no injury")

plt.show()
"""['QB', 'Missing Data', 'WR', 'ILB', 'RB', 'DE', 'TE', 'FS', 'CB',

       'G', 'T', 'OLB', 'DT', 'SS', 'MLB', 'C', 'NT', 'DB', 'K', 'LB',

       'S', 'HB', 'P']"""

position = "CB"

position_plays = np.random.choice(PlayList[(PlayList["Position"] == position) & (PlayList["FieldType"] == "Synthetic")]["PlayKey"].tolist(), 10)

plot_plays(position_plays, path_color='orange', 

           event_filter=[],

          annotate=True, title="{} plays".format(position))

plt.show()

position_plays = np.random.choice(PlayList[(PlayList["Position"] == position) & (PlayList["FieldType"] == "Natural")]["PlayKey"].tolist(), 10)

plot_plays(position_plays, path_color='red', 

           event_filter=[],

          annotate=True, title="{} plays".format(position))

plt.show()
def summarize(group):

    times = group.loc[group["event"].notnull()]["time"].tolist()

    events = group.loc[group["event"].notnull()]["event"].tolist()

    summary = ' -> '.join(''.join(str(x)) for x in zip(times,events))

    play_id = group["PlayKey"].iloc[0]

    info = merge.loc[merge["PlayKey"] == play_id, ["RosterPosition", "FieldType", "BodyPart"]].to_string(index=False, header=False)

    return " ".join([summary, info])



summaries = injury_plays.groupby("PlayKey").apply(lambda x: summarize(x))

for summary in summaries:

    print(summary)