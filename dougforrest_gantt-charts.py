from itertools import cycle

import json

import os

import pandas as pd

import plotly.figure_factory as ff

import plotly.express as px

import sys

from textwrap import wrap



def load_tables(data_dir=os.path.join('/kaggle/input/data-science-bowl-2019'),

                tables=[],

                kwargs={},

                namespace=sys.modules[__name__],

                file_types='csv'):



    data_files = os.listdir(data_dir)

    tables_filname_mapping = {f.rsplit('.')[0]:

                              f for f in data_files if

                              f.rsplit('.')[1] == file_types}



    for table_i in tables:

        print(f"Loading {table_i}")

        table_i_kwargs = kwargs.get(table_i, {})

        table_i_name = table_i.lower()

        table_i_path = os.path.join(data_dir,

                                    tables_filname_mapping[table_i])



        df_i = pd.read_csv(table_i_path,

                           **table_i_kwargs)



        print(f'{table_i_name} has '

              '{:,} rows and {:,} cols'.format(df_i.shape[0],

                                               df_i.shape[1]))

        print(f"{df_i.dtypes} \n")



        setattr(namespace, table_i_name, df_i)

        del df_i



        

load_tables(

                  tables=['specs', 'train'],

                  kwargs= {'train': {

                                     'parse_dates': ['timestamp'],

                                      'dtype': {

                                          'event_code': 'category',

                                          'title': 'category',

                                          'type': 'category',

                                          'world': 'category'

                                        }

                                     },

                         'test': {

                                     'parse_dates': ['timestamp'],

                                      'dtype': {

                                          'event_code': 'category',

                                          'title': 'category',

                                          'type': 'category',

                                          'world': 'category'

                                        }

                                     }                           

                        },

                  namespace=sys.modules[__name__])
def user_gantt(df, user=None, session=None, hours_threshold=2):

    """Plot a Gantt chart for a dataframe, optionally provide a

    installation_id and game_session for filtering. Hover over bars for detail.



    Args:

        df (TYPE): Description

        user (None, optional): installation_id

        session (None, optional): game_session

        hours_threshold (int, optional): user inactivity threshold to break plots



    Returns:

        None - shows plots

    """



    if session and user:

        df = df.loc[(df.installation_id == user) & (df.game_session == session)]

    elif user:

        df = df.loc[(df.installation_id == user)]

    else:

        df



    df['event_data_json'] = df['event_data'].map(lambda e: json.loads(e))

    df['misses'] = df.event_data_json.map(lambda e: e.get('misses', None))

    df['correct'] = df.event_data_json.map(lambda e: e.get('correct', None))



    colors = {'Assessment': 'rgb(220, 0, 0)',

              'Activity': (1, 0.9, 0.16),

              'Game': 'rgb(0, 255, 100)',

              'Clip': 'rgb(0, 100, 100)'}



    sessions = df.groupby(['game_session'], sort=False)

    output = {}



    i = 0



    for session_id, session in sessions:

        d = {}

        if len(set(session.title.values)) > 1:

            print(f'Warning found multiple session titles {session.title.unique()}')



        if len(set(session.type.values)) > 1:

            print(f'Warning found multiple session type {session.type.unique()}')



        d['Task'] = session.iloc[0].title

        d['Start'] = session.timestamp.min()

        d['Finish'] = session.timestamp.max()

        d['Resource'] = session.iloc[0].type

        timedelta = (d['Finish'] - d['Start']).total_seconds() / 60

        d['Misses'] = list(session.loc[~pd.isnull(session.misses), 'misses'].values)

        d['Correct'] = list(session.loc[~pd.isnull(session.correct), 'correct'].values)

        d['Description'] = f"Session: {session_id}<br>Time spent {timedelta:.2f} mins<br>"

        if d['Misses']:

            d['Description'] += f"Misses: {d['Misses']}<br>"



        if d['Correct']:

            d['Description'] += f"Correct: {d['Correct']}<br>"



        if output:

            last_finish = output[i][-1]['Finish']

            idle_gap_hours = (d['Start'] - last_finish).total_seconds() / 60 / 60

        else:

            idle_gap_hours = None



        if idle_gap_hours and idle_gap_hours > hours_threshold:

            i += 1



        try:

            output[i].append(d)

        except KeyError:

            output[i] = [d]



    for k, v in output.items():

        if k != 0:

            idle_gap_hours = (output[k][0]['Start'] - output[k-1][-1]['Finish']).total_seconds() / 60 / 60

            print(f"Idle time {idle_gap_hours:.02f} hours")



        start = output[k][0]['Start'].strftime("%m-%d-%Y")

        fig = ff.create_gantt(v,

                              colors=colors,

                              index_col='Resource',

                              show_colorbar=True,

                              group_tasks=True,

                              title=f"Activity on {start}")

        fig.show()



    # return(output)





def format_event_data(event, width=30):

    if isinstance(event, str):

        event = json.loads(event)

    output = ''

    for k, v in event.items():

        value_str = "<br>".join(wrap(str(v), width=width))

        output += f"    {k}: {value_str}<br>"



    return(f"Event Data:<br>{output}")











def user_session_gantt(df, user=None, session=None, seconds_threshold=100, specs=None):

    """Plot a detailed Gantt chart for a given installation_id and game_session.

     Hover over bars for detail.



    Args:

        df (pd.DataFrame): A dataframe

        user (str): installation_id

        session (str): game_session

        seconds_threshold (int, optional): threshold in seconds to break into multiple charts

        specs (pd.DataFrame): The specs dataframe for informational hover



    Returns:

        None - plots shown

    """

    df = df.loc[(df.installation_id == user) & (df.game_session == session)]



    color_pallete = px.colors.qualitative.Dark24

    event_codes = df.event_code.unique()

    zip_list =zip(event_codes, cycle(color_pallete)) if len(event_codes) > len(color_pallete) else zip(event_codes, color_pallete)



    colors = {k:v for k,v in zip_list}





    output = {}



    i = 0

    previous_timestamp = None

    for idx, row in df.iterrows():

        d = {}

        title = row.title

        d['Task'] = row.event_code

        if not previous_timestamp:

            previous_timestamp = row.timestamp

        d['Start'] = previous_timestamp

        d['Finish'] = row.timestamp

        previous_timestamp = row.timestamp

        d['Resource'] = row.event_code

        timedelta = (d['Finish'] - d['Start']).total_seconds()

        info = ''

        if isinstance(specs, pd.DataFrame):

            info = specs.loc[specs.event_id == row.event_id, 'info'].values[0]

            info = "<br>".join(wrap(info, width=40))

            info = f"Info: {info}"



        event_data = format_event_data(row.event_data)



        d['Description'] = f"Event Code: {row.event_code}<br>Event ID: {row.event_id}<br>Time spent: {timedelta:.2f} seconds<br>{info}<br>{event_data}"

        if output:

            last_finish = output[i][-1]['Finish']

            idle_gap = d['Start'] - last_finish

        else:

            idle_gap = None



        if idle_gap and idle_gap.days > seconds_threshold:

            i += 1



        try:

            output[i].append(d)

        except KeyError:

            output[i] = [d]



    for k, v in output.items():

        if k != 0:

            idle_gap = output[k][0]['Start'] - output[k-1][-1]['Finish']

            print(f"Idle time {idle_gap.days} days")



        start = output[k][0]['Start'].strftime("%m-%d-%Y")

        fig = ff.create_gantt(v,

                              colors=colors,

                              index_col='Resource',

                              show_colorbar=True,

                              group_tasks=True,

                              title=f"{title} on {start}")

        fig.show()



    # return(output)

sample_session = train.sample(1, random_state=9)

installation_id = sample_session.iloc[0]['installation_id']

game_session = sample_session.iloc[0]['game_session']

print(f"User {installation_id} session {game_session}")
# help(user_gantt)

user_gantt(train, user=installation_id)
# help(user_session_gantt)

user_session_gantt(train, user=installation_id, session=game_session, specs=specs)