import pandas as pd

pd.set_option('max_colwidth', 999)



train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
train[(train.type=="Game") & (train.event_code==2030)]
specs[specs.event_id=="08fd73f3"]
def cnt_miss(df):

    cnt = 0

    for e in range(len(df)):

        x = df['event_data'].iloc[e]

        y = json.loads(x)['misses']

        cnt += y

    return cnt



if session_type=="Game":

    misses_cnt = cnt_miss(g_session[g_session.event_code == 2030] )

    type_dict['accumulated_game_miss'] += misses_cnt

train[(train.installation_id=="0001e90f") & (train.game_session=="f11eb823348bfa23")]
# For particular game_session

try:

    game_level = json.loads(g_session['event_data'].iloc[-1])["level"]

    type_dict['mean_game_level'] = (type_dict['mean_game_level'] + game_level) / 2.0

except:

    pass



try:

    game_round = json.loads(g_session['event_data'].iloc[-1])["round"]

    type_dict['mean_game_round'] = (type_dict['mean_game_round'] + game_round) / 2.0

except:

    pass



try:

    game_duration = json.loads(g_session['event_data'].iloc[-1])["duration"]

    type_dict['mean_game_duration'] = (type_dict['mean_game_duration'] + game_duration ) / 2.0

except:

    pass 
train[(train.type=="Assessment") & (train.event_code==4020) ]
specs[specs.event_id=="5f0eb72c"]
def get_4020_acc(df):

     

    counter_dict = {'Cauldron Filler (Assessment)_4020_accuracy':0,

                    'Mushroom Sorter (Assessment)_4020_accuracy':0,

                    'Bird Measurer (Assessment)_4020_accuracy':0,

                    'Chest Sorter (Assessment)_4020_accuracy':0 }

        

    for e in ['Cauldron Filler (Assessment)','Bird Measurer (Assessment)','Mushroom Sorter (Assessment)','Chest Sorter (Assessment)']:

        

        Assess_4020 = df[(df.event_code == 4020) & (df.title==activities_map[e])]   

        true_attempts_ = Assess_4020['event_data'].str.contains('true').sum()

        false_attempts_ = Assess_4020['event_data'].str.contains('false').sum()



        measure_assess_accuracy_ = true_attempts_/(true_attempts_+false_attempts_) if (true_attempts_+false_attempts_) != 0 else 0

        counter_dict[e+"_4020_accuracy"] += (counter_dict[e+"_4020_accuracy"] + measure_assess_accuracy_) / 2.0

    

    return counter_dict
train[ (train.event_code==4025) & (train.title == 'Cauldron Filler (Assessment)')]
specs[specs.event_id=="91561152"]
def calculate_accuracy(session):

    Assess_4025 = session[(session.event_code == 4025) & (session.title=='Cauldron Filler (Assessment)')]   

    true_attempts_ = Assess_4025['event_data'].str.contains('true').sum()

    false_attempts_ = Assess_4025['event_data'].str.contains('false').sum()



    accuracy_ = true_attempts_/(true_attempts_+false_attempts_) if (true_attempts_+false_attempts_) != 0 else 0