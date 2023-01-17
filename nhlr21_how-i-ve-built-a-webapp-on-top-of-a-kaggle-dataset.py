import pandas as pd



def get_player_attributes(playername):

    df = pd.read_csv('../input/data.csv')

    df = df[df.drop(

        ['Value', 'Wage', 'Special', 'Joined', 'Loaned From', 'Contract Valid Until', 'LS', 'ST', 'RS',

         'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM',

         'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', ], 1).columns.tolist()[2:-1]]



    temp_wr = df['Work Rate'].str.split('/', expand=True)

    df['Work Rate'] = temp_wr[0] + ' / ' + temp_wr[1]



    temp_pic = df['Photo'].str.split('players/4', expand=True)

    df['Photo'] = temp_pic[0] + 'players/10' + temp_pic[1]



    temp_logo = df['Club Logo'].str.split('teams/2', expand=True)

    df['Club Logo'] = temp_logo[0] + 'teams/10' + temp_logo[1]



    return df[df['Name'] == playername].to_dict('r')[0]
get_player_attributes('P. Pogba')