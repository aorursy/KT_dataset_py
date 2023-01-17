# useful imports
import numpy as np
import pandas as pd
# importing the data
video_review = pd.read_csv("../input/video_review.csv")
player_role  = pd.read_csv("../input/play_player_role_data.csv")
player_punt  = pd.read_csv("../input/player_punt_data.csv")
play_info = pd.read_csv("../input/play_information.csv")
ngs_2016_1 = pd.read_csv("../input/NGS-2016-post.csv")
ngs_2016_2 = pd.read_csv("../input/NGS-2016-pre.csv")
ngs_2016_3 = pd.read_csv("../input/NGS-2016-reg-wk1-6.csv")
ngs_2016_4 = pd.read_csv("../input/NGS-2016-reg-wk13-17.csv")
ngs_2016_5 = pd.read_csv("../input/NGS-2016-reg-wk7-12.csv")
ngs_2017_1 = pd.read_csv("../input/NGS-2017-post.csv")
ngs_2017_2 = pd.read_csv("../input/NGS-2017-pre.csv")
ngs_2017_3 = pd.read_csv("../input/NGS-2017-reg-wk1-6.csv")
ngs_2017_4 = pd.read_csv("../input/NGS-2017-reg-wk13-17.csv")
ngs_2017_5 = pd.read_csv("../input/NGS-2017-reg-wk7-12.csv")
full_players = pd.merge(player_punt, player_role, on=['GSISID'], how = 'left')
full_set = pd.merge(full_players, play_info, on=['GameKey','PlayID'], how = 'left')
set_df=full_set.dropna()

set_df['Home_Team_Visit_Team'] = set_df['Home_Team_Visit_Team'].astype(str)
set_df['Score_Home_Visiting'] = set_df['Score_Home_Visiting'].astype(str)
set_df=set_df.join(set_df['Home_Team_Visit_Team'].str.split('-', 1, expand=True).rename(columns={0:'Home',1:'Away'}))
set_df=set_df.join(set_df['Score_Home_Visiting'].str.split(' - ', 1, expand=True).rename(columns={0:'Home_score',1:'Away_score'}))

# Date
set_df["Game_Date"] = pd.to_datetime(set_df["Game_Date"], format = '%m/%d/%Y')

# drop columns that were split
set_df = set_df.drop(['Home_Team_Visit_Team'], axis = 1)
set_df = set_df.drop(['Score_Home_Visiting'], axis = 1)
set_df['PlayDescription'] = set_df['PlayDescription'].astype(str)
# fair catch
fair_catch = []
for row in set_df['PlayDescription'].str.contains('fair catch'):
    if row == True:
        fair_catch.append(1)
    else:
        fair_catch.append(0)

# injury
injury = []
for row in set_df['PlayDescription'].str.contains('injured'):
    if row == True:
        injury.append(1)
    else:
        injury.append(0)

# downed
downed = []
for row in set_df['PlayDescription'].str.contains('downed'):
    if row == True:
        downed.append(1)
    else:
        downed.append(0)
# fumbles
fumbles = []
for row in set_df['PlayDescription'].str.contains('FUMBLES'):
    if row == True:
        fumbles.append(1)
    else:
        fumbles.append(0)
# muffs
muffs = []
for row in set_df['PlayDescription'].str.contains('MUFFS'):
    if row == True:
        muffs.append(1)
    else:
        muffs.append(0)

# Touchback
touchback = []
for row in set_df['PlayDescription'].str.contains('Touchback'):
    if row == True:
        touchback.append(1)
    else:
        touchback.append(0)
        
# Touchdown
touchdown = []
for row in set_df['PlayDescription'].str.contains('TOUCHDOWN'):
    if row == True:
        touchdown.append(1)
    else:
        touchdown.append(0)

# Out of bounds
oob = []
for row in set_df['PlayDescription'].str.contains('bounds'):
    if row == True:
        oob.append(1)
    else:
        oob.append(0)

# add new columns to the df 
set_df["fair_catch"] = fair_catch
set_df["injury"] = injury
set_df["downed"] = downed
set_df["fumble"] = fumbles
set_df["muff"] = muffs
set_df["touchback"] = touchback
set_df["touchdown"] = touchdown
set_df['out_bounds'] = oob
corr_columns = ['fair_catch', 'injury', 'downed', 'fumble', 'muff', 'touchback', 
                'touchdown', 'out_bounds']
df_corr = set_df[corr_columns]
corr = df_corr.corr()
corr.style.background_gradient().set_precision(2)
# sorting the values to improve visibility
corr['injury'].sort_values()
pos_role = pd.merge(video_review, full_players, on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'], how='left')
pos_role.Primary_Partner_GSISID = pos_role.Primary_Partner_GSISID.astype(str)
full_players.GSISID = full_players.GSISID.astype(str)
pos_role_partner = pd.merge(pos_role, full_players, how='left', left_on=['Season_Year', 'GameKey', 'PlayID', 'Primary_Partner_GSISID'], 
                            right_on=['Season_Year', 'GameKey', 'PlayID', 'GSISID'])
pos_role_partner = pos_role_partner.dropna().drop_duplicates(subset=['Season_Year', 'GameKey', 'PlayID', 'GSISID_x', 'Player_Activity_Derived', 'Turnover_Related', 'Primary_Impact_Type','Primary_Partner_GSISID']).reset_index(drop=True)
author = []
for row in (pos_role_partner['Player_Activity_Derived'].
            str.contains('|'.join(['Blocking', 'Tackling']))):
    if row == True:
        author.append(1)
    else:
        author.append(0)

receiving = []
for row in (pos_role_partner['Player_Activity_Derived'].
            str.contains('|'.join(['Blocked', 'Tackled']))):
    if row == True:
        receiving.append(1)
    else:
        receiving.append(0)

pos_role_partner['Author'] = author
pos_role_partner['Receiver'] = receiving
RB_act = []
WR_act = []
LB_act = []
DB_act = []
TE_act = []
ST_act = []

RB_rec = []
WR_rec = []
LB_rec = []
DB_rec = []
TE_rec = []
ST_rec = []
for row in range(len(pos_role_partner)):
    if pos_role_partner.iat[row, -2] == 1:
        if pos_role_partner.iat[row, -8] not in ('RB', 'FB'):
            RB_act.append(0)
        else:
            RB_act.append(1)

        if pos_role_partner.iat[row, -8] != 'WR':
            WR_act.append(0)
        else:
            WR_act.append(1)

        if pos_role_partner.iat[row, -8] not in ('ILB', 'OLB', 'MLB'):
            LB_act.append(0)
        else:
            LB_act.append(1)

        if pos_role_partner.iat[row, -8] not in ('CB', 'SS', 'FS', 'S'):
            DB_act.append(0)
        else:
            DB_act.append(1)

        if pos_role_partner.iat[row, -8] != 'TE':
            TE_act.append(0)
        else:
            TE_act.append(1)

        if pos_role_partner.iat[row, -8] not in ('DE', 'LS', 'P'):
            ST_act.append(0)
        else:
            ST_act.append(1)
        if pos_role_partner.iat[row, -4] not in ('RB', 'FB'):
            RB_rec.append(0)
        else:
            RB_rec.append(1)

        if pos_role_partner.iat[row, -4] != 'WR':
            WR_rec.append(0)
        else:
            WR_rec.append(1)

        if pos_role_partner.iat[row, -4] not in ('ILB', 'OLB', 'MLB'):
            LB_rec.append(0)
        else:
            LB_rec.append(1)

        if pos_role_partner.iat[row, -4] not in ('CB', 'SS', 'FS', 'S'):
            DB_rec.append(0)
        else:
            DB_rec.append(1)

        if pos_role_partner.iat[row, -4] != 'TE':
            TE_rec.append(0)
        else:
            TE_rec.append(1)

        if pos_role_partner.iat[row, -4] not in ('DE', 'LS', 'P'):
            ST_rec.append(0)
        else:
            ST_rec.append(1)

    if pos_role_partner.iat[row, -2] == 0:
        if pos_role_partner.iat[row, -8] not in ('RB', 'FB'):
            RB_rec.append(0)
        else:
            RB_rec.append(1)

        if pos_role_partner.iat[row, -8] != 'WR':
            WR_rec.append(0)
        else:
            WR_rec.append(1)

        if pos_role_partner.iat[row, -8] not in ('ILB', 'OLB', 'MLB'):
            LB_rec.append(0)
        else:
            LB_rec.append(1)

        if pos_role_partner.iat[row, -8] not in ('CB', 'SS', 'FS', 'S'):
            DB_rec.append(0)
        else:
            DB_rec.append(1)

        if pos_role_partner.iat[row, -8] != 'TE':
            TE_rec.append(0)
        else:
            TE_rec.append(1)

        if pos_role_partner.iat[row, -8] not in ('DE', 'LS', 'P'):
            ST_rec.append(0)
        else:
            ST_rec.append(1)
        if pos_role_partner.iat[row, -4] not in ('RB', 'FB'):
            RB_act.append(0)
        else:
            RB_act.append(1)

        if pos_role_partner.iat[row, -4] != 'WR':
            WR_act.append(0)
        else:
            WR_act.append(1)

        if pos_role_partner.iat[row, -4] not in ('ILB', 'OLB', 'MLB'):
            LB_act.append(0)
        else:
            LB_act.append(1)

        if pos_role_partner.iat[row, -4] not in ('CB', 'SS', 'FS', 'S'):
            DB_act.append(0)
        else:
            DB_act.append(1)

        if pos_role_partner.iat[row, -4] != 'TE':
            TE_act.append(0)
        else:
            TE_act.append(1)

        if pos_role_partner.iat[row, -4] not in ('DE', 'LS', 'P'):
            ST_act.append(0)
        else:
            ST_act.append(1)
pos_role_partner['RB_act'] = RB_act
pos_role_partner['WR_act'] = WR_act
pos_role_partner['LB_act'] = LB_act
pos_role_partner['DB_act'] = DB_act
pos_role_partner['TE_act'] = TE_act
pos_role_partner['ST_act'] = ST_act

pos_role_partner['RB_rec'] = RB_rec
pos_role_partner['WR_rec'] = WR_rec
pos_role_partner['LB_rec'] = LB_rec
pos_role_partner['DB_rec'] = DB_rec
pos_role_partner['TE_rec'] = TE_rec
pos_role_partner['ST_rec'] = ST_rec
corr_columns_pos_2 = ['RB_act', 'WR_act', 'LB_act', 'DB_act', 'TE_act', 'ST_act', 'RB_rec', 'WR_rec',
       'LB_rec', 'DB_rec', 'TE_rec', 'ST_rec']
df_corr_pos_2 = pos_role_partner[corr_columns_pos_2]
corr_pos_2 = df_corr_pos_2.corr()
corr_pos_2.style.background_gradient().set_precision(2)