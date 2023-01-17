import numpy as np # algebra

import pandas as pd # data manipulation

import warnings; warnings.simplefilter('ignore')

df = pd.read_csv('../input/NFL Play by Play 2009-2017 (v4).csv')

df.PassOutcome[df.PassOutcome == 'Complete'] = 1

df.PassOutcome[df.PassOutcome == 'Incomplete Pass'] = 0
#create dataframes for passes to the left & right

pass_right = df[(df['PassLocation'] == 'right') & (df['PlayType'] == 'Pass') & 

                (df['Season'] > 2015) ]

pass_left = df[(df['PassLocation'] == 'left') & (df['PlayType'] == 'Pass') & 

               (df['Season'] > 2015)]



#group all passes by the passer

group_r = pass_right.groupby('Passer') 

#sum the pass attempts and completions

group_r = group_r.agg({'PassOutcome' : 'sum', 'PassAttempt' : 'sum'}).reset_index()

#calculate completion %

group_r['Right_Comp_Percent'] = group_r.apply(lambda row: row.PassOutcome / row.PassAttempt, axis=1)

#rename columns

group_r = group_r.rename(columns={'PassOutcome':'Right_Completions','PassAttempt':'Right_Attempts' })



#same as above but for passes to the left side of the field

group_l = pass_left.groupby('Passer')

group_l = group_l.agg({'PassOutcome' : 'sum', 'PassAttempt' : 'sum'}).reset_index()

group_l['Left_Comp_Percent'] = group_l.apply(lambda row: row.PassOutcome / row.PassAttempt, axis=1)

group_l = group_l.rename(columns={'PassOutcome':'Left_Completions','PassAttempt':'Left_Attempts' })





#Combine left and right passing data

group_agg = pd.merge(group_r,group_l, on='Passer', how='outer')



#Only select passers who have more than 100 attempts to each side of the field

group_agg = group_agg[(group_agg['Right_Attempts'] > 100) &

                      (group_agg['Left_Attempts'] > 100)]



#Create the percent difference metric between the left and right completion percentages

group_agg['Difference'] = group_agg.apply(lambda row: abs(row.Right_Comp_Percent - row.Left_Comp_Percent) , axis=1)



#Select columns we want to display

group_agg = group_agg[['Passer', 'Right_Comp_Percent', 'Left_Comp_Percent', 'Difference' ]]



pd.options.display.float_format = '{:.2%}'.format #format as percents



output = group_agg.sort_values(['Difference'], ascending= False) #Sort values



output.head(25)