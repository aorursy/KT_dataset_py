import numpy as np

import pandas as pd



import plotly.express as px

import plotly.graph_objects as go



df = pd.read_csv("/kaggle/input/na-lcs-summer-2020-player-proximities/proximities.csv")

df.drop("Unnamed: 0", inplace=True, axis=1)

teams = df['game_won'].drop_duplicates().tolist()
df.head()
df_sup = df[df['player_role']=="support"]
df3 = pd.DataFrame(index=teams)



for role in ['top', 'jgl', 'mid', 'adc']:

    df3[role] = df_sup[df_sup['teammate_role']==role].groupby('team').mean()['proximity'].tolist()



    

for role in ['top','jgl','mid','adc']:

    df3[role] -= df3[role].min()

    df3[role] /= df3[role].max()

df3
means = [0.0]*4

for i,role in enumerate(['top','jgl','mid','adc']):

    means[i] = (df3[role].mean() - df3[role].min()) / (df3[role].max() - df3[role].min())
n = 2



nums = df3.iloc[n].tolist()

nums.append(nums[0])

means.append(means[0])



fig = go.Figure()





fig.add_trace(go.Scatterpolar(r = means, 

                    theta = ['Top','  Jungle','Mid  ','ADC', 'Top'],

                             name = "League average"))



fig.add_trace(go.Scatterpolar(r = nums, 

                    theta = ['Top','  Jungle','Mid  ','ADC', 'Top'],

                             fill = "toself",

                             name = "%s Support" % df3.index[n].upper()))





fig.update_layout(title="%s: Support proximities" % df3.index[n].upper())

fig.update_layout(

  polar=dict(

    radialaxis=dict(

      visible=True,

    range = [0,1],

        tickfont_size = 9,

        nticks=3,

        angle=180,

        tickangle=-180

    ),

  ),

  showlegend=True

)





fig.show()
for n in range(10):

    nums = df3.iloc[n].tolist()

    nums.append(nums[0])



    fig = go.Figure()





    fig.add_trace(go.Scatterpolar(r = means, 

                        theta = ['Top','  Jungle','Mid  ','ADC', 'Top'],

                                 name = "League average"))



    fig.add_trace(go.Scatterpolar(r = nums, 

                        theta = ['Top','  Jungle','Mid  ','ADC', 'Top'],

                                 fill = "toself",

                                 name = "%s Support" % df3.index[n].upper()))





    fig.update_layout(title="%s: Support proximities" % df3.index[n].upper())

    fig.update_layout(

      polar=dict(

        radialaxis=dict(

          visible=True,

        range = [0,1],

            tickfont_size = 9,

            nticks=3,

            angle=180,

            tickangle=-180

        ),

      ),

      showlegend=True

    )





    fig.show()