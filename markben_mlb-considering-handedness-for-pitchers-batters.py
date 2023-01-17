import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os



atbats = pd.read_csv('../input/atbats.csv')
# Add new column to determine if atbat resulted in an out 

# sac fly / sac bunt are considered in favor of the batter (productive outs)

# fielders choice / error in favor of the pitcher

atbats['out'] = (

    atbats['event'].str.contains('out') | 

    atbats['event'].str.contains('Out') | 

    atbats['event'].str.contains('DP') |

    atbats['event'].str.contains('Double Play') | 

    atbats['event'].str.contains('Triple Play') | 

    atbats['event'].str.contains('Fielders Choice') |

    atbats['event'].str.contains('Error') |

    atbats['event'].str.contains('Batter Interference')

)
df1 = atbats.groupby(['p_throws','stand','out'])

d = df1['out'].count()

print(d)
# LHP vs. LHH 

ll_outs = d.iloc[1]

ll_total = d.iloc[0] + d.iloc[1]

ll_out_pct = ll_outs/ll_total

ll_hits = ll_total - ll_outs

ll_hit_pct = ll_hits/ll_total



fig1, ax1 = plt.subplots()

ax1.pie([ll_out_pct, ll_hit_pct],

        autopct='%1.1f%%', shadow=True, startangle=90, 

        textprops={'color':'w','size':'xx-large'})

ax1.set_title('Left-handed Pitcher vs. Left-handed Batter\n2015 - 2018')

ax1.legend(title="Outcome of At Bat",

           loc="upper right", 

           bbox_to_anchor=(1.85,1),

           labels=['Pitcher victorious', 'Batter victorious'])

plt.show()
# LHP vs. RHH 

lr_outs = d.iloc[3]

lr_total = d.iloc[2] + d.iloc[3]

lr_out_pct = lr_outs/lr_total

lr_hits = lr_total - lr_outs

lr_hit_pct = lr_hits/lr_total



fig1, ax1 = plt.subplots()

ax1.pie([lr_out_pct, lr_hit_pct],

        autopct='%1.1f%%', shadow=True, startangle=90, 

        textprops={'color':'w','size':'xx-large'})

ax1.set_title('Left-handed Pitcher vs. Right-handed Batter\n2015 - 2018')

ax1.legend(title="Outcome of At Bat",

           loc="upper right", 

           bbox_to_anchor=(1.5,1),

           labels=['Pitcher victorious', 'Batter victorious'])

plt.show()
# RHP vs. RHH 

rr_outs = d.iloc[7]

rr_total = d.iloc[6] + d.iloc[7]

rr_out_pct = rr_outs/rr_total

rr_hits = rr_total - rr_outs

rr_hit_pct = rr_hits/rr_total



fig1, ax1 = plt.subplots()

ax1.pie([rr_out_pct, rr_hit_pct], 

        autopct='%1.1f%%', shadow=True, startangle=90, 

        textprops={'color':'w','size':'xx-large'})

ax1.set_title('Right-handed Pitcher vs. Right-handed Batter\n2015 - 2018')

ax1.legend(title="Outcome of At Bat",

           loc="upper right", 

           bbox_to_anchor=(1.5,1),

           labels=['Pitcher victorious', 'Batter victorious'])

plt.show()
# RHP vs. LHH 

rl_outs = d.iloc[5]

rl_total = d.iloc[4] + d.iloc[5]

rl_out_pct = rl_outs/rl_total

rl_hits = rl_total - rl_outs

rl_hit_pct = rl_hits/rl_total



fig1, ax1 = plt.subplots()

ax1.pie([rl_out_pct, rl_hit_pct], 

        autopct='%1.1f%%', shadow=True, startangle=90, 

        textprops={'color':'w','size':'xx-large'})

ax1.set_title('Right-handed Pitcher vs. Left-handed Batter\n2015 - 2018')

ax1.legend(title="Outcome of At Bat",

           loc="upper right", 

           bbox_to_anchor=(1.5,1),

           labels=['Pitcher victorious', 'Batter victorious'])

plt.show()