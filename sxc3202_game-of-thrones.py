import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import *

import seaborn as sns

from collections import Counter

sns.set_style("white")

%matplotlib inline
battles=pd.read_csv('../input/game-of-thrones/battles.csv')

pd.options.display.max_columns=1000 #显示全部列

battles
battles.info()
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

# 攻击方国王

attacker=battles.groupby('attacker_king').size()

# 防御方国王

defender=battles.groupby('defender_king').size()

# 两者相加统计国王参战次数

king_total= attacker.add(defender,fill_value=0)

# 绘制饼图

king_total.plot.pie(labels=['Balon/Euron Greyjoy', 'Joffrey/Tommen Baratheon', 'Mance Rayder', 'Renly Baratheon', 'Robb Stark', 'Stannis Baratheon'], 

                                             autopct='%.1f%%', fontsize=16, figsize=(6, 6),style = dict, title ='battle_King')
# 攻击方获胜

attacker_win = battles[battles['attacker_outcome'] == 'win'].groupby('attacker_king').size()

# 防御方获胜

defender_win = battles[battles['attacker_outcome'] == 'loss'].groupby('defender_king').size()

# 总共胜利战役数量

king_win = attacker_win.add(defender_win,fill_value=0)

# 柱状图

king_win.div(king_total,fill_value=0).plot.barh(figsize=(6,5),fontsize=16,title = 'rate of winning')
# 各王作为攻击方的次数柱状图

attacker.plot.barh(figsize=(6,5),fontsize=16,title = 'attacker_king')
# 战争类别统计

battle_type=battles.groupby('battle_type').size()

battle_type.plot.pie(labels=['ambush', 'pitched battle', 'razing', 'siege'], 

                                             autopct='%.1f%%', fontsize=16, figsize=(6, 6),style = dict, title ='battle_type')

c = list(Counter([tuple(set(x)) for x in battles.dropna(subset = ["attacker_king", "defender_king"])[["attacker_king", "defender_king"]].values if len(set(x)) > 1]).items())

p = pd.DataFrame(c).sort_values(1).plot.barh(figsize = (10, 6))

p.set(yticklabels = ["%s vs. %s" % (x[0], x[1]) for x in list(zip(*c))[0]], xlabel = "No. of Battles"), p.legend("")
## 从attacker_1 2 3 4和defender_1 2中提取所有的家族

a1=battles.groupby('attacker_1').size()

a2=battles.groupby('attacker_2').size()

a3=battles.groupby('attacker_3').size()

a4=battles.groupby('attacker_4').size()

d1=battles.groupby('defender_1').size()

d2=battles.groupby('defender_2').size()

# 所有series相加

a1.add(a2,fill_value=0).add(a3,fill_value=0).add(a4,fill_value=0).add(d1,fill_value=0).add(d2,fill_value=0)

a1.plot.barh()
# Tully 为attacker时的attacker_king是谁？

attacker_king_tully = battles[(battles['attacker_1']=='Tully')|(battles['attacker_2']=='Tully')|(battles['attacker_3']=='Tully')|

       (battles['attacker_4']=='Tully')].groupby('attacker_king').size()



# Tully 为defender时的defender_king是谁？

defender_king_tully = battles[(battles['defender_1']=='Tully')|(battles['defender_2']=='Tully')].groupby('defender_king').size()



# Tully 家族在战争中支持的国王

attacker_king_tully.add(defender_king_tully,fill_value=0)
# Tyrell 为attacker时的attacker_king是谁？

attacker_king_tyrell = battles[(battles['attacker_1']=='Tyrell')|(battles['attacker_2']=='Tyrell')|(battles['attacker_3']=='Tyrell')|

       (battles['attacker_4']=='Tyrell')].groupby('attacker_king').size()



# Tyrell 为defender时的defender_king是谁？

defender_king_tyrell = battles[(battles['defender_1']=='Tyrell')|(battles['defender_2']=='Tyrell')].groupby('defender_king').size()



# Tyrell家族在战争中支持的国王

attacker_king_tyrell.add(defender_king_tyrell,fill_value=0)
totalday = battles.groupby('year').size()

totalday
# 每年的夏天的占比

summer = battles[battles.summer == 1].groupby('year').size()

summer.div(totalday,fill_value=0)
df0 = battles.dropna(subset = ['major_death', 'major_capture'])

#给每个区域分组并计算major_death major_capture的和

data = df0.groupby('region').sum()[['major_death', 'major_capture']]

p = pd.concat([data, df0.region.value_counts().to_frame()], axis = 1) 

p = p.sort_values('region', ascending = False)

p.plot.barh()

plt.xlabel('count')

plt.title('attacker_outcome_size')
data = battles.dropna(axis = 0, subset = ["attacker_size", "defender_size", "attacker_outcome"]).copy(deep = True)

colors = [sns.color_palette()[0] if x == "win" else "lightgray" for x in data.attacker_outcome.values]

p = data.plot.scatter("attacker_size", "defender_size", c = colors, s = 100, lw = 2.)

_ = p.set(xlabel = "Attacker Size", ylabel = "Defender Size")
import statsmodels.api as sm

battles['count']=1

battletype=battles.pivot_table('count',index='battle_type',columns='attacker_outcome',aggfunc=sum,fill_value=0).reset_index()

battlesize = battles[["attacker_size", "defender_size", "attacker_outcome","count"]].dropna(axis = 0)

battlesize['size_diff']=battlesize['attacker_size'] /battlesize['defender_size']

battlesize[['loss','win']]=pd.get_dummies(battlesize['attacker_outcome'])

battlesize=battlesize.drop('loss',axis=1)

battlesize['intercept']=1

logit_mod=sm.Logit(battlesize['win'],battlesize[['intercept','size_diff']])

result=logit_mod.fit()

result.summary()