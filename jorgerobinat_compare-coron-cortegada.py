import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



coron=pd.read_csv('/kaggle/input/wind-coron/coron_all.csv',parse_dates=["time"]).set_index("time")

cortegada=pd.read_csv('/kaggle/input/wind-coron/cortegada_all.csv',parse_dates=["time"]).set_index("time")

join = cortegada.join(coron, lsuffix='_corte', rsuffix='_coron')

join.columns

correlations=[]

#select variable threshold

vars_threshold=['mod_corte','mod_coron',"spd_o_coron","spd_o_corte"]

var_threshold=vars_threshold[1]

for threshold in range(0,8,2):

    correlation=join[join[var_threshold]>threshold][['dir_o_corte', 'dir_corte','dir_o_coron','dir_coron',]].corr()

    correlations.append(correlation)

 

fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8),)





sns.heatmap(correlations[0],annot = True, fmt='.2f',vmin=0, vmax=1, center= 0.5,ax=ax,)

ax.set_title('Correlation wind predicted intensity more than {}'.format(0), )

sns.heatmap(correlations[1],annot = True, fmt='.2f',vmin=0, vmax=1, center= 0.5,ax=ax2)

ax2.set_title('Correlation wind predicted intensity more than {}'.format(2))

sns.heatmap(correlations[2],annot = True, fmt='.2f',vmin=0, vmax=1, center= 0.5,ax=ax3)

ax3.set_title('Correlation wind predicted intensity more than {}'.format(4))

sns.heatmap(correlations[3],annot = True, fmt='.2f',vmin=0, vmax=1, center= 0.5,ax=ax4)

ax4.set_title('Correlation wind predicted intensity more than {}'.format(6))

plt.show()

threshold=4

#select variable threshold

vars_threshold=['mod_corte','mod_coron',"spd_o_coron","spd_o_corte"]

var_threshold=vars_threshold[1]

g = sns.PairGrid(join[join[var_threshold]>threshold][['dir_o_corte', 'dir_corte','dir_o_coron','dir_coron',]].sample(8000))

g = g.map_diag(plt.hist)

g = g.map_offdiag(plt.scatter)
join_am=join.between_time("00:00","12:00")

join_pm=join.between_time("13:00","23:00")

fig, axs = plt.subplots(2,figsize = (8,6))

threshold=3

#select variable threshold

vars_threshold=['mod_corte','mod_coron',"spd_o_coron","spd_o_corte"]

var_threshold=vars_threshold[1]

print("Correlation wind predicted intensity more than {} m/s at AM  and PM hours".format(threshold))

sns.heatmap(join_am[join_am[var_threshold]>=threshold][['dir_o_corte', "dir_corte",'dir_o_coron','dir_coron',]].corr(),annot=True,cmap="YlGnBu",

            ax=axs[0],fmt='.3f',vmin=0, vmax=1, center= 0.5,)

sns.heatmap(join_pm[join_pm["mod_corte"]>=threshold][['dir_o_corte', 'dir_corte','dir_o_coron','dir_coron',]].corr(),annot=True,cmap="YlGnBu",

            ax=axs[1],fmt='.3f',vmin=0, vmax=1, center= 0.5,)

plt.show()
correlations=join[["spd_o_corte",'mod_corte',"spd_o_coron",'mod_coron']].corr()

ax=sns.heatmap(correlations,annot = True, fmt='.2f',vmin=0, vmax=1, center= 0.5,)

ax.set_title('Correlation wind intensity ')

plt.show()
g = sns.PairGrid(join[["spd_o_corte",'mod_corte',"spd_o_coron",'mod_coron']].sample(8000))

g = g.map_diag(plt.hist)

g = g.map_offdiag(plt.scatter)
join_am_corr=join.between_time("00:00","12:00")

join_pm_corr=join.between_time("13:00","23:00")

threshold=0

#select variable threshold

vars_threshold=['mod_corte','mod_coron',"spd_o_coron","spd_o_corte"]

var_threshold=vars_threshold[1]

fig, axs = plt.subplots(2,figsize = (7,6))

print("Correlation wind intensity at AM and PM hours variable threshold: {} threshold: {}".format(var_threshold,threshold))

sns.heatmap(join_am[join_am[var_threshold]>=threshold][["spd_o_corte",'mod_corte',"spd_o_coron",'mod_coron']].corr(),annot=True,cmap="YlGnBu",ax=axs[0],fmt='.3f',vmin=0, vmax=1, center= 0.5,)

sns.heatmap(join_pm[join_pm[var_threshold]>=threshold][["spd_o_corte",'mod_corte',"spd_o_coron",'mod_coron']].corr(),annot=True,cmap="YlGnBu",ax=axs[1],fmt='.3f',vmin=0, vmax=1, center= 0.5,)

plt.show()
g1=(join[["spd_o_corte","spd_o_coron"]].between_time("00:00","12:00")).hist(bins=20)

g2=join[["spd_o_corte","spd_o_coron"]].between_time("13:00","23:00").hist(bins=20)
result=pd.DataFrame(index=join.index)

result["spd_cortegada_am"]=join["spd_o_corte"].between_time("00:00","12:00")

result["spd_coron_am"]=join["spd_o_coron"].between_time("00:00","12:00")

result["spd_cortegada_pm"]=join["spd_o_corte"].between_time("13:00","23:00")

result["spd_coron_pm"]=join["spd_o_coron"].between_time("13:00","23:00")

g=result.plot(kind="box",grid=True,figsize=(8,4))

result.describe()