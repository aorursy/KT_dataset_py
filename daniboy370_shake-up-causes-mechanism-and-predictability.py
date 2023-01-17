import os

import math

import time

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.style as style

import matplotlib.pyplot as plt



PATH_root = '/kaggle/input/' # --> '/world-data-by-country-2020'

# os.chdir(PATH_root) # os.listdir()



df = pd.read_csv( PATH_root+'/competitions-shakeup/df_Elo.csv')
df_dn = df[df['Shake'] <0]['Shake']

df_up = df[df['Shake']>=0]['Shake']

df_stats = df['Shake'].describe()

df_len, df_mean, df_median, df_std, df_min, df_max = df_stats[0], df_stats[1], df_stats[5], df_stats[2], df_stats[3], df_stats[-1]

labels = ['Shake-up', 'No-Shake', 'Shake-down']



# --------- Mean and median extraction -------- #

df_up_g = df[df['Shake']>0]['Shake']

df_eq = df[df['Shake']==0]['Shake']

df_up_mean, df_up_median = np.mean( df_up ), np.median( df_up )

df_dn_mean, df_dn_median = np.mean( df_dn ), np.median( df_dn )



df_dn_len, df_eq_len, df_up_g_len = len(df_dn), len(df_eq), len(df_up_g)

sizes = pd.Series([df_up_g_len, df_eq_len, df_dn_len])*100/df_len



explode = (0.02, 0.2, 0.02)

colors = ['#66b3ff','#99ff99', '#ff9999']



fig, ax = plt.subplots( figsize=(8, 8) )

ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%', shadow=True, startangle=-110, 

        labeldistance=1.1, pctdistance=0.6, radius=1, textprops={'fontsize': 20})



# Equal aspect ratio ensures that pie is drawn as a circle

ax.axis('equal')

plt.tight_layout()

plt.show()
print('Total participants : ', int(df_len))

print('Range : [', int(df_min),',', int(df_max),']')

print(f'Mean : {df_mean:.2f}')

print('Median : ', int(df_median))

print(f'STD : {df_std:.2f}')
rug_kws={"color": "g"}

kde_kws={"color": "k", "lw": 3, "label": "KDE"}

hist_kws={"linewidth": 4, "alpha": 0.75, "color": "g"}

sns.set(style="white", palette="muted", color_codes=True)



bins_num, df_len = 50, len(df)

plt.figure(figsize=(15, 6))



ax = plt.gca()

ax.set_facecolor('#fffccc')

sns.set(font_scale = 1.5)

sns.set_style('whitegrid')

sns.distplot(df['Shake'], color="blue", kde=False, hist=True, bins=bins_num, label='Shake', kde_kws=kde_kws, rug_kws=rug_kws, hist_kws=hist_kws)



plt.title('Shake Histogram ( bin-width = '+str(int(df_len/bins_num))+' places )', fontsize = 20) 

plt.grid(linestyle='-', linewidth=1.25)

plt.xlabel('$\Delta LB$', fontsize = 20)

plt.ylabel('Frequency', fontsize = 20)

plt.xlim([-500, 500])

plt.xticks(fontsize=18)

plt.yticks(fontsize=18)

plt.axvline(x=df_mean, label='Mean', c='k', linestyle='dashed', linewidth=3.5)

plt.axvline(x=df_median, label='Median', c='k', linestyle='-', linewidth=2.5)

plt.legend(fontsize = 18)

plt.show()
rug_kws={"color": "g"}

kde_kws={"color": "k", 'linestyle':'--', "lw": 4, "label": "KDE"}

hist_kws={"linewidth": 3.5, "alpha": 0.75, "color": "g"}

sns.set(style="white", palette="muted", color_codes=True)



bins_num = 100

plt.figure(figsize=(15, 7))



ax = plt.gca()

ax.set_facecolor('#fffccc')

sns.set(font_scale = 1.5)

sns.set_style('whitegrid')

sns.distplot(df['Shake'], color="blue", kde=True, hist=True, bins=bins_num, label='Shake', kde_kws=kde_kws, rug_kws=rug_kws, hist_kws=hist_kws)





plt.title('Shake PDF ( bin-ratio = '+str( np.round( 100/bins_num, 3 ))+' [%] )', fontsize = 20) 

plt.grid(linestyle='-', linewidth=1.0)

plt.xlabel('$\Delta LB$', fontsize = 20)

plt.ylabel('Density (KDE)', fontsize = 20)

plt.xlim([-1000, 1100])

# plt.xlim([df_min*0.6, df_max*0.7])

plt.xticks(fontsize=18)

plt.yticks(fontsize=18)

plt.legend(fontsize = 18)

plt.show()
df_hist = np.histogram( df['Shake'], bins=1000 )



Range = [[-29, 29], [-143, 143], [-369, 369]]



for i,t in enumerate( Range ):

    x_min, x_max = Range[i][0], Range[i][1]

    a = df_hist[1] > x_min

    b = df_hist[1] < x_max

    prob_tot = df_hist[1][ a & b ]

    prob_min, prob_max = np.min(prob_tot), np.max(prob_tot)

    prob_min_s, prob_max_s = str(int(prob_min)), str(int(prob_max))

    prob_CDF = str( np.round( 100*( df_hist[0][ (a & b)[:-1]] / df_len ).sum() , 3) )

    print('P('+ prob_min_s+ ' <= X <= '+ prob_max_s+ ') = '+ prob_CDF+ ' [%]')
plt.figure(figsize=(15, 7))

ax = plt.gca()

ax.set_facecolor('#fffef0')

sns.set(font_scale = 1.5)

sns.set_style('whitegrid')

hist_b={"linewidth": 2, "alpha": 0.75, "color": "b"}

hist_r={"linewidth": 2, "alpha": 0.75, "color": "r"}



# ----------- Mean and median lines ----------- #

plt.axvline(x=df_up_median, label='SU-Median', c='b', linestyle='dashed', linewidth=3.5, alpha=0.8)

plt.axvline(x=df_dn_median, label='SD-Median', c='r', linestyle='dashed', linewidth=3.5, alpha=0.8)

plt.axvline(x=df_up_mean, label='SU-Mean', c='b', linestyle='-', linewidth=3, alpha=0.6)

plt.axvline(x=df_dn_mean, label='SD-Mean', c='r', linestyle='-', linewidth=3, alpha=0.6)

plt.legend(fontsize = 18)

sns.distplot(df_up, color="blue", kde=True, bins=90, label='Shake-up', kde_kws=dict(linewidth=0), hist_kws=hist_b)

sns.distplot(df_dn, color="red", kde=True, bins=150, label='Shake-down', kde_kws=dict(linewidth=0), hist_kws=hist_r)



# --------------------------------------------- #

plt.title("Shake Histogram", fontsize = 25)

plt.ylabel('Frequency', fontsize = 20)  #  Density (KDE)

plt.xlabel('$\Delta LB$', fontsize = 20)

plt.grid(linestyle='-', linewidth=1.0)

plt.xlabel('$\Delta LB$', fontsize = 20)

plt.ylabel('Density (KDE)', fontsize = 20)

plt.xlim([-600, 600])

# plt.xlim([df_min*0.3, df_max*0.35])

plt.xticks(fontsize=18)

plt.yticks(fontsize=18)

plt.show()
print( '# -------- Shake-up ------- #')

print( 'Median =  %d     Mean =  %.2f' %(int(df_up_median), df_up_mean))

print( '\n# -------- Shake-down ----- #')

print( 'Median = %d     Mean = %.2f' %(int(df_dn_median), df_dn_mean))
plt.figure(figsize=(16, 8))

ax = plt.gca()

ax.set_facecolor('#fffef0')



Df_up = df[df['Shake']>=0]

Df_dn = df[df['Shake']<0]



plt.scatter( Df_up['Rank_public'], Df_up['Rank_private'], label='Shake-up', s=65, c='b', alpha=1, linewidths=1, edgecolors='w')

plt.scatter( Df_dn['Rank_public'], Df_dn['Rank_private'], label='Shake-down', s=65, c='r', alpha=0.8, linewidths=1, edgecolors='w')

plt.legend(fontsize = 22, loc='lower right')



h_medal = np.round( 0.1*df_len )

plt.hlines(y=h_medal, xmin=0, xmax=0.8*df_len, color='k', linestyle='dashed', linewidth=2.5, alpha=0.6)

plt.axvline(x=h_medal, label='Medals', c='k', alpha=0.5, linestyle='dashed', linewidth=2.5)



# --------------------------------------------- #

plt.title('Private rank vs. Public rank', fontsize = 25)

plt.ylabel('Private LB', fontsize = 20)

plt.xlabel('Public LB', fontsize = 20)

plt.grid(linestyle='-', linewidth=1.0)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.show()
plt.figure(figsize=(16, 8))

ax = plt.gca()

ax.set_facecolor('#fffef0')



Df_up = df[df['Shake']>=0]

Df_dn = df[df['Shake']<0]



plt.scatter( Df_up['Rank_public'], Df_up['Shake'], label='Shake-up', s=65, c='b', alpha=1, linewidths=1, edgecolors='w')

plt.scatter( Df_dn['Rank_public'], Df_dn['Shake'], label='Shake-down', s=65, c='r', alpha=0.8, linewidths=1, edgecolors='w')

plt.legend(fontsize = 18, loc='lower right')



h_medal = np.round( 0.1*df_len )

plt.axvline(x=h_medal, label='Medals', c='k', alpha=0.5, linestyle='dashed', linewidth=2.5)



# --------------------------------------------- #

plt.title('$\Delta LB$ vs. Public rank', fontsize = 25)

plt.xlabel('Public LB', fontsize = 20)

plt.ylabel('$\Delta LB$', fontsize = 20)

plt.grid(linestyle='-', linewidth=1.0)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.ylim([np.min(Df_dn['Shake'])*0.9, np.max(Df_up['Shake'])*1.25])

plt.show()
h_bronze = h_medal

h_silver = np.round( 0.05*df_len )

h_gold   = np.round( 10 + 0.002*df_len )



# shortcut for np.array casting

def logical_list( l1, l2 ):

    L1, L2 = np.array( l1 ), np.array( l2 )

    return pd.Series( np.logical_and( L1, L2) )

     

df_bronze = df[ logical_list( (df['Rank_public'] > h_silver), (df['Rank_public'] <= h_bronze) ) ]

df_silver = df[ logical_list( (df['Rank_public'] > h_gold), (df['Rank_public'] <= h_silver) ) ]

df_gold   = df[ logical_list( (df['Rank_public'] >= 0), (df['Rank_public'] <= h_gold) ) ]



# ----------------- Visualization ----------------- #

plt.figure(figsize=(15, 7))

ax = plt.gca()

ax.set_facecolor('#f2f9fa')



plt.scatter( df_bronze['Rank_public'], df_bronze['Rank_private'], label='Bronze Medal', s=90, c='brown', alpha=1, linewidths=0.8, edgecolors='k')

plt.scatter( df_silver['Rank_public'], df_silver['Rank_private'], label='Silver Medal', s=90, c='silver', alpha=1, linewidths=1, edgecolors='k')

plt.scatter( df_gold['Rank_public'],   df_gold['Rank_private'],   label='Gold Medal', s=90, c='gold', alpha=1, linewidths=1, edgecolors='k')



# --------------------------------------------- #

plt.legend(fontsize = 20, loc='upper right')

plt.hlines(y=h_medal, xmin=0, xmax=np.max(df_bronze['Rank_public']), color='k', linestyle='dashed', linewidth=2.5, alpha=0.7)

plt.title('Re-ranking of Public LB medalists ', fontsize = 23)

plt.xlabel('Public LB', fontsize = 21)

plt.ylabel('Private LB', fontsize = 21)

plt.grid(linestyle='-', linewidth=1.0)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

# plt.ylim([-50, 600])

# plt.ylim([np.min(Df_dn['Shake'])*0.7, np.max(Df_up['Shake'])*1.25])

plt.show()
h_bronze = h_medal

h_silver = np.round( 0.05*df_len )

h_gold   = np.round( 10 + 0.002*df_len )



# shortcut for np.array casting

def logical_list( l1, l2 ):

    L1, L2 = np.array( l1 ), np.array( l2 )

    return pd.Series( np.logical_and( L1, L2) )

     

df_bronze = df[ logical_list( (df['Rank_private'] > h_silver), (df['Rank_private'] <= h_bronze) ) ]

df_silver = df[ logical_list( (df['Rank_private'] > h_gold), (df['Rank_private'] <= h_silver) ) ]

df_gold   = df[ logical_list( (df['Rank_private'] >= 0), (df['Rank_private'] <= h_gold) ) ]



# ----------------- Visualization ----------------- #

plt.figure(figsize=(15, 7))

ax = plt.gca()

ax.set_facecolor('#f2f9fa')



plt.scatter( df_bronze['Rank_public'], df_bronze['Rank_private'], label='Bronze Medal', s=90, c='brown', alpha=1, linewidths=0.8, edgecolors='k')

plt.scatter( df_silver['Rank_public'], df_silver['Rank_private'], label='Silver Medal', s=90, c='silver', alpha=1, linewidths=1, edgecolors='k')

plt.scatter( df_gold['Rank_public'],   df_gold['Rank_private'],   label='Gold Medal', s=90, c='gold', alpha=1, linewidths=1, edgecolors='k')



# --------------------------------------------- #

plt.legend(fontsize = 20, loc='lower right')

plt.axvline(x=h_medal, label='Medals', c='k', alpha=0.7, linestyle='dashed', linewidth=2.5)

plt.title('Re-ranking of Private LB medalists', fontsize = 23)

plt.xlabel('Public LB', fontsize = 21)

plt.ylabel('Private LB', fontsize = 21)

plt.grid(linestyle='-', linewidth=1.0)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

# plt.ylim([np.min(Df_dn['Shake'])*0.7, np.max(Df_up['Shake'])*1.25])

plt.show()