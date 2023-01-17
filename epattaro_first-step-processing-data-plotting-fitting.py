## importing libraries and functions ##



import pandas as pd

import numpy as np

import copy

import matplotlib.pyplot as plt

import statsmodels.api as sm



import pylab



from scipy.optimize import curve_fit



%matplotlib inline
DF=pd.read_csv('../input/shot_logs.csv')



#DF=copy.deepcopy(pd.read_csv('shot_logs.csv'))
print ('columns:', DF.shape[1], '\n')

print ('DF length:', len(DF), '\n')



print ('printing columns names and sample data:\n')



for i in DF.columns:

    print (i,DF[i][0],type(DF[i][0]))
DF.isnull().any()
print (len (DF))

print (len (DF[DF.SHOT_CLOCK.isnull()==True]))



print (round(float(len (DF[DF.SHOT_CLOCK.isnull()==True]))/float(len (DF)),2)*100,'%')
print (len (DF[DF['TOUCH_TIME']<0]))

print (round(len (DF[DF['TOUCH_TIME']<0])/float(len (DF)),3)*100,'%')
del DF['SHOT_RESULT']

del DF['W']



DF=DF[DF['TOUCH_TIME']>0]
def data_split(x):

    

    (a,b)=x.split('-')

    a=a.strip()

    

    return a

    

def home_team_split (x):



    (a,b)=x.split('-')

    

    if '@' in b:

        (b1,b2)=b.split('@')

    if 'vs.' in b:

        (b1,b2)=b.split('vs.')

        

    b1=b1.strip()

    return b1

    

def adversary_team_split (x):

    

    (a,b)=x.split('-')

    

    if '@' in b:

        (b1,b2)=b.split('@')

    if 'vs.' in b:

        (b1,b2)=b.split('vs.')



    b2=b2.strip()

    return b2
DF['date']=DF['MATCHUP'].apply(data_split)

DF['date']=DF['date'].apply(pd.to_datetime)

DF['home_team']=DF['MATCHUP'].apply(home_team_split)

DF['adv_team']=DF['MATCHUP'].apply(adversary_team_split)
list(DF.columns)
text_opts={'fontsize':20,'fontweight':'bold'}
def count_shots(x):

    y=len(DF[DF['player_id']==x])

    return y



def count_shots_made(x):

    dummy_DF=DF[DF['FGM']==1]

    y=len(dummy_DF[dummy_DF['player_id']==x])

    return y



def count_games(x):

    y=(DF[DF['player_id']==x])

    z=len(y.groupby('GAME_ID'))

    return z



def max_attempts_in_game(x):

    y=DF[DF['player_id']==x]

    z=y.groupby('GAME_ID').count()

    k=np.max(z)[0]

    return k



players=pd.DataFrame(list(set(DF['player_id'])))

players.columns=['player_id']



players['total_attempts']=players['player_id'].apply(count_shots)

players['FGM']=players['player_id'].apply(count_shots_made)

players['ratio_FGM']=players['FGM']/players['total_attempts']



players['ratio_FGM_low'],players['ratio_FGM_upp']=sm.stats.proportion_confint(players['FGM'], players['total_attempts'], method='jeffrey')



players['ratio_FGM_low']=players['ratio_FGM']-players['ratio_FGM_low']

players['ratio_FGM_upp']=players['ratio_FGM_upp']-players['ratio_FGM']



players['total_games']=players['player_id'].apply(count_games)

players['avg_attempts_per_game']=players['total_attempts']/players['total_games']

players['avg_FGM_per_game']=players['FGM']/players['total_games']

players['max_attempts_in_game']=players['player_id'].apply(max_attempts_in_game)







print (len(players), 'different players')



players=players.sort_values('ratio_FGM')
dummy=players.sort_values('ratio_FGM', ascending=False)



plt.figure(figsize=(20,10))



#plt.scatter(dummy.index,dummy.ratio_FGM.values)



plt.plot(dummy.ratio_FGM.values, 'ko')



plt.errorbar(np.arange(len(dummy)), dummy.ratio_FGM.values, yerr=[dummy['ratio_FGM_low'],dummy['ratio_FGM_upp']])



plt.grid()

plt.xticks([], **text_opts)

plt.yticks(**text_opts)



plt.ylim(0,1)

plt.xlim(-5,290)



plt.title('FGM % by player', **text_opts)

plt.ylabel('FGM %', **text_opts)

plt.xlabel('different players', **text_opts)
set (DF[DF['player_id'].isin(players[players['ratio_FGM']>.65]['player_id'])].player_name)
players.head()
plt.figure(figsize=(20,10))



dummy=players.sort_values('avg_attempts_per_game', ascending=False)



plt.plot(dummy['avg_attempts_per_game'].values, 'ko', color='black')

plt.plot(dummy['avg_attempts_per_game'].values*dummy['ratio_FGM'].values, 'ko', color='green')



plt.grid()

plt.xticks([], **text_opts)

plt.yticks(**text_opts)

plt.xlim(-5,290)



plt.title('players atempts, FGM and efficiency', **text_opts)

plt.ylabel('attempts & FGM', **text_opts)

plt.xlabel('different players', **text_opts)



plt.legend(['attempts','FGM'],markerscale=2, loc='upper left', prop={'size':24})



plt.twinx()



plt.plot(dummy['ratio_FGM'].values, 'ko', color='red')

plt.ylim(0,1)

plt.yticks(color='red', **text_opts)



plt.legend(['efficiency'],markerscale=2, prop={'size':24})
mades=DF[DF['FGM']==1]

missed=DF[DF['FGM']==0]



max_touch_time=np.max(DF['TOUCH_TIME'])

max_shot_dist=np.max(DF['SHOT_DIST'])



shot_distance_DF=pd.DataFrame(np.zeros(len(np.arange(0,max_shot_dist+0.1,0.1))))

touch_time_DF=pd.DataFrame(np.zeros(len(np.arange(0,max_touch_time+0.1,0.1))))



shot_distance_DF['distance']=np.arange(0,max_shot_dist+0.1,0.1)

touch_time_DF['time']=np.arange(0,max_touch_time+0.1,0.1)



def num_attempts_SHOT_DIST (x):

    

    z=DF[DF['SHOT_DIST']==x]

    k=len(z)

    

    return k



def num_attempts_TOUCH_TIME (x):

    

    z=DF[DF['TOUCH_TIME']==x]

    k=len(z)

    

    return k



def num_fgm_SHOT_DIST (x):

    

    z=mades[mades['SHOT_DIST']==x]

    k=len(z)

    

    return k



def num_fgm_TOUCH_TIME (x):

    

    z=mades[mades['TOUCH_TIME']==x]

    k=len(z)

    

    return k



shot_distance_DF['attempts']=shot_distance_DF['distance'].apply(num_attempts_SHOT_DIST)

touch_time_DF['attempts']=touch_time_DF['time'].apply(num_attempts_TOUCH_TIME)



shot_distance_DF['fgm']=shot_distance_DF['distance'].apply(num_fgm_SHOT_DIST)

touch_time_DF['fgm']=touch_time_DF['time'].apply(num_fgm_TOUCH_TIME)



shot_distance_DF['ratio']=shot_distance_DF['fgm']/shot_distance_DF['attempts'].fillna(0)

touch_time_DF['ratio']=touch_time_DF['fgm']/touch_time_DF['attempts'].fillna(0)



(shot_distance_DF['ratio_uncertainty_low'],shot_distance_DF['ratio_uncertainty_upp']) = sm.stats.proportion_confint(shot_distance_DF['fgm'], shot_distance_DF['attempts'], method='jeffrey')



shot_distance_DF['ratio_uncertainty_low']=shot_distance_DF['ratio']-shot_distance_DF['ratio_uncertainty_low']

shot_distance_DF['ratio_uncertainty_upp']=shot_distance_DF['ratio_uncertainty_upp']-shot_distance_DF['ratio']



(touch_time_DF['ratio_uncertainty_low'],touch_time_DF['ratio_uncertainty_upp']) = sm.stats.proportion_confint(touch_time_DF['fgm'], touch_time_DF['attempts'], method='jeffrey')



touch_time_DF['ratio_uncertainty_low']=touch_time_DF['ratio']-touch_time_DF['ratio_uncertainty_low']

touch_time_DF['ratio_uncertainty_upp']=touch_time_DF['ratio_uncertainty_upp']-touch_time_DF['ratio']



shot_distance_DF=shot_distance_DF.fillna(0)

touch_time_DF=touch_time_DF.fillna(0)



shot_distance_DF=shot_distance_DF[shot_distance_DF['attempts']>0]

touch_time_DF=touch_time_DF[touch_time_DF['attempts']>0]
tolerance=0.1



DF_1=shot_distance_DF[(tolerance>=shot_distance_DF['ratio_uncertainty_low'])&(tolerance>=shot_distance_DF['ratio_uncertainty_upp'])]

DF_2=shot_distance_DF[(tolerance<shot_distance_DF['ratio_uncertainty_low'])&(tolerance<shot_distance_DF['ratio_uncertainty_upp'])]



DF_3=touch_time_DF[(tolerance>=touch_time_DF['ratio_uncertainty_low'])&(tolerance>=touch_time_DF['ratio_uncertainty_upp'])]

DF_4=touch_time_DF[(tolerance<touch_time_DF['ratio_uncertainty_low'])&(tolerance<touch_time_DF['ratio_uncertainty_upp'])]



DF_5=DF_3[DF_3['time']>1]





# DF_1=shot_distance_DF[(shot_distance_DF['ratio']*tolerance>=shot_distance_DF['ratio_uncertainty_low'])&(shot_distance_DF['ratio']*tolerance>=shot_distance_DF['ratio_uncertainty_upp'])]

# DF_2=shot_distance_DF[(shot_distance_DF['ratio']*tolerance<shot_distance_DF['ratio_uncertainty_low'])&(shot_distance_DF['ratio']*tolerance<shot_distance_DF['ratio_uncertainty_upp'])]



# DF_3=touch_time_DF[(touch_time_DF['ratio']*tolerance>=touch_time_DF['ratio_uncertainty_low'])&(touch_time_DF['ratio']*tolerance>=touch_time_DF['ratio_uncertainty_upp'])]

# DF_4=touch_time_DF[(touch_time_DF['ratio']*tolerance<touch_time_DF['ratio_uncertainty_low'])&(touch_time_DF['ratio']*tolerance<touch_time_DF['ratio_uncertainty_upp'])]
plt.figure(figsize=(20,10))



plt.scatter(DF_1.distance,DF_1.ratio, color='blue')

plt.errorbar(DF_1.distance,DF_1.ratio, yerr=[DF_1['ratio_uncertainty_low'],DF_1['ratio_uncertainty_upp']], color='blue')



plt.scatter(DF_2.distance,DF_2.ratio, color='red')

plt.errorbar(DF_2.distance,DF_2.ratio, yerr=[DF_2['ratio_uncertainty_low'],DF_2['ratio_uncertainty_upp']], color='red')



plt.xticks(**text_opts)

plt.yticks(**text_opts)



plt.ylabel('ratio (%)', **text_opts)

plt.xlabel('SHOT_DISTANCE', **text_opts)



plt.grid()



plt.title('FGM ratio per distance', **text_opts)



plt.legend([('uncertainties <= %s' %(tolerance*100) +' %'),('uncertainties > %s' %(tolerance*100) +' %')],markerscale=3, prop={'size':24})
def sigmoid(x, x0, k, a, c):

     y = a / (1 + np.exp(-k*(x-x0))) + c

     return y
plt.figure(figsize=(20,10))



xdata = DF_1['distance']

ydata = DF_1['ratio']



popt, pcov = curve_fit(sigmoid, xdata, ydata)



x = np.linspace(0, 25, 50)

y = sigmoid(x, *popt)



plt.scatter(xdata, ydata, s=25)

plt.errorbar(xdata,ydata,yerr=[DF_1['ratio_uncertainty_low'],DF_1['ratio_uncertainty_upp']])

plt.plot(x,y, color='green', linewidth=5)

plt.legend(['fit','data'], markerscale=3, prop={'size':24})

plt.grid()



plt.xticks(**text_opts)

plt.yticks(**text_opts)



plt.ylabel('ratio (%)', **text_opts)

plt.xlabel('distance', **text_opts)



plt.title('distance with fit', **text_opts)



plt.tight_layout()



#########

#########



plt.figure(figsize=(20,10))



plt.title('residues of fit', **text_opts)



plt.scatter(xdata, ydata-sigmoid(xdata, *popt), s=25)

plt.errorbar(xdata, ydata-sigmoid(xdata, *popt),yerr=[DF_1['ratio_uncertainty_low'],DF_1['ratio_uncertainty_upp']])

plt.plot(xdata,np.zeros(len(xdata)), linewidth=5, color='black')

plt.xticks(**text_opts)

plt.yticks(**text_opts)



plt.grid()



plt.ylabel('residue', **text_opts)

plt.xlabel('distance', **text_opts)



plt.legend(['fit','residue'], markerscale=3, prop={'size':24})



plt.tight_layout()
plt.figure(figsize=(20,10))



plt.scatter(DF_3.time,DF_3.ratio, color='blue')

plt.errorbar(DF_3.time,DF_3.ratio, yerr=[DF_3['ratio_uncertainty_low'],DF_3['ratio_uncertainty_upp']], color='blue')



plt.scatter(DF_4.time,DF_4.ratio, color='red')

plt.errorbar(DF_4.time,DF_4.ratio, yerr=[DF_4['ratio_uncertainty_low'],DF_4['ratio_uncertainty_upp']], color='red')



plt.grid()





plt.title('FGM ratio per touch time', **text_opts)



plt.xticks(**text_opts)

plt.yticks(**text_opts)



plt.ylabel('ratio (%)', **text_opts)

plt.xlabel('touch time (s)', **text_opts)



plt.legend([('uncertainty <= %s per cent of ratio' %(tolerance*100)),('uncertainty > %s per cent of ratio' %(tolerance*100))],markerscale=3, prop={'size':24})
plt.figure(figsize=(20,10))



xdata = DF_5['time']

ydata = DF_5['ratio']



#popt, pcov = curve_fit(sigmoid, xdata, ydata)

(a,b)=np.polyfit(xdata,ydata,1)



x=np.linspace(1,10,50)

y=x*a+b



plt.scatter(xdata, ydata, s=25)

plt.errorbar(xdata,ydata,yerr=[DF_5['ratio_uncertainty_low'],DF_5['ratio_uncertainty_upp']])

plt.plot(x,y, color='green', linewidth=5)

plt.legend(['fit','data'], markerscale=3, prop={'size':24})

plt.grid()



plt.xticks(**text_opts)

plt.yticks(**text_opts)



plt.ylabel('ratio (%)', **text_opts)

plt.xlabel('touch time (s)', **text_opts)



plt.title('touch time with fit', **text_opts)



plt.tight_layout()

######

######





plt.figure(figsize=(20,10))



plt.title('residues of fit', **text_opts)



plt.scatter(xdata, ydata-(xdata*a+b), s=25)

plt.errorbar(xdata, ydata-(xdata*a+b),yerr=[DF_5['ratio_uncertainty_low'],DF_5['ratio_uncertainty_upp']])

plt.plot(xdata,np.zeros(len(xdata)), linewidth=5, color='black')

plt.xticks(**text_opts)

plt.yticks(**text_opts)



plt.ylabel('ratio (%)', **text_opts)

plt.xlabel('touch time (s)', **text_opts)



plt.legend(['fit','residue'], markerscale=3, prop={'size':24})



plt.grid()



plt.tight_layout()