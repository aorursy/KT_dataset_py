from os import listdir
from scipy.stats import ks_2samp
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc as pltrc

'''
get_end_event check NGS data to see how the play ends.
- parameters:
    - thisplay, NGS data of this play for all the players
- return:
    a string that could be 'tackle', 'touchdown', 'touchback', 'fair_catch', 'punt_downed', 
    'out_of_bounds', 'out_of_bounds_direct' (punter punted the ball directly out of bounds),
    and 'no_play' (e.g., false start, delay of game)
'''
def get_end_event(thisplay): 
    end_evts = ['tackle', 'touchdown', 'fair_catch', 'punt_downed', 'out_of_bounds', 'touchback']
    evts = thisplay.Event.unique()
    
    match = set(evts).intersection(end_evts)
    if len(match) == 0:
        return 'no_play'
    else:
        match = match.pop()

        if match == 'out_of_bounds':
            if any(evts == 'punt_received'):
                return match
            else:
                return 'out_of_bounds_direct'
        else:
            return match
    
'''
check the play description if it is a no_play
'''    
def print_no_play_description(playsum, gamekey, playid):    
    
    # show play description
    play_cond = (
        (playsum.GameKey==gamekey) &
        (playsum.PlayID==playid))
    pd.set_option('display.max_colwidth', -1)
    display(playsum.loc[play_cond, ['PlayDescription']])

derived_summary_file = Path('../working/data_fair_catch.csv')
if derived_summary_file.is_file():
    playsum = pd.read_csv(derived_summary_file)
    prepare_data = False
else:
    prepare_data = True
datapath = '../input/'
if prepare_data:
    playsum = pd.read_csv(datapath + 'play_information.csv') # play summary table
    playsum["End_Event"] = "" # how this play ends
    playsum["NumV"] = ""      # number of jammers (V)
    playsum["NumG"] = ""      # number of gunners (G)
if prepare_data:
    # loop through all the NGS files
    ngsfiles = [filename for filename in listdir(datapath) if filename.startswith("NGS")]
    for playfile in ngsfiles:
        print(playfile)
        ngsplays = pd.read_csv(datapath + playfile, low_memory=False)

        # get a concise list of the plays in this ngs file
        playf = ngsplays.drop_duplicates(subset=['GameKey','PlayID'], keep='first').copy()
        playf.reset_index(drop=True, inplace=True)

        # loop through all the plays
        for play_ind in range(len(playf)):
            # check ngsplays to see how this play ends
            play_cond = (
                (ngsplays.GameKey==playf.loc[play_ind,'GameKey']) &
                (ngsplays.PlayID==playf.loc[play_ind,'PlayID']))

            # get the end event
            thisplay = ngsplays[play_cond]
            endevt = get_end_event(thisplay)

            # if you'd like to print out play descriptions of no_play, uncomment below
#             if endevt == 'no_play':
#                 print_no_play_description(playsum, 
#                                           playf.loc[play_ind,'GameKey'], 
#                                           playf.loc[play_ind,'PlayID'])

            # update play summary
            play_cond = (
                (playsum.GameKey==playf.loc[play_ind,'GameKey']) &
                (playsum.PlayID==playf.loc[play_ind,'PlayID']))
            playsum.loc[play_cond, 'End_Event'] = endevt

        ngsplays = None
        playf = None
if prepare_data:
    players = pd.read_csv(datapath + 'play_player_role_data.csv')

    for ind in range(playsum.shape[0]):
        gamekey = playsum.loc[ind, 'GameKey']
        playid = playsum.loc[ind, 'PlayID']

        thisplay = players[(players.GameKey==gamekey)&(players.PlayID==playid)]
        v = set(thisplay['Role']).intersection(['VLi','VLo','VRi','VRo','VR','VL'])
        g = set(thisplay['Role']).intersection(['GLi','GLo','GRi','GRo','GR','GL'])

        playsum.loc[ind,'NumV'] = len(v)
        playsum.loc[ind,'NumG'] = len(g)
if prepare_data:
    playsum = playsum.drop(playsum.index[(playsum.End_Event=='')])   
    playsum.reset_index(drop=True, inplace=True)

    # save the data frame
    playsum.to_csv(derived_summary_file)

# print out how plays end
vc = playsum.End_Event.value_counts()
print(vc)
# playsum = playsum.drop(playsum.index[(playsum.End_Event=='no_play')])   
# playsum = playsum.drop(playsum.index[(playsum.End_Event=='out_of_bounds_direct')])   
# playsum = playsum.drop(playsum.index[(playsum.End_Event=='touchback')])   
# playsum.reset_index(drop=True, inplace=True)
vc = playsum.End_Event.value_counts()
print('percentage of plays')
print(vc/sum(vc))

prob_faircatch = vc['fair_catch']/sum(vc)
vc = playsum.NumG.value_counts()
vc/sum(vc)
nvs = [2,3,4]
fc = []
for nv in nvs:
    pp = playsum.loc[(playsum.NumV==nv) & (playsum.NumG==2)]
    vc = pp.End_Event.value_counts()
    fc.append(vc['fair_catch']/sum(vc))    
    print('{} jammers, {:.2f}% ({}) plays were fair catch. 2 jammers were {:.2f} times more'.format(nv, 
                                                                      100*fc[-1], 
                                                                      vc['fair_catch'],                                                                      
                                                                      fc[0]/fc[-1]))
    
n = playsum.shape[0]
nrun = 100

fair_prob = {'2':[], '3':[], '4':[]}
for i in range(nrun):
    pboot_ind = np.ceil(n * np.random.rand(n))
    pboot = playsum.reindex(pboot_ind)

    for nv in nvs:
        pp = pboot.loc[(pboot.NumV==nv) & (pboot.NumG==2)]
        vc = pp.End_Event.value_counts()
        fair_prob[str(nv)].append(vc['fair_catch']/sum(vc))        
for nv1 in [2, 3]:
    for nv2 in range(nv1+1, 5):
        value, pvalue = ks_2samp(fair_prob[str(nv1)], fair_prob[str(nv2)])
        print('{} vs {} jammers, fair catch probabilities are {:.3f} vs {:.3f}, bonferroni-corrected p-value = {:.4f}'.format(nv1,  nv2, 
                                                                             np.median(fair_prob[str(nv1)]), 
                                                                             np.median(fair_prob[str(nv2)]),                                                                              
                                                                             3*pvalue))


fc = []
fnvs = np.array(nvs)
fnvs = fnvs[::-1]

for nv in fnvs:
    fc.append(np.percentile(fair_prob[str(nv)], [0.005, 0.5, 0.995]))

fc = np.array(fc)    

# set font for figures
font = {'weight' : 'bold',
        'size'   : 14}
pltrc('font', **font)

# plot it
plt.bar(fnvs, fc[:,1], yerr=np.diff(fc).T, align='center', alpha=0.3, width=0.35, color='blue')
plt.xlim([1,5])
plt.xticks(nvs, nvs)
plt.ylabel('percentage of fair catch plays')
plt.xlabel('number of jammers')
plt.show()
review = pd.read_csv(datapath+'video_review.csv')
video = pd.read_csv(datapath+'video_footage-injury.csv')

fc = pd.merge(playsum, review, left_on=['PlayID','GameKey'], right_on=['PlayID','GameKey'])
v = fc.End_Event.value_counts()
inj_faircatch = v['fair_catch']/sum(v)

# plotting
fig, ax = plt.subplots()
index = np.array([1, 2])
bar_width = 0.2
opacity = 0.5

rects1 = plt.bar(index, [prob_faircatch, inj_faircatch], bar_width,
                alpha=opacity, color='b', label='fair_catch')

rects2 = plt.bar(index+bar_width, [1-prob_faircatch, 1-inj_faircatch], bar_width,
                alpha=opacity, color='g', label='others')

plt.xticks(index+0.5*bar_width, ('all plays', 'injuried plays'))
plt.ylabel('percentage of plays')
plt.legend(loc=2)
plt.xlim([0.5,2.75])
fc = fc[(fc.End_Event=='fair_catch')]
fc = pd.merge(fc, video, left_on=['PlayID','GameKey'], right_on=['playid','gamekey'])

pd.set_option('display.max_colwidth', -1)
fc['PREVIEW LINK (5000K)']