import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir
from matplotlib import rc as pltrc

'''
This function first load the video_footage-injury.csv to identify injury plays,
and then loop through all the NGS files to gather NGS data of these plays.

- parameters:
    - outname, the file name to save NGS data of injury plays
    - datapath, where original NGS data are stored
- return:
    - data frame of NGS data for the injury plays
'''
def extract_injury_play_NGS(outname='ngs_injury.csv', datapath='../input/'):

    task = 'injury'    
    if task == 'injury':
        videos = pd.read_csv(datapath + 'video_footage-injury.csv')        
    else:
        videos = pd.read_csv(datapath + 'video_footage-control.csv')
        
    ngsfiles = [filename for filename in listdir(datapath) if filename.startswith("NGS")]

    out = None
    for playfile in ngsfiles:
    
        print(playfile)
        p0 = pd.read_csv(datapath + playfile, low_memory=False)
        
        for i in range(videos.shape[0]):
            p1 = p0.loc[(p0['GameKey']==videos.gamekey[i]) & 
                        (p0['PlayID']==videos.playid[i])].copy()
            if p1.shape[0] > 0:
                if out is None:
                    out = p1
                else:
                    out = out.append(p1)
                    
        p0 = None
        print('injury NGS data frame shape: ', end='')    
        print(out.shape)
                    
    out.to_csv(outname)
    print('save to ', outname)
    
    return out
 
'''
Getting index of NGS data where collision happened.  It assumes collision happened when
the distance between the two players were the shortest.  This assumption is reasonable
but imperfect. So the index of one play was corrected manually by visual inspection.
- parameters:
    - concuss, NGS data of the injured player
    - partner, NGS data of the partner
- return:
    - index of collision from the NGS data 
'''    
def get_index_of_collision(concuss, partner):    
    if partner.shape[0] == 0:
        return np.float('nan')
    
    dist = np.sqrt(pow(concuss['x']-partner['x'],2)+
                   pow(concuss['y']-partner['y'],2))    
    
    ind = np.where(dist == dist.min())
    ind = ind[0][0]
    
    if concuss.at[0,'GameKey']==364 and concuss.at[0,'PlayID']==2489:     
        ind = 192 # hack for this particular play, because their distance was shorter at line of scrimmage
                            
    return ind
        
'''
Get GSISID of a player by its role in a play
- parameters:
    - players, all the players of a play from play_player_role_data.csv
    - role, role of interest
- return:
    - GSISID of the player of the role
'''
def get_GSISID_by_role(players, role):
    x = players.loc[(players.Role==role),'GSISID'].copy()
    x.reset_index(drop=True, inplace=True)
    return x[0]

'''
Convert angle 'o' and 'dir' from NGS to the x and y component of the direction, 
multiplied by speed.  From my experiment, 'o' and 'dir' seem to be different from 
how the manual describes them. They both increase clockwise, instead of counter-clockwise
in the manual.  The 0 degree of 'o' is toward the positive x direction instead of 
positive y.  Nevertheless, the following conversion makes the x and y components
reasonable among all the plays.
- parameters:
    - theta, either 'o' or 'dir'
    - speed, speed at the direction of theta
    - is_dir, is 'dir' provided instead of 'o'
- return:
    - u, x component of the direction
    - v, y component of the direction
'''
def theta_to_UV(theta, speed, is_dir):
    if is_dir:
        theta = np.deg2rad(theta-90) # zero degree is toward the positive y direction      
    else: 
        theta = np.deg2rad(theta) # zero degree is toward the positive x direction
    
    # it seems that theta increases clockwise
    u = speed * np.cos(theta)
    v = speed * -np.sin(theta)
    
    return u, v
    
'''
NGS data of a route of a player
- parameters:
    - ngs, NGS data of a play
    - player_id, GSISID of the player of interest
- return:
    - route of the player in NGS format
'''    
def get_route(ngs, player_id):
    route = ngs[(ngs.GSISID==player_id)].copy()
    
    if route.empty:
        return route
    
    # Reorder by Time and reset index
    route.sort_values(by=['Time'], inplace=True)
    route.reset_index(drop=True, inplace=True)    
    
    # end event of the play
    endevent = route.loc[(route.Event == 'tackle') |                                                
                         (route.Event == 'touchdown') |                        
                         (route.Event=='fair_catch') |                        
                         (route.Event=='punt_downed') |              
                         (route.Event == 'out_of_bounds')]
    
    # takes out routes after the play ended
    route = route.drop(route.index[endevent.index[0]+1:route.shape[0]])
    return route

'''
remove part of the route that was logged after the end of the play
- parameter:
    - route, route of a play in NGS format
    - keep_len, length of the route to keep
- return:
    - truncated route
'''
def remove_route_after_play(route, keep_len):
    return route.drop(route.index[(keep_len+1):route.shape[0]])

'''
plot the route in simple line, and an arrow in the end of the route to indicate direction
- parameters:
    - route, route of a player in a play in NGS format
    - is_dir, True to plot movement direction in the end; False to plot head direction
'''
def plot_route_line(route, is_dir=True):

    plt.plot(route.loc[:,'x'], route.loc[:,'y'], color='black')
    
    if is_dir:
        theta = route.loc[:,'dir']
    else:
        theta = route.loc[:,'o']    
    u, v = theta_to_UV(theta, route.loc[:,'dis']/0.1, is_dir)
    
    last_ind = route.shape[0]-1
#     last_ind = np.arange(10,last_ind, 20)
    plt.quiver(route.loc[last_ind,'x'], route.loc[last_ind,'y'], u[last_ind], v[last_ind],
              scale_units='xy', scale=3, width=0.005)

'''
plot the position and head direction of all 22 players: punting team in orange, 
and return team in blue.
- parameters:
    - init_pos, NGS data of all the players at a point of time
'''
def plot_init_xy_head(init_pos):
    init_return = init_pos.loc[(init_pos.Role.isin(return_team_roles))]
    init_punt = init_pos.loc[~(init_pos.Role.isin(return_team_roles))]    
    
    # plot return team
    u, v = theta_to_UV(init_return['o'], 5*np.ones(init_return['o'].shape), False)
    plt.quiver(init_return['x'], init_return['y'], u, v,
              scale_units='xy', scale=3, width=0.005, color=[0,0.6,1])
    
    # plot punting team
    u, v = theta_to_UV(init_punt['o'], 5*np.ones(init_punt['o'].shape), False)
    plt.quiver(init_punt['x'], init_punt['y'], u, v,
              scale_units='xy', scale=3, width=0.005, color=[1,0.6,0])    

'''
label events on the route of the injured player, partner, and the punt returner.
Events such as ball snap (o), and when the ball was received (+)
- parameters:
    - concuss, NGS data of a play of the injured player 
    - partner, NGS data of a play of the partner
    - returner, NGS data of a play of the punt returner
'''
def label_event_on_route(concuss, partner, returner):
    eoi = ['ball_snap', 'punt_received']
    eoi_shape = ['o', '+']
    returner_s = [None, 200]
    for i in range(len(eoi)):
        concuss_evt = concuss[(concuss.Event == eoi[i])]
        plt.scatter(concuss_evt['x'], concuss_evt['y'], c='red', marker=eoi_shape[i])
        returner_evt = returner[(returner.Event == eoi[i])]        
        plt.scatter(returner_evt['x'], returner_evt['y'], c='black', marker=eoi_shape[i], s=returner_s[i])        
        if not np.isnan(gsisid['partner']):
            partner_evt = partner[(partner.Event == eoi[i])]
            plt.scatter(partner_evt['x'], partner_evt['y'], c='green', marker=eoi_shape[i])      

'''
plot collision on the routes. The blue arrow indicates the direction and speed of the blocker, 
and the orange arrow indicates the head direction and speed of the blocked player.
- parameters:
    - concuss, NGS data of the injured player in a play
    - partner, NGS data of the partner in the same play
    - blocking_id, GSISID that initiated the block
    - contact_ind, the index of when the collision happened
    - advance, how many data points ahead to plot the directions. E.g., if the contact index
               is 100 and advance is 3, then the plot shows the direction at index 97
- return:
    - blocked_uv, the head direction of the blocked player in x, y components
    - blocking_uv, the movement direction of the blocking player in x, y components
'''
def plot_collision(concuss, partner, blocking_id, contact_ind, advance=0):            
    
    if blocking_id == concuss.at[0,'GSISID']:
        blocking = concuss
        blocked = partner        
    else:
        blocking = partner
        blocked = concuss
    
    # plot the head direction who's blocked    
    u, v = theta_to_UV(blocked.loc[contact_ind-advance,'o'], blocked.loc[contact_ind-advance,'dis']/0.1, False)
    plt.quiver(blocked.loc[contact_ind,'x'], blocked.loc[contact_ind, 'y'], u, v, color=[1,0.6,0])
    blocked_uv = [u, v]

    # plot the movement direction of the blocker
    u, v = theta_to_UV(blocking.loc[contact_ind-advance,'dir'], blocking.loc[contact_ind-advance,'dis']/0.1, True)
    plt.quiver(blocking.loc[contact_ind,'x'], blocking.loc[contact_ind, 'y'], u, v, color=[0,0.6,1])
    blocking_uv = [u, v]
    
    return blocked_uv, blocking_uv
    
'''
Get the GSISID of the player who initiated the block
- parameters:
    - block_injury, a play from video_review.csv
    - gsisid, a dict of GSISID of players of interest
'''    
def who_is_blocking(block_injury, gsisid):    
    # who initiate the block? (blocking)
    if block_injury.at['Player_Activity_Derived']=='Blocked':
        gsisid['blocking'] = gsisid['partner']
    elif block_injury.at['Player_Activity_Derived']=='Blocking':
        gsisid['blocking'] = gsisid['injury']
    else:
        gsisid['blocking'] = None
    
    return gsisid

'''
Check whether the collision occured at the line of scrimmage
- paramters:
    - ngs, full NGS data
    - gsisid, a dict of GSISID of players of interest
- return:
    - True or False
'''
def injury_at_return_team_backfield(ngs, gsisid):
    concuss = get_route(ngs, gsisid['injury'])
    partner = get_route(ngs, gsisid['partner'])
    
    # get the index of the route when the concussed player and the partner collided
    contact_ind = get_index_of_collision(concuss, partner)
    
    # punter's initial position
    punter_init = ngs.loc[(ngs.GSISID==gsisid['punter'])&(ngs.Event=='ball_snap')]

    diffx = abs(np.array(concuss.loc[contact_ind, ['x']]) - np.array(punter_init['x']))
    if max(diffx) < 15:
        return True
    else:
        return False    


'''
Plot football field, credit to https://www.kaggle.com/coakeson/watch-the-videos-showing-injury
'''
def plot_field():
    fontsize = 18
    
    # Normal length of field is 120 yards
    plt.xlim(-10, 130)
    plt.xticks(np.arange(0, 130, step=10),
               ['End', 'Goal Line', '10', '20', '30', '40', '50', '40', '30', '20', '10', 'Goal Line', 'End'])
    # Normal width is 53.3 yards
    plt.ylim(-10, 65)
    plt.yticks(np.arange(0, 65, 53.3), ['Sideline', 'Sideline'])
    plt.title('Playing Field', fontsize=fontsize)
    plt.xlabel('yardline', fontsize=fontsize)
    plt.ylabel('width of field', fontsize=fontsize)       
    
'''
Plot a play
- parameters:
    - ngs, NGS data of all players in a play
    - gsisid, a dict of GSISID of players of interest
    - players, a play from play_player_role_data.csv
'''    
def plot_play(ngs, gsisid, players):
    # get the route of concussed player and the partner
    concuss = get_route(ngs, gsisid['injury'])
    partner = get_route(ngs, gsisid['partner'])
    
    # compute their speed
    speed_concuss = concuss['dis']/0.1
    speed_partner = partner['dis']/0.1    
    
    # get the index of the route when the concussed player and the partner collided
    contact_ind = get_index_of_collision(concuss, partner)
    
    # plot the route of the concussed player and the partner, and color coding the speed. 
    # Red for concussed player and blue for the partner
    sns.set()
    plt.figure(figsize=(15,7.5))
    cmap = plt.get_cmap('Spectral')
    vmax = 10
    plt.scatter(concuss['x'], concuss['y'], c=-1.0*speed_concuss, cmap=cmap, alpha=0.8, vmin=-vmax, vmax=vmax) 
    if not np.isnan(gsisid['partner']):
        plt.scatter(partner['x'], partner['y'], c=speed_partner, cmap=cmap, alpha=0.8, vmin=-vmax, vmax=vmax)        
    plt.clim(-vmax, vmax)
    plt.colorbar(label='yards/sec')
    
    # plot initial player location
    init_pos = ngs.loc[(ngs.Event=='ball_snap'), ['GSISID', 'x','y','o', 'dir']]  
    init_pos = pd.merge(init_pos, players, on=['GSISID'])
    plot_init_xy_head(init_pos)
    
    # plot collision location, head direction of the blocked player, and movement direction of the blocker
    if not np.isnan(contact_ind):        
        blocked_uv, blocking_uv = plot_collision(
                concuss, partner, gsisid['blocking'], contact_ind, advance=3)    
        
    # plot route of the punt returner
    returner = get_route(ngs, gsisid['returner'])
    returner = remove_route_after_play(returner, concuss.shape[0])
    plot_route_line(returner, True)    
    
    # label events of interest
    label_event_on_route(concuss, partner, returner)      
    
    plot_field()
    plt.show()   

    # statistics of interest for quantitative analysis
    out = {'blocking_dir': blocking_uv,
           'blocked_o': blocked_uv,
           'punter_init': init_pos.loc[(init_pos.Role=='P',['x','y'])],
           'returner_init': init_pos.loc[(init_pos.Role=='PR',['x','y'])],
           'is_blocker_injured': gsisid['blocking'] == concuss.at[0,'GSISID']}

    return out


# set font for figures
font = {'weight' : 'bold',
        'size'   : 18}
pltrc('font', **font)
pd.set_option('display.max_colwidth', -1)
datapath = '../input/'
    
# load data frames in need
player_roles = pd.read_csv(datapath + 'play_player_role_data.csv')
video = pd.read_csv(datapath + 'video_footage-injury.csv')
review = pd.read_csv(datapath + 'video_review.csv')

# convert Primary_Partner_GSISID to numeric values, and discard plays that do not have a partner
review.loc[:,'Primary_Partner_GSISID'] = pd.to_numeric(
    review.loc[:,'Primary_Partner_GSISID'].copy(), errors='coerce')
review = review.loc[~pd.isna(review.Primary_Partner_GSISID)]

# select only blocking-induced injuries, and exclude those were caused by friendly-fire
block_injury = review[(review.Player_Activity_Derived.isin(['Blocked','Blocking'])&
                       (review.Friendly_Fire=='No'))]
block_injury.reset_index(drop=True, inplace=True)

# get NGS data for the injury plays
ngs_injury_file = Path('ngs_injury.csv')
if ngs_injury_file.is_file():
    ngs_concussion = pd.read_csv(ngs_injury_file)
else:
    ngs_concussion = extract_injury_play_NGS(outname=ngs_injury_file, 
                                             datapath=datapath)
# list all the return team positions
return_team_roles = ['PDL2', 'PDR3', 'PLR2', 'PDR4', 'VRi', 'VRo', 'VLo', 
                     'PDL3', 'PLL', 'PLL2', 'PDR2', 'PDL5', 'PLM', 'PR', 
                     'PDL4', 'VL', 'PDL1', 'PDR1', 'PDR5', 'PDM', 'PDL6', 
                     'PDR6', 'PFB', 'PLL1', 'VR', 'PLR', 'PLR3', 'PLR1', 'PLM1', 
                     'PLL3', 'VLi']  
    
# loop through each play do plot the play and extract statistics of interest    
soi = [] # statistics of interest
for i in range(len(block_injury)):

    # print info of this play
    print('***** ', i, ' *****')
    print(block_injury.iloc[i])

    # Get necessary values for query of NGS data
    gamekey = block_injury.loc[i, 'GameKey']
    playid = block_injury.loc[i, 'PlayID']
    players = player_roles[(player_roles.GameKey==gamekey)&
                            (player_roles.PlayID==playid)].copy()
    players.reset_index(drop=True, inplace=True)
    
    # select NGS data of this play 
    play_ngs = ngs_concussion[(ngs_concussion.GameKey==gamekey)&
                             (ngs_concussion.PlayID==playid)].copy()
    play_ngs.reset_index(drop=True, inplace=True)
        
    # players of interest in this plot
    gsisid = {'injury': block_injury.loc[i, 'GSISID'], 
              'partner': block_injury.loc[i, 'Primary_Partner_GSISID'], 
              'punter': get_GSISID_by_role(players, 'P'), 
              'returner': get_GSISID_by_role(players, 'PR')}
    gsisid = who_is_blocking(block_injury.iloc[i], gsisid)
    
    # discard plays where injuried occured at line of scrimmage
    if injury_at_return_team_backfield(play_ngs, gsisid):
        print('discard the play because injury occured at the line of scrimmage')
        continue
    
    # plot the play and gather statistics
    out = plot_play(play_ngs, gsisid, players)
    soi.append(out)

    # print out the video of this play
    print(video.loc[(video.gamekey==gamekey)&(video.playid==playid), 'PREVIEW LINK (5000K)'])    
n_inj = len(soi) # the number of blocking-induced injury plays
blocking_dir = np.zeros((n_inj,2))
blocked_o = np.zeros((n_inj,2)) 
is_blocker_injured = np.zeros((n_inj,1))
    
# reformat the list of dict into np arrays    
for i in range(n_inj):
    
    # align punting to the same direction 
    if soi[i]['punter_init']['x'].unique() > soi[i]['returner_init']['x'].unique():
        reverse = 1
    else:
        reverse = -1
        
    blocking_dir[i,0] = reverse*soi[i]['blocking_dir'][0]    
    blocking_dir[i,1] = soi[i]['blocking_dir'][1]        
    blocked_o[i,0] = reverse*soi[i]['blocked_o'][0]    
    blocked_o[i,1] = soi[i]['blocked_o'][1]       
    is_blocker_injured[i] = soi[i]['is_blocker_injured'] 

# set font for figures
font = {'weight' : 'bold',
        'size'   : 18}
pltrc('font', **font)    
    
# plot it    
plt.figure(figsize=(8,5))
plt.quiver(np.zeros((n_inj,1)), np.zeros((n_inj,1)), blocked_o[:,0], blocked_o[:,1],
           color=[1,0.6,0], units='width', scale=35)
plt.quiver(np.zeros((n_inj,1)), np.zeros((n_inj,1)), blocking_dir[:,0], blocking_dir[:,1],
           color=[0,0.6,1], units='width', scale=35)
plt.xticks([0],[])
plt.yticks([0],[])
plt.legend(['blocked, head direction','blocking, movement direction'], 
           fontsize=14, loc=1)

n_blocked_backdir = sum(blocked_o[:,0]<=0)
n_blocking_backdir = sum(blocking_dir[:,0]<=0)

print('Out of {} injuries, '.format(n_inj))     
print('{} ({:.2f}%) blocking in the backfield direction,'.format(n_blocking_backdir, 100*n_blocking_backdir/n_inj))
print('{} ({:.2f}%) blocked players were facing backfield'.format(n_blocked_backdir, 100*n_blocked_backdir/n_inj))
print('* Backfield from the perspective of the return team')
# %%
plt.figure(figsize=(5,5))
plt.bar(['Blocked', 'Blocking'], 
        [np.sum(is_blocker_injured==0)/is_blocker_injured.size, 
         np.sum(is_blocker_injured==1)/is_blocker_injured.size],
        alpha=0.3, width=0.35, color='blue')
plt.ylim([0,1])
plt.xlim([-1,2])
plt.ylabel('P(X|concussion)', fontsize=18)
