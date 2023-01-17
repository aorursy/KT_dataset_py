try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json

import numpy as np
import matplotlib.pyplot as plt
trajectory_file = '../input/UDysRS_UPDRS_Export/Communication_all_export.txt'

with open(trajectory_file, 'r') as infile:
    comm_dict = json.load(infile)

len(comm_dict.keys())
list(comm_dict.keys())[:20]
list(comm_dict['26-1'].keys())
sorted(comm_dict['26-1']['position'].keys())
%matplotlib inline

plt.plot(np.array(comm_dict['52-3']['position']['Lsho'])[:,0]) # visualizing the horizontal movement
trajectory_file = '../input/UDysRS_UPDRS_Export/LA_split_all_export.txt'

with open(trajectory_file, 'r') as infile:
    la_dict = json.load(infile)

len(la_dict.keys())
list(la_dict.keys())[:5]
list(la_dict['217'].keys())
sorted(la_dict['217']['position'].keys())
%matplotlib inline

plt.plot(np.array(la_dict['217']['position']['Lank_act'])[:,1]) # visualizing the vertical movement
plt.ylim(380, 500)
%matplotlib inline

plt.plot(np.array(la_dict['217']['position']['Lank_rst'])[:,1])
plt.ylim(380, 500)
trajectory_file = '../input/UDysRS_UPDRS_Export/TT_opt_flow_export.txt'

with open(trajectory_file, 'r') as infile:
    tt_dict = json.load(infile)

len(tt_dict.keys())
list(tt_dict.keys())[:5]
list(tt_dict['217'].keys())
%matplotlib inline

plt.plot(tt_dict['217']['Lank'][1])
rating_file = '../input/UDysRS_UPDRS_Export/UDysRS.txt'

with open(rating_file, 'r') as infile:
    ratings = json.load(infile)

ratings.keys()
len(ratings['Communication'].keys())
list(ratings['Communication'].keys())[:10]
ratings['Communication']['215']
rating_file = '../input/UDysRS_UPDRS_Export/UPDRS.txt'

with open(rating_file, 'r') as infile:
    ratings = json.load(infile)

sorted(ratings.keys())
rating_file = '../input/UDysRS_UPDRS_Export/CAPSIT.txt'

with open(rating_file, 'r') as infile:
    ratings = json.load(infile)

sorted(ratings.keys())
ratings['3.10']['214']
sn_file = '../input/UDysRS_UPDRS_Export/sn_numbers.txt'

with open(sn_file, 'r') as infile:
    subj_num = json.load(infile)

len(subj_num.keys())
subj_num['217']
%matplotlib inline

trial_num = '56-2'

plt.plot(np.array(comm_dict[trial_num]['position']['Rsho'])[:,0])
rating_file = '../input/UDysRS_UPDRS_Export/UDysRS.txt'

with open(rating_file, 'r') as infile:
    ratings = json.load(infile)

# If using communication task, will have to split the trial num to get the key
# split can also be done using str.split(), but regexp allows split on space or hyphen
import re

trial_key = re.split("\s|-", trial_num)[0]  
ratings['Communication'][trial_key]
subj_num[trial_key]
