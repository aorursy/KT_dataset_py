import numpy as np

import pandas as pd

from tqdm.auto import tqdm



import warnings

warnings.filterwarnings("ignore")



train = pd.read_csv('../input/train.csv', dtype={'time_to_failure': np.float32}, usecols=[1])

train.head()
T_chunk = 0.001064

S_block = 1280

S_chunk = 4096

frames_per_chunk = S_chunk - 1/S_block

T_frame = T_chunk/frames_per_chunk

ttf_mean = 6.4

ttf_diffs = [1.4697, 11.54102, 14.18108, 8.85708, 12.69404, 8.056066, 7.05905, 16.10807, 7.906067, 9.63804, 11.42708, 11.02503, 8.829078, 8.56705, 14.75209345, 9.4601, 11.619]

bounds = [0, 37, 212, 333, 532, 697, 925, 1096, 1250, 1457, 1638, 1826, 2052, 2255, 2502, 2795, 3078, 3305, 3525, 3726, 3903, 4146, 4193]

ttf_durations = ttf_diffs[:]

ttf_durations[0] = np.array(ttf_diffs[1:]).mean()

ttf_lower_bound = 0.002



size = 150000

segments = (train.shape[0]) // size



Y_tr = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])



ttf_idx = 0

y = 0

start = 0

bounds_index = 0

for segment in tqdm(range(segments)):

    seg = train.iloc[start:start+size]

    y -= size * T_frame

    

    #if bounds[bounds_index] == segment:

    #    bounds_index += 1

    #    duration = y + 4



    while y < ttf_lower_bound:

        y += ttf_diffs[ttf_idx]

        duration = ttf_durations[ttf_idx]

        ttf_idx += 1

    start += size

    

    diff = y - seg['time_to_failure'].values[-1]

    if abs(diff) > 0.0005985:

        print(ttf_idx, segment, diff, y, seg['time_to_failure'].values)

    Y_tr.loc[segment, 'time_to_failure'] = y

    Y_tr.loc[segment, 'quake_duration'] = duration



Y_tr.to_csv('time_prediction.csv', index=False)