!pip install banet
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import scipy.io as sio

from banet.core import InOutPath

from banet.predict import predict_one
# List weight files to use to generate the model outputs

weights_path = '/kaggle/input/banet-pretrained-weights/'

weight_files = ['banetv0.20-val2017-fold0.pth','banetv0.20-val2017-fold1.pth','banetv0.20-val2017-fold2.pth']

weight_files = [weights_path + o for o in weight_files]
region = 'AU2020'

path = InOutPath('/kaggle/input/australia-bushfires-viirs-750m-daily', 'output')

times = pd.DatetimeIndex([pd.Timestamp(o.stem.split('_')[-1]) for o in sorted((path/region).src.ls())])

print(times[0], times[-1])
tstart, tend = times.min(), times.max()

month_start = (tstart + pd.Timedelta(days=31)).month
# Generate the monthly product for the months of October 2019 to January 2020

ptimes = ['2019-10-01', '2019-11-01', '2019-12-01', '2020-01-01']

ptimes = [pd.Timestamp(o) for o in ptimes]

preds_all = []

for time in ptimes:

    time_start = pd.Timestamp((time - pd.Timedelta(days=30)).strftime('%Y-%m-15')) # Day 15, previous month

    times = pd.date_range(time_start, periods=64, freq='D')

    preds = predict_one(path, times, weight_files, region)

    preds = preds[times.month == time.month]

    preds_all.append(preds)

preds_all = np.concatenate(preds_all, axis=0)
preds_all.shape
# Calculate confidence level and dates of burning

ba = preds_all.sum(0)

ba[ba>1] = 1

ba[ba<0.05] = np.nan # Set as nan pixels with confidence level bellow 0.05

bd = preds_all.argmax(0)

bd = bd.astype(float)

bd[np.isnan(ba)] = np.nan
# Plot results

fig, axes = plt.subplots(ncols=2, figsize=(12,7), dpi=300)

im = axes[0].imshow(ba, cmap='jet', vmin=0, vmax=1)

fig.colorbar(im, ax=axes[0])

axes[0].set_title('Confidence level')

im = axes[1].imshow(bd, cmap='RdYlGn_r')

axes[1].set_title('Burndate (starting at 2019-10-01)')

fig.colorbar(im, ax=axes[1])

fig.tight_layout()