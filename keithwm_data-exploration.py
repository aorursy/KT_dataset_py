import numpy as np

import pandas as pd

from os.path import join as opj

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = 10, 10
df = pd.read_json(opj('..', 'input', 'train.json'))

bands_1 = np.stack([np.array(band).reshape(75, 75) for band in df['band_1']], axis=0)

bands_2 = np.stack([np.array(band).reshape(75, 75) for band in df['band_2']], axis=0)



v_max = max(abs(bands_1).max(), abs(bands_2).max())

v_min = -v_max
samples_ship = df.index[df['is_iceberg'] == 0]

samples_iceberg = df.index[df['is_iceberg'] == 1]
N_plots = 5

fig, axss = plt.subplots(N_plots, 4)

for i_plot, axs in enumerate(axss):

    sample_ship = samples_ship[i_plot]

    sample_iceberg = samples_iceberg[i_plot]

    axs[0].imshow(bands_1[sample_ship], vmin=v_min, vmax=v_max)

    axs[1].imshow(bands_2[sample_ship], vmin=v_min, vmax=v_max)

    axs[2].imshow(bands_1[sample_iceberg], vmin=v_min, vmax=v_max)

    axs[3].imshow(bands_2[sample_iceberg], vmin=v_min, vmax=v_max)

    

fig, axss = plt.subplots(N_plots, 2)

for i_plot, axs in enumerate(axss):

    sample_ship = samples_ship[i_plot]

    sample_iceberg = samples_iceberg[i_plot]

    axs[0].imshow(bands_1[sample_ship] - bands_2[sample_ship])

    axs[1].imshow(bands_1[sample_iceberg] - bands_2[sample_iceberg])

    

    

fig, axs = plt.subplots(1, 4)

axs[0].imshow(bands_1[samples_ship].sum(axis=0)/samples_ship.size, vmin=v_min, vmax=v_max)

axs[1].imshow(bands_2[samples_ship].sum(axis=0)/samples_ship.size, vmin=v_min, vmax=v_max)

axs[2].imshow(bands_1[samples_iceberg].sum(axis=0)/samples_iceberg.size, vmin=v_min, vmax=v_max)

axs[3].imshow(bands_2[samples_iceberg].sum(axis=0)/samples_iceberg.size, vmin=v_min, vmax=v_max)

    

fig, axs = plt.subplots(1, 2)

axs[0].imshow(bands_1[samples_ship].sum(axis=0)/samples_ship.size -

              bands_1[samples_iceberg].sum(axis=0)/samples_iceberg.size)

axs[1].imshow(bands_2[samples_ship].sum(axis=0)/samples_ship.size - 

              bands_2[samples_iceberg].sum(axis=0)/samples_iceberg.size)



fig, axs = plt.subplots(1, 2)

axs[0].imshow(bands_1[samples_ship].sum(axis=0)/samples_ship.size -

              bands_2[samples_iceberg].sum(axis=0)/samples_iceberg.size)

axs[1].imshow(bands_2[samples_ship].sum(axis=0)/samples_ship.size - 

              bands_2[samples_iceberg].sum(axis=0)/samples_iceberg.size)



alpha = 0.5

fig, axss = plt.subplots(1, 2, squeeze=False, figsize=(10, 5))

_ = axss[0,0].hist(bands_1[samples_ship, :, :].flatten(), 

                   bins=100, normed=True, alpha=alpha, label='ship')

_ = axss[0,1].hist(bands_2[samples_ship, :, :].flatten(), 

                   bins=100, normed=True, alpha=alpha, label='ship')

_ = axss[0,0].hist(bands_1[samples_iceberg, :, :].flatten(), 

                   bins=100, normed=True, alpha=alpha, label='iceberg')

_ = axss[0,1].hist(bands_2[samples_iceberg, :, :].flatten(), 

                   bins=100, normed=True, alpha=alpha, label='iceberg')

axss[0, 0].set_title('band_1')

axss[0, 1].set_title('band_2')

axss[0, 1].legend()

inc_angle = pd.to_numeric(df['inc_angle'], errors='coerce').as_matrix()
fig, axs = plt.subplots(1, 2)

_ = axs[0].plot(inc_angle, bands_1.mean(axis=(1, 2)), '.')

_ = axs[0].plot(inc_angle, bands_1.var(axis=(1, 2)), '.')

_ = axs[0].plot(inc_angle, bands_1.max(axis=(1, 2)), '.')

_ = axs[1].plot(inc_angle, bands_2.mean(axis=(1, 2)), '.')

_ = axs[1].plot(inc_angle, bands_2.var(axis=(1, 2)), '.')

_ = axs[1].plot(inc_angle, bands_2.max(axis=(1, 2)), '.')