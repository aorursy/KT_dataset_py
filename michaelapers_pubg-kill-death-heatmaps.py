# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

from scipy.misc.pilutil import imread


df = pd.read_csv('../input/PUBG_MatchData_Flattened.tsv', sep='\t')

edf = df.loc[df['map_id'] == 'ERANGEL']
mdf = df.loc[df['map_id'] == 'MIRAMAR']

# print(edf.head())
# print(mdf.head())

def killer_victim_df_maker(df):
    victim_x_df = df.filter(regex='victim_position_x')
    victim_y_df = df.filter(regex='victim_position_y')
    killer_x_df = df.filter(regex='killer_position_x')
    killer_y_df = df.filter(regex='killer_position_y')

    victim_x_s = pd.Series(victim_x_df.values.ravel('F'))
    victim_y_s = pd.Series(victim_y_df.values.ravel('F'))
    killer_x_s = pd.Series(killer_x_df.values.ravel('F'))
    killer_y_s = pd.Series(killer_y_df.values.ravel('F'))

    vdata={'x': victim_x_s, 'y':victim_y_s}
    kdata={'x': killer_x_s, 'y':killer_y_s}

    victim_df = pd.DataFrame(data = vdata).dropna(how='any')
    victim_df = victim_df[victim_df['x']>0]
    killer_df = pd.DataFrame(data = kdata).dropna(how='any')
    killer_df = killer_df[killer_df['x']>0]
    return killer_df,victim_df

ekdf,evdf = killer_victim_df_maker(edf)
mkdf,mvdf = killer_victim_df_maker(mdf)

# print(ekdf.head())
# print(evdf.head())
# print(mkdf.head())
# print(mvdf.head())
# print(len(ekdf), len(evdf), len(mkdf), len(mvdf))



plot_data_ev = evdf[['x', 'y']].values
plot_data_ek = ekdf[['x', 'y']].values
plot_data_mv = mvdf[['x', 'y']].values
plot_data_mk = mkdf[['x', 'y']].values

plot_data_ev = plot_data_ev*4040/800000
plot_data_ek = plot_data_ek*4040/800000
plot_data_mv = plot_data_mv*976/800000
plot_data_mk = plot_data_mk*976/800000
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def heatmap(x, y, s, bins=100):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent
bg = imread('../input/erangel.jpg')
hmap, extent = heatmap(plot_data_ev[:,0], plot_data_ev[:,1], 1.5, bins=800)
alphas = np.clip(Normalize(0, hmap.max()/100, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(hmap.max()/100, hmap.max()/20, clip=True)(hmap)
colors = cm.bwr(colors)
colors[..., -1] = alphas
hmap2, extent2 = heatmap(plot_data_ek[:,0], plot_data_ek[:,1], 1.5, bins=800)
alphas2 = np.clip(Normalize(0, hmap2.max()/100, clip=True)(hmap2)*1.5, 0.0, 1.)
colors2 = Normalize(hmap2.max()/100, hmap2.max()/20, clip=True)(hmap2)
colors2 = cm.RdBu(colors2)
colors2[..., -1] = alphas2
fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.bwr, alpha=0.5)
# ax.imshow(colors2, extent=extent2, origin='lower', cmap=cm.RdBu, alpha=0.5)
plt.gca().invert_yaxis()
fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
# ax.imshow(colors, extent=extent, origin='lower', cmap=cm.bwr, alpha=0.5)
ax.imshow(colors2, extent=extent2, origin='lower', cmap=cm.RdBu, alpha=0.5)
plt.gca().invert_yaxis()
fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.bwr, alpha=0.5)
ax.imshow(colors2, extent=extent2, origin='lower', cmap=cm.RdBu, alpha=0.5)
plt.gca().invert_yaxis()
bg = imread('../input/miramar.jpg')
hmap, extent = heatmap(plot_data_mv[:,0], plot_data_mv[:,1], 1.5, bins=800)
alphas = np.clip(Normalize(0, hmap.max()/200, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(hmap.max()/100, hmap.max()/20, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas
hmap2, extent2 = heatmap(plot_data_mk[:,0], plot_data_mk[:,1], 1.5, bins=800)
alphas2 = np.clip(Normalize(0, hmap2.max()/200, clip=True)(hmap2)*1.5, 0.0, 1.)
colors2 = Normalize(hmap2.max()/100, hmap2.max()/20, clip=True)(hmap2)
colors2 = cm.rainbow(colors2)
colors2[..., -1] = alphas2
fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
ax.imshow(colors2, extent=extent2, origin='lower', cmap=cm.rainbow, alpha=0.5)
#plt.scatter(plot_data_er[:,0], plot_data_er[:,1])
plt.gca().invert_yaxis()

def divbutnotbyzero(a,b):
    c = np.zeros(a.shape)
    for i, row in enumerate(b):
        for j, el in enumerate(row):
            if el==0:
                c[i][j] = a[i][j]
            else:
                c[i][j] = a[i][j]/el
    return c
bg = imread('../input/erangel.jpg')
hmap, extent = heatmap(plot_data_ev[:,0], plot_data_ev[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_ek[:,0], plot_data_ek[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, hmap3.max()/100, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(hmap3.max()/100, hmap3.max()/20, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 4096); ax.set_ylim(0, 4096)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()
print(hmap3.mean())
bg = imread('../input/miramar.jpg')
hmap, extent = heatmap(plot_data_mv[:,0], plot_data_mv[:,1], 0, bins=800)
hmap2, extent2 = heatmap(plot_data_mk[:,0], plot_data_mk[:,1], 0, bins=800)
hmap3 = divbutnotbyzero(hmap,hmap2)
alphas = np.clip(Normalize(0, hmap3.max()/100, clip=True)(hmap)*1.5, 0.0, 1.)
colors = Normalize(hmap3.max()/100, hmap3.max()/20, clip=True)(hmap)
colors = cm.rainbow(colors)
colors[..., -1] = alphas


fig, ax = plt.subplots(figsize=(24,24))
ax.set_xlim(0, 1000); ax.set_ylim(0, 1000)
ax.imshow(bg)
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.rainbow, alpha=0.5)
plt.gca().invert_yaxis()
print(hmap3.mean())