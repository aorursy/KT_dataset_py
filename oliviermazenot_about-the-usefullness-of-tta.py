import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/kaggle/input/alaska2-image-steganalysis/Cover/00002.jpg')
img = img[:,:,::-1]

fig, axs = plt.subplots(2,4, figsize=(12,6))

for i in range(8):
    ax = axs[i//4, i%4]
    im = img.transpose((i//4,1-i//4,2)) # optionnal rotation
    if i%2 == 1:
        im = im[::-1,:,:] # optionnal vertical flip
    if (i//2)%2 == 1:
        im = im[:,::-1,:] # optionnal horizontal flip
    
    ax.set_title(f'isometry_{i}')
    ax.imshow(im)
    ax.axis('off')

plt.show()

df_results_by_iso = pd.read_csv('/kaggle/input/resultsbyiso/results_by_iso.csv')
df_results_iso_0 = pd.read_csv('/kaggle/input/resultsiso0/results_iso_0.csv').rename(columns={'compression JPEG':'compression_JPEG'})
df_results_iso_2 = pd.read_csv('/kaggle/input/results-iso-2/results_iso_2.csv')


# errors in dataframe names :

mask = df_results_by_iso.isometry == 'mean of 8 Iso_0'
df_results_by_iso.loc[mask, 'isometry'] = 'mean of 8 isometries'

for i in range(8):
    mask = df_results_by_iso.isometry == str(i)
    df_results_by_iso.loc[mask, 'isometry'] = f'iso_{i}'
    
df_plot = pd.concat([df_results_by_iso.iloc[-9:,:], df_results_iso_0.iloc[-1:,:], df_results_iso_2.iloc[-1:,:]])
mask = df_results_by_iso.compression_JPEG == 'all'
df_results_by_iso[mask].iloc[:,1:]
df = pd.concat([df_results_iso_0, df_results_iso_2])
mask = df.compression_JPEG == 'all'
df[mask].iloc[:,1:]
fig, ax = plt.subplots(figsize=(12,6))

ax.bar(range(11), df_plot.weighted_AUC, width=0.7)

ax.set_xticks(range(11))
ax.set_xticklabels(df_plot.isometry, rotation=50)

plt.title('Competition metric scores : one isometry vs TTA')
plt.ylim([0.91, 0.924])
plt.ylabel('weighted AUC')

plt.show()