import matplotlib.pyplot as plt
from skimage.measure import label # for labeling regions
from skimage.measure import regionprops # for shape analysis
import numpy as np # for matrix operations and array support
from skimage.color import label2rgb # for making overlay plots
import matplotlib.patches as mpatches # for showing rectangles and annotations
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread # for reading images
# the radiomics / texture analysis tools
from SimpleITK import GetImageFromArray
import radiomics
from radiomics.featureextractor import RadiomicsFeaturesExtractor # This module is used for interaction with pyradiomic
import logging
logging.getLogger('radiomics').setLevel(logging.CRITICAL + 1)  # this tool makes a whole TON of log noise
em_image_vol = imread('../input/training.tif')[:20, ::2, ::2]
em_thresh_vol = imread('../input/training_groundtruth.tif')[:20, ::2, ::2]
print("Data Loaded, Dimensions", em_image_vol.shape,'->',em_thresh_vol.shape)
# Instantiate the extractor
texture_extractor = RadiomicsFeaturesExtractor(verbose=False)
texture_extractor.disableAllFeatures()
_text_feat = {ckey: [] for ckey in texture_extractor.getFeatureClassNames()}
texture_extractor.enableFeaturesByName(**_text_feat)

print('Extraction parameters:\n\t', texture_extractor.settings)
print('Enabled filters:\n\t', texture_extractor._enabledImagetypes) 
print('Enabled features:\n\t', texture_extractor._enabledFeatures) 
%%time
results = texture_extractor.execute(GetImageFromArray(em_image_vol[:10, ::4, ::4]),
                            GetImageFromArray(em_thresh_vol.clip(0,1).astype(np.uint8)[:10, ::4, ::4]))
pd.DataFrame([results]).T
em_idx = np.random.permutation(range(em_image_vol.shape[0]))[0]
em_slice = em_image_vol[em_idx]
em_thresh = em_thresh_vol[em_idx]
print("Slice Loaded, Dimensions", em_slice.shape)
# show the slice and threshold
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (11, 5))
ax1.imshow(em_slice, cmap = 'gray')
ax1.axis('off')
ax1.set_title('Image')
ax2.imshow(em_thresh, cmap = 'gray')
ax2.axis('off')
ax2.set_title('Segmentation')
# here we mark the threshold on the original image

ax3.imshow(label2rgb(em_thresh, em_slice, bg_label=0))
ax3.axis('off')
ax3.set_title('Overlayed')
xx,yy = np.meshgrid(np.arange(em_slice.shape[1]),np.arange(em_slice.shape[0]))
region_labels=np.floor(xx/16)*64+np.floor(yy/16)
plt.matshow(region_labels,cmap='rainbow');
# compute some GLCM properties each patch
from tqdm import tqdm
from collections import defaultdict
prop_imgs = defaultdict(lambda : np.zeros_like(em_slice, dtype=np.float32))

score_img = np.zeros_like(em_slice, dtype=np.float32)
out_df_list = []
for patch_idx in tqdm(np.unique(region_labels)):
    xx_box, yy_box = np.where(region_labels==patch_idx)
    c_block = em_image_vol[:, xx_box.min():xx_box.max(), 
                                 yy_box.min():yy_box.max()]
    c_label = np.ones_like(c_block).astype(np.uint8)
    
    out_row = texture_extractor.execute(GetImageFromArray(c_block),
                            GetImageFromArray(c_label))
    
    mean_score = np.mean(em_thresh[region_labels == patch_idx])
    score_img[region_labels == patch_idx] = mean_score
    for k,v in out_row.items():
        if isinstance(v, (float, np.floating)):
            prop_imgs[k][region_labels == patch_idx] = v
    out_row['score'] = mean_score
    out_df_list += [out_row]
# show the slice and threshold
fig, m_axs = plt.subplots(8, 4, figsize = (20, 40))
print('Radiomic Images:', len(prop_imgs))
n_axs = m_axs.flatten()
ax1 = n_axs[0]
ax2 = n_axs[1]
ax1.imshow(em_slice, cmap = 'gray')
ax1.axis('off')
ax1.set_title('Image')
ax2.imshow(score_img,cmap='gray')
ax2.axis('off')
ax2.set_title('Mitochondria')
np.random.seed(2018)
for c_ax, c_prop in zip(n_axs[2:], np.random.permutation(list(prop_imgs.keys()))):
    c_ax.imshow(prop_imgs[c_prop], cmap = 'viridis')
    c_ax.axis('off')
    c_ax.set_title('{}'.format(c_prop.replace('original_','').replace('_', '\n')))
out_df=pd.DataFrame(out_df_list)
out_df['positive_score']=out_df['score'].map(lambda x: 'FG' if x>0 else 'BG')
out_df.describe().T
import seaborn as sns
sub_df = out_df[[x for x in out_df.columns if (x in ['positive_score']) or x.startswith('original_glrlm')]]
sub_df.columns = [x.replace('original_glrlm', '').replace('_', ' ') for x in sub_df.columns]
sns.pairplot(sub_df,
             hue='positive score',
             diag_kind="kde",
            diag_kws=dict(shade=True),
                        kind="reg")
out_df.to_csv('em_radiomics.csv')
from skimage.util.montage import montage2d
full_monty = montage2d(np.pad(np.stack([(v-v.mean())/v.std() for v in [em_slice]+list(prop_imgs.values())],0), [(0,0), (5, 5), (5,5)], mode = 'minimum'))
print(full_monty.shape)
fig, ax1 = plt.subplots(1,1,figsize = (20, 20))
ax1.imshow(full_monty, cmap = 'viridis', vmin = -3, vmax = 3)
help(np.pad)
