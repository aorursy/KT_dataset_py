import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

from skimage.io import imread

import matplotlib.pyplot as plt

import seaborn as sns
# load input images

all_tif_images=glob('../input/BBBC010_v1_images/*_w1_*.tif')

all_fg_images=glob('../input/BBBC010_v1_foreground/*.png')

# put input images paths into pandas DataFrame

image_df=pd.DataFrame([{'gfp_path': f} for f in all_tif_images])



image_df.iloc[1,0]
# define mapping function

def _get_light_path(in_path):

    w2_path='_w2_'.join(in_path.split('_w1_'))

    glob_str='_'.join(w2_path.split('_')[:-1]+['*.tif'])

    m_files=glob(glob_str)

    if len(m_files)>0:

        return m_files[0]

    else:

        return None

# create new columns with yet another paths

image_df['light_path']=image_df['gfp_path'].map(_get_light_path)

image_df=image_df.dropna()

image_df['base_name']=image_df['gfp_path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])



image_df['base_name'][1]
# Extract data from the image name



# clearly this is not the case

# <plate>_<wellrow>_<wellcolumn>_<wavelength>_<fileid>.tif

# Columns 1-12 are positive controls treated with ampicillin. Columns 13-24 are untreated negative controls.

# we apply a new rule

# 1649_1109_0003_Amp5-1_B_20070424_A01_w1_9E84F49F-1B25-4E7E-8040-D1BB2D7E73EA.tif

# junk_junk_junk_junk_junk_date_RowCol_wavelength_id.tif



image_df['plate_rc']=image_df['base_name'].map(lambda x: x.split('_')[6])

image_df['row']=image_df['plate_rc'].map(lambda x: x[0:1])

image_df['column']=image_df['plate_rc'].map(lambda x: int(x[1:]))

image_df['treated']=image_df['column'].map(lambda x: x<13)

image_df['wavelength']=image_df['base_name'].map(lambda x: x.split('_')[7])



image_df['mask_path']=image_df['plate_rc'].map(lambda x: '../input/BBBC010_v1_foreground/{}_binary.png'.format(x))

print('Loaded',image_df.shape[0],'datasets')

# pd.df.sample - Returns a random sample of items from an axis of object.

image_df.sample(3)
%matplotlib inline

test_image_row=list(image_df.query('treated').sample(1).T.to_dict().values())[0]

test_img=imread(test_image_row['light_path'])

test_gfp=imread(test_image_row['gfp_path'])

test_bg=imread(test_image_row['mask_path'])

print('Test Image:',test_img.shape)



fig, ((ax_light,ax_gfp, ax4),(ax2 ,ax3, _)) = plt.subplots(2,3, figsize = (10,6))

ax_light.imshow(test_img,cmap='gray')

ax_light.set_title('Light-field Image'.format(**test_image_row))



ax_gfp.imshow(np.sqrt(test_gfp),cmap='BuGn')

ax_gfp.set_title('GFP Image'.format(**test_image_row))



ax2.hist(test_img.ravel())

ax2.set_title('Light Distribution')



ax3.hist(test_gfp.ravel())

ax3.set_title('GFP Distribution')



ax4.imshow(test_bg, cmap = 'bone')

ax4.set_title('Segmented')
%matplotlib inline

test_image_row=list(image_df.query('treated==False').sample(1).T.to_dict().values())[0]

test_img=imread(test_image_row['light_path'])

test_gfp=imread(test_image_row['gfp_path'])

test_bg=imread(test_image_row['mask_path'])

print('Test Image:',test_img.shape)



fig, ((ax_light,ax_gfp, ax4),(ax2 ,ax3, _)) = plt.subplots(2,3, figsize = (10,6))

ax_light.imshow(test_img,cmap='gray')

ax_light.set_title('Light-field Image'.format(**test_image_row))



ax_gfp.imshow(np.sqrt(test_gfp),cmap='BuGn')

ax_gfp.set_title('GFP Image'.format(**test_image_row))



ax2.hist(test_img.ravel())

ax2.set_title('Light Distribution')



ax3.hist(test_gfp.ravel())

ax3.set_title('GFP Distribution')



ax4.imshow(test_bg, cmap = 'bone')

ax4.set_title('Segmented')
worm_df=pd.read_csv('../input/BBBC010_v1_foreground_eachworm.csv')

worm_df.sample(3)
worm_summary_df=worm_df.groupby('plate_rc').agg({'worm_id':'max', 'worm_pixel_area': 'sum'}).reset_index().rename(columns={'worm_id':'worm_count'})

worm_summary_df.sample(4)
# The npz was saved directly from a pandas df and so it is a bit uglier (hence the additional ravel), h5 was unfortunately too poorly compressed

with np.load('../input/BBBC010_v1_foreground_eachworm.npz') as mask_npz:

    mask_df=pd.DataFrame({k:[iv for ik,iv in v.ravel()[0].items()] 

                          for k,v in mask_npz.items() if 'path' not in k})

mask_df.sample(4)
# Rename logocal column

image_df['treated']=image_df['treated'].map(lambda x: 'ampicillin' if x else 'negative control')

image_df.sample(3)
# Get very basic statistics of individual images

#%%time

image_df['light_mean']=image_df['light_path'].map(lambda x: np.mean(imread(x)))

image_df['gfp_mean']=image_df['gfp_path'].map(lambda x: np.mean(imread(x)))

image_df['light_sd']=image_df['light_path'].map(lambda x: np.std(imread(x)))

image_df['gfp_sd']=image_df['gfp_path'].map(lambda x: np.std(imread(x)))

image_df.sample(3)



# The standard deviation is as high as mean ... wide spread of values
# this function obviously takes only numeric values from supplied data frame and plots their combinations

# the color is given by whether the sampel was treated or not

sns.pairplot(image_df.drop(['column'],1),hue='treated')
full_df=image_df.merge(worm_summary_df,on='plate_rc')

sns.pairplot(full_df.drop(['column'],1),hue='treated')
from scipy.stats import ttest_ind, ttest_rel

# https://plot.ly/python/t-test/

# t_stats, p_vals = ttest_ind(full_df['gfp_mean'].values, full_df['worm_pixel_area'].values)



## http://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas

treated = full_df[full_df['treated'] == 'ampicillin'].select_dtypes(['number']).drop('column', axis=1)

# full_df[full_df['treated'] == 'ampicillin'].select_dtypes([np.number])



control = full_df[full_df['treated'] != 'ampicillin'].select_dtypes(['number']).drop('column', axis=1)

# full_df[full_df['treated'] != 'ampicillin'].select_dtypes([np.number])
# probably need to normalize the data first?

def normalise(df):

    dfNorm = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return dfNorm



treatNorm = normalise(treated)

contNorm  = normalise(control)
t_stats, p_val = ttest_ind(treatNorm.values, #['gfp_mean']

                           contNorm.values) #['worm_pixel_area']



print("The t-statistic is {} and the p-value is {} ".format(t_stats, p_val))
df_group = full_df.drop('column', axis=1).groupby('treated') #.select_dtypes(['number'])

df_group.describe(percentiles = [.01, .05, .95, .99])



# treatNorm.describe(percentiles = [.01, .05, .95, .99]), contNorm.describe(percentiles = [.01, .05, .95, .99])
df_group.corr()
df_group.skew()

# [treated.kurt(),   control.kurt()]
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

from skimage.io import imread

import matplotlib.pyplot as plt

import seaborn as sns
# load input images

all_tif_images=glob('../input/BBBC010_v1_images/*_w1_*.tif')

all_fg_images=glob('../input/BBBC010_v1_foreground/*.png')

# put input images paths into pandas DataFrame

image_df=pd.DataFrame([{'gfp_path': f} for f in all_tif_images])



image_df.iloc[1,0]
# define mapping function

def _get_light_path(in_path):

    w2_path='_w2_'.join(in_path.split('_w1_'))

    glob_str='_'.join(w2_path.split('_')[:-1]+['*.tif'])

    m_files=glob(glob_str)

    if len(m_files)>0:

        return m_files[0]

    else:

        return None

# create new columns with yet another paths

image_df['light_path']=image_df['gfp_path'].map(_get_light_path)

image_df=image_df.dropna()

image_df['base_name']=image_df['gfp_path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])



image_df['base_name'][1]
# Extract data from the image name



# clearly this is not the case

# <plate>_<wellrow>_<wellcolumn>_<wavelength>_<fileid>.tif

# Columns 1-12 are positive controls treated with ampicillin. Columns 13-24 are untreated negative controls.

# we apply a new rule

# 1649_1109_0003_Amp5-1_B_20070424_A01_w1_9E84F49F-1B25-4E7E-8040-D1BB2D7E73EA.tif

# junk_junk_junk_junk_junk_date_RowCol_wavelength_id.tif



image_df['plate_rc']=image_df['base_name'].map(lambda x: x.split('_')[6])

image_df['row']=image_df['plate_rc'].map(lambda x: x[0:1])

image_df['column']=image_df['plate_rc'].map(lambda x: int(x[1:]))

image_df['treated']=image_df['column'].map(lambda x: x<13)

image_df['wavelength']=image_df['base_name'].map(lambda x: x.split('_')[7])



image_df['mask_path']=image_df['plate_rc'].map(lambda x: '../input/BBBC010_v1_foreground/{}_binary.png'.format(x))

print('Loaded',image_df.shape[0],'datasets')

# pd.df.sample - Returns a random sample of items from an axis of object.

image_df.sample(3)
%matplotlib inline

test_image_row=list(image_df.query('treated').sample(1).T.to_dict().values())[0]

test_img=imread(test_image_row['light_path'])

test_gfp=imread(test_image_row['gfp_path'])

test_bg=imread(test_image_row['mask_path'])

print('Test Image:',test_img.shape)



fig, ((ax_light,ax_gfp, ax4),(ax2 ,ax3, _)) = plt.subplots(2,3, figsize = (10,6))

ax_light.imshow(test_img,cmap='gray')

ax_light.set_title('Light-field Image'.format(**test_image_row))



ax_gfp.imshow(np.sqrt(test_gfp),cmap='BuGn')

ax_gfp.set_title('GFP Image'.format(**test_image_row))



ax2.hist(test_img.ravel())

ax2.set_title('Light Distribution')



ax3.hist(test_gfp.ravel())

ax3.set_title('GFP Distribution')



ax4.imshow(test_bg, cmap = 'bone')

ax4.set_title('Segmented')
%matplotlib inline

test_image_row=list(image_df.query('treated==False').sample(1).T.to_dict().values())[0]

test_img=imread(test_image_row['light_path'])

test_gfp=imread(test_image_row['gfp_path'])

test_bg=imread(test_image_row['mask_path'])

print('Test Image:',test_img.shape)



fig, ((ax_light,ax_gfp, ax4),(ax2 ,ax3, _)) = plt.subplots(2,3, figsize = (10,6))

ax_light.imshow(test_img,cmap='gray')

ax_light.set_title('Light-field Image'.format(**test_image_row))



ax_gfp.imshow(np.sqrt(test_gfp),cmap='BuGn')

ax_gfp.set_title('GFP Image'.format(**test_image_row))



ax2.hist(test_img.ravel())

ax2.set_title('Light Distribution')



ax3.hist(test_gfp.ravel())

ax3.set_title('GFP Distribution')



ax4.imshow(test_bg, cmap = 'bone')

ax4.set_title('Segmented')
worm_df=pd.read_csv('../input/BBBC010_v1_foreground_eachworm.csv')

worm_df.sample(3)
worm_summary_df=worm_df.groupby('plate_rc').agg({'worm_id':'max', 'worm_pixel_area': 'sum'}).reset_index().rename(columns={'worm_id':'worm_count'})

worm_summary_df.sample(4)
# The npz was saved directly from a pandas df and so it is a bit uglier (hence the additional ravel), h5 was unfortunately too poorly compressed

with np.load('../input/BBBC010_v1_foreground_eachworm.npz') as mask_npz:

    mask_df=pd.DataFrame({k:[iv for ik,iv in v.ravel()[0].items()] 

                          for k,v in mask_npz.items() if 'path' not in k})

mask_df.sample(4)
# Rename logocal column

image_df['treated']=image_df['treated'].map(lambda x: 'ampicillin' if x else 'negative control')

image_df.sample(3)
# Get very basic statistics of individual images

#%%time

image_df['light_mean']=image_df['light_path'].map(lambda x: np.mean(imread(x)))

image_df['gfp_mean']=image_df['gfp_path'].map(lambda x: np.mean(imread(x)))

image_df['light_sd']=image_df['light_path'].map(lambda x: np.std(imread(x)))

image_df['gfp_sd']=image_df['gfp_path'].map(lambda x: np.std(imread(x)))

image_df.sample(3)



# The standard deviation is as high as mean ... wide spread of values
# this function obviously takes only numeric values from supplied data frame and plots their combinations

# the color is given by whether the sampel was treated or not

sns.pairplot(image_df.drop(['column'],1),hue='treated')
full_df=image_df.merge(worm_summary_df,on='plate_rc')

sns.pairplot(full_df.drop(['column'],1),hue='treated')
from scipy.stats import ttest_ind, ttest_rel

# https://plot.ly/python/t-test/

# t_stats, p_vals = ttest_ind(full_df['gfp_mean'].values, full_df['worm_pixel_area'].values)



## http://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas

treated = full_df[full_df['treated'] == 'ampicillin'].select_dtypes(['number']).drop('column', axis=1)

# full_df[full_df['treated'] == 'ampicillin'].select_dtypes([np.number])



control = full_df[full_df['treated'] != 'ampicillin'].select_dtypes(['number']).drop('column', axis=1)

# full_df[full_df['treated'] != 'ampicillin'].select_dtypes([np.number])
# probably need to normalize the data first?

def normalise(df):

    dfNorm = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return dfNorm



treatNorm = normalise(treated)

contNorm  = normalise(control)
t_stats, p_val = ttest_ind(treatNorm.values, #['gfp_mean']

                           contNorm.values) #['worm_pixel_area']



print("The t-statistic is {} and the p-value is {} ".format(t_stats, p_val))
df_group = full_df.drop('column', axis=1).groupby('treated') #.select_dtypes(['number'])

df_group.describe(percentiles = [.01, .05, .95, .99])



# treatNorm.describe(percentiles = [.01, .05, .95, .99]), contNorm.describe(percentiles = [.01, .05, .95, .99])
df_group.corr()
df_group.skew()

# [treated.kurt(),   control.kurt()]