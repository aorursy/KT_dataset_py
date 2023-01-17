%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import h5py
from sklearn.model_selection import train_test_split
from skimage.io import imsave, imread
h5_dir = os.path.join('..', 'input', 'train_segs.h5')
!ls -lh {h5_dir}
skip_cols = ['image', 'mask']
out_df = {}
array_cols = {}
with h5py.File(h5_dir, 'r') as f:
    for k in f.keys():
        print(k, f[k].shape, f[k].dtype)
        if k not in skip_cols:
            out_df[k] = f[k].value
        else:
            array_cols[k] = f[k].value
out_df = pd.DataFrame(out_df).reset_index()
str_cols = out_df.select_dtypes(['O']).columns
for c_col in str_cols:
    out_df[c_col] = out_df[c_col].map(lambda x: x.decode())
out_df.sample(3)
fig, maxs = plt.subplots(1, 4, figsize = (20, 10))
for c_ax, i in zip(maxs, ['PatientAge', 'PatientSex', 'ViewPosition', 'boxes']):
    out_df.groupby([i]).size().plot.bar(ax=c_ax)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
mean_cxr = np.mean(array_cols['image'], 0)[:, :, 0]
ax1.imshow(mean_cxr, cmap='bone')
mean_mask = np.mean(array_cols['mask'], 0)[:, :, 0]
mean_mask = mean_mask/mean_mask.sum()
imsave('opacity_prior.tif', mean_mask.astype('float32'))
ax2.imshow(mean_mask, cmap='viridis')
xy_vec = np.stack([x.ravel() for x in np.meshgrid(range(512), range(512), indexing='ij')], -1)
xy_prob = mean_mask.ravel()
sample_lung_idx = lambda points: np.random.choice(range(xy_vec.shape[0]), size=points, p=xy_prob)
sample_lung_regions = lambda points: xy_vec[sample_lung_idx(points), :]
mask_xy = sample_lung_regions(500)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(mean_cxr, cmap='bone')
ax1.plot(mask_xy[:, 1], mask_xy[:, 0], 'r+')
ax2.imshow(mean_mask, cmap='viridis')
ax2.plot(mask_xy[:, 1], mask_xy[:, 0], 'r+')
train_df, test_df = train_test_split(out_df, 
                                     test_size=0.5, 
                                     random_state=2018, 
                                     stratify=out_df[['PatientSex', 'ViewPosition', 'boxes']])
print(train_df.shape, test_df.shape)
def generate_tile(in_df, tile_count=100, tile_dims=(64, 64)):
    np.random.seed(2018)
    lung_idx = sample_lung_idx(tile_count)
    out_rows = []
    for (_, c_row), l_pos_idx in zip(
        in_df.sample(tile_count, replace=True).iterrows(), 
        lung_idx):
        x, y = xy_vec[l_pos_idx, :]
        loc_prior = xy_prob[l_pos_idx]
        sx = max(0, x-tile_dims[0]//2)
        sy = max(0, y-tile_dims[1]//2)
        ex = min(512, sx+tile_dims[0])
        ey = min(512, sy+tile_dims[1])
        sx = ex-tile_dims[0]
        sy = ey-tile_dims[1]
        new_row = dict(c_row.items())
        idx = new_row.pop('index')
        new_row['x'] = x
        new_row['y'] = y
        new_row['bbox'] = [sx, ex, sy, ey]
        new_row['image'] = array_cols['image'][idx, sx:ex, sy:ey, 0]
        new_row['mask'] = array_cols['mask'][idx, sx:ex, sy:ey, 0]
        new_row['opacity'] = new_row['mask'].mean()
        new_row['opacity_prior'] = loc_prior*mean_mask.sum()
        out_rows += [new_row]
    out_df = pd.DataFrame(out_rows)
    cut_off = 0.01
    return out_df[(out_df['opacity']<=cut_off) | (out_df['opacity']>=(1-cut_off))].copy()
%%time
train_tiles_df = generate_tile(train_df, 30000)
test_tiles_df = generate_tile(test_df, 30000)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
train_tiles_df['opacity_prior'].plot.hist(ax=ax1)
ax1.set_title('Training Prior')
test_tiles_df['opacity_prior'].plot.hist(ax=ax2)
ax2.set_title('Test Prior')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
train_tiles_df['opacity'].plot.hist(ax=ax1)
ax1.set_title('Training')
test_tiles_df['opacity'].plot.hist(ax=ax2)
ax2.set_title('Test')
from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
from keras import models, layers
full_train_stack = np.stack(train_tiles_df['image'].values, 0)
color_image_stack = np.stack([full_train_stack, full_train_stack, full_train_stack], -1).astype(float)
pp_color_image_stack = preprocess_input(color_image_stack)
c_model = models.Sequential()
c_model.add(PTModel(include_top=False, 
                    input_shape=pp_color_image_stack.shape[1:], 
                    weights='imagenet'))
c_model.add(layers.GlobalAvgPool2D())
vgg_features = c_model.predict(pp_color_image_stack, verbose=True, batch_size=64)
del pp_color_image_stack
del full_train_stack
del color_image_stack
from sklearn.ensemble import RandomForestRegressor
# intensity model
y_vec = np.stack(train_tiles_df['opacity'], 0)
rf_int = RandomForestRegressor(random_state=2018)
rf_int.fit(np.stack([train_tiles_df['image'].map(np.mean).values, 
                     train_tiles_df['image'].map(np.std).values,
                     train_tiles_df['opacity_prior'].values
                    ], -1), y_vec)
def int_score_diff(in_df):
    in_img = np.stack(in_df['image'].values, 0)
    pred_score = rf_int.predict(np.stack([np.mean(in_img, (1, 2)), 
                                          np.std(in_img, (1, 2)),
                                         in_df['opacity_prior'].values], -1))
    return np.abs(np.stack(in_df['opacity'].values, 0)-pred_score)
plt.hist(int_score_diff(train_tiles_df))
# vgg model
rf_vgg = RandomForestRegressor(random_state=2018)
rf_vgg.fit(vgg_features, y_vec)
def vgg_score_diff(in_df):
    in_img = np.stack(in_df['image'].values, 0)
    rgb_img = np.stack([in_img, in_img, in_img], -1).astype(float)
    pred_score = rf_vgg.predict(c_model.predict(preprocess_input(rgb_img), 
                                                      batch_size=64, 
                                                      verbose=True))
    return np.abs(np.stack(in_df['opacity'].values, 0)-pred_score)
plt.hist(vgg_score_diff(train_tiles_df)-int_score_diff(train_tiles_df), 50)
def clear_and_balance(new_df, cut_off=1e-2):
    """keep only the cases where VGG does much better than RF on features"""
    new_df['model_error'] = vgg_score_diff(new_df) - 0.7*int_score_diff(new_df)
    new_df['model_error'].plot.hist()
    new_df = new_df[new_df['model_error']<cut_off].drop('model_error', 1)
    new_df['opacity'] = new_df['opacity'].astype(bool).astype(int)
    min_count = np.min(new_df['opacity'].value_counts())
    return new_df.groupby(['opacity']).\
        apply(lambda x: x.sample(min_count, random_state=2018)).\
        reset_index(drop=True)
train_tiles_df = clear_and_balance(train_tiles_df)
test_tiles_df = clear_and_balance(test_tiles_df)
print(train_tiles_df.shape[0], test_tiles_df.shape[0])
print(train_tiles_df['opacity'].value_counts())
print(test_tiles_df['opacity'].value_counts())
sns.pairplot(test_tiles_df, hue='opacity')
train_tiles_df.head(2)
test_tiles_df.head(2)
def show_rows(in_df, rows=3):
    fig, m_axs = plt.subplots(rows, 2, figsize = (15, 5*rows))
    for (ax1, ax2), (_, c_row) in zip(m_axs, in_df.sample(rows, random_state=2018).iterrows()):
        ax1.imshow(c_row['image'], cmap='bone')
        ax1.set_title('{PatientAge} {PatientSex}\n{x}, {y}'.format(**c_row))
        ax2.imshow(c_row['mask'], vmin=0, vmax=1)
        ax2.set_title('{opacity}'.format(**c_row))
show_rows(train_tiles_df)
show_rows(test_tiles_df)
def package_output(tile_df, base_name, random_state=0):
    new_tile_df = tile_df.sample(tile_df.shape[0], random_state=random_state) # shuffle order
    new_tile_df['slice_idx'] = range(new_tile_df.shape[0])
    new_tile_df['tile_id']  = new_tile_df.apply(lambda x: 'tile_{slice_idx}_{x}_{y}'.format(**x), 1)
    tile_arr = np.stack(new_tile_df['image'], 0)
    imsave('{base_name}.tif'.format(base_name=base_name), tile_arr)
    new_tile_df[['tile_id', 'slice_idx', 'PatientAge', 'PatientSex', 'ViewPosition', 'opacity_prior', 'opacity']].to_csv('{base_name}_all.csv'.format(base_name=base_name), index=False)
    new_tile_df[['tile_id', 'slice_idx', 'PatientAge', 'PatientSex', 'ViewPosition', 'opacity_prior']].to_csv('{base_name}_info.csv'.format(base_name=base_name), index=False)
    new_tile_df[['tile_id', 'opacity']].to_csv('{base_name}_answer.csv'.format(base_name=base_name), index=False)
    def_guess_df = new_tile_df[['tile_id']].copy()
    def_guess_df['opacity'] = 0.5
    def_guess_df.to_csv('{base_name}_baseline.csv'.format(base_name=base_name), index=False)
    def_guess_df['opacity'] = 1.0
    def_guess_df.to_csv('{base_name}_always_yes.csv'.format(base_name=base_name), index=False)
    def_guess_df['opacity'] = 0.0
    def_guess_df.to_csv('{base_name}_always_no.csv'.format(base_name=base_name), index=False)
    def_guess_df['opacity'] = np.random.uniform(0, 1, def_guess_df.shape[0])
    def_guess_df.to_csv('{base_name}_random.csv'.format(base_name=base_name), index=False)
package_output(train_tiles_df, 'train')
package_output(test_tiles_df, 'test')
!ls -lh *
from skimage.util.montage import montage2d
nice_image = montage2d(imread('train.tif')[:625]).astype(np.uint8)
imsave('nice_montage.png', nice_image)
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(nice_image, cmap='bone')
ax1.axis('off')
fun_df = pd.read_csv('train_all.csv')
fun_df['intensity'] = np.mean(imread('train.tif'), (1, 2))
fun_df.head(1)
sns.pairplot(fun_df[['intensity', 'opacity_prior', 'PatientAge', 'opacity']], hue='opacity')
