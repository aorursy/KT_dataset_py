import os
img_dir = '../input/extracting-obb-images/'
img_zip = os.path.join(img_dir, 'images.zip')
!unzip -q {img_zip} -d ../working/
# not needed in Kaggle, but required in Jupyter
%matplotlib inline 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
from glob import glob
from skimage.io import imread
img_paths = {
    int(os.path.basename(x).split('_')[0]): x
    for x in 
    glob(os.path.join('..', 'working', '*.png'))
}
print('loaded', len(img_paths))
img_df = pd.read_csv(os.path.join(img_dir, 'image_subset.csv'))
img_df.columns = ['idx', 'jnk'] + img_df.columns.tolist()[2:]
img_df['local_path'] = img_df['idx'].map(img_paths.get)
img_df.drop(['jnk'], axis=1, inplace=True)
img_df.dropna(inplace=True)
img_df = img_df.sample(8000).sort_values(['timecode'])
print(img_df.shape)
img_df.head(3)
for _, c_row in img_df.sample(5).iterrows():
    print(c_row['local_path'], imread(c_row['local_path']).shape)
from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
MIN_DIM_SIZE = 32
BATCH_SIZE = 64
from keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (336, 512) # slightly smaller than vgg16 normally expects
core_idg = ImageDataGenerator(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = False, 
                              vertical_flip = False, 
                              height_shift_range = 0.01, 
                              width_shift_range = 0.01, 
                              rotation_range = 0, 
                              shear_range = 0.00,
                              fill_mode = 'nearest',
                              zoom_range=0.01,
                             preprocessing_function = preprocess_input)
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen
base_pretrained_model = PTModel(input_shape = (None, None)+(3,), 
                                include_top = False, 
                                weights = 'imagenet')
feature_output_shape = base_pretrained_model.predict(np.zeros((1,)+IMG_SIZE+(3,))).shape[1:]
print('Features Shape:', feature_output_shape)
from tqdm import tqdm_notebook
data_loops = 1
MASTER_CROP_OFFSET = 4
out_frame = []
out_xy = []
out_feat = []
for scale_down_idx in range(1):
    # make the images a multiple scales
    scale_down = 2**scale_down_idx
    train_gen = flow_from_dataframe(core_idg, img_df, 
                             path_col = 'local_path',
                            y_col = 'idx', 
                            target_size = (IMG_SIZE[0]//scale_down, IMG_SIZE[1]//scale_down),
                             color_mode = 'rgb',
                            batch_size = BATCH_SIZE,
                               shuffle = False)
    CROP_OFFSET = MASTER_CROP_OFFSET//scale_down
    for i, (c_x, c_idx) in zip(tqdm_notebook(range(data_loops*train_gen.n//train_gen.batch_size)), 
                            train_gen):
        
        c_y_vec = base_pretrained_model.predict(c_x)
        frame_coords = np.arange(c_y_vec.shape[0])
        x_coords = np.linspace(0, 1, c_y_vec.shape[1]+2*CROP_OFFSET)[CROP_OFFSET:-CROP_OFFSET]
        y_coords = np.linspace(0, 1, c_y_vec.shape[2]+2*CROP_OFFSET)[CROP_OFFSET:-CROP_OFFSET]
        fr_c, xx_c, yy_c = np.meshgrid(frame_coords, x_coords, y_coords, indexing = 'ij')
        out_frame += [c_idx[fr_c.ravel()]]
        out_xy += [np.stack([xx_c.ravel(), yy_c.ravel()], -1)]
        out_feat += [c_y_vec.reshape((-1, c_y_vec.shape[-1]))]
out_frame = np.concatenate(out_frame,0)
out_xy = np.concatenate(out_xy, 0)
out_feat = np.concatenate(out_feat,0)
sig_img_dir = '../input/oebb-signal/'
sig_images = {os.path.splitext(f)[0]: imread(os.path.join(base_path, f))[:, :, :3]
     for base_path, _, files in os.walk(sig_img_dir) 
     for f in files
    if os.path.splitext(f.upper())[1][1:] in ['JPG', 'PNG']}
fig, m_axs = plt.subplots(3, 6, figsize = (20, 10))
for c_ax, (c_id, c_img) in zip(m_axs.flatten(), sig_images.items()):
    c_ax.imshow(c_img)
    c_ax.set_title(c_id)
    c_ax.axis('off')
ref_img = sig_images['Hauptsignal Ausleger']
from scipy.ndimage import zoom
def image_to_multiscale_features(in_img, scale_steps = 3):
    img_prep = preprocess_input(in_img) 
    max_ds = np.floor(np.min(img_prep.shape[:2])/MIN_DIM_SIZE)
    min_ds = np.ceil(np.max(img_prep.shape[:2])/np.min(IMG_SIZE))
    c_vec = []
    scale_vec = np.unique(np.linspace(min_ds, max_ds, scale_steps))
    print(scale_vec)
    for c_s in scale_vec:
        c_tensor = np.expand_dims(zoom(img_prep, (1/c_s, 1/c_s, 1), order=2), 0)
        c_feat = base_pretrained_model.predict(c_tensor)
        c_feat = c_feat.reshape((-1, c_feat.shape[-1]))
        c_vec += [c_feat]
        c_vec += [np.mean(c_feat, 0, keepdims=True)] # global average pooling
    return np.concatenate(c_vec, 0)
ref_vec = image_to_multiscale_features(ref_img)
print(ref_vec.shape)
def compare_features(all_vecs, feat_db, comb_func = np.max):
    """
    all_vecs are all of the image vectors from the current search query
    feat_db is all of the vectors in the database
    comb_func is the function to combine the scores of all the query vectors together
     - np.max takes the best match in each image
     - np.mean takes the best average match across multiple scales
    """
    dot_score = []
    for in_vec in all_vecs:
        dot_score += [np.dot(feat_db, in_vec)]
    # combine dot_scores
    dot_score = np.stack(dot_score, 0)
    dot_score = comb_func(dot_score, 0)
    dot_score = dot_score/dot_score.max() # normalize by maximum score
    return dot_score, np.argsort(-1*dot_score)
%%time
f_score, f_rank = compare_features(ref_vec, out_feat)
def get_reg(in_img, in_reg):
    x_dim, y_dim = in_img.shape[0], in_img.shape[1]
    return in_img[int(x_dim*in_reg[0]):int(x_dim*in_reg[1]),
                  int(y_dim*in_reg[2]):int(y_dim*in_reg[3])]
def plot_matches(query_img, rank_vec, score_vec, top_matches=5):
    x_dim = 0.25
    y_dim = x_dim*query_img.shape[1]/query_img.shape[0]
    fig, m_axs = plt.subplots(top_matches, 3, figsize = (15, 6*top_matches))
    for c_ax in m_axs.flatten():
        c_ax.axis('off')
    for k_idx, (ax_in, ax_full, ax_out) in zip(rank_vec, m_axs):
        ax_in.imshow(query_img)
        ax_in.set_title('Search Query')
        c_frame = out_frame[k_idx]
        c_x, c_y = out_xy[k_idx]
        c_path = img_df[img_df['idx']==c_frame]['local_path'].values[0]
        in_img = imread(c_path)
        ax_full.imshow(in_img)
        ax_full.set_title(c_path.split('/')[-2:])
        cur_reg = (100*np.clip([c_x-x_dim, c_x+x_dim, c_y-y_dim, c_y+y_dim], 0, 1)).astype(int)/100
        ax_out.imshow(get_reg(in_img, cur_reg))
        ax_out.set_title('Score: {:2.1f}%\n{}'.format(100*score_vec[k_idx], cur_reg))
plot_matches(ref_img, f_rank, f_score, 4)
score_vec = {c_row['idx']: dict(**c_row) 
                   for _, c_row in img_df.iterrows()}
score_df = pd.DataFrame([dict(**score_vec[f], score=s) for f, s in zip(out_frame, f_score)])
score_df.sample(5)
score_grp_df = score_df.groupby(['idx', 'easting', 'northing']).agg({'score': 'max'}).reset_index()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
cut_off = np.percentile(score_grp_df['score'], 99.5)
below_df = score_grp_df[score_grp_df['score']<cut_off]
above_df = score_grp_df[score_grp_df['score']>=cut_off]
ax1.plot(below_df['easting'], below_df['northing'], 'b.-', alpha=0.2, label='Below')
ax1.plot(above_df['easting'], above_df['northing'], 'rs', alpha=1, ms=10, label='Above')
ax1.legend()
ax2.hist(score_grp_df['score'], 30);
ax2.axvline(cut_off, c='r')
sig_vec_scores = {}
for c_ref, c_ref_img in tqdm_notebook(sig_images.items()):
    c_ref_vec = image_to_multiscale_features(c_ref_img)
    sig_vec_scores[c_ref] = compare_features(c_ref_vec, out_feat)
for c_ref, c_ref_img in tqdm_notebook(sig_images.items()):
    cf_score, cf_rank = sig_vec_scores[c_ref] 
    plot_matches(c_ref_img, cf_rank, cf_score, 2)
# clean-up temporary files
!rm -rf ../working/*
import h5py
with h5py.File('search_index.h5', 'w') as f:
    f.create_dataset('frame', data=out_frame, compression = 5)
    f.create_dataset('xy_pos', data=out_xy, compression = 5)
    f.create_dataset('features', data=out_feat, compression = 8)
!ls -lh *.h5
col_list_mat = [np.expand_dims(out_frame, -1), out_xy]
col_name_mat = ['idx', 'img_x_pos', 'img_y_pos']
for c_ref, c_ref_img in sig_images.items():
    cf_score, cf_rank = sig_vec_scores[c_ref] 
    col_list_mat+=[np.expand_dims(cf_score, -1)]
    col_name_mat+=['Match_Score_{}'.format(c_ref)]

feature_map_df = pd.DataFrame(np.concatenate(col_list_mat, -1), 
                              columns=col_name_mat)
feature_map_df['idx'] = feature_map_df['idx'].map(int)
feature_map_df.sample(3)
img_space_df = pd.merge(img_df, feature_map_df, on='idx')
img_space_df.sample(2).T
img_space_df.to_csv('match_results.csv')
