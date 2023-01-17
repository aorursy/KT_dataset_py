!ls -R ../input | head -n 10
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.util.montage import montage2d as montage
from skimage.io import imread
import os
base_dir = '../input'
from glob import glob
all_path_df = pd.DataFrame(dict(path = 
                                glob(os.path.join(base_dir,
                                                  'aorta-data', 'data',
                                                  '*', '*', '*', 'image.png'))))
all_path_df['patient_id'] = all_path_df['path'].map(lambda x: x.split('/')[-2])
all_path_df['train_group'] = all_path_df['path'].map(lambda x: x.split('/')[-4])
all_path_df['mask_path'] = all_path_df['path'].map(lambda x: x.replace('image.', 'mask.'))
all_path_df.sample(2)
t_img = imread(all_path_df['path'].values[0])
t_mask = imread(all_path_df['mask_path'].values[0])
print(t_img.shape, t_img.min(), t_img.max(), t_img.mean())
print(t_mask.shape, t_mask.min(), t_mask.max())
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(t_img)
ax2.imshow(t_mask)
all_path_df['mask_image'] = all_path_df['mask_path'].map(lambda x: imread(x)[:, :, 0])
all_path_df['image'] = all_path_df['path'].map(lambda x: imread(x)[:, :, 0])
def pad_nd_image(in_img,  # type: np.ndarray
                 out_shape,  # type: List[Optional[int]]
                 mode='reflect',
                 **kwargs):
    # type: (...) -> np.ndarray
    """
    Pads an array to a specific size
    :param in_img:
    :param out_shape: the desired outputs shape
    :param mode: the mode to use in numpy.pad
    :param kwargs: arguments for numpy.pad
    :return:
    >>> pprint(pad_nd_image(np.eye(3), [7,7]))
    [[ 1.  0.  0.  0.  1.  0.  0.]
     [ 0.  1.  0.  1.  0.  1.  0.]
     [ 0.  0.  1.  0.  0.  0.  1.]
     [ 0.  1.  0.  1.  0.  1.  0.]
     [ 1.  0.  0.  0.  1.  0.  0.]
     [ 0.  1.  0.  1.  0.  1.  0.]
     [ 0.  0.  1.  0.  0.  0.  1.]]
    >>> pprint(pad_nd_image(np.eye(3), [2,2])) # should return the same
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    >>> t_mat = np.ones((2, 27, 29, 3))
    >>> o_img = pad_nd_image(t_mat, [None, 32, 32, None], mode = 'constant', constant_values=0)
    >>> o_img.shape
    (2, 32, 32, 3)
    >>> pprint(o_img.mean())
    0.7646484375
    >>> pprint(o_img[0,3,:,0])
    [ 0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
      1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.]
    """
    pad_dims = []
    for c_shape, d_shape in zip(in_img.shape, out_shape):
        pad_before, pad_after = 0, 0
        if d_shape is not None:
            if c_shape < d_shape:
                dim_diff = d_shape - c_shape
                pad_before = dim_diff // 2
                pad_after = dim_diff - pad_before
        pad_dims += [(pad_before, pad_after)]
    return np.pad(in_img, pad_dims, mode=mode, **kwargs)

def force_array_dim(in_img,  # type: np.ndarray
                    out_shape,  # type: List[Optional[int]]
                    pad_mode='reflect',
                    crop_mode='center',
                    **pad_args):
    # type: (...) -> np.ndarray
    """
    force the dimensions of an array by using cropping and padding
    :param in_img:
    :param out_shape:
    :param pad_mode:
    :param crop_mode: center or random (default center since it is safer)
    :param pad_args:
    :return:
    >>> np.random.seed(2018)
    >>> pprint(force_array_dim(np.eye(3), [7,7], crop_mode = 'random'))
    [[ 1.  0.  0.  0.  1.  0.  0.]
     [ 0.  1.  0.  1.  0.  1.  0.]
     [ 0.  0.  1.  0.  0.  0.  1.]
     [ 0.  1.  0.  1.  0.  1.  0.]
     [ 1.  0.  0.  0.  1.  0.  0.]
     [ 0.  1.  0.  1.  0.  1.  0.]
     [ 0.  0.  1.  0.  0.  0.  1.]]
    >>> pprint(force_array_dim(np.eye(3), [2,2], crop_mode = 'center'))
    [[ 1.  0.]
     [ 0.  1.]]
    >>> pprint(force_array_dim(np.eye(3), [2,2], crop_mode = 'random'))
    [[ 1.  0.]
     [ 0.  1.]]
    >>> pprint(force_array_dim(np.eye(3), [2,2], crop_mode = 'random'))
    [[ 0.  0.]
     [ 1.  0.]]
    >>> get_error(force_array_dim, in_img = np.eye(3), out_shape = [2,2], crop_mode = 'junk')
    'Crop mode must be random or center: junk'
    >>> t_mat = np.ones((1, 7, 9, 3))
    >>> o_img = force_array_dim(t_mat, [None, 12, 12, None], pad_mode = 'constant', constant_values=0)
    >>> o_img.shape
    (1, 12, 12, 3)
    >>> pprint(o_img.mean())
    0.4375
    >>> pprint(o_img[0,3,:,0])
    [ 0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.  0.]
    """
    assert crop_mode in ['random', 'center'], "Crop mode must be random or " \
                                              "center: {}".format(crop_mode)

    pad_image = pad_nd_image(in_img, out_shape, mode=pad_mode, **pad_args)
    crop_dims = []
    for c_shape, d_shape in zip(pad_image.shape, out_shape):
        cur_slice = slice(0, c_shape)  # default
        if d_shape is not None:
            assert d_shape <= c_shape, \
                "Padding command failed: {}>={} - {},{}".format(d_shape,
                                                                c_shape,
                                                                pad_image.shape,
                                                                out_shape
                                                                )
            if d_shape < c_shape:
                if crop_mode == 'random':
                    start_idx = np.random.choice(
                        range(0, c_shape - d_shape + 1))
                    cur_slice = slice(start_idx, start_idx + d_shape)
                else:
                    start_idx = (c_shape - d_shape) // 2
                    cur_slice = slice(start_idx, start_idx + d_shape)
        crop_dims += [cur_slice]
    return pad_image.__getitem__(crop_dims)
all_path_df['mask_image'] = all_path_df['mask_image'].map(lambda x: force_array_dim(x, (224, 224)))
all_path_df['image'] = all_path_df['image'].map(lambda x: force_array_dim(x, (224, 224)))
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(montage(np.stack(all_path_df['image'], 0)))
fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(montage(np.stack(all_path_df['mask_image'], 0)))
train_df = all_path_df.query('train_group=="train"')
test_df = all_path_df.query('train_group=="test"')
def df_to_block(in_df):
    return np.stack(in_df['image'], 0)/255.0, np.stack(in_df['mask_image'], 0)/255.0
train_X = df_to_block(train_df)
test_X = df_to_block(test_df)
print(train_X[0].shape,test_X[0].shape)
def meshgridnd_like(in_img,
                    rng_func=range):
    new_shape = list(in_img.shape)
    all_range = [rng_func(i_len) for i_len in new_shape]
    return tuple([x_arr.swapaxes(0, 1) for x_arr in np.meshgrid(*all_range)])
def volume_as_feature_vector(in_vol, out_seg):
    coord_vecs = meshgridnd_like(in_vol[0])
    return np.concatenate([np.stack([c_slice.ravel()]+[c_coord.ravel() for c_coord in coord_vecs], -1) for c_slice in in_vol],0), np.expand_dims(out_seg.ravel(), -1)
train_vec, train_y = volume_as_feature_vector(*train_X)
test_vec, test_y = volume_as_feature_vector(*test_X)
single_img, single_img_seg = test_X[0][0:1], test_X[1][0:1] 
single_img_vec, _ = volume_as_feature_vector(single_img, single_img_seg)
print(test_vec.shape, test_y.shape)
print('single image', single_img_vec.shape)
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
def add_boundary(in_img, in_seg, cmap = 'bone', norm = True, add_labels = True):
    if norm:
        n_img = (1.0*in_img-in_img.min())/(1.1*(in_img.max()-in_img.min()))
    else:
        n_img = in_img
    rgb_img = plt.cm.get_cmap(cmap)(n_img)[:, :, :3]
    if add_labels:
        return label2rgb(image = rgb_img, label = in_seg.astype(int), bg_label = 0)
    else:
        return mark_boundaries(image = rgb_img, label_img = in_seg.astype(int), color = (1, 0, 0), mode = 'thick')
    
def show_results(fitted_model):
    pred_prob = fitted_model.predict_proba(test_vec)
    if isinstance(pred_prob, list):
        pred_prob = pred_prob[0]
    pred_cat = np.argmax(pred_prob, -1)
    fpr, tpr, _ = roc_curve(test_y, pred_prob[:, 1])
    auc = roc_auc_score(test_y, pred_prob[:, 1])
    fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2, 2, figsize = (20, 20))
    ax1.plot(fpr, tpr, 'b.-', label = 'ROC Curve')
    ax1.set_title('ROC (AUC:{:2.2f})'.format(auc))
    c_mat = confusion_matrix(test_y, pred_cat)
    sns.heatmap(c_mat, 
                fmt = 'd',
                annot = True,
                cbar = False,
                ax = ax2)
    
    print('Accuracy {:2.2f}%'.format(100*accuracy_score(test_y, pred_cat)))
    print('\n')
    print(classification_report(test_y, pred_cat))
    # show a single image
    pred_prob = fitted_model.predict_proba(single_img_vec)
    if isinstance(pred_prob, list):
        pred_prob = pred_prob[0]
    prob_as_img = pred_prob[:, 1].reshape(single_img.shape)
    ax3.imshow(add_boundary(single_img[0], single_img_seg[0]))
    ax3.set_title('Image with Ground Truth')
    ax4.imshow(add_boundary(prob_as_img[0], single_img_seg[0], add_labels = False, cmap = 'viridis'))
    ax4.set_title('Prediction with Ground Truth')
class just_intensity():
    def predict_proba(self, in_vec):
        n_vec = in_vec[:, 0]/in_vec[:, 0].max()
        return [np.stack([n_vec,n_vec], -1)]
ji_model = just_intensity()
show_results(ji_model)
class inversed_intensity():
    def predict_proba(self, in_vec):
        n_vec = in_vec[:, 0]/in_vec[:, 0].max()
        n_vec *= -1
        return [np.stack([n_vec,n_vec], -1)]
ii_model = inversed_intensity()
show_results(ii_model)
from sklearn.dummy import DummyClassifier
dc = DummyClassifier()
dc.fit(train_vec, train_y)
show_results(dc)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs = 4)
rf.fit(train_vec, train_y)
show_results(rf)
from sklearn.ensemble import ExtraTreesClassifier
et = RandomForestClassifier(n_jobs = 4)
et.fit(train_vec, train_y)
show_results(et)
from skimage.filters import edges, gaussian as gaussian_filter
from skimage.morphology import closing, white_tophat, disk
from collections import defaultdict
feat_list = defaultdict(Sobel = edges.sobel, 
             Gauss3 = lambda x: gaussian_filter(x, sigma = 3),
            Gauss5 = lambda x: gaussian_filter(x, sigma = 5),
             Gauss10 = lambda x: gaussian_filter(x, sigma = 10),
             Gauss3Sobel = lambda x: edges.sobel(gaussian_filter(x, sigma = 3)),
                        TopHat4 = lambda x: white_tophat(x, disk(4)),
                        TopHat8 = lambda x: white_tophat(x, disk(8)),
                        Closing4 = lambda x: closing(x, disk(4)),
                        Closing8 = lambda x: closing(x, disk(8))
                       )
fig, m_axs = plt.subplots(3, 3, figsize = (15, 15))
for (feat_name, feat_func), c_ax in zip(feat_list.items(), m_axs.flatten()):
    c_ax.imshow(feat_func(train_X[0][0, :, :]))
    c_ax.set_title(feat_name)
def volume_as_feature_vector(in_vol, out_seg):
    coord_vecs = meshgridnd_like(in_vol[0])
    return np.concatenate([np.stack([c_slice.ravel()]+
                                    [feat_func(c_slice).ravel() for feat_func in feat_list.values()]+ # features
                                    [c_coord.ravel() for c_coord in coord_vecs], -1) for c_slice in in_vol],0), np.expand_dims(out_seg.ravel(), -1)
train_vec, train_y = volume_as_feature_vector(*train_X)
test_vec, test_y = volume_as_feature_vector(*test_X)
single_img, single_img_seg = test_X[0][0:1], test_X[1][0:1] 
single_img_vec, _ = volume_as_feature_vector(single_img, single_img_seg)
print(test_vec.shape, test_y.shape)
print('single image', single_img_vec.shape)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs = 4)
rf.fit(train_vec, train_y)
show_results(rf)
from sklearn.ensemble import ExtraTreesClassifier
et = RandomForestClassifier(n_jobs = 4)
et.fit(train_vec, train_y)
show_results(et)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures, Normalizer
rf_poly = Pipeline([('Normalize', Normalizer()), 
                          ('PolynomialFeatures', PolynomialFeatures(2)),
                          ('RandomForest', RandomForestClassifier(n_jobs = 4, verbose = True))])
rf_poly
%%time
rf_poly.fit(train_vec, train_y)
show_results(rf_poly)
