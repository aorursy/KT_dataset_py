IMG_SIZE = (512, 512)
BOX_MAX_SIZE = 0.25
BOX_THRESHOLD = 0.5
TOP_BOXES = 10
CLASS_THRESHOLD = 0.5
GAUSSIAN_DEGREE = 4
!ls ../input
%matplotlib inline
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import pydicom
from keras import layers, models
from glob import glob
from scipy.ndimage import zoom
rsna_comp_dir = '../input/rsna-pneumonia-detection-challenge'
# calculate test image paths and ids
test_dicoms = glob(os.path.join(rsna_comp_dir, 'stage_2_test_images', '*.dcm'))
test_dicoms_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in test_dicoms}
def process_dicom(in_path, out_shape = IMG_SIZE):
    c_dicom = pydicom.read_file(in_path, stop_before_pixels=False)
    base_img = c_dicom.pixel_array
    c_size = base_img.shape
    x_fact = out_shape[0]/c_size[0]
    y_fact = out_shape[1]/c_size[1]
    rs_img = zoom(base_img, (x_fact, y_fact))
    return rs_img
class_prod_df = pd.read_csv('../input/lung-opacity-classification-transfer-learning/image_level_class_probs.csv')
class_prod_df['Lung Opacity'].plot.hist()
class_prod_df.sample(3)
print(class_prod_df[class_prod_df['Lung Opacity']>CLASS_THRESHOLD].shape[0], '/', class_prod_df.shape[0])
class_prod_df.sample(3)
# does not work pred_model = models.load_model('../input/lung-opacity-inception-sharp-boxnet/boxnet.h5', compile=False)
coord_model = models.load_model('../input/lung-opacity-inception-sharp-boxnet/coordinate_model.h5', compile=False)
def run_prediction(in_id):
    in_path = test_dicoms_dict.get(in_id, in_id)
    dicom_array = process_dicom(in_path)
    dicom_tensor = np.expand_dims(np.expand_dims(dicom_array, 0), -1)
    return coord_model.predict(dicom_tensor)[0]
train_df = pd.read_csv(os.path.join(rsna_comp_dir, 'stage_2_train_labels.csv')).query('Target>0')
train_dir = os.path.join(rsna_comp_dir, 'stage_2_train_images')
train_df['path'] = train_df['patientId'].map(lambda x: os.path.join(train_dir, '{}.dcm'.format(x)))
train_df.sample(3)
def project_gaussians(gaus_coord,  # type: tf.Tensor
                      proj_grid # type: tf.Tensor 
                       ):
    # type: (...) -> tf.Tensor
    """
    Project M gaussians on a grid of points
    :param gaus_coord: the n, m, 5 (x, y, w, h, I)
    :param proj_grid: the xx yy grid to project on (n, R, C, 2)
    :return:
    """
    with tf.variable_scope('gauss_proj'):
        batch_size = tf.shape(gaus_coord)[0]
        n_gaus = tf.shape(gaus_coord)[1]
        xg_wid = tf.shape(proj_grid)[1]
        yg_wid = tf.shape(proj_grid)[2]
        with tf.variable_scope('create_m_grids'):
            """create a grid for each gaussian"""
            grid_prep = lambda x: tf.tile(tf.expand_dims(x, 1), [1, n_gaus, 1, 1])
            xx_grid = grid_prep(proj_grid[:, :, :, 0])
            yy_grid = grid_prep(proj_grid[:, :, :, 1])

        with tf.variable_scope('create_rc_coords'):
            """create coordinates for each position and """
            coord_prep = lambda x: tf.tile(tf.expand_dims(tf.expand_dims(x, 2), 3), 
                                      [1, 1, xg_wid, yg_wid])
            c_x = coord_prep(gaus_coord[:, :, 0])
            c_y = coord_prep(gaus_coord[:, :, 1])
            c_w = BOX_MAX_SIZE*coord_prep(0.5+0.45*gaus_coord[:, :, 2])
            c_h = BOX_MAX_SIZE*coord_prep(0.5+0.45*gaus_coord[:, :, 3])
            c_norm_max=gaus_coord[:, :, 4]
            c_max = coord_prep(0.5+0.5*c_norm_max)
        with tf.variable_scope('transform_coords'):
            x_trans = (xx_grid-c_x)/c_w
            xe_trans = tf.exp(-tf.pow(x_trans, GAUSSIAN_DEGREE))
            y_trans = (yy_grid-c_y)/c_h
            ye_trans = tf.exp(-tf.pow(y_trans, GAUSSIAN_DEGREE))
            all_gauss = c_max*xe_trans*ye_trans
            sum_gauss = tf.clip_by_value(tf.reduce_sum(all_gauss, 1), 0, 1)
            return tf.expand_dims(sum_gauss, -1)
w_scale_factor = np.sqrt(2)
fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))
x = np.linspace(-1, 1, 512)
sigma = 0.1
y = np.exp(-np.power(x/sigma, GAUSSIAN_DEGREE))
ax1.plot(x, y)
W = w_scale_factor*sigma
ax1.axvline(-W/2)
ax1.axvline(W/2)
from IPython.display import display
def box_vec_as_df(in_boxs, out_size):
    out_df = pd.DataFrame(in_boxs, columns = ['y', 'x', 'height', 'width', 'score'])
    for c_s, c_col in zip(out_size, 'xy'):
        out_df[c_col] = out_df[c_col]*c_s/2+c_s/2
    for c_s, c_col in zip(out_size, ['width', 'height']):
        out_df[c_col] = BOX_MAX_SIZE*(0.5+0.45*out_df[c_col])
        out_df[c_col] = out_df[c_col]*c_s/2+c_s/2
    out_df['score'] = 0.5+0.5*out_df['score']
    #out_df['width'] = w_scale_factor*out_df['width']
    #out_df['height'] = w_scale_factor*out_df['height']
    out_df['x'] = out_df['x']-out_df['width']/2
    out_df['y'] = out_df['y']-out_df['height']/2
    return out_df
def bbox_str(in_boxes):
    new_boxes = in_boxes[in_boxes['score']>BOX_THRESHOLD].sort_values('score', ascending=False).head(TOP_BOXES)
    out_str_list = new_boxes.apply(lambda x: '{score:.2f} {x:.0f} {y:.0f} {width:.0f} {height:.0f}'.format(**x), 1)
    return ' '.join(out_str_list)
test_df = train_df[train_df['patientId']==train_df['patientId'].iloc[0]]
test_path = test_df['path'].iloc[0]
test_array = pydicom.read_file(test_path).pixel_array
test_boxes_df = box_vec_as_df(run_prediction(test_path), (1024, 1024))
display(test_boxes_df)
print(bbox_str(test_boxes_df))
test_df
from matplotlib.patches import Rectangle
fig, m_axs = plt.subplots(1, 3, figsize = (20, 10))
for c_ax, c_rows in zip(m_axs, [test_df, 
                                test_boxes_df, 
                                test_boxes_df.sort_values('score', ascending=False).head(TOP_BOXES)]):
    c_ax.imshow(test_array, cmap='bone')
    for i, (_, c_row) in enumerate(c_rows.dropna().iterrows()):
        c_ax.plot(c_row['x'], c_row['y'], 's')
        confidence = 0.5*c_row.get('score', 1.0)
        c_ax.add_patch(Rectangle(xy=(c_row['x'], c_row['y']),
                                width=c_row['width'],
                                height=c_row['height'], 
                                 alpha = confidence))
%%time
def full_pred(in_row):
    if in_row['Lung Opacity']>CLASS_THRESHOLD:
        c_boxes = box_vec_as_df(run_prediction(in_row['patientId']), (1024, 1024))
        return bbox_str(c_boxes)
    else:
        return ''
class_prod_df['PredictionString'] = class_prod_df.apply(full_pred, 1)
class_prod_df[['patientId','PredictionString']].head(10)
class_prod_df[['patientId','PredictionString']].to_csv('submission.csv', index=False)
