import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
base_dir = os.path.join('..', 'input')

all_files = glob(os.path.join(base_dir, 'TrainingSet', 'TrainingSet', '*', '*', '*'))
all_df = pd.DataFrame(dict(path = all_files))
all_df['folder'] = all_df['path'].map(lambda x: x.split('/')[-2])
all_df['patient'] = all_df['path'].map(lambda x: x.split('/')[-3])
all_df['file_id'] = all_df['path'].map(lambda x: os.path.splitext(os.path.split(x)[1])[0])
all_df['file_ext'] = all_df['path'].map(lambda x: os.path.splitext(x)[1][1:])
all_df['slice'] = all_df['file_id'].map(lambda x: int(x.split('-')[1]))
all_df['size'] = all_df['path'].map(lambda x: os.stat(x).st_size)
all_df['data_type'] = all_df.apply(lambda c_row: 'dcm' if c_row['file_ext']=='dcm' else c_row['file_id'].split('-')[-2],1)
print(all_df.shape[0], all_df.query('size>0', inplace = True), '->', all_df.shape[0])
all_df.sample(10)
all_images_df = pd.pivot_table(all_df, 
               columns = 'data_type', 
               values = 'path',
               index = ['patient', 'slice'],
              aggfunc = 'first').reset_index()
valid_pts = ~all_images_df.icontour.isnull()
valid_pts &= ~all_images_df.ocontour.isnull()
valid_pts &= ~all_images_df.dcm.isnull()
labeled_df = all_images_df[valid_pts].copy()
print(labeled_df.shape)
labeled_df.head(3)
from skimage.measure import grid_points_in_poly
from dicom import read_file
def read_contour(in_path):
    c_df = pd.read_table(in_path,sep = '\s+', header=None)
    c_df.columns = ['x', 'y']
    return c_df
_, t_row = next(labeled_df.sample(1).iterrows())
t_img = read_file(t_row['dcm']).pixel_array
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))
ax1.imshow(t_img, cmap = 'bone')
i_df = read_contour(t_row['icontour'])
o_df = read_contour(t_row['ocontour'])
ax1.plot(i_df['x'], i_df['y'], '.-', label = 'inner contour')
ax1.plot(o_df['x'], o_df['y'], '.-', label = 'outer contour')
ax1.legend()
ax2.imshow(grid_points_in_poly(t_img.shape, i_df[['y', 'x']].values))
ax3.imshow(grid_points_in_poly(t_img.shape, o_df[['y', 'x']].values))
labeled_df['image'] = labeled_df['dcm'].map(lambda x: read_file(x).pixel_array)
labeled_df['icontour_pts'] = labeled_df['icontour'].map(read_contour)
labeled_df['ocontour_pts'] = labeled_df['ocontour'].map(read_contour)
labeled_df['mask'] = labeled_df.apply(lambda c_row: [
    grid_points_in_poly(c_row['image'].shape, c_row['icontour_pts'][['y', 'x']].values),
    grid_points_in_poly(c_row['image'].shape, c_row['ocontour_pts'][['y', 'x']].values)
], 1)
def crop_copy(in_img, size = (256, 256)):
    out_img = np.zeros(size)
    out_img[0:min(in_img.shape[0], size[0]),
           0:min(in_img.shape[1], size[1])] = in_img[0:min(in_img.shape[0], size[0]),
           0:min(in_img.shape[1], size[1])]
    return out_img
all_img_stack = np.expand_dims(np.stack(labeled_df['image'].map(crop_copy).values,0), -1)
all_mask_icontour = np.expand_dims(np.stack(labeled_df['mask'].map(lambda x: crop_copy(x[0])).values,0), -1)
print(all_img_stack.shape, all_mask_icontour.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(all_img_stack, all_mask_icontour, random_state = 2018, test_size = 0.25)
print(x_train.shape, x_test.shape)
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Input, concatenate
in_node = Input(all_img_stack.shape[1:], name = 'ImageIn')
bn_node = BatchNormalization()(in_node)
c_node = Conv2D(8, kernel_size = (3,3), padding = 'same')(bn_node)
c_node = Conv2D(16, kernel_size = (3,3), padding = 'same')(c_node)
dil_layers = [c_node]
for i in [2, 4, 6, 8, 12]:
    dil_layers += [Conv2D(16,
                          kernel_size = (3, 3), 
                          dilation_rate = (i, i), 
                          padding = 'same',
                         activation = 'relu')(c_node)]
c_node = concatenate(dil_layers)
c_node = Conv2D(1, kernel_size = (1,1), padding = 'same', activation = 'sigmoid')(c_node)
s_seg_model = Model(inputs = [in_node], outputs = [c_node])
s_seg_model.compile(optimizer = 'adam', loss = 'mse', metrics = ['binary_crossentropy'])
s_seg_model.summary()
def show_model(in_model, idx = None):
    if idx is None:
        idx = np.random.choice(range(x_test.shape[0]))
    out_res = in_model.predict(x_test[idx:(idx+1)])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4), dpi = 200)
    ax1.imshow(x_test[idx,:,:,0], cmap = 'bone')
    ax1.set_title('In MRI')
    ax2.imshow(y_test[idx,:,:,0], cmap = 'RdBu')
    ax2.set_title('Manual Labels')
    ax3.imshow(out_res[0,:,:,0], cmap = 'RdBu')
    ax3.set_title('Predictions')
show_model(s_seg_model)
    
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('mri_tools')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
s_seg_model.fit(x_train, y_train, 
                validation_data=(x_test, y_test), 
                verbose = 1,
               epochs = 50, 
                callbacks = callbacks_list)
show_model(s_seg_model)
s_seg_model.load_weights(weight_path)
show_model(s_seg_model)
