import os

from skimage.io import imread # for reading images

import matplotlib.pyplot as plt # for showing plots

from skimage.filters import median # for filtering the data

from skimage.measure import label # for labeling bubbles

from skimage.morphology import disk # for morphology neighborhoods

from skimage.morphology import erosion, dilation, opening # for disconnecting bubbles

import numpy as np # for matrix operations and array support

from skimage.util import montage as montage2d

from skimage import img_as_float
%matplotlib inline

import matplotlib.pyplot as plt # setup plots nicely

import seaborn as sns

plt.rcParams["figure.figsize"] = (8, 8)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})
image_path = '../input/training.tif'

em_image = imread(image_path)

print("Data Loaded, Dimensions", em_image.shape)
%matplotlib inline



em_idx = np.random.permutation(range(em_image.shape[0]))[0]

em_slice = em_image[em_idx]

print("Slice Loaded, Dimensions", em_slice.shape)

# show the slice and histogram

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))

ax1.imshow(em_slice, cmap = 'gray')

ax1.axis('off')

ax2.hist(em_slice.ravel()) # make it 1d to make a histogram

ax2.set_title('Intensity Histogram')
%matplotlib inline

# create the basic coordinates

xx, yy = np.meshgrid(np.linspace(-1,1, em_slice.shape[1]), 

                    np.linspace(-1,1, em_slice.shape[0]))

def _make_slope(scale, ang_min = -np.pi, ang_max = np.pi):

    x_angle = np.random.uniform(ang_min, ang_max)

    return scale*(np.cos(x_angle)*xx+np.sin(x_angle)*yy)



def _make_periodic_slope(scale_h, scale_w, escale = 2):

    x_angle = np.random.uniform(-0.2,0.2)

    n_x = scale_w*(np.cos(x_angle)*xx+np.sin(x_angle)*yy)

    return scale_h*np.exp(escale*np.sin(n_x))



def _make_periodic_artifact(scale_h, scale_w, scale_w2):

    x_angle = np.random.uniform(-0.2,0.2)

    n_xx = xx + np.random.uniform(-0.01, 0.01, size = xx.shape)

    n_yy = yy + np.random.uniform(-0.01, 0.01, size = xx.shape)

    n_x = (np.cos(x_angle)*n_xx+np.sin(x_angle)*n_yy)

    n_sx = np.abs(0.5*(np.sin(scale_w*n_x)+np.sin(scale_w2*n_x)))-0.5

    return 1+scale_h*n_sx



def _make_gaussian_bump(width, height, depth):

    x_cent, y_cent = np.random.uniform(-1,1, size = 2)

    return depth*np.exp(-(np.power((xx-x_cent)/width, 2)+

                                 np.power((yy-y_cent)/height,2)

                                )

                       )

def _make_bumps(count, depth, min_width = 0.1, max_width = 0.8):

    out_img = None

    for _ in range(count):

        out_bump = _make_gaussian_bump(np.random.uniform(min_width, max_width),

                                      np.random.uniform(min_width, max_width),

                                      depth)

        out_img = out_bump if out_img is None else out_img+out_bump

    return out_img

fig, m_axs = plt.subplots(2,3, figsize = (9,6))

for c_bump_ax, c_img_ax in m_axs.T:

    s_bumps = _make_periodic_slope(1.0, 30,1)

    

    s_bumps = _make_periodic_artifact(0.5, 30, 35)

    a_bumps = _make_bumps(6, 15, 0.2, 0.9)

    c_bump = 60*s_bumps + a_bumps

    c_bump_ax.imshow(c_bump, cmap = 'magma')

    c_bump_ax.axis('off')

    c_bump_ax.set_title('Bump Map')

    c_bump_slice = (s_bumps*em_slice+a_bumps).clip(0,255).astype(np.uint8)

    c_img_ax.imshow(c_bump_slice, cmap='gray')

    c_img_ax.axis('off')

    c_img_ax.set_title('Slice w/BG')
# create a library of test images from one slice

%matplotlib inline

X_train = np.stack([(em_slice*_make_periodic_artifact(0.5, 30, 35)+_make_bumps(6, 15, 0.2, 0.9)).clip(0,255).astype(np.uint8) for i in range(9)],0)

plt.imshow(montage2d(X_train), cmap = 'gray')
from skimage import img_as_float



from skimage.morphology import reconstruction

from skimage.filters import gaussian, median, rank



from skimage.morphology import disk



def rolling_ball_background(in_image, radius):

    bg_image = rank.mean(in_image, disk(radius))

    out_image = img_as_float(in_image) - img_as_float(bg_image)

    return out_image
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

ax1.imshow(em_slice, cmap = 'gray')

ax1.set_title('Original Image')

ax2.imshow(X_train[0], cmap = 'gray')

ax2.set_title('Image w/BG')

ax3.imshow(median(X_train[0], disk(3)), cmap = 'gray')

ax3.set_title('Background Corrected')
radii = np.linspace(1, 300, 8)

mse = []

for i, radius in enumerate(radii):

    mse += [ np.mean(np.power((

        median(img_as_float(X_train[0]), disk(radius))-img_as_float(em_slice)

                              ).ravel()

                               ,2))]

plt.plot(radii,mse)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

ax1.imshow(em_slice, cmap = 'gray')

ax1.set_title('Original Image')

ax2.imshow(X_train[0], cmap = 'gray')

ax2.set_title('Image w/BG')

ax3.imshow(rolling_ball_background(X_train[0], 100), cmap = 'gray')

ax3.set_title('Background Corrected')
radii = np.linspace(20, 220, 8)

mse = []

for i, radius in enumerate(radii):

    mse += [ np.mean(np.power((rolling_ball_background(X_train[0], radius)-

                               img_as_float(em_slice) # convert the ground truth to a float as well

                              ).ravel()

                               ,2))]

plt.plot(radii,mse)
%%time

from itertools import product

train_idx = np.random.choice(range(em_image.shape[0]),size=8)

aug_count = 8

X_train = np.stack([(em_image[i]*_make_periodic_artifact(0.5, 30, 35)+

                     _make_bumps(6, 15, 0.2, 0.9)).clip(0,255).astype(np.uint8) for i, _ in 

                    product(train_idx,range(aug_count))],0)



Y_train = np.stack([em_image[i] for i, _ in product(train_idx,range(aug_count))],0)



# convert to float

X_train = np.expand_dims(img_as_float(X_train),-1) 

Y_train = np.expand_dims(img_as_float(Y_train),-1) 



mean_x_val = X_train.mean()

mean_y_val = Y_train.mean()

print('X_offset', mean_x_val, 'Y_offset', mean_y_val)

X_train -= mean_x_val

Y_train -= mean_y_val



_, x_wid, y_wid, c_wid = Y_train.shape

print('train', X_train.shape)
import keras.backend as K

K.set_image_dim_ordering('tf')

from keras.models import Sequential, Model, Input

from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, concatenate, merge, Conv2DTranspose

from keras.optimizers import SGD, Adam

DEFAULT_OPT = Adam(lr=6e-3)

OUT_ACT = 'tanh'

smooth = 0.5

def dice_coef(y_true, y_pred):

    #y_true_f = K.flatten(y_true)

    #y_pred_f = K.flatten(y_pred)

    y_true_f = K.batch_flatten(y_true)

    y_pred_f = K.batch_flatten(y_pred)

    return (2. * K.dot(y_true_f, K.transpose(y_pred_f)) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)

def compile_model(i_model):

    i_model.compile(optimizer = DEFAULT_OPT, loss = 'mse', metrics = ['mse', dice_coef])
def _build_one_model(in_shape, out_chan, kernel_size=1):

    raw_img = Input(shape = in_shape, name = 'InputImage')

    simple_filter = Convolution2D(filters=out_chan,

                                  kernel_size=(kernel_size,kernel_size), 

                                  padding='same', 

                                  name = '{0}x{0}Filter'.format(kernel_size),

                                  activation=OUT_ACT)(raw_img)

    one_cnn_model = Model(inputs = [raw_img], outputs=[simple_filter])

    compile_model(one_cnn_model)

    return one_cnn_model



one_cnn_model = _build_one_model((x_wid, y_wid, c_wid), 1)

# andrej says 4e-3 but that seems too high

loss_history = []

one_cnn_model.summary()
%matplotlib inline

im_args = dict(cmap='gray', vmin=-1,vmax=1)

def show_prediction(c_model = one_cnn_model, show_idx = 0):

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (9, 3))

    im_args = dict(cmap='gray', vmin=-1,vmax=1)

    ax1.imshow(X_train[show_idx,:,:,0],**im_args)

    ax1.axis('off')

    ax1.set_title('Noisy Input')

    ax2.imshow(c_model.predict(X_train[show_idx:(show_idx+1)])[0,:,:,0], **im_args)

    ax2.set_title('Filter Output')

    ax2.axis('off')

    ax3.imshow(Y_train[show_idx,:,:,0],**im_args)

    ax3.axis('off')

    ax3.set_title('Ground Truth')



def show_validations(c_model = one_cnn_model, show_idx = 2):

    fig, m_axs = plt.subplots(show_idx,3, figsize = (9, 3*show_idx))

    im_args = dict(cmap='gray', vmin=-1,vmax=1)

    for (ax1, ax2, ax3) in m_axs:

        v_idx = np.random.choice(range(em_image.shape[0]),size=1)

        gt_img = em_image[v_idx]

        in_img = (gt_img*_make_periodic_artifact(0.5, 30, 35)+

                     _make_bumps(6, 15, 0.2, 0.9)).clip(0,255).astype(np.uint8) 



        in_img=np.expand_dims(img_as_float(in_img),-1)-mean_x_val

        gt_img = img_as_float(gt_img)-mean_y_val

        print(gt_img.shape)

        ax1.imshow(in_img[0,:,:,0],**im_args)

        ax1.axis('off')

        ax1.set_title('Validiting Input')

        ax2.imshow(c_model.predict(in_img)[0,:,:,0], **im_args)

        ax2.set_title('Filter Output')

        ax2.axis('off')

        ax3.imshow(gt_img[0],**im_args)

        ax3.axis('off')

        ax3.set_title('Ground Truth')
loss_history += [one_cnn_model.fit(X_train, Y_train, epochs = 1)]

show_prediction()
loss_history += [one_cnn_model.fit(X_train, Y_train, epochs = 10, shuffle = True, verbose = False)]

show_prediction()
from scipy.ndimage import gaussian_filter

large_one_cnn_model = _build_one_model((x_wid, y_wid, c_wid), 1, kernel_size=65)

# initialize more intelligently

[w, b] = large_one_cnn_model.layers[-1].get_weights()

w = 5e-2*np.random.uniform(-1, 1, size=w.shape)

w[32, 32] = 1.5

# smooth out the weights

w[:, :, 0, 0] = gaussian_filter(w[:, :, 0, 0], 1.25)

# add smaller noise on top of it

w+=5e-3*np.random.uniform(-1, 1, size=w.shape)

large_one_cnn_model.layers[-1].set_weights([w, b])

large_one_cnn_model.compile(optimizer=Adam(3e-4), loss='mse', metrics=['mse', dice_coef])

loss_history = []

large_one_cnn_model.summary()

show_validations(large_one_cnn_model)
loss_history += [large_one_cnn_model.fit(X_train, Y_train, epochs=5)]

show_prediction(large_one_cnn_model)
cnn_weights = large_one_cnn_model.layers[-1].get_weights()[0][:,:, 0, 0]

max_val = np.percentile(np.abs(cnn_weights), 100)

plt.colorbar(

    plt.matshow(

        cnn_weights,

        cmap='RdBu',

        vmin=-max_val,

        vmax=max_val

    )

)
loss_history += [large_one_cnn_model.fit(X_train, Y_train, epochs=25, shuffle=True, verbose=False)]

show_validations(large_one_cnn_model)
cnn_weights = large_one_cnn_model.layers[-1].get_weights()[0][:,:, 0, 0]

max_val = np.percentile(np.abs(cnn_weights), 100)

plt.colorbar(

    plt.matshow(

        cnn_weights,

        cmap='RdBu',

        vmin=-max_val,

        vmax=max_val

    )

)
def _build_mult_model(in_shape, out_chan, layers = 2, kernel_size=3):

    raw_img = Input(shape = in_shape, name = 'InputImage')

    last_img = raw_img

    for i in range(2):

        last_img = Convolution2D(filters=np.power(2,i+1),

                                      kernel_size=(kernel_size,kernel_size), 

                                 padding='same', 

                                 name = '{0}x{0}Filter_{1}'.format(kernel_size, i),

                                      activation='relu')(last_img)

    

    last_img = concatenate([raw_img, last_img])

    

    last_filter = Convolution2D(filters=out_chan,

                                  kernel_size=(1,1), padding='valid', name = '1x1Filter_Out',activation=OUT_ACT)(last_img)

    

    mult_1x1_model = Model(inputs = [raw_img], outputs=[last_filter])

    compile_model(mult_1x1_model)

    return mult_1x1_model



mult_1x1_model = _build_mult_model((x_wid, y_wid, c_wid), 1)

# andrej says 4e-3 but that seems too high

loss_history = []

mult_1x1_model.summary()
from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG

# Define model

vmod = model_to_dot(mult_1x1_model, show_shapes=True)

vmod.write_svg('mult_model.svg')

SVG('mult_model.svg')
loss_history += [mult_1x1_model.fit(X_train, Y_train, epochs = 10, shuffle = True, batch_size=8)]

show_prediction(mult_1x1_model)
mult_33x33_model = _build_mult_model((x_wid, y_wid, c_wid), 1, kernel_size=33)

# andrej says 4e-3 but that seems too high

loss_history = []

mult_33x33_model.summary()
loss_history += [mult_33x33_model.fit(X_train, Y_train, epochs = 10, shuffle = True, batch_size=16)]

show_prediction(mult_33x33_model)
# DeepConCat Model

def _build_concat_model(in_shape, out_chan, layers = 2, blocks = 4, use_deconv = False):

    raw_img = Input(shape = in_shape, name = 'InputImage')

    start_img = raw_img

    last_img = raw_img

    for k in range(blocks):

        ds_fact = np.power(2,k)

        clayers = layers if (ds_fact == 1) or (not use_deconv) else layers - 1

        if ds_fact>1:

            last_img = MaxPooling2D(pool_size=(ds_fact, ds_fact), name = 'Pooling_B{}'.format(k))(last_img)

        for i in range(clayers):

            last_img = Convolution2D(filters=np.power(2,i+1)+k,

                                          kernel_size=(3,3), 

                                     padding='same', 

                                     name = '3x3Filter_B{}_L{}'.format(k,i),

                                          activation='relu')(last_img)

        if ds_fact>1:

            if not use_deconv:

                last_img = UpSampling2D(size=(ds_fact, ds_fact), name = 'UpSampling_B{}'.format(k))(last_img)

            else:

                last_img = Conv2DTranspose(filters = np.power(2,layers)+k, kernel_size = (ds_fact, ds_fact), 

                                       padding='same',

                                      strides = (ds_fact, ds_fact), activation = 'relu', 

                                       data_format = K.image_data_format(),name= 'Deconvolution_B{}'.format(k))(last_img)

        last_img = concatenate([start_img, last_img])

        start_img = last_img

    last_img = concatenate([raw_img, last_img])

    last_filter = Convolution2D(filters=out_chan,

                                  kernel_size=(1,1), 

                                padding='valid', 

                                name = '1x1Filter_Out',

                                activation=OUT_ACT)(last_img)

    

    deep_cc_model = Model(inputs = [raw_img], outputs=[last_filter])

    compile_model(deep_cc_model)

    return deep_cc_model



dcc_model = _build_concat_model((x_wid, y_wid, c_wid), 1)

# andrej says 4e-3 but that seems too high

loss_history = []

dcc_model.summary()
from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG

# Define model

vmod = model_to_dot(dcc_model, show_shapes=True)

vmod.write_svg('deepcc_model.svg')

SVG('deepcc_model.svg')
loss_history += [dcc_model.fit(X_train, Y_train, epochs = 10, shuffle = True, batch_size=1)]

show_prediction(dcc_model)
from skimage.util import montage as montage2d

#plt.cm.R

mt_args = dict(cmap='gray', vmin=-0.5, vmax = 0.5)

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (20,9))

ax1.imshow(montage2d(X_train[:,:,:,0]), **mt_args)

ax2.imshow(montage2d(dcc_model.predict(X_train, batch_size = 2)[:,:,:,0]), **mt_args)

ax3.imshow(montage2d(Y_train[:,:,:,0]), **mt_args)
%matplotlib inline

epich = np.cumsum(np.concatenate([np.linspace(0.5,1,len(mh.epoch)) for mh in loss_history]))

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (20,5))

_ = ax1.plot(epich,np.concatenate([mh.history['loss'] for mh in loss_history]),'b-')

           # epich,np.concatenate([mh.history['val_loss'] for mh in loss_history]),'r-')

ax1.legend(['Training', 'Validation'])

ax1.set_title('Loss')





_ = ax3.semilogy(epich,np.concatenate([mh.history['mean_squared_error'] for mh in loss_history]),'b-')

    #epich,np.concatenate([mh.history['val_mean_squared_error'] for mh in loss_history]),'r-')

ax3.legend(['Training', 'Validation'])

ax3.set_title('MSE')



_ = ax2.plot(epich,np.concatenate([mh.history['dice_coef'] for mh in loss_history]),'b-')

    #epich,np.concatenate([mh.history['val_dice_coef'] for mh in loss_history]),'r-')

ax2.legend(['Training', 'Validation'])

ax2.set_title('Dice Coefficient')
fig_cnt = int(np.sqrt(len(dcc_model.weights)))

fig, c_axs = plt.subplots(fig_cnt, fig_cnt, figsize = (20, 20))

im_settings = {'vmin': -0.25, 'vmax': 0.25, 'cmap': 'RdBu', 'interpolation': 'none'}

for n_weight, n_weight_tensor, c_ax in zip(dcc_model.get_weights(), dcc_model.weights, c_axs.flatten()):

    c_mat = n_weight.squeeze()

    #print(c_mat.shape)

    if len(c_mat.shape) == 1:

        ind = np.array(range(len(c_mat)))

        c_ax.bar(ind, c_mat)

    elif len(c_mat.shape) == 2:

        c_ax.imshow(c_mat, **im_settings)

    elif len(c_mat.shape) == 3:

        print(c_mat.shape)

        c_ax.imshow(montage2d(c_mat), **im_settings)

    elif len(c_mat.shape) == 4:

        c_ax.imshow(montage2d(np.stack([montage2d(c_layer) for c_layer in c_mat],0)), **im_settings)

    c_ax.set_title('{}\n{}'.format(n_weight_tensor.name, c_mat.shape))

    c_ax.axis('off')
# deep residual Model

from keras.layers import add, Activation

def _build_resnet_model(in_shape, out_chan, 

                        layers = 2, blocks = 5, 

                        start_block = 0,

                        use_deconv = False,

                       always_conv_skip = False,

                       max_depth = 16):

    raw_img = Input(shape = in_shape, name = 'InputImage')

    start_img = raw_img

    last_img = raw_img

    layer_depth = lambda i,k: np.clip(np.power(2,i+k+1-start_block),1,max_depth)

    for k in range(start_block,blocks):

        ds_fact = np.power(2,k)

        

        clayers = layers if (ds_fact == 1) or (not use_deconv) else layers - 1

        if ds_fact>1:

            last_img = MaxPooling2D(pool_size=(ds_fact, ds_fact), name = 'Pooling_B{}'.format(k))(last_img)

        for i in range(clayers):

            last_img = Convolution2D(filters=layer_depth(i,k),

                                          kernel_size=(3,3), 

                                     padding='same', 

                                     name = '3x3_Filter_B{}_L{}'.format(k,i),

                                          activation='linear')(last_img)

            last_img = BatchNormalization(name = 'BN_B{}_L{}'.format(k,i))(last_img)

            if i<(layers-1):

                last_img = Activation('relu')(last_img)

        if ds_fact>1:

            if not use_deconv:

                last_img = UpSampling2D(size=(ds_fact, ds_fact), name = 'UpSampling_B{}'.format(k))(last_img)

            else:

                last_img = Conv2DTranspose(filters = layer_depth(layers-1,k), 

                                           kernel_size = (ds_fact, ds_fact), 

                                       padding='same',

                                      strides = (ds_fact, ds_fact), activation = 'linear', 

                                       data_format = K.image_data_format(),name= 'Deconvolution_B{}'.format(k))(last_img)

        

        cur_depth = layer_depth(layers-1,k)

        if (start_img._keras_shape[-1]!=cur_depth) or (always_conv_skip):

            # only perform the convolution on the last input if necessary

            start_img_match = Convolution2D(filters=cur_depth,

                                          kernel_size=(1,1), 

                                     padding='same', 

                                     name = '1x1_Filter_B{}_L{}'.format(k,i),

                                          activation='linear')(start_img)

        else:

            start_img_match = start_img

            

        last_img = add([start_img_match, last_img], name = 'Add_B{}'.format(k))

        last_img = Activation('relu')(last_img)

        start_img = last_img

    last_filter = Convolution2D(filters=out_chan,

                                  kernel_size=(1,1), 

                                padding='valid', 

                                name = '1x1Filter_Out',

                                activation=OUT_ACT)(last_img)

    

    deep_res_model = Model(inputs = [raw_img], outputs=[last_filter])

    compile_model(deep_res_model)

    return deep_res_model



rs_model = _build_resnet_model((x_wid, y_wid, c_wid), 1, 

                               blocks = 6,

                               start_block = 0)

loss_history = []

rs_model.summary()
from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG

# Define model

vmod = model_to_dot(rs_model, show_shapes=True)

vmod.write_svg('deepres_model.svg')

SVG('deepres_model.svg')
from IPython.display import clear_output

for i in range(5):

    clear_output()

    loss_history += [rs_model.fit(X_train, Y_train, epochs = 4, shuffle = True, batch_size=1)]

    plt.close('all')

    show_prediction(rs_model)
show_validations(rs_model, 4)
fig_cnt = int(np.sqrt(len(rs_model.weights)))

fig, c_axs = plt.subplots(fig_cnt, fig_cnt, figsize = (25, 25))

im_settings = {'vmin': -0.25, 'vmax': 0.25, 'cmap': 'RdBu', 'interpolation': 'none'}

for n_weight, n_weight_tensor, c_ax in zip(rs_model.get_weights(), rs_model.weights, c_axs.flatten()):

    try:

        c_mat = n_weight.transpose([2,3,0,1])

    except:

        c_mat = n_weight#.swapaxes(0,-1)

    #print(c_mat.shape)

    if len(c_mat.shape) == 1:

        ind = np.array(range(len(c_mat)))

        c_ax.bar(ind, c_mat)

    elif len(c_mat.shape) == 2:

        c_ax.imshow(c_mat, **im_settings)

    elif len(c_mat.shape) == 3:

        print(c_mat.shape)

        c_ax.imshow(montage2d(c_mat), **im_settings)

    elif len(c_mat.shape) == 4:

        c_ax.imshow(montage2d(np.stack([montage2d(c_layer) for c_layer in c_mat],0)), **im_settings)

    c_ax.set_title('{}\n{}'.format(n_weight_tensor.name, c_mat.shape))

    c_ax.axis('off')
def small_tile_gen(tile_x, tile_y):

    while True:

        slice_idx = np.random.choice(train_x.shape[0])

        x_pos = np.random.choice(range(0, train_x.shape[1]-tile_x))

        y_pos = np.random.choice(range(0, train_x.shape[2]-tile_y))

        yield (train_x[slice_idx:slice_idx+1, x_pos:(x_pos+tile_x), y_pos:(y_pos+tile_y)],

              train_y[slice_idx:slice_idx+1, x_pos:(x_pos+tile_x), y_pos:(y_pos+tile_y)])
fig, m_ax = plt.subplots(2, 4)

[iax.axis('off') for iax in m_ax.flatten()]

(ax_in, ax_out) = m_ax

for c_iax, c_oax, (c_x, c_y) in zip(ax_in, ax_out, small_tile_gen(96, 96)):

    c_iax.imshow(c_x[0,:,:,0], **im_args)

    c_iax.set_title('In')

    c_oax.imshow(c_y[0,:,:,0], **im_args)

    c_oax.set_title('Out')