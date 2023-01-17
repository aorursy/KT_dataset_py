%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import pandas as pd
from glob import glob
from skimage.segmentation import mark_boundaries
def read_swc(in_path):
    swc_df = pd.read_csv(in_path, sep = ' ', comment='#', 
                         header = None)
    # a pure guess here
    swc_df.columns = ['id', 'junk1', 'x', 'y', 'junk2', 'width', 'next_idx']
    return swc_df[['x', 'y', 'width']]
DATA_ROOT = '../input/road'
tile_size = (128, 128)
%%time
image_files = glob(os.path.join(DATA_ROOT, '*.tif'))
image_mask_files = [(c_file, '.swc'.join(c_file.split('.tif'))) for c_file in image_files]
fig, m_axes = plt.subplots(5,3, figsize = (12, 20), dpi = 200)
all_img_list = []
for (im_path, mask_path), c_ax in zip(image_mask_files, m_axes.flatten()):
    im_data = imread(im_path)
    mk_data = read_swc(mask_path)
    mk_im_data = np.zeros(im_data.shape[:2], dtype = np.uint8)
    xx, yy = np.meshgrid(np.arange(im_data.shape[1]), np.arange(im_data.shape[0]))
    for _, c_row in mk_data.sample(frac = 0.7).iterrows():
        mk_im_data[np.square(xx-c_row['x'])+np.square(yy-c_row['y'])<np.square(c_row['width'])] = 1
    c_ax.imshow(mark_boundaries(image = im_data[:, :, :3], label_img = mk_im_data, mode = 'thick', color = (1, 0, 0)))
    all_img_list += [(im_data, mk_im_data)]
fig.savefig('overview.png')
train_tiles = all_img_list[:10]
test_tiles = all_img_list[10:]
print("Training Data Loaded, Dimensions", len(train_tiles),[x[1].shape for x in train_tiles])
print("Testing Data Loaded, Dimensions", len(test_tiles),[x[1].shape for x in test_tiles])
def g_random_tile(tile_list):
    # w e need two steps since each tile is differently shaped
    z_dim = len(tile_list)
    z_pos = np.random.choice(range(z_dim))
    c_img, c_seg = tile_list[z_pos]
    x_dim, y_dim = c_seg.shape
    x_pos = np.random.choice(range(x_dim-tile_size[0]))
    y_pos = np.random.choice(range(y_dim-tile_size[1]))
    return c_img[x_pos:(x_pos+tile_size[0]), y_pos:(y_pos+tile_size[1])]/255.0, \
            np.expand_dims(c_seg[x_pos:(x_pos+tile_size[0]), y_pos:(y_pos+tile_size[1])],-1).astype(float)
np.random.seed(2018)
t_x, t_y = g_random_tile(test_tiles)
print('x:', t_x.shape, 'Range:', t_x.min(), '-', t_x.max())
print('y:', t_y.shape, 'Range:', t_y.min(), '-', t_y.max())
np.random.seed(2017)
t_img, m_img = g_random_tile(test_tiles)
fig, (ax_img, ax_mask) = plt.subplots(1,2, figsize = (12, 6))
ax_img.imshow(np.clip(255*t_img, 0, 255).astype(np.uint8) if t_img.shape[2]==3 else t_img[:,:,0],
              interpolation = 'none', cmap = 'bone')
ax_mask.imshow(m_img[:,:,0], cmap = 'bone')
import conx as cx
!mkdir MiniUNet.conx
cfg_str = '{"font_size": 12, "font_family": "monospace", "border_top": 25, "border_bottom": 25, "hspace": 300, "vspace": 50, "image_maxdim": 200, "image_pixels_per_unit": 50, "activation": "linear", "arrow_color": "black", "arrow_width": "2", "border_width": "2", "border_color": "black", "show_targets": true, "show_errors": false, "pixels_per_unit": 1, "precision": 2, "svg_scale": 1.0, "svg_rotate": true, "svg_preferred_size": 400, "svg_max_width": 800, "dashboard.dataset": "Train", "dashboard.features.bank": "conv_4", "dashboard.features.columns": 8, "dashboard.features.scale": 0.5, "config_layers": {"input": {"visible": true, "minmax": null, "vshape": [32, 32], "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "bnorm": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "conv_0": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "conv_1": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "pool1": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "conv_2": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "conv_3": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "pool2": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "conv_4": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "conv_5": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "up2": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "cat2": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "conv_6": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "conv_7": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "up1": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "cat1": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "conv_8": {"visible": true, "minmax": null, "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": null, "feature": 0, "max_draw_units": 20}, "output": {"visible": true, "minmax": [0.0, 1.0], "vshape": null, "image_maxdim": null, "image_pixels_per_unit": null, "colormap": "bone", "feature": 0, "max_draw_units": 20}}}'
with open('MiniUNet.conx/config.json', 'w') as f:
    f.write(cfg_str)
net = cx.Network("MiniUNet")
base_depth = 8
net.add(cx.ImageLayer("input", tile_size, t_img.shape[-1])) 
net.add(cx.BatchNormalizationLayer("bnorm"))
c2 = lambda i, j, act = "relu": cx.Conv2DLayer("conv_{}".format(i, j), j, (3, 3), padding='same', activation=act)
net.add(c2(0, base_depth))
net.add(c2(1, base_depth))
net.add(cx.MaxPool2DLayer("pool1", pool_size=(2, 2), dropout=0.25))
net.add(c2(2, 2*base_depth))
net.add(c2(3, 2*base_depth))
net.add(cx.MaxPool2DLayer("pool2", pool_size=(2, 2), dropout=0.25))
net.add(c2(4, 4*base_depth))
net.add(c2(5, 4*base_depth))
net.add(cx.UpSampling2DLayer("up2", size = (2,2)))
net.add(cx.ConcatenateLayer("cat2"))
net.add(c2(6, 2*base_depth))
net.add(c2(7, 2*base_depth))
net.add(cx.UpSampling2DLayer("up1", size = (2,2)))
net.add(cx.ConcatenateLayer("cat1"))
net.add(c2(8, 2*base_depth))
net.add(cx.Conv2DLayer("output", 1, (1, 1), padding='same', activation='sigmoid'));
net.connect('input', 'bnorm')
net.connect('bnorm', 'conv_0')
net.connect('bnorm', 'cat1')
net.connect('conv_0', 'conv_1')
net.connect('conv_1', 'pool1')
net.connect('pool1', 'conv_2')
net.connect('conv_2', 'conv_3')
net.connect('conv_3', 'pool2')
net.connect('pool2', 'conv_4')
net.connect('conv_4', 'conv_5')
net.connect('conv_5', 'up2')
net.connect('up2', 'cat2')
net.connect('conv_3', 'cat2')
net.connect('cat2', 'conv_6')
net.connect('conv_6', 'conv_7')
net.connect('conv_7', 'up1')
net.connect('up1', 'cat1')
net.connect('cat1', 'conv_8')
net.connect('conv_8', 'output')
net.compile(error="binary_crossentropy", optimizer="adam")
net.picture(t_img, dynamic = True, rotate = True, show_targets = True, scale = 1.25)
net.dataset.clear()
ip_pairs = [g_random_tile(train_tiles) for _ in range(5000)]
net.dataset.append(ip_pairs)
net.dataset.split(0.25)
net.train(epochs=30, record=True)
net.propagate_to_image("conv_5", t_img)
net.picture(t_img, dynamic = True, rotate = True, show_targets = True, scale = 1.25)
net.dashboard()
net.movie(lambda net, epoch: net.propagate_to_image("conv_5", t_img, scale = 3), 
                'mid_conv.gif', mp4 = False)
net.movie(lambda net, epoch: net.propagate_to_image("conv_8", t_img, scale = 3), 
                'hr_conv.gif', mp4 = False)
net.movie(lambda net, epoch: net.propagate_to_image("output", t_img, scale = 3), 
                'output.gif', mp4 = False)
