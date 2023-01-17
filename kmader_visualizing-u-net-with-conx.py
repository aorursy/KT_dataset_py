%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
tile_size = (64, 64)
train_em_image_vol = imread('../input/training.tif')[:40, ::2, ::2]
train_em_seg_vol = imread('../input/training_groundtruth.tif')[:40, ::2, ::2]>0
test_em_image_vol = imread('../input/training.tif')[:40, ::2, ::2]
test_em_seg_vol = imread('../input/training_groundtruth.tif')[:40, ::2, ::2]>0
print("Data Loaded, Dimensions", train_em_image_vol.shape,'->',train_em_seg_vol.shape)
def g_random_tile(em_image_vol, em_seg_vol):
    z_dim, x_dim, y_dim = em_image_vol.shape
    z_pos = np.random.choice(range(z_dim))
    x_pos = np.random.choice(range(x_dim-tile_size[0]))
    y_pos = np.random.choice(range(y_dim-tile_size[1]))
    return np.expand_dims(em_image_vol[z_pos, x_pos:(x_pos+tile_size[0]), y_pos:(y_pos+tile_size[1])],-1), \
            np.expand_dims(em_seg_vol[z_pos, x_pos:(x_pos+tile_size[0]), y_pos:(y_pos+tile_size[1])],-1).astype(float)
np.random.seed(2018)
t_x, t_y = g_random_tile(test_em_image_vol, test_em_seg_vol)
print('x:', t_x.shape, 'Range:', t_x.min(), '-', t_x.max())
print('y:', t_y.shape, 'Range:', t_y.min(), '-', t_y.max())
np.random.seed(2018)
t_img, m_img = g_random_tile(test_em_image_vol, test_em_seg_vol)
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
ip_pairs = [g_random_tile(train_em_image_vol, train_em_seg_vol) for _ in range(1000)]
net.dataset.append(ip_pairs)
net.dataset.split(0.25)
net.train(epochs=20, record=True)
net.propagate_to_image("conv_5", t_img)
net.picture(t_img, dynamic = True, rotate = True, show_targets = True, scale = 1.25)
net.dashboard()
net.movie(lambda net, epoch: net.propagate_to_image("conv_5", t_img, scale = 3), 
                'mid_conv.gif', mp4 = False)
net.movie(lambda net, epoch: net.propagate_to_image("conv_8", t_img, scale = 3), 
                'hr_conv.gif', mp4 = False)
net.movie(lambda net, epoch: net.propagate_to_image("output", t_img, scale = 3), 
                'output.gif', mp4 = False)
