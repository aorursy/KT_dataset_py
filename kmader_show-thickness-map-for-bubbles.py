from skimage.io import imread, imsave
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # a nice progress bar
import pandas as pd
stack_image = imread('../input/plateau_border.tif')
print(stack_image.shape, stack_image.dtype)
%matplotlib inline
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))
for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):
    cax.imshow(np.sum(stack_image,i).squeeze(), interpolation='none', cmap = 'bone_r')
    cax.set_title('%s Projection' % clabel)
    cax.set_xlabel(clabel[0])
    cax.set_ylabel(clabel[1])
from skimage.morphology import binary_opening, convex_hull_image as chull
bubble_image = np.stack([chull(csl>0) & (csl==0) for csl in stack_image])
plt.imshow(bubble_image[5]>0, cmap = 'bone')
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
from skimage.filters import gaussian
from skimage import measure
py.init_notebook_mode(connected=True)
smooth_bubble_image = gaussian(bubble_image[::2, ::3, ::3]*1.0/bubble_image.max(), 1.5)
print('downsampling:', smooth_bubble_image.shape)
verts, faces, _, _ = measure.marching_cubes_lewiner(
    smooth_bubble_image, smooth_bubble_image.mean())
x, y, z = zip(*verts)
ff_fig = FF.create_trisurf(x=x, y=y, z=z,
                           simplices=faces,
                           title="Foam Bubbles Borde",
                           aspectratio=dict(x=1, y=1, z=1),
                           plot_edges=False)
c_mesh = ff_fig['data'][0]
c_mesh.update(lighting=dict(ambient=0.18,
                            diffuse=1,
                            fresnel=0.1,
                            specular=1,
                            roughness=0.1,
                            facenormalsepsilon=1e-6,
                            vertexnormalsepsilon=1e-12))
c_mesh.update(flatshading=False)
py.iplot(ff_fig)
%%time
from scipy.ndimage.morphology import distance_transform_edt as distmap
low_res_bubble_image = bubble_image[:, ::3, ::3]
bubble_dist = distmap(low_res_bubble_image)
%matplotlib inline
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))
ax1.imshow(bubble_dist[100], interpolation='none', cmap = 'jet')
ax1.set_title('YZ Slice')
ax2.imshow(bubble_dist[:,100], interpolation='none', cmap = 'jet')
ax2.set_title('XZ Slice')
ax3.imshow(bubble_dist[:,:,100], interpolation='none', cmap = 'jet')
ax3.set_title('XY Slice')
%matplotlib inline
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))
for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):
    cax.imshow(np.max(bubble_dist,i).squeeze(), interpolation='none', cmap = 'magma')
    cax.set_title('%s Projection' % clabel)
    cax.set_xlabel(clabel[0])
    cax.set_ylabel(clabel[1])
from skimage.feature import peak_local_max
bubble_candidates = peak_local_max(bubble_dist, min_distance=12)
print('Found', len(bubble_candidates), 'bubbles')
thickness_map = np.zeros(bubble_dist.shape, dtype = np.float32)
xx, yy, zz = np.meshgrid(np.arange(bubble_dist.shape[1]),
                         np.arange(bubble_dist.shape[0]),
                         np.arange(bubble_dist.shape[2])
                        )
# sort candidates by size
sorted_candidates = sorted(bubble_candidates, key = lambda xyz: bubble_dist[tuple(xyz)])
for label_idx, (x,y,z) in enumerate(tqdm(sorted_candidates),1):
    cur_bubble_radius = bubble_dist[x,y,z]
    cur_bubble = (np.power(xx-float(y),2)+
                  np.power(yy-float(x),2)+
                  np.power(zz-float(z),2))<=np.power(cur_bubble_radius,2)
    thickness_map[cur_bubble] = cur_bubble_radius
%matplotlib inline
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))
for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):
    cax.imshow(np.max(thickness_map,i).squeeze(), interpolation='none', cmap = 'jet')
    cax.set_title('%s Projection' % clabel)
    cax.set_xlabel(clabel[0])
    cax.set_ylabel(clabel[1])
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def show_3d_mesh(image, thresholds):
    p = image[::-1].swapaxes(1,2)
    cmap = plt.cm.get_cmap('nipy_spectral_r')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, c_threshold in tqdm(list(enumerate(thresholds))):
        verts, faces, _, _ = measure.marching_cubes_lewiner(p, c_threshold)
        mesh = Poly3DCollection(verts[faces], alpha=0.5, edgecolor='none', linewidth = 0.1)
        mesh.set_facecolor(cmap(i / len(thresholds))[:3])
        ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    ax.view_init(45, 45)
    return fig
%%time
show_3d_mesh(thickness_map, list(reversed(np.linspace(1,thickness_map.max()-2,4))));
plt.hist(thickness_map[thickness_map>0])
train_values = pd.read_csv('../input/bubble_volume.csv')
print(train_values.shape)
train_values.sample(3)
%matplotlib inline
fig, (ax1) = plt.subplots(1,1, figsize = (8, 4))
vol_bins = np.logspace(np.log10(train_values['volume'].min()), np.log10(train_values['volume'].max()), 20)
ax1.hist(4/3*np.pi*np.power(thickness_map[thickness_map>0], 3), bins = vol_bins, normed = True, label = 'Thickness Map Volumes')
ax1.hist(train_values['volume'], bins = vol_bins, alpha = 0.5, normed = True, label = 'Validated Volumes')
ax1.set_xscale('log')
ax1.legend()
ax1.set_title('Volume Comparison');
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (12, 12))
ax = fig.add_subplot(111, projection='3d')
x_list, y_list, z_list = [], [], []
for (x,y,z) in sorted_candidates:
    x_list += [x]
    y_list += [y]
    z_list += [z]
ax.scatter(x_list, y_list, z_list, label = 'distance map maximum')
ax.scatter(train_values['x'], train_values['y']/3, train_values['z']/3, label = 'validation points')
ax.legend()
