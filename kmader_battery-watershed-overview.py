%matplotlib inline
import os
import numpy as np
from skimage.io import imread
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage import filters as skthresh
from skimage.morphology import opening, closing, disk
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt
from skimage.morphology import label
from skimage.feature import peak_local_max
from skimage.segmentation import mark_boundaries, watershed
base_dir = os.path.join('..', 'input')
all_tiffs = glob(os.path.join(base_dir, 'nmc*/*/grayscale/*'))
tiff_df = pd.DataFrame(dict(path = all_tiffs))
tiff_df['frame'] = tiff_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
tiff_df['experiment'] = tiff_df['frame'].map(lambda x: '_'.join(x.split('_')[0:-1]))
tiff_df['slice'] = tiff_df['frame'].map(lambda x: int(x.split('_')[-1]))
print('Images Found:', tiff_df.shape[0])
tiff_df.sample(3)
%matplotlib inline
random_path = tiff_df.sample(1, random_state = 123)['path'].values[0]
bw_img = imread(random_path)[250:-250, 250:-250]
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10), dpi=120)
ax1.imshow(bw_img, cmap='bone')
ax1.set_title('Gray Scale');
try:
    skthresh.try_all_threshold(bw_img[450:1100, 450:1110], figsize = (24, 15))
except RuntimeError as re:
    print('Try all threshold is one whiny function: {}'.format(re))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10), dpi=120)
ax1.imshow(bw_img, cmap='bone')
ax1.set_title('Gray Scale')

thresh_img = bw_img > skthresh.threshold_triangle(bw_img)
ax2.imshow(thresh_img, cmap='bone')
ax2.set_title('Segmentation')
bw_seg_img = closing(
    closing(
        opening(thresh_img, disk(3)),
        disk(1)
    ), disk(1)
)
bw_seg_img = binary_fill_holes(bw_seg_img)
ax3.imshow(bw_seg_img, cmap='bone')
ax3.set_title('Clean Segments');
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(24, 12), dpi=100)
bw_lab_img = label(bw_seg_img)
ax1.imshow(bw_lab_img, cmap = 'nipy_spectral')
# find boundaries
ax3.imshow(mark_boundaries(label_img = bw_lab_img, image = bw_img))
ax3.set_title('Boundaries');
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,  36), dpi=100)

bw_dmap = distance_transform_edt(bw_seg_img)

ax1.imshow(bw_dmap, cmap = 'nipy_spectral')

bw_peaks = label(peak_local_max(bw_dmap, indices=False, footprint=np.ones((3, 3)),
                            labels=bw_seg_img, exclude_border = True))

ws_labels = watershed(-bw_dmap, bw_peaks, mask=bw_seg_img)

ax2.imshow(ws_labels, cmap = 'gist_earth')
# find boundaries
ax3.imshow(mark_boundaries(label_img = ws_labels, image = bw_img))
ax3.set_title('Boundaries')
from skimage.morphology import dilation
from skimage.measure import perimeter
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
get_roi = lambda x: x[400:800, 400:700]
im_crop = get_roi(bw_img)
dist_map = get_roi(bw_dmap)
node_id_image = get_roi(ws_labels)

ax1.imshow(im_crop)

node_dict = {}
for c_node in np.unique(node_id_image[node_id_image>0]):
    y_n, x_n = np.where(node_id_image==c_node)
    node_dict[c_node] = {'x': np.mean(x_n),
                        'y': np.mean(y_n),
                        'width': np.mean(dist_map[node_id_image==c_node]),
                        'perimeter': perimeter(node_id_image==c_node)}
    ax1.plot(np.mean(x_n), np.mean(y_n), 'rs')

edge_dict = {}

for i in node_dict.keys():
    i_grow = dilation(node_id_image==i, np.ones((3,3)))
    for j in node_dict.keys():
        if i<j:
            j_grow = dilation(node_id_image==j, np.ones((3,3)))
            interface_length = np.sum(i_grow & j_grow)
            if interface_length>0:
                v_nodes = [i,j]
    
                edge_dict[(i,j)] = {'start': v_nodes[0], 
                                    'start_perimeter': node_dict[v_nodes[0]]['perimeter'],
                                    'end_perimeter': node_dict[v_nodes[-1]]['perimeter'],
                                     'end': v_nodes[-1],
                                    'interface_length': interface_length,
                                     'euclidean_distance': np.sqrt(np.square(node_dict[v_nodes[0]]['x']-
                                                                             node_dict[v_nodes[-1]]['x'])+
                                                                   np.square(node_dict[v_nodes[0]]['y']-
                                                                             node_dict[v_nodes[-1]]['y'])
                                                                  ),
                                    'max_width': np.max(dist_map[i_grow & j_grow]),
                                    'mean_width': np.mean(dist_map[i_grow & j_grow])}
                s_node = node_dict[v_nodes[0]]
                e_node = node_dict[v_nodes[-1]]
                ax1.plot([s_node['x'], e_node['x']], 
                         [s_node['y'], e_node['y']], 'b-', 
                         linewidth = np.max(dist_map[i_grow & j_grow]), alpha = 0.5)

ax2.imshow(mark_boundaries(label_img = node_id_image, image = im_crop))
ax2.set_title('Borders');
import pandas as pd
edge_df = pd.DataFrame(list(edge_dict.values()))
edge_df.head(5)
delete_edges = edge_df.query(
    'interface_length>0.33*(start_perimeter+end_perimeter)')
print('Found', delete_edges.shape[0], '/', edge_df.shape[0], 'edges to delete')
delete_edges.head(5)
node_id_image = get_roi(ws_labels)
for _ in range(3):
    # since some mappings might be multistep
    for _, c_row in delete_edges.iterrows():
        node_id_image[node_id_image==c_row['end']] = c_row['start']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

ax1.imshow(im_crop)

node_dict = {}
for c_node in np.unique(node_id_image[node_id_image>0]):
    y_n, x_n = np.where(node_id_image==c_node)
    node_dict[c_node] = {'x': np.mean(x_n),
                        'y': np.mean(y_n),
                        'width': np.mean(dist_map[node_id_image==c_node]),
                        'perimeter': perimeter(node_id_image==c_node)}
    ax1.plot(np.mean(x_n), np.mean(y_n), 'rs')

edge_dict = {}

for i in node_dict.keys():
    i_grow = dilation(node_id_image==i, np.ones((3,3)))
    for j in node_dict.keys():
        if i<j:
            j_grow = dilation(node_id_image==j, np.ones((3,3)))
            interface_length = np.sum(i_grow & j_grow)
            if interface_length>0:
                v_nodes = [i,j]
    
                edge_dict[(i,j)] = {'start': v_nodes[0], 
                                    'start_perimeter': node_dict[v_nodes[0]]['perimeter'],
                                    'end_perimeter': node_dict[v_nodes[-1]]['perimeter'],
                                     'end': v_nodes[-1],
                                    'interface_length': interface_length,
                                     'euclidean_distance': np.sqrt(np.square(node_dict[v_nodes[0]]['x']-
                                                                             node_dict[v_nodes[-1]]['x'])+
                                                                   np.square(node_dict[v_nodes[0]]['y']-
                                                                             node_dict[v_nodes[-1]]['y'])
                                                                  ),
                                    'max_width': np.max(dist_map[i_grow & j_grow]),
                                    'mean_width': np.mean(dist_map[i_grow & j_grow])}
                s_node = node_dict[v_nodes[0]]
                e_node = node_dict[v_nodes[-1]]
                ax1.plot([s_node['x'], e_node['x']], 
                         [s_node['y'], e_node['y']], 'b-', 
                         linewidth = np.max(dist_map[i_grow & j_grow]), alpha = 0.5)

ax2.imshow(mark_boundaries(label_img = node_id_image, image = im_crop))
ax2.set_title('Borders');
import networkx as nx
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
G = nx.Graph()
for k, v in node_dict.items():
    G.add_node(k, weight=v['width'])
for k, v in edge_dict.items():
    G.add_edge(v['start'], v['end'], **v)
nx.draw_shell(G, ax=ax1, with_labels=True,
               node_color=[node_dict[k]['width'] for k in sorted(node_dict.keys())],
               node_size=400,
               cmap=plt.cm.autumn,
              edge_color=[G.edges[k]['interface_length'] for k in list(G.edges.keys())],
        width=[2*G.edges[k]['max_width'] for k in list(G.edges.keys())],
              edge_cmap = plt.cm.Greens)
ax1.set_title('Randomly Organized Graph')
ax2.imshow(im_crop)
nx.draw(G,
        pos={k: (v['x'], v['y']) for k, v in node_dict.items()},
        ax=ax2,
        node_color=[node_dict[k]['width'] for k in sorted(node_dict.keys())],
        node_size=50,
        cmap=plt.cm.Greens,
       edge_color=[G.edges[k]['interface_length'] for k in list(G.edges.keys())],
        width=[2*G.edges[k]['max_width'] for k in list(G.edges.keys())], 
        edge_cmap=plt.cm.autumn, 
        alpha = 0.75,
        with_labels=False)
