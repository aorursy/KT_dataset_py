%matplotlib inline
import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from skimage.util.montage import montage2d
from pydicom import read_file as read_dicom
import SimpleITK as sitk
from tqdm import tqdm_notebook
base_dir = os.path.join('..', 'input')
# some dicom processing code
from collections import namedtuple
type_info = namedtuple('type_info',
                       ['inferrable', 'realtype', 'has_nulltype', 'length',
                        'is_complex'])
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.valuerep import PersonName3
from pydicom.uid import UID
_dicom_conv_dict = {MultiValue: lambda x: np.array(x).tolist(),
                        Sequence: lambda seq: [
                            [(str(d_ele.tag), str(d_ele.value)) for d_ele in
                             d_set] for
                            d_set in seq],
                        PersonName3: lambda x: str(x),
                        UID: lambda x: str(x)}
def dicom_to_dict(in_dicom):
    temp_dict = {a.name: a.value for a in in_dicom.iterall()}
    if in_dicom.__dict__.get('_pixel_array', None) is not None:
        temp_dict['Pixel Array'] = in_dicom.pixel_array.tolist()
    return pd.DataFrame([temp_dict]).T.to_dict()[0]
all_dicom_paths = glob(os.path.join(base_dir, '*', '*', '*', '*', '*'))
print(len(all_dicom_paths), 'dicom files')
dicom_df = pd.DataFrame(dict(path = all_dicom_paths))
dicom_df['SliceNumber'] = dicom_df['path'].map(lambda x: int(os.path.splitext(x.split('/')[-1])[0][2:]))
dicom_df['SeriesName'] = dicom_df['path'].map(lambda x: x.split('/')[-2])
dicom_df['StudyID'] = dicom_df['path'].map(lambda x: x.split('/')[-3])
dicom_df['PatientID'] = dicom_df['path'].map(lambda x: x.split('/')[-4].split(' ')[0])
dicom_df.sample(3)
dicom_folder_df = dicom_df.groupby(['PatientID', 'SeriesName']).agg('first').reset_index()
print(dicom_folder_df.shape[0], 'folders')
full_dicom_folder_df = pd.DataFrame([dict(**c_row, 
                                          **dicom_to_dict(read_dicom(c_row['path'], 
                                                                              stop_before_pixels=True))) for _, c_row in tqdm_notebook(dicom_folder_df.iterrows())])
print(full_dicom_folder_df.shape[1], 'columns')
full_dicom_folder_df.sample(3)
full_dicom_folder_df.describe(include = 'all').T
def read_dicom_stack(in_path):
    series_reader = sitk.ImageSeriesReader()
    # series_reader.LoadPrivateTagsOn()
    dir_name = os.path.dirname(in_path)
    dicom_names = series_reader.GetGDCMSeriesFileNames(dir_name)
    series_reader.SetFileNames(dicom_names)
    out_img = series_reader.Execute()
    # make sure it is float32
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(sitk.sitkFloat32)
    out_arr = sitk.GetArrayFromImage(castFilter.Execute(out_img))
    out_arr = out_arr[::-1] # flip z axis
    return out_arr, np.roll(out_img.GetSpacing(), -2)
fig, m_axs = plt.subplots(5, 4, figsize = (20, 6*5))
for (ax1, ax2, ax3, c_ax), (_, c_row) in zip(m_axs, 
                                             full_dicom_folder_df.sample(9, random_state = 0).iterrows()):
    try:
        c_img, c_dim = read_dicom_stack(c_row['path'])
        for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], [(1,2), (0,2), (0, 1)])):
            cax.imshow(np.max(c_img,i).squeeze(), interpolation='lanczos', cmap = 'bone')
            cax.set_title('%s%s Projection' % ('xyz'[clabel[0]], 'xyz'[clabel[1]]))
            cax.set_xlabel('xyz'[clabel[0]])
            cax.set_ylabel('xyz'[clabel[1]])
            cax.set_aspect(c_dim[clabel[0]]/c_dim[clabel[1]])
            cax.axis('off')
        c_ax.hist(c_img.ravel())
        c_ax.set_title('{PatientID}-{SeriesName}\n{shape}'.format(shape = c_img.shape, size = c_dim, **c_row))
    except Exception as e:
        c_ax.set_title('{}'.format(str(e)[:40]))
        print(e)
        c_ax.axis('off')
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.ndimage import zoom
from scipy.ndimage import binary_fill_holes, label, binary_closing
skin_seg = c_img>-120
skin_seg = binary_closing(skin_seg, np.ones((3,3,3)))
skin_seg = binary_fill_holes(skin_seg)
lab_seg, max_label  = label(skin_seg)
biggest_comp = np.argmax(np.histogram(lab_seg[lab_seg>0], range(max_label+1))[0])
skin_seg = (lab_seg==biggest_comp)

# Use marching cubes to obtain the surface mesh of these ellipsoids
# make it isotropic 3x3x3
zoom_fact = c_dim/np.array([3.0, 3.0, 3.0])
d_img = zoom(skin_seg.astype(np.float32), zoom_fact)
verts, faces, normals, values = measure.marching_cubes_lewiner(d_img, 0.5, spacing = tuple(c_dim/zoom_fact))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, 45)
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                cmap='bone', linewidth=0, antialiased=True, edgecolor='none')
for i, (_, c_row) in enumerate(full_dicom_folder_df.sample(2, random_state = 0).iterrows()):
    c_img, c_dim = read_dicom_stack(c_row['path'])
    skin_seg = c_img>-120
    skin_seg = binary_closing(skin_seg, np.ones((3,3,3)))
    skin_seg = binary_fill_holes(skin_seg)
    lab_seg, max_label  = label(skin_seg)
    biggest_comp = np.argmax(np.histogram(lab_seg[lab_seg>0], range(max_label+1))[0])
    skin_seg = (lab_seg==biggest_comp)

    # Use marching cubes to obtain the surface mesh of these ellipsoids
    # make it isotropic 3x3x3
    zoom_fact = c_dim/np.array([2.0, 2.0, 2.0])
    d_img = zoom(skin_seg.astype(np.float32), zoom_fact)
    verts, faces, normals, values = measure.marching_cubes_lewiner(d_img, 0.5, spacing = tuple(c_dim/zoom_fact))
    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                    cmap='bone', linewidth=0, antialiased=False, edgecolor='none')
    ax.view_init(135, 90)
    fig.savefig('%08d.png' % i)
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
py.init_notebook_mode()
x, y, z = zip(*verts)
ff_fig = FF.create_trisurf(x=x, y=y, z=z,
                           simplices=faces,
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
bone_seg = (c_img>200) & (c_img<2500)
bone_seg = binary_closing(bone_seg, np.ones((3,3,3)))
bone_seg = binary_fill_holes(bone_seg)
lab_seg, max_label  = label(bone_seg)
biggest_comp = np.argmax(np.histogram(lab_seg[lab_seg>0], range(max_label+1))[0])
bone_seg = (lab_seg==biggest_comp)
fig, ax1 = plt.subplots(1,1, figsize = (20, 20))
ax1.imshow(montage2d(bone_seg), cmap = 'gray')
# Use marching cubes to obtain the surface mesh of these ellipsoids
# make it isotropic 2x2x2
zoom_fact = c_dim/np.array([2.0, 2.0, 2.0])
d_img = zoom(bone_seg.astype(np.float32), zoom_fact)
verts, faces, normals, values = measure.marching_cubes_lewiner(d_img, 0.5, spacing = tuple(c_dim/zoom_fact))
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(135, 90)
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                cmap='bone_r', linewidth=0, antialiased=True, edgecolor='none', alpha = 0.5)
x, y, z = zip(*verts)
ff_fig = FF.create_trisurf(x=x, y=y, z=z,
                           simplices=faces,
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
from skimage.morphology.convex_hull import convex_hull_image
def safe_convex_hull(x):
    if not np.max(x):
        return np.zeros(x.shape, dtype=bool)
    return convex_hull_image(x)
skull_mask = np.stack([safe_convex_hull(x) for x in bone_seg],0)
brain_mask = (c_img>-120)*skull_mask
brain_mask[bone_seg] = 0
n_img = c_img.copy()
n_img[~brain_mask] = -2000
fig, ax1 = plt.subplots(1,1, figsize = (20, 20))
ax1.imshow(montage2d(n_img), vmin = -40, vmax = 50, cmap = 'gray')
d_img = zoom(n_img.astype(np.float32), zoom_fact)
verts, faces = measure.marching_cubes_classic(d_img, 0, spacing = tuple(c_dim/zoom_fact))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(135, 90)
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                cmap='bone_r', linewidth=0, antialiased=True, edgecolor='none', alpha = 0.5)
