import h5py

import numpy as np

import pandas as pd

import plotly.graph_objects as go

import plotly.offline
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.tri as mtri

from IPython.display import Image, display, SVG, clear_output, HTML

plt.rcParams["figure.figsize"] = (6, 6)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

plt.rcParams['image.cmap'] = 'gray' # grayscale looks better
with h5py.File('../input/face-dataset/face_ds.h5', 'r') as h:

    for k in h.keys():

        print(k, h[k].shape)

    sample_face = h['bs104_PR_SU_0'][()]
trace=dict(type='scatter3d',

           x=sample_face[:, 0],

           y=sample_face[:, 1],

           z=sample_face[:, 2],

           mode='markers',

           marker=dict(color=[f'rgb({r}, {g}, {b})' 

                              for r, g, b in sample_face[:, 3:]],

                       size=3)

          )

fig = go.Figure(

    data=[trace],

    layout_title_text="3D Face Map"

)

fig
from scipy.signal import periodogram

from scipy.interpolate import interp1d

img_vec = np.mean(sample_face[:, 3:6], axis=1)

r = np.floor(np.sqrt(sample_face.shape[0])).astype(int)



f, S = periodogram(img_vec)

target_sizes = np.arange(r//2, 2*r)

period_score = interp1d(1/f, S, assume_sorted=False, fill_value=0, bounds_error=False)(target_sizes)



peak_size = target_sizes[np.argmax(period_score)]

# show graph

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(img_vec)

ax2.plot(target_sizes, period_score)

ax2.axvline(peak_size, color='k', alpha=0.5, label='Peak Size {}'.format(peak_size))

ax2.legend()
v_len = sample_face.shape[0]//peak_size

reshape_flat = lambda x: x.reshape((v_len, peak_size, x.shape[1]), order='C').astype('uint8').squeeze()[::-1]

rgb_image = reshape_flat(sample_face[:peak_size*v_len, 3:])

xyz_image = reshape_flat(sample_face[:peak_size*v_len, :3])

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

ax1.imshow(rgb_image)

ax2.imshow(xyz_image[:, :, 0], cmap='RdBu')

ax3.imshow(xyz_image[:, :, 1], cmap='RdBu')

ax4.imshow(xyz_image[:, :, 2], cmap='RdBu')
from PIL import Image

img_as_8bit = lambda x: np.array(Image.fromarray(x).convert('P', palette='WEB', dither=None))

dum_img = Image.fromarray(np.ones((3,3,3), dtype='uint8')).convert('P', palette='WEB')

idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]

depth_map = xyz_image[:, :, 2].copy().astype('float')

depth_map[depth_map<20] = np.nan

trace=go.Surface(

           z=depth_map,

        surfacecolor=img_as_8bit(rgb_image),

        cmin=0, 

        cmax=255,

        colorscale=colorscale

          )

fig = go.Figure(

    data=[trace],

    layout_title_text="3D Face Surface"

)

fig
from skimage.io import imread

hl_img = (imread('../input/headsetfovimages/hl2_eye.png', as_gray=True)*255).astype('uint8')

plt.imshow(hl_img)
from scipy.ndimage import zoom

match_img = np.mean(rgb_image[:, :rgb_image.shape[1]//2], axis=2).astype('uint8')

match_img = zoom(match_img, 5)

plt.imshow(match_img)
import numba



#@numba.jit

def proj(a, h):

    a_proj = a @ h[:2, :2]

    a_proj[:, 0] += h[0, 2]

    a_proj[:, 1] += h[1, 2]

    return a_proj
import cv2

MAX_FEATURES = 1000

GOOD_MATCH_PERCENT = 0.25

def alignImages(im1Gray, im2Gray):

    # Detect ORB features and compute descriptors.

    orb = cv2.ORB_create(MAX_FEATURES)

    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)

    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score

    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)

    matches = matches[:numGoodMatches]

    # Draw top matches

    imMatches = cv2.drawMatches(im1Gray, keypoints1, im2Gray, keypoints2, matches, None)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (24,8))

    ax1.imshow(imMatches)

    # Extract location of good matches

    points1 = np.zeros((len(matches), 2), dtype=np.float32)

    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):

        points1[i, :] = keypoints1[match.queryIdx].pt

        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography

    h, mask = cv2.findHomography(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=0.1) # cv2.RANSAC or cv2.LMEDS

    points_proj = proj(points2, h)

    ax2.plot(points1[:, 0], points1[:, 1], 'r.', label='Points1')

    ax2.plot(points2[:, 0], points2[:, 1], 'gs', label='Points2')

    ax2.plot(points_proj[:, 0], points_proj[:, 1], 'b+', label='Reprojected Points')

    ax2.legend()

    # Use homography

    height, width = im2Gray.shape

    im1Reg = cv2.warpPerspective(im1Gray, h, (width, height))

    ax3.imshow(im1Reg)

    return im1Reg, h, (points1, points2)
print(hl_img.dtype, match_img.dtype)

a_img, h_mat, (pt1, pt2) = alignImages(hl_img, match_img) #hl_img[-40::-1, -30::-1])

print(a_img.shape, pt1.shape, pt2.shape)
from scipy.optimize import fmin

from sklearn.neighbors import KDTree

def reproj_error(a, b, h, verbose=True):

    c_graph = KDTree(b)

    a_proj = proj(a, h)

    dist, idx = c_graph.query(a_proj, k=1)

    if verbose:

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

        ax1.plot(b[:, 0], b[:, 0], '-')

        ax1.plot(a_proj[:, 0], b[:, 0], '.')

        ax2.plot(b[:, 1], b[:, 1], '-')

        ax2.plot(a_proj[:, 1], b[:, 1], '.')

        

        ax3.plot(b[:, 0], b[:, 1], '+', label='Original')

        ax3.plot(a_proj[:, 0], a_proj[:, 1], '.', label='Reprojected')

        ax3.legend()

    return np.mean(np.sort(dist)[:-10])
reproj_error(pt1, pt2, np.eye(3))
new_mat = fmin(lambda x: reproj_error(pt1, pt2, x.reshape((3,3)), verbose=False), h_mat.ravel(), ftol=1e-6, xtol=1e-6)

reproj_error(pt1, pt2, new_mat.reshape((3,3)))
(h_mat*100).astype(int)
import SimpleITK as sitk

def register_img_generic(fixed_arr, 

                 moving_arr,

                registration_func,

                        show_transform = True):

    fixed_image = sitk.GetImageFromArray(fixed_arr)

    moving_image = sitk.GetImageFromArray(moving_arr)

    ff_img = sitk.Cast(fixed_image, sitk.sitkFloat32)

    mv_img = sitk.Cast(moving_image, sitk.sitkFloat32)

    

    final_transform_v1 = registration_func(ff_img, mv_img)

    resample = sitk.ResampleImageFilter()

    resample.SetReferenceImage(fixed_image)

    

    # SimpleITK supports several interpolation options, we go with the simplest that gives reasonable results.     

    resample.SetInterpolator(sitk.sitkBSpline)  

    resample.SetTransform(final_transform_v1)

    if show_transform:

        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))

        xx, yy = np.meshgrid(range(moving_arr.shape[0]), range(moving_arr.shape[1]))

        test_pattern = (((xx % 40)>30)|((yy % 40)>30)).astype(np.float32)

        ax1.imshow(test_pattern, cmap = 'bone_r')

        ax1.set_title('Test Pattern')

        test_pattern_img = sitk.GetImageFromArray(test_pattern)

        skew_pattern = sitk.GetArrayFromImage(resample.Execute(test_pattern_img))

        ax2.imshow(skew_pattern, cmap = 'bone_r')

        ax2.set_title('Registered Pattern')

    return sitk.GetArrayFromImage(resample.Execute(moving_image))



def bspline_intra_modal_registration(fixed_image, moving_image, grid_physical_spacing =  [15.0]*3):

    registration_method = sitk.ImageRegistrationMethod()

    # Determine the number of BSpline control points using the physical spacing we want for the control grid. 

    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]

    mesh_size = [int(image_size/grid_spacing + 0.5) \

                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]

    print('Using Mesh Size', mesh_size)

    initial_transform = sitk.BSplineTransformInitializer(image1 = fixed_image, 

                                                         transformDomainMeshSize = mesh_size, order=3)    

    registration_method.SetInitialTransform(initial_transform)

        

    registration_method.SetMetricAsMeanSquares()

    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be 

    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the

    # whole image.

    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)

    registration_method.SetMetricSamplingPercentage(0.01)

    

    # Multi-resolution framework.            

    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])

    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])

    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()



    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100)

    return registration_method.Execute(fixed_image, moving_image)
%%time

itk_reg_img = register_img_generic(hl_img, match_img, registration_func=bspline_intra_modal_registration)
plt.imshow(itk_reg_img)