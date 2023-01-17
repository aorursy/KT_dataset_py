import os
import tempfile
from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import CsaOdfModel
from dipy.direction import peaks_from_model
from dipy.direction import ProbabilisticDirectionGetter
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model
from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.tracking import utils
from dipy.reconst import peaks, shm
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors
from dipy.tracking.streamline import Streamlines
from dipy.tracking.eudx import EuDX
from nilearn.plotting import plot_anat, plot_roi, plot_stat_map
from nilearn.image import index_img, iter_img, new_img_like, math_img
from IPython.display import Image
from xvfbwrapper import Xvfb
import nibabel as nb
import pylab as plt
import numpy as np
# helper function for plotting woth dipy and VTK on a headless system
def show_image(actor, size=(1000,1000)):
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_filename = os.path.join(tmp_dir, 'tmp.png')
        with Xvfb() as xvfb:
            ren = window.Renderer()
            ren.add(actor)
            window.record(ren, n_frames=1, out_path=temp_filename, size=size)
            window.clear(ren)
        return Image(filename=temp_filename) 
img = nb.load('../input/hardi150.nii/HARDI150.nii')
data = img.get_data()
data.shape
gtab = gradient_table('../input/HARDI150.bval', '../input/HARDI150.bvec')
(gtab.bvals == 0).sum()
gtab.bvecs.shape
show_image(actor.point(gtab.gradients, window.colors.blue, point_radius=100))
i = 0
cur_img = index_img(img, i)
plot_anat(cur_img, cut_coords=(0,0,2), draw_cross=False, figure=plt.figure(figsize=(18,4)), cmap='magma', 
              vmin=0, vmax=1600, title="bval = %g, bvec=%s"%(gtab.bvals[i], str(np.round(gtab.bvecs[i,:],2))))

i = 38
cur_img = index_img(img, i)
plot_anat(cur_img, cut_coords=(0,0,2), draw_cross=False, figure=plt.figure(figsize=(18,4)), cmap='magma', 
              vmin=0, vmax=400, title="bval = %g, bvec=%s"%(gtab.bvals[i], str(np.round(gtab.bvecs[i,:],2))))

i = 70
cur_img = index_img(img, i)
plot_anat(cur_img, cut_coords=(0,0,2), draw_cross=False, figure=plt.figure(figsize=(18,4)), cmap='magma', 
              vmin=0, vmax=400, title="bval = %g, bvec=%s"%(gtab.bvals[i], str(np.round(gtab.bvecs[i,:],2))))
csa_model = CsaOdfModel(gtab, sh_order=8)
data_small = data[30:50, 65:85, 38:39]
csa_fit_small = csa_model.fit(data_small)
csa_odf_small = csa_fit_small.odf(peaks.default_sphere)
fodf_spheres_small = actor.odf_slicer(csa_odf_small, sphere=peaks.default_sphere, scale=0.9, norm=False, colormap='plasma')
show_image(fodf_spheres_small)
csd_peaks_small = peaks_from_model(model=csa_model,
                                   data=data_small,
                                   sphere=peaks.default_sphere,
                                   relative_peak_threshold=.5,
                                   min_separation_angle=25,
                                   parallel=True)

fodf_peaks_small = actor.peak_slicer(csd_peaks_small.peak_dirs, csd_peaks_small.peak_values)
show_image(fodf_peaks_small)
labels_img = nb.load("../input/aparc-reduced.nii/aparc-reduced.nii")
plot_roi(math_img("(labels == 1) | (labels == 2)", labels=labels_img), index_img(img, 0),figure=plt.figure(figsize=(18,4)),)
labels = labels_img.get_data()
white_matter = (labels == 1) | (labels == 2)
csa_model = shm.CsaOdfModel(gtab, 6)
csa_fit = csa_model.fit(data)
csa_peaks = peaks.peaks_from_model(model=csa_model,
                                   data=data,
                                   sphere=peaks.default_sphere,
                                   relative_peak_threshold=.8,
                                   min_separation_angle=45,
                                   mask=white_matter)
gfa_img = nb.Nifti1Image(csa_peaks.gfa, img.affine)
plot_stat_map(gfa_img, index_img(img,0),figure=plt.figure(figsize=(18,4)))
classifier = ThresholdTissueClassifier(csa_peaks.gfa, .25)
plot_roi(math_img("x == 2", x=labels_img), index_img(img, 0), figure=plt.figure(figsize=(18,4)))
seed_mask = labels == 2
seeds = utils.seeds_from_mask(seed_mask, density=[2, 2, 2], affine=img.affine)
# Initialization of LocalTracking. The computation happens in the next step.
streamlines_generator = LocalTracking(csa_peaks, classifier, seeds, img.affine, step_size=.5)

# Generate streamlines object
streamlines = Streamlines(streamlines_generator)

color = line_colors(streamlines)
streamlines_actor = actor.line(streamlines, line_colors(streamlines))
show_image(streamlines_actor)
prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csa_fit.shm_coeff,
                                                    max_angle=30.,
                                                    sphere=peaks.default_sphere)

streamlines_generator = LocalTracking(prob_dg, classifier, seeds, img.affine,
                                      step_size=.5, max_cross=1)

# Generate streamlines object.
streamlines = Streamlines(streamlines_generator)
streamlines_actor = actor.line(streamlines, line_colors(streamlines))
show_image(streamlines_actor)
seeds = utils.seeds_from_mask(white_matter, density=2)
streamline_generator = EuDX(csa_peaks.peak_values, csa_peaks.peak_indices,
                            odf_vertices=peaks.default_sphere.vertices,
                            a_low=.05, step_sz=.5, seeds=seeds)
affine = streamline_generator.affine

streamlines = Streamlines(streamline_generator, buffer_size=512)

show_image(actor.line(streamlines, line_colors(streamlines)))
len(streamlines)
cc_slice = labels == 2
cc_streamlines = utils.target(streamlines, cc_slice, affine=affine)
cc_streamlines = Streamlines(cc_streamlines)

other_streamlines = utils.target(streamlines, cc_slice, affine=affine,
                                 include=False)
other_streamlines = Streamlines(other_streamlines)
assert len(other_streamlines) + len(cc_streamlines) == len(streamlines)
len(cc_streamlines)
plot_roi(labels_img, index_img(img, 0), figure=plt.figure(figsize=(18,4)))
np.unique(np.array(labels))
plot_roi(math_img("x == 0", x=labels_img), index_img(img, 0), figure=plt.figure(figsize=(18,4)))
plot_roi(math_img("x == 1", x=labels_img), index_img(img, 0), figure=plt.figure(figsize=(18,4)))
M, grouping = utils.connectivity_matrix(cc_streamlines, labels, affine=affine,
                                        return_mapping=True,
                                        mapping_as_streamlines=True)
M[:3, :] = 0
M[:, :3] = 0
plt.imshow(np.log1p(M), interpolation='nearest')
np.argmax(M)
from numpy import unravel_index
new_M = M.copy()
#new_M[11,54] = 0
#new_M[54,11] = 0
unravel_index(new_M.argmax(), new_M.shape)
from nilearn.plotting import plot_stat_map
source_region = 32
target_region = 75
lr_superiorfrontal_track = grouping[source_region, target_region]
shape = labels.shape
dm = utils.density_map(lr_superiorfrontal_track, shape, affine=affine)
dm_img = nb.Nifti1Image(dm.astype("int16"), img.affine)
pl = plot_stat_map(dm_img, index_img(img,0), figure=plt.figure(figsize=(18,4)))
pl.add_contours(math_img("x == %d"%source_region, x=labels_img))
pl.add_contours(math_img("x == %d"%target_region, x=labels_img))