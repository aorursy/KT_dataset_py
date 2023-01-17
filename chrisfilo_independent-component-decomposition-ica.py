%%capture
from nilearn.decomposition import CanICA
from nilearn.plotting import plot_prob_atlas
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, plot_glass_brain, show
from glob import glob
func_filenames = glob("../input/sub-*/*_preproc.nii")
canica = CanICA(n_components=10, smoothing_fwhm=6., threshold=3., 
                verbose=10, random_state=0)
canica.fit(func_filenames)
components_img = canica.components_img_
for i, cur_img in enumerate(iter_img(components_img)):
    plot_glass_brain(cur_img, title="IC %d" % i)
    show()
