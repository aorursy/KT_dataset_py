!pip install git+git://github.com/neurostuff/NiMARE.git@608516ec3034e356326dfe70df5e9ed77efd2be8
import os
import json
import numpy as np
import nibabel as nb
import tempfile
from glob import glob
from os.path import basename, join, dirname, isfile

import pandas as pd
import nibabel as nib
import pylab as plt
from scipy.stats import t
from nilearn.masking import apply_mask
from nilearn.plotting import plot_stat_map

import nimare
from nimare.meta.ibma import (stouffers, fishers, weighted_stouffers,
                              rfx_glm, ffx_glm)
from nimare.utils import t_to_z
pd.read_csv('../input/coordinates.csv').head()
pd.read_csv('../input/studies.csv').head()
dset_dict = {}
coords_df = pd.read_csv('../input/coordinates.csv')
for i, row in pd.read_csv('../input/studies.csv').iterrows():
    this_study_coords = coords_df[coords_df['study_id'] == row[0]]
    contrast = {"sample_sizes": [row[1]],
                "coords": { "space": this_study_coords['space'].unique()[0],
                            "x": list(this_study_coords['x']),
                            "y": list(this_study_coords['y']),
                            "z": list(this_study_coords['z'])}}
    dset_dict[row[0]] = {"contrasts": {"1": contrast }}
with tempfile.NamedTemporaryFile(mode='w', suffix=".json") as fp:
    json.dump(dset_dict, fp)
    fp.flush()
    db = nimare.dataset.Database(fp.name)
    dset = db.get_dataset()
mask_img = dset.mask
dset.data['pain_01']
ale = nimare.meta.cbma.ALE(dset, ids=dset.ids)
ale.fit(n_iters=10)
plot_stat_map(ale.results.images['ale'], cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r', figure=plt.figure(figsize=(18,4)))
plot_stat_map(ale.results.images['z_vfwe'], cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r', figure=plt.figure(figsize=(18,4)))
mkda = nimare.meta.cbma.MKDADensity(dset, ids=dset.ids, kernel__r=10)
mkda.fit(n_iters=10)
plot_stat_map(mkda.results.images['vfwe'], cut_coords=[0, 0, -8],
              draw_cross=False, cmap='RdBu_r', figure=plt.figure(figsize=(18,4)))
z_imgs = []
sample_sizes = []
for study in dset_dict.keys():
    z_map_path = "../input/stat_maps/%s.nidm/ZStatistic_T001.nii.gz"%study
    t_map_path = "../input/stat_maps/%s.nidm/TStatistic.nii.gz"%study
    sample_size = dset_dict[study]["contrasts"]["1"]["sample_sizes"][0]
    if os.path.exists(z_map_path):
        z_imgs.append(nb.load(z_map_path))
        sample_sizes.append(sample_size)
    elif os.path.exists(t_map_path):
        t_map_nii = nb.load(t_map_path)
        # assuming one sided test
        z_map_nii = nb.Nifti1Image(t_to_z(t_map_nii.get_fdata(), sample_size-1), t_map_nii.affine)
        z_imgs.append(z_map_nii)
        sample_sizes.append(sample_size)
        
z_data = apply_mask(z_imgs, mask_img)
sample_sizes = np.array(sample_sizes)
for z_img in z_imgs[:5]:
    plot_stat_map(z_img, threshold=0, cut_coords=[0, 0, -8], 
                  draw_cross=False, figure=plt.figure(figsize=(18,4)))
result = fishers(z_data, mask_img)
plot_stat_map(result.images['ffx_stat'], threshold=0,
              cut_coords=[0, 0, -8], draw_cross=False,
              figure=plt.figure(figsize=(18,4)))
plot_stat_map(result.images['log_p'], threshold=-np.log(.05),
              cut_coords=[0, 0, -8], draw_cross=False,
              cmap='RdBu_r', figure=plt.figure(figsize=(18,4)))
result = weighted_stouffers(z_data, sample_sizes, mask_img)
plot_stat_map(result.images['log_p'], threshold=-np.log(.05),
              cut_coords=[0, 0, -8], draw_cross=False,
              cmap='RdBu_r', figure=plt.figure(figsize=(18,4)))