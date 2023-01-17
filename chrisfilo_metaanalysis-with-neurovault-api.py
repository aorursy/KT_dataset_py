# This needs to be installed inside the notebook for compatibility with Kaggle

!pip install git+git://github.com/neurostuff/nimare.git@608516ec3034e356326dfe70df5e9ed77efd2be8#egg=nimare
%matplotlib inline



import io

import requests

import pandas as pd

from nilearn.image import resample_to_img, math_img

from nilearn.datasets import load_mni152_template, load_mni152_brain_mask

import nibabel as nb

from gzip import GzipFile

from nimare.meta.ibma import stouffers, weighted_stouffers

from nilearn.masking import apply_mask

from nilearn.plotting import plot_roi, plot_stat_map, plot_glass_brain



template_nii = load_mni152_template()

template_mask_nii = load_mni152_brain_mask()
def get_images_metadata(collection_ids):

    images = []

    for collection_id in collection_ids:

        url = "http://neurovault.org/api/collections/%d/images/?format=json"%collection_id

        while url:

            r = requests.get(url)

            d = r.json()

            images += d['results']

            url = d['next']

    return pd.DataFrame(images)
collection_ids = [3235]
all_images_df = get_images_metadata(collection_ids)

all_images_df.columns
ss_images_df = all_images_df[all_images_df.analysis_level == 'single-subject']

ss_images_df.describe()
def perform_metaanalysis(images_df):

    z_imgs = []

    for i, row in images_df.iterrows():

        download_url = row['file']

        print("Downloading %s"%download_url)

        r = requests.get(download_url)

        fp = io.BytesIO(r.content)

        gzfileobj = GzipFile(filename="tmp.nii.gz", mode='rb', fileobj=fp)

        nii = nb.Nifti1Image.from_file_map({'image': nb.FileHolder("tmp.nii.gz", gzfileobj)})



        # making sure all images have the same size

        resampled_nii = resample_to_img(nii, template_nii)

        z_imgs.append(resampled_nii)



    z_data = apply_mask(z_imgs, template_mask_nii)

    results = stouffers(z_data, template_mask_nii, inference='ffx', null='theoretical', corr='FWE', two_sided=True)

    return results
results = perform_metaanalysis(ss_images_df)
results.images.keys()
plot_roi(math_img(formula = '(a < 0.05)*mask', a=results.images['p'], mask=template_mask_nii))
ss_above_18_df = ss_images_df[ss_images_df.age > 18]
len(ss_above_18_df)
results = perform_metaanalysis(ss_above_18_df)

plot_roi(math_img(formula = '(a < 0.05)*mask', a=results.images['p'], mask=template_mask_nii))