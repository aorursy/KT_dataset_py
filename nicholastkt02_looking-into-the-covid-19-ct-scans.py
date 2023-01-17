import pandas as pd

import numpy as np

from glob import glob



from nilearn import image, plotting

import nibabel as nib
import warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
metadata_df_path = "../input/covid19-ct-scans/metadata.csv"

metadata_df = pd.read_csv(metadata_df_path)

metadata_df.head()
len(metadata_df.ct_scan.unique())
metadata_df.ct_scan[0]

ct_scan_ex_img = metadata_df.ct_scan[0]
ct_scan_example_img = image.smooth_img(ct_scan_ex_img, fwhm = 3)
print("ct_scans directory example image")

plotting.plot_img(ct_scan_example_img)
infect_mask = metadata_df.infection_mask[0]

infect_mask
infect_mask = image.smooth_img(infect_mask, fwhm = 3)

plotting.plot_img(infect_mask)
lung_infec_mask = metadata_df.lung_and_infection_mask[0]

lung_infec_mask
lung_infec_mask_img = image.smooth_img(lung_infec_mask, fwhm = 3)

plotting.plot_img(lung_infec_mask_img)
lung_mask = metadata_df.lung_mask[0]

lung_mask

lung_mask_img = image.smooth_img(lung_mask, fwhm = 3)

plotting.plot_img(lung_mask_img)
test_path = metadata_df.ct_scan[10]

test_path_img = nib.load(test_path)

test_array2 = test_path_img.get_data()

test_array = np.array(test_path_img)

test_array

plotting.plot_img(test_path_img)
test_array2.shape