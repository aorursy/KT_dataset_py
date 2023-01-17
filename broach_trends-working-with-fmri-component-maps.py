"""

From: https://www.kaggle.com/bbradt/loading-and-exploring-spatial-maps

"""

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from scipy import stats

from scipy.ndimage.morphology import generate_binary_structure

from scipy.ndimage.measurements import label

import numpy as np # linear algebra

import nilearn as nl

import nilearn.plotting as nlplt

import nibabel as nib

import h5py

import matplotlib.pyplot as plt

from nilearn.masking import apply_mask

from nilearn.masking import unmask



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

mask_filename = '../input/trends-assessment-prediction/fMRI_mask.nii'

#subject_filename = '../input/trends-assessment-prediction/fMRI_train/10004.mat'

subject_filename = '../input/trends-assessment-prediction/fMRI_test/10030.mat'

# smri_filename = 'ch2better.nii'

mask_niimg = nl.image.load_img(mask_filename)



def load_subject(filename, mask_niimg):

    """

    Load a subject saved in .mat format with

        the version 7.3 flag. Return the subject

        niimg, using a mask niimg as a template

        for nifti headers.

        

    Args:

        filename    <str>            the .mat filename for the subject data

        mask_niimg  niimg object     the mask niimg object used for nifti headers

    """

    subject_data = None

    with h5py.File(subject_filename, 'r') as f:

        subject_data = f['SM_feature'][()]

    # It's necessary to reorient the axes, since h5py flips axis order

    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])

    subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)

    return subject_niimg
#example...



subject_niimg = load_subject(subject_filename, mask_niimg)

print("Image shape is %s" % (str(subject_niimg.shape)))

num_components = subject_niimg.shape[-1]

print("Detected {num_components} spatial maps".format(num_components=num_components))
type(subject_niimg)
maskData = apply_mask(subject_niimg, mask_niimg)

type(maskData)
print('Total number of voxels:' + str(53*63*52))

print('Number of voxels in standardized brain mask:' + str(maskData.shape[1]))

df_train_scores = pd.read_csv('../input/trends-assessment-prediction/train_scores.csv')

df_train_scores['age_bins'] = pd.cut(x=df_train_scores['age'], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90], 

                                     labels=['teens','twenties','thirties','forties','fifties','sixties','seventies','eighties'])

skf = StratifiedKFold(n_splits=25, shuffle=True, random_state=5272020)

for train_index, test_index in skf.split(df_train_scores, df_train_scores['age_bins']):

     print("TRAIN length:", len(train_index), "TEST length:", len(test_index))
#this is just a test, so let's try component 5 (ADN) for fun:

myComp = 5

#initialize np array for the test subjects:

sMat = np.zeros(shape=(len(test_index), maskData.shape[1]))
i = 0

for id in test_index:

    subject_filename = '../input/trends-assessment-prediction/fMRI_train/' + str(df_train_scores['Id'].iloc[id]) + '.mat'

    subject_niimg = load_subject(subject_filename, mask_niimg)

    maskData = apply_mask(subject_niimg, mask_niimg)

    sMat[i,]= maskData[myComp,]

    i += 1
t = stats.ttest_1samp(sMat, 0, axis=0)

tmap = unmask(t.statistic, mask_niimg).get_fdata()



t_img = nib.Nifti1Image(tmap, header=mask_niimg.header, affine=mask_niimg.affine)

nlplt.plot_stat_map(t_img, title="IC %d" % myComp, threshold=20.2, colorbar=True)
#generate the binary structure:

struct = generate_binary_structure(3,3)

labeled_array, num_features = label(tmap>20, struct)

    

#label the clusters

L_img = nib.Nifti1Image(labeled_array, header=mask_niimg.header, affine=mask_niimg.affine)



nlplt.plot_roi(L_img, colorbar=True, cmap='Paired')



num_features
affine = mask_niimg.affine

label_img = nib.Nifti1Image(labeled_array, affine)

clustMask = apply_mask(label_img, mask_niimg)
clustMask.shape
RightHemMean = np.mean(sMat[:,clustMask==1], axis=1)

LeftHemMean = np.mean(sMat[:,clustMask==2], axis=1)
r = stats.linregress(RightHemMean, LeftHemMean)



print(r)
plt.plot(RightHemMean, LeftHemMean, 'o', label='original data')

plt.plot(RightHemMean, r.intercept + r.slope*RightHemMean, 'r', label='fitted line')

plt.legend()

plt.show()
X =  np.zeros(shape=len(test_index))

i = 0

for id in test_index:

    X[i]= df_train_scores['age'].iloc[id]

    i += 1
r = stats.linregress(RightHemMean, X)



print(r)

plt.plot(RightHemMean, X, 'o', label='original data')

plt.plot(RightHemMean, r.intercept + r.slope*RightHemMean, 'r', label='fitted line')

plt.legend()

plt.show()