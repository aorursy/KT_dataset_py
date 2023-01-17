# Install dicom-csv, utils for gathering, aggregation and handling metadata from DICOM files.

# This allows us to join DICOMs into volumes in a correct order without relying on file names.



!pip install git+https://github.com/neuro-ml/dicom-csv.git
# install deep-pipe, collection of tools for dl experiments,

# here just for the vizualisation



!pip install deep-pipe
from dicom_csv import join_tree, aggregate_images, load_series
# Single row of df corresponds to a single DICOM file (unique SOPInstanceUID in DICOM's slang)

# First run takes ~3-4 mins, since it actually opens all available DICOMs inside a directory,

# so I recommend you to store the resulting DataFrame



df = join_tree('/kaggle/input/osic-pulmonary-fibrosis-progression/train', verbose=True, relative=False, force=True)

df.head(3)
df['NoError'].value_counts()
# Single row of df_images corresponds to a single volume (unique SeriesInstanceUID in DICOM's slang)





df_images = aggregate_images(df.query('NoError == True'))

df_images.head(3)
df_images.RescaleSlope.value_counts().sort_index()
df_images.RescaleIntercept.value_counts().sort_index()
image = load_series(df_images.iloc[0], orientation=False)
from dpipe.im.visualize import slice3d
# There is a slide bar you could move to go over different slices (run notebook to see it).



slice3d(image)
import matplotlib.pyplot as plt

%matplotlib inline
df_images.columns
df_images['PixelArrayShape'].value_counts()
df_images['PatientSex'].value_counts()
df_images['PixelSpacing0'].value_counts().sort_index()
plt.hist(df_images['PixelSpacing0'], bins=15)

plt.xlabel('Pixel spacing, mm.')

plt.ylabel('Number of images.');
df_images['SlicesCount'].value_counts().sort_index()
plt.hist(df_images['SlicesCount'], bins=15)

plt.xlabel('Number of slices in a single 3D image.')

plt.ylabel('Number of images.');