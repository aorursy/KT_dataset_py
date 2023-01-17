import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from kaggle_datasets import KaggleDatasets

GCS_PATH = KaggleDatasets().get_gcs_path()
!gsutil ls $GCS_PATH
!gsutil version -l