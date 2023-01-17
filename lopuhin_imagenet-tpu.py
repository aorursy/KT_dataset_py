! git clone https://github.com/lopuhin/tpu-imagenet.git

! cd tpu-imagenet && git show -s
! cd tpu-imagenet && git pull
! export PYTHONPATH=tpu-imagenet
! ls /kaggle/input/
from pathlib import Path

from kaggle_datasets import KaggleDatasets



gcs_paths = [KaggleDatasets().get_gcs_path(p.name)

             for p in Path('/kaggle/input/').iterdir()]

gcs_paths
! tpu-imagenet/train.py {' '.join(gcs_paths)} --epochs 3