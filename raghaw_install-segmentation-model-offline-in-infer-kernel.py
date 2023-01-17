import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!ls ../input/segmentation-models-zip/
!mkdir -p /tmp/pip/cache/

!cp ../input/segmentation-models-zip/pretrainedmodels-0.7.4.xyz /tmp/pip/cache/pretrainedmodels-0.7.4.tar.gz

!cp ../input/segmentation-models-zip/segmentation_models_pytorch-0.0.1-py2.py3-none-any.whl /tmp/pip/cache/

!cp ../input/segmentation-models-zip/torchfile-0.1.0.xyz /tmp/pip/cache/torchfile-0.1.0.tar.gz

!cp ../input/segmentation-models-zip/torchnet-0.0.4.xyz /tmp/pip/cache/torchnet-0.0.4.tar.gz

!cp ../input/segmentation-models-zip/torchvision-0.2.2-py2.py3-none-any.whl /tmp/pip/cache/

!cp ../input/segmentation-models-zip/tqdm-4.19.9-py2.py3-none-any.whl /tmp/pip/cache/

!cp ../input/segmentation-models-zip/visdom-0.1.8.8.xyz /tmp/pip/cache/visdom-0.1.8.8.tar.gz
!pip install --no-index --find-links /tmp/pip/cache/ segmentation-models-pytorch

#!pip install --find-links /tmp/pip/cache/ segmentation-models-pytorch
import segmentation_models_pytorch as smp