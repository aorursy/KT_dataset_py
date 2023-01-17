# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install /kaggle/input/fastai-200/dataclasses-0.6-py3-none-any.whl
!pip install /kaggle/input/fastai-200/fastcore-1.0.0-py3-none-any.whl
!pip install /kaggle/input/fastai-200/torch-1.6.0-cp37-cp37m-manylinux1_x86_64.whl
!pip install /kaggle/input/fastai-200/torchvision-0.7.0-cp37-cp37m-manylinux1_x86_64.whl
!pip install /kaggle/input/fastai-200/fastai-2.0.0-py3-none-any.whl
!pip install /kaggle/input/fastai-200/fastscript-1.0.0-py3-none-any.whl
!pip install /kaggle/input/fastai-200/nbdev-1.0.1-py3-none-any.whl
!pip install /kaggle/input/fastai-200/isodate-0.6.0-py2.py3-none-any.whl
!pip install /kaggle/input/fastai-200/msrest-0.6.18-py2.py3-none-any.whl
!pip install /kaggle/input/fastai-200/msrestazure-0.6.4-py2.py3-none-any.whl
!pip install /kaggle/input/fastai-200/azure_common-1.1.25-py2.py3-none-any.whl
!pip install /kaggle/input/fastai-200/azure_cognitiveservices_search_imagesearch-2.0.0-py2.py3-none-any.whl
!pip install /kaggle/input/fastai-200/sentencepiece-0.1.86-cp37-cp37m-manylinux1_x86_64.whl
!pip install /kaggle/input/fastai-200/fastbook-0.0.8-py3-none-any.whl
#!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
dest = '/kaggle/input/bears-model/grizzly.jpg'
im = Image.open(dest)
im.to_thumb(224,224)
path = Path('/kaggle/input/bears-model/')
path.ls(file_exts='.pkl')
learn_inf = load_learner(path/'export.pkl')
learn_inf.predict(path/'grizzly.jpg')
learn_inf.dls.vocab