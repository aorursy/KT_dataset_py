# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
! wget 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt' 

! wget 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json' 

! wget 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin'

! wget 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt'

! wget 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json' 

! wget 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin' 
!ls /kaggle/working/bert-model