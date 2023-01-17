# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Set your own project id here
PROJECT_ID = 'kernel47d170785e' #your-google-cloud-project
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
for dirname, _, filenames in os.walk('/kaggle/metadata.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
