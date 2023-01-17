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
{
    
  "id": "c3b81304-0d8e-49f8-bfdd-4964ff3696fa",
  "realName": "buildings",
  "name": "HERE Buildings layer",
  "shortDescription": "All countries ingested",
  "version": 1527233555,
  "description": "Layer containing data about buildings",
  "geoProperties": [
    {
      "id": "5e467099-578e-4857-9bbc-4b7fd5ef5d4a",
      "name": "name - Building name",
      "dataType": "string",
      "shortDescription": "Name of the building in the corresponding language",
      "description": "Name of the building in the corresponding language"
        }
      
      ]
    
    }