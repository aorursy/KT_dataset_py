# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import turicreate as tc
! mkdir temp
!ls
tc.config.set_runtime_config('TURI_CACHE_FILE_LOCATIONS', 'temp')
data =  tc.SFrame('../input/BreastCancer2.csv')
train_data, test_data = data.random_split(0.8)
model = tc.classifier.create(train_data, target='class', features = ['thickness', 'size','shape','adhesion','single','nuclei','chromatin','nucleoli','mitosis'])
predictions = model.classify(test_data)
predictions
# obtain statistical results for the model by model.evaluate method 
results = model.evaluate(test_data)
results
