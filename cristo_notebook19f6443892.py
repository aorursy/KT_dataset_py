# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.io



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_1 = scipy.io.loadmat("../input/train_1/1_1_1.mat", struct_as_record=False, squeeze_me=True)

test_1 = scipy.io.loadmat("../input/test_1/1_1.mat", struct_as_record=False, squeeze_me=True)
struct = train_1['dataStruct']

print(struct.channelIndices)

print(struct.data.shape)

print(struct.iEEGsamplingRate)

print(struct.nSamplesSegment)

print(struct.sequence)



struct = test_1['dataStruct']

print(struct.channelIndices)

print(struct.data.shape)

print(struct.iEEGsamplingRate)

print(struct.nSamplesSegment)

print(struct.sequence)
type(struct.data)
np.ndarray(shape=(0))