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
import os



os.listdir("../input/apnea-ecg")
# !pip install wfdb

import wfdb

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
recordname = "../input/apnea-ecg/a04"



record = wfdb.rdsamp(recordname)
record.p_signals
annotation = wfdb.rdann(recordname, extension="apn")



annotation.contained_labels
annotation.get_label_fields()
annotation.symbol[:10]
np.unique(annotation.symbol, return_counts=True)
record_small = wfdb.rdsamp(recordname, sampto = 5999)

annotation_small = wfdb.rdann(recordname, extension="apn", sampto = 5999)

wfdb.plotrec(record_small, annotation=annotation_small,

             title='Record ' + recordname + ' from Apnea Database',

             timeunits = 'seconds', ecggrids = 'all')