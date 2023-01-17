# !pip install wfdb

import wfdb

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
recordname = "../input/apnea-ecg/a04"



record = wfdb.rdsamp(recordname)
annotation = wfdb.rdann(recordname, extension="apn")



annotation.contained_labels
annotation.get_label_fields()
annotation.symbol[:10]
np.unique(annotation.symbol, return_counts=True)
record_small = wfdb.rdsamp(recordname, sampto = 5999)

annotation_small = wfdb.rdann(recordname, extension="apn", sampto = 5999)

wfdb.plot_all_records("../input/apnea-ecg")