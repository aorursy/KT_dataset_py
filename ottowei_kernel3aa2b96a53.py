

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import wfdb

import matplotlib.pyplot as plt

import os





# for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

os.listdir("../input/apneaecg/apnea-ecg")





record = wfdb.rdrecord('../input/apneaecg/apnea-ecg/a01') 

wfdb.plot_wfdb(record, title='Record a01 from Physionet Kaggle Apnea ECG') 

display(record.__dict__)





record2 = wfdb.rdrecord('../input/apneaecg/apnea-ecg/a05') 

wfdb.plot_wfdb(record, title='Record c10 from Physionet Kaggle Apnea ECG') 

display(record2.__dict__)





recordname = "../input/apneaecg/apnea-ecg/a04"

record3 = wfdb.rdsamp(recordname)

annotation = wfdb.rdann(recordname, extension="apn")



annotation.contained_labels

annotation.get_label_fields()

annotation.symbol[:10]

np.unique(annotation.symbol, return_counts=True)
import wfdb

import numpy as np



record = wfdb.rdrecord(record_name='../input/apneaecg/apnea-ecg/a01', sampfrom=0, sampto=1000,channels=None, physical=True, pb_dir=None, m2s=True, smooth_frames=True, ignore_skew=False, return_res=16, force_channels=True, channel_names=None, warn_empty=False)



wfdb.plot_wfdb(record, title='Record a01 from Physionet Kaggle Apnea ECG') 

display(record.__dict__)