!pip install wfdb
import numpy as np
import pandas as pd
import os
import wfdb
import matplotlib.pyplot as plt
all_signal=[]
meta_data=[]
for file in sorted(os.listdir("/kaggle/input/stress-recognition-in-automobile-drivers/physionet.org/files/drivedb/1.0.0/")):
    if file.endswith(".dat"):
        signals, fields = wfdb.rdsamp("/kaggle/input/stress-recognition-in-automobile-drivers/physionet.org/files/drivedb/1.0.0/" + os.path.splitext(file)[0])
        all_signal.append(signals)
        meta_data.append(fields)
my_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
plt.style.use('ggplot')
plt.figure(figsize=(50, 24))
plt.suptitle("Driver 1", fontsize=40)
for i in range(all_signal[0].shape[1]):
    plt.subplot(2, 3, i + 1)
    plt.ylabel(meta_data[0]['units'][i], fontsize=18)
    plt.xlabel("samples (fs = 15.5)", fontsize=18)
    plt.plot(all_signal[0].T[i], color=my_colors[i])
    plt.title(meta_data[0]['sig_name'][i],fontsize=26)
my_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
plt.style.use('ggplot')
plt.figure(figsize=(50, 24))
plt.suptitle("Driver 1 - first 60 seconds", fontsize=40)
for i in range(all_signal[0].shape[1]):
    plt.subplot(2, 3, i + 1)
    plt.ylabel(meta_data[0]['units'][i], fontsize=18)
    plt.xlabel("samples (fs = 15.5)", fontsize=18)
    plt.plot(all_signal[0].T[i][0:int(15.5 * 60)], color=my_colors[i])
    plt.title(meta_data[0]['sig_name'][i],fontsize=26)