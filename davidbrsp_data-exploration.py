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

# Any results you write to the current directory are saved as outpu
data_all = pd.read_excel("/kaggle/input/covid19/dataset.xlsx") # load excel file

# supressing columns for better visualization
columns_to_drop = ["Patient ID", "Patient age quantile", "Patient addmited to regular ward (1=yes, 0=no)", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Mycoplasma pneumoniae", "Promyelocytes","Metamyelocytes", "Myelocytes", "Myeloblasts", "Fio2 (venous blood gas analysis)", "Urine - Sugar", "Urine - Red blood cells", "Partial thromboplastin time (PTT) ", "Prothrombin time (PT), Activity", "D-Dimer", "Vitamin B12", "Albumin"]

#data_all =
#data_all.dropna()
#data_all.infer_objects()

#data_all.loc[data_all['SARS-Cov-2 exam result'] == 'positive'] = 1
#data_all.loc[data_all['SARS-Cov-2 exam result'] == 'negative'] = 0

data_positive = data_all.loc[data_all["SARS-Cov-2 exam result"] == 'positive'][:] # select only covid-19 positive cases
#data_positive = data_all.loc[data_all["SARS-Cov-2 exam result"] == 1][:] # select only covid-19 positive cases
data_positive.drop(columns=columns_to_drop, inplace=True)
# select only covid-19 negative cases
data_negative = data_all.loc[data_all["SARS-Cov-2 exam result"] == 'negative'][:]
#data_negative = data_all.loc[data_all["SARS-Cov-2 exam result"] == 0][:]
data_negative.drop(columns=columns_to_drop, inplace=True)

Dat


# data_all.boxplot(rot=90, fontsize=10, figsize=(16,6)) # boxplot of all cases (negative and positive)
import matplotlib.pyplot as plt

my_medianprops = dict(linestyle='-', linewidth=2, color='red')
my_fontsize=10
my_showfliers=False

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 6), sharey=True)

box_positive = data_positive.boxplot(return_type='dict', ax=axes[0], rot=90, fontsize=my_fontsize, showfliers=my_showfliers, medianprops=my_medianprops) # draw boxplot of positive cases
box_negative = data_negative.boxplot(return_type='dict', ax=axes[1], rot=90, fontsize=my_fontsize, showfliers=my_showfliers, medianprops=my_medianprops) # draw boxplot of positive cases

# https://stackoverflow.com/questions/18861075/overlaying-the-numeric-value-of-median-variance-in-boxplots
x_pos, y_pos = [[], []]
for line in box_positive['medians']:
    # get position data for median line
    x0, y0 = line.get_xydata()[0] # top of median line
#    x0, y0, x1, y1 = line.get_xydata() # top of median line
    x1, y1 = line.get_xydata()[1] # bottom of median line
    x_pos.append((x0 + x1) / 2)
    y_pos.append(y0)

x_neg, y_neg = [[], []]
for line in box_negative['medians']:
    # get position data for median line
    x0, y0 = line.get_xydata()[0] # top of median line
#    x1, y1, x2, y2 = line.get_xydata() # top of median line
    x1, y1 = line.get_xydata()[1] # bottom of median line
    x_neg.append((x0 + x1) / 2)
    y_neg.append(y1)

    
from scipy.interpolate import make_interp_spline, BSpline

# 300 represents number of points to make between min and max
x_pos_sp = np.linspace(min(x_pos), max(x_pos), 300)
spl = make_interp_spline(x_pos, y_pos, k=3)  # type: BSpline
y_pos_sp = spl(x_pos_sp)

# 300 represents number of points to make between min and max
x_neg_sp = np.linspace(min(x_neg), max(x_neg), 300)
spl = make_interp_spline(x_neg, y_neg, k=3)  # type: BSpline
y_neg_sp = spl(x_neg_sp)

#plt.plot(xnew, power_smooth)    
    
    
axes[0].plot(x_pos_sp, y_pos_sp, color='red', linewidth=1)
axes[1].plot(x_neg_sp, y_neg_sp, color='green',linewidth=1)

axes[0].set_title('covid-19 positive n=(' + str(data_positive.shape[0]) + ')')
axes[0].set_xticklabels('')
axes[1].set_title('covid-19 negative n=(' + str(data_negative.shape[0]) + ')')

fig.subplots_adjust(hspace=0.2)
plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#X_std = StandardScaler().fit_transform(X)

X_std = StandardScaler().fit_transform(data_positive)

pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
