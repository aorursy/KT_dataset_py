# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
aa_full = pd.read_csv("../input/tglf-new-coefficients/aa.csv").iloc[:, 1]

bb_full = pd.read_csv("../input/tglf-new-coefficients/bb.csv").iloc[:, 1]

cc_full = pd.read_csv("../input/tglf-new-coefficients/cc.csv").iloc[:, 1]



aa_michele = pd.read_csv("../input/tglf-new-coefficients/aa_michele.csv").iloc[:, 1]

bb_michele = pd.read_csv("../input/tglf-new-coefficients/bb_michele.csv").iloc[:, 1]

cc_michele = pd.read_csv("../input/tglf-new-coefficients/cc_michele.csv").iloc[:, 1]



aa_slow = pd.read_csv("../input/tglf-new-coefficients/aa_slow.csv").iloc[:, 1]

bb_slow = pd.read_csv("../input/tglf-new-coefficients/bb_slow.csv").iloc[:, 1]

cc_slow = pd.read_csv("../input/tglf-new-coefficients/cc_slow.csv").iloc[:, 1]



aa_wo_relu = pd.read_csv("../input/tglf-new-coefficients/aa_wo_relu.csv").iloc[:, 1]

bb_wo_relu = pd.read_csv("../input/tglf-new-coefficients/bb_wo_relu.csv").iloc[:, 1]

cc_wo_relu = pd.read_csv("../input/tglf-new-coefficients/cc_wo_relu.csv").iloc[:, 1]



aa_scaled = pd.read_csv("../input/tglf-new-coefficients/aa_scaled.csv").iloc[:, 1]

bb_scaled = pd.read_csv("../input/tglf-new-coefficients/bb_scaled.csv").iloc[:, 1]

cc_scaled = pd.read_csv("../input/tglf-new-coefficients/cc_scaled.csv").iloc[:, 1]
taml = pd.read_csv("../input/tglf-and-ml/TAML_v1.csv")[:-6]

aa0 = taml["aa0"]

aa1 = taml["aa1"]

bb0 = taml["bb0"]

bb1 = taml["bb1"]

cc0 = taml["cc0"]

cc1 = taml["cc1"]



# aa_full = sigmoid(aa_full)

# bb_full = sigmoid(bb_full)

# cc_full = sigmoid(cc_full)
import matplotlib.pyplot as plt

bins = 20

alpha=0.5



comparison = [aa_scaled,bb_scaled,cc_scaled]
plt.hist(comparison[0], bins, alpha=alpha, label='aa_new')

plt.hist(aa0, bins, alpha=alpha, label='aa0')

plt.legend(loc='upper right')

plt.show()
plt.hist(comparison[0], bins, alpha=alpha, label='aa_new')

plt.hist(aa1, bins, alpha=alpha, label='aa1')

plt.legend(loc='upper right')

plt.show()
plt.hist(comparison[1], bins, alpha=alpha, label='bb_new')

plt.hist(bb0, bins, alpha=alpha, label='bb0')

plt.legend(loc='upper right')

plt.show()
plt.hist(comparison[1], bins, alpha=alpha, label='bb_new')

plt.hist(bb1, bins, alpha=alpha, label='bb1')

plt.legend(loc='upper right')

plt.show()
plt.hist(comparison[2], bins, alpha=alpha, label='cc_new')

plt.hist(cc0, bins, alpha=alpha, label='cc0')

plt.legend(loc='upper right')

plt.show()
plt.hist(comparison[2], bins, alpha=alpha, label='cc_new')

plt.hist(cc1, bins, alpha=alpha, label='cc1')

plt.legend(loc='upper right')

plt.show()