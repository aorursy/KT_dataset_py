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
import os
import pandas as pd
import numpy as np
import math
import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
import gc
pd.set_option("display.max_row", 500)
import matplotlib.pyplot as plt

class sigm():
    def __init__(self, s1, s2, a1 = 0, a2 = 0, b = 0, c1 = 0, c2 = 0, i = 0):
        self.rv = list()
        self.rl = list()
        self.a1 = a1
        self.a2 = a2
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.s1 = s1.values
        self.s2 = s2.values
        self.iter = i
        self.set_t1()

    def set_t1(self):
        ph = np.ndarray(shape = [self.s1.shape[0]])
        s = ph.shape[0]
        ph[:] = [ - s + i for i in range(s)]
        self.t = ph

    def set_t2(self):
        ph = np.ndarray(shape = [self.s1.shape[0]])
        s = ph.shape[0]
        ph[:] = [ i for i in range(s)]
        self.t = ph

    def x1(self):
        return self.a1 + \
               self.b * (self.t - self.c1) * self.b - \
               np.log(np.exp(self.b * (self.t - self.c1)) + 1)

    def x2(self):
        return self.a2 + \
               self.b * (self.t - self.c2) * self.b - \
               np.log(np.exp(self.b * (self.t - self.c2)) + 2)



    def gd(self):

        self.x11 = np.exp(self.b * (self.t - self.c1))
        self.x12 = self.x11 * math.exp(self.a1)

        self.A1 = self.x11 + self.x12 + 1
        self.B1 = self.x11 + 1
        self.x14 = 2 * (np.log(self.A1 / self.B1) - self.s1)

        self.a1_gd = (self.x12 * self.x14 / (self.A1))
        self.x13 = self.a1_gd / self.B1
        self.b1_gd = (self.x13 * self.t)
        self.c1_gd = (-self.x13 * self.b)


        self.x21 = np.exp(self.b * (self.t - self.c2))
        self.x22 = self.x21 * math.exp(self.a2)

        self.A2 = self.x21 + self.x22 + 1
        self.B2 = self.x21 + 1
        self.x24 = 2 * (np.log(self.A2 / self.B2) - self.s2)

        self.a2_gd = (self.x22 * self.x24 / (self.A2))
        self.x23 = self.a2_gd / self.B2
        self.b2_gd = (self.x23 * self.t)
        self.c2_gd = (-self.x23 * self.b)
        self.b_gd = self.b1_gd + self.b2_gd

        self.a1_gd = self.a1_gd.sum()
        self.a2_gd = self.a2_gd.sum()
        self.b_gd = self.b_gd.sum()
        self.c1_gd = self.c1_gd.sum()
        self.c2_gd = self.c2_gd.sum()
        self.iter += 1
        self.l2 = ((self.x14 ** 2).sum() + (self.x24 ** 2).sum()) / 4

    def mv(self):
        vn = self.a1_gd ** 2 + \
             self.a2_gd ** 2 + \
             self.b_gd ** 2 + \
             self.c1_gd ** 2 + \
             self.c2_gd ** 2
        st = min((self.l2 / vn) / 5,
                 1 / (vn * math.log(self.iter + 10)))
        self.rl.append(self.l2)
        self.rv.append([self.a1, self.a2, self.b, self.c1, self.c2])
        self.a1 = self.a1 - st * self.a1_gd
        self.a2 = self.a2 - st * self.a2_gd
        self.b = self.b - st * self.b_gd
        self.c1 = self.c1 - st * self.c1_gd
        self.c2 = self.c2 - st * self.c2_gd

tmp_dir = "/home/lsy/Project/kaggle/"
os.chdir(tmp_dir)
os.listdir("src_data")
train_data = pd.read_csv("./src_data/train.csv")
train_data["Province_State"].  fillna("", inplace = True)
train_data["sym"] = train_data["Country_Region"] + "@" + train_data["Province_State"]
train_data["c1"] = train_data["ConfirmedCases"]
train_data["c2"] = train_data["Fatalities"]
train_data["c3"] = (pd.to_datetime(train_data["Date"]) - datetime(2020, 1, 1)).apply(lambda x:x.days)
train_data["p_c1"] = np.log(train_data["c1"] + 1)
train_data["p_c2"] = np.log(train_data["c2"] + 1)




df1 = train_data.pivot(index = "sym", columns = "c3", values = "p_c1")
df2 = train_data.pivot(index = "sym", columns = "c3", values = "p_c2")


##df1 = train_data.pivot(index = "sym", columns = "c3", values = "c1")
##df2 = train_data.pivot(index = "sym", columns = "c3", values = "c2")
import os
pks = os.listdir(".")
res_dict = dict()
for k in range(df1.shape[0]):
    _name = df1.index[k]
    print(_name)
    if "{0}.pkl".format(_name) in pks:
        continue
    s1 = df1.iloc[k]
    s2 = df2.iloc[k]

    s1_1 = s1[:]
    s2_1 = s2[:]
    _cnt = 0
    while True:
        _cnt += 1
        print(rlmin[ - 1])
        if not "China" in _name:
            sp = sigm(s1 = s1_1, s2 = s2_1, a1 = 6.5, a2 = 4.5, b = 0.2, c1 = -55, c2 = -50)

        else:
            sp = sigm(s1 = s1_1, s2 = s2_1, a1 = 5.5, a2 = 3.5, b = 0.2, c1 = 3, c2 = 8)
        sp.gd()
        rlmin = [sp.l2]
        try:
            for i in range(100000):
                sp.gd()
                sp.mv()
            rl = pd.Series(sp.rl)
            rv = pd.DataFrame(sp.rv)
            rlmin.append(rl[ - 100000:].min())
        except:
            break

        if not rlmin[ - 1] < rlmin[ - 2]:
            break
        if _cnt > 5:
            break
    res_dict[_name] = dict()
    res_dict[_name]["s1"] = rv.loc[rl.index[rl == rl.min()][ - 1]]
    res_dict[_name]["s2"] = rl.loc[rl.index[rl == rl.min()][ - 1]]

    pd.Series(res_dict[_name]).to_pickle("{0}.pkl".format(_name))
    del sp, rl, rv
    gc.collect()


pd.read_csv("/kaggle/input/submission/submission.csv")[["ForecastId","ConfirmedCases","Fatalities"]].to_csv("/kaggle/working/submission.csv",index=False)
