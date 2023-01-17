# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy.matlib

import numpy as np

from tqdm import tqdm
for i in tqdm(range(50000)):

    W = 200

    N = 50

    X = np.zeros(N)

    X[0] = np.random.rand(1) * (2*W / N) 

    for num in range(1,N-1):

        B_k = 2/(N-num) * (W-np.sum(X[:num]))

        X[num] = np.random.rand(1) * B_k

    X[49] = W - sum(X[:49])

    if i == 0:

        X_array = X

    else:

        X_array = np.row_stack((X_array,X))

    
np.mean(X_array,axis=0)
comparsion_num = 0

for _ in range(10000):

    random_num = np.random.rand(1)

    if random_num < 0.05:

        comparsion_num += 1

    else:

        if random_num < 0.1:

            comparsion_num += 2

        else:

            if random_num < 0.2:

                comparsion_num += 3

            else:

                if random_num <0.3:

                    comparsion_num += 4

                else:

                    if random_num < 0.9:

                        comparsion_num += 5

                    else:

                        comparsion_num += 5

comparsion_num/10000
comparsion_num = 0

for _ in range(10000):

    random_num = np.random.rand(1)

    if random_num < 0.6:

        comparsion_num += 1

    else:

        if random_num < 0.7:

            comparsion_num += 2

        else:

            if random_num < 0.8:

                comparsion_num += 3

            else:

                if random_num <0.9:

                    comparsion_num += 4

                else:

                    if random_num < 0.95:

                        comparsion_num += 5

                    else:

                        comparsion_num += 5

comparsion_num/10000
gussian_array = np.random.multivariate_normal(mean=[0,0],cov=[[1,0.8],[0.8,1]],size=[100000,2])
%matplotlib inline



import seaborn as sns

from scipy import stats
mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.8], 

                                                     [0.8, 1.]])

x = mvnorm.rvs(100000)
h = sns.jointplot(x[:, 0], x[:, 1], kind='kde', stat_func=None)

h.set_axis_labels('X1', 'X2', fontsize=16)
norm = stats.norm()

x_unif = norm.cdf(x)

h = sns.jointplot(x_unif[:, 0], x_unif[:, 1], kind='hex', stat_func=None)

h.set_axis_labels('Y1', 'Y2', fontsize=16)
m1 = stats.expon()

m2 = stats.weibull_max(2)

m2 = stats.beta(a=10, b=2)



x1_trans = m1.ppf(x_unif[:, 0])

x2_trans = m2.ppf(x_unif[:, 1])



h = sns.jointplot(x1_trans, x2_trans, kind='kde', xlim=(-6, 2), ylim=(.6, 1.0), stat_func=None);

h.set_axis_labels('Maximum river level', 'Probablity of flooding', fontsize=16);
mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.2], 

                                                     [0.2, 1.]])

x = mvnorm.rvs(100000)
h = sns.jointplot(x[:, 0], x[:, 1], kind='kde', stat_func=None)

h.set_axis_labels('X1', 'X2', fontsize=16)
norm = stats.norm()

x_unif = norm.cdf(x)

h = sns.jointplot(x_unif[:, 0], x_unif[:, 1], kind='hex', stat_func=None)

h.set_axis_labels('Y1', 'Y2', fontsize=16)
m1 = stats.expon()

m2 = stats.weibull_max(2)

m2 = stats.beta(a=10, b=2)



x1_trans = m1.ppf(x_unif[:, 0])

x2_trans = m2.ppf(x_unif[:, 1])



h = sns.jointplot(x1_trans, x2_trans, kind='kde', xlim=(-6, 2), ylim=(.6, 1.0), stat_func=None);

h.set_axis_labels('Maximum river level', 'Probablity of flooding', fontsize=16);
mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., -0.8], 

                                                     [-0.8, 1.]])

x = mvnorm.rvs(100000)
h = sns.jointplot(x[:, 0], x[:, 1], kind='kde', stat_func=None)

h.set_axis_labels('X1', 'X2', fontsize=16)
norm = stats.norm()

x_unif = norm.cdf(x)

h = sns.jointplot(x_unif[:, 0], x_unif[:, 1], kind='hex', stat_func=None)

h.set_axis_labels('Y1', 'Y2', fontsize=16)
m1 = stats.expon()

m2 = stats.weibull_max(2)

m2 = stats.beta(a=10, b=2)



x1_trans = m1.ppf(x_unif[:, 0])

x2_trans = m2.ppf(x_unif[:, 1])



h = sns.jointplot(x1_trans, x2_trans, kind='kde', xlim=(-6, 2), ylim=(.6, 1.0), stat_func=None);

h.set_axis_labels('Maximum river level', 'Probablity of flooding', fontsize=16);
def lambd(t):

    return 1 + np.sin(t)
t = 0

T = 20

i = 0

sequence = []

while t < 20:

    u = np.random.rand(1)

    t = t - 1/lambd(t) * np.log(u)

    if t < T:

        i += 1

        sequence.append(t)

for i in range(len(sequence)):

    print("customer_num:"+str(i+1)+"   arrive_time:"+str(sequence[i]))