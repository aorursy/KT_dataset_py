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
df_vacuum = pd.read_csv("/kaggle/input/vacuumvoltage/U-t (1).dat", delim_whitespace=True)

#df_aluminium.set_index('Laikas (s)', drop=True, inplace=True)

df_vacuum = df_vacuum.drop([0])



#Use p from the given table

def torr_to_pa(torr):

    return torr * 133.322368



df_vacuum['p(Torr)'] = [1, 1, 0.744, 0.599, 0.359, 0.278, 0.215, 0.200, 0.190, 0.158, 0.136, 0.126, 0.117, 0.111, 0.105, 0.068, 0.046, 0.031, 0.026, 0.021, 0.018, 0.017]

df_vacuum['p(pa)'] = [torr_to_pa(torr) for torr in df_vacuum['p(Torr)']]



def rate(i):

    V = 1

    p1 = df_vacuum['p(pa)'][i]

    p2 = df_vacuum['p(pa)'][i+1]

    return V * np.log(p1/p2)



def rate_S(i):

    if i + 1 > len(df_vacuum['p(pa)']) or i < 1:

        im1 = rate(i - 1)

        if im1 != 0:

            return im1

    elif i <= 1:

        ip1 = rate(i + 1)

        if ip1 != 0:

            return ip1

    

    return rate(i)





df_vacuum['S'] = [rate_S(i) for i in range(1, len(df_vacuum) + 1)]



df_vacuum
df_vacuum.plot(x='t(s)', y='p(Torr)', grid=True )

df_vacuum.plot(x='t(s)', y='p(Torr)', grid=True, logy=True )

df_vacuum.plot(x='t(s)', y='p(pa)', grid=True )

df_vacuum.plot(x='t(s)', y='p(pa)', grid=True, logy=True )
df_vacuum.plot(x='t(s)', y='S', grid=True, title='S(t)' )

df_vacuum.plot(x='p(pa)', y='S', grid=True, title='S(p)' )

df_vacuum.plot(x='p(Torr)', y='S', grid=True, title='S(p)' )
def free_path(p):

    # Calculate the powers of 10 by hand to increase accuracy

    k_b = 1.38 #* 10**(-23)

    T = 3 #* 10**2

    d2 = 0.675 #* 10**-20

    #* 10^(-1)

    return (k_b * T) / (np.sqrt(2) * np.pi * p * d2) * 0.1



def knudsen_number(l):

    D = 0.01

    return l / D



df_vacuum['l'] = [free_path(p) for p in df_vacuum['p(pa)']]

df_vacuum['Kn'] = [knudsen_number(l) for l in df_vacuum['l']]



df_vacuum