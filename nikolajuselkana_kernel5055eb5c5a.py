import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Plot data

from scipy import stats

from scipy.signal import savgol_filter



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
headers = ["t(s)", "T(C)"]



specific_heat_copper = np.array([[0, 100, 200, 300, 400], [0.3810, 0.3935, 0.4082, 0.4220, 0.4359]])

df_specific_heat_copper = pd.DataFrame(data=specific_heat_copper, index=['T, (C)', 'c_1'])



df_aluminium = pd.read_csv("/kaggle/input/aluminiumandcopperheatloss/lab28_d07_Al.dat", delim_whitespace=True, names=headers)

df_aluminium['T(K)'] = [273 + T for T in df_aluminium['T(C)']]

df_copper = pd.read_csv("/kaggle/input/aluminiumandcopperheatloss/lab28_d07_Cu.dat",sep='\s+', names=headers)

df_copper['T(K)'] = [273 + T for T in df_copper['T(C)']]



df_aluminium
def get_c_specific (T):

    t = df_specific_heat_copper

    index_lower = int(T/100)

    T_lower_bound = t[index_lower][0]

    c_lower_bound = t[index_lower][1]

    delta_c = t[index_lower + 1][1] - c_lower_bound

    

    return delta_c / 100 * (T - T_lower_bound) + c_lower_bound



df_copper['c specific'] = [get_c_specific(T) for T in df_copper['T(C)']]

df_copper.plot(x='T(K)', y='c specific', grid=True )
def diff_T(i1, i2, df_metal):

    return df_metal['T(C)'][i2] - df_metal['T(C)'][i1]



def diff_t(i1, i2, df_metal):

    return df_metal['t(s)'][i2] - df_metal['t(s)'][i1]



def rate_Tt(i1, i2, df_metal):

    if i1 < 0 or i2 < 0 or i1 >= len(df_metal) or i2 >= len(df_metal):

        return 0

    

    return diff_T(i1, i2, df_metal)/diff_t(i1, i2, df_metal)



def rate_T(i, df_metal):

    return 0.5 * ( rate_Tt(i, i + 1, df_metal) + rate_Tt(i - 1, i, df_metal) )



def rate(df_metal):

    length = len(df_metal.index)

    list_rate_metal = [rate_T(i, df_metal) for i in df_metal.index]

    



    df_metal['Rate dT/dt'] = list_rate_metal



    # Smooth the noise

    df_metal['Rate with Savitzky–Golay filter'] = savgol_filter(df_metal['Rate dT/dt'], 61, 2, mode="nearest")



rate(df_copper)

rate(df_aluminium)



df_copper.plot(x='t(s)', y=['Rate dT/dt', 'Rate with Savitzky–Golay filter'], title='Vario vėsimo sparta')

df_aluminium.plot(x='t(s)', y=['Rate dT/dt', 'Rate with Savitzky–Golay filter'], title='Aliuminio vėsimo sparta')
def get_c_specific_from_known(T, df_known, df_not):

    if not T in df_known[df_known['T(C)'] == T].values:

        return 0

    

    # rates are t indexed

    c_known = df_known[df_known['T(C)'] == T]['c specific'].values[0]

    m_known = 1

    m_not = 1

    rate_known = df_known[df_known['T(C)'] == T]['Rate with Savitzky–Golay filter'].values[0]

    rate_not = df_not[df_not['T(C)'] == T]['Rate with Savitzky–Golay filter'].values[0]

    

    return c_known * m_known / m_not * rate_known / rate_not



#df_copper.iloc[:, 0][0: 50]

#get_c_specific_from_known(390.0, df_copper, df_aluminium)

df_aluminium['c specific'] = [get_c_specific_from_known(T, df_copper, df_aluminium) for T in df_aluminium['T(C)']]

df_aluminium
df_copper.plot(x='t(s)', y='T(K)', grid=True, title='Vario T(t)' )

df_aluminium.plot(x='t(s)', y='T(K)', grid=True, title='Aliuminio T(t)' )

df_copper.plot(x='t(s)', y='T(K)', grid=True, title='Vario ln(T(t))', logy=True )

df_copper.plot(x='T(K)', y='c specific', grid=True, title='Vario c(T)')

df_aluminium.plot(x='T(K)', y='c specific', grid=True, title='Aliuminio c(T)' )