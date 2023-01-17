import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # to plot linear regression

from scipy import stats # to get linear regression values



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_viscosity = pd.read_csv("/kaggle/input/liquidviscosity/lab22_d07.dat", delim_whitespace=True, names=['T(C)', 't(s)'] )



data_reference = [

    [0, 0.9998, 1.781],

    [5, 1.0000, 1.518],

    [10, 0.9997, 1.307],

    [15, 0.9991, 1.139],

    [20, 0.9982, 1.002],

    [25, 0.9970, 0.890],

    [30, 0.9957, 0.798],

    [35, 0.9940, 0.719],

    [40, 0.9922, 0.653],

    [45, 0.9902, 0.595],

    [50, 0.9880, 0.547],

    [60, 0.9832, 0.466],

    [70, 0.9778, 0.404],

    [80, 0.9718, 0.354],

    [90, 0.9653, 0.315],

    [100, 0.9584, 0.282]

]



df_reference = pd.DataFrame(data_reference, columns=['T(C)', 'rho', 'n'])





time_mean_20 = df_viscosity[df_viscosity['T(C)'] == 20]['t(s)'].mean()

df_viscosity[df_viscosity['T(C)'] == 20]

time_mean_20

rho_r = 2221 #kg/m3



K = (df_reference[df_reference['T(C)'] == 20]['n'].values[0]) / (time_mean_20 * (2221 - 998.2) )



def viscosity_from_i(df_viscosity, i, rho_changing):

    T = df_viscosity['T(C)'][i]

    t = df_viscosity['t(s)'][i]

    

    rho_ball =  2221

    rho_liquid = df_reference[df_reference['T(C)'] == T]['rho'].values[0] if rho_changing else 1

    rho_liquid *= 1000

    

    return K * t * (rho_ball - rho_liquid)

    

df_viscosity['η'] = [viscosity_from_i(df_viscosity, i, False) for i in df_viscosity.index]

df_viscosity['ln(η)'] = [np.log(n) for n in df_viscosity['η']]

df_viscosity['η (temperature dependent ρ)'] = [viscosity_from_i(df_viscosity, i, True) for i in df_viscosity.index]

df_viscosity['ln(η (temperature dependent ρ))'] = [np.log(n) for n in df_viscosity['η (temperature dependent ρ)']]

df_viscosity['1/T'] = [1 / (T + 273) for T in df_viscosity['T(C)']]



#df_viscosity.plot.scatter(x='1/T', y='η', logy=True )

#df_viscosity.plot.scatter(x='1/T', y=['η'], logy=True )

#df_viscosity.plot(x='1/T', y='η (temperature dependent ρ)', logy=True )



ax1 = df_viscosity.plot(kind='scatter', x='1/T', y='ln(η)', color='r', label='η')    

ax2 = df_viscosity.plot(kind='scatter', x='1/T', y='ln(η (temperature dependent ρ))', label='η (nuo temperatūros priklausantis ρ)', color='g', ax=ax1)    

ax1.set_xlabel("1/T (1/K)")

ax1.set_ylabel("ln(η)")





coef = np.polyfit(df_viscosity['1/T'], df_viscosity['ln(η (temperature dependent ρ))'], 1)

poly1d_fn = np.poly1d(coef)

ax1.plot(df_viscosity['1/T'], poly1d_fn(df_viscosity['1/T']), '-b')

print(coef, np.exp(coef[1]), coef[0])



a1 = sns.lmplot(x='1/T', y='ln(η (temperature dependent ρ))', data=df_viscosity, markers=["X"], fit_reg=True)

ax_final1 = df_viscosity.plot(kind='scatter', x='T(C)', y='η (temperature dependent ρ)', marker='x', label='Teorinė hipotezė')    

#ax2 = df_viscosity.plot(kind='scatter', x='T(C)', y='η (temperature dependent ρ)', label='η (nuo temperatūros priklausantis ρ)', color='g', ax=ax1)    

#ax1.set_xlabel("1/T (1/K)")

#ax1.set_ylabel("ln(η)")





ax_final2 = df_reference[5:11].plot(kind='scatter', x='T(C)', y='n', marker='o', color='Red', label='η pamatuotas', ax=ax_final1)

df_viscosity['ns'] = [ np.exp(coef[1]) * np.exp( coef[0]/(T + 273) )  for T in df_viscosity['T(C)']]

ax_final3 = df_viscosity.plot(x='T(C)', y='ns', label='η žinyno', color='DarkGreen', ax=ax_final2)

ax_final1.set_xlabel("t (C)")

ax_final1.set_ylabel("η (Pa * s)")

#df_viscosity