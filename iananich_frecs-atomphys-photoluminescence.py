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
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
U = 5 # V

T = 1620 # C

lambda_max = 2339 # A

I_max = 60



phi_max = 1 / lambda_max * 1e10 # Hz



df = pd.DataFrame({

    'd': [2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700],

    '\u03bb, \u00c5': [5170, 5240, 5300, 5370, 5440, 5520, 5590, 5675, 5750, 5850, 5975, 6090, 6190, 6315, 6470],

    'I': [1, 2, 3, 9, 25, 46, 59, 59, 55, 52, 46, 37, 28, 23, 18],

    'I (підс.)': [170, 190, 210, 240, 270, 300, 330, 370, 400, 440, 490, 530, 510, 590, 610],

    'I (барв.)': [20, 20, 30, 40, 60, 120, 200, 250, 310, 360, 410, 440, 470, 490, 500],

    'I (дист. вода)': [100, 110, 120, 140, 150, 170, 190, 210, 220, 240, 260, 280, 290, 300, 300],

})
def normalize(arr):

    mx = arr.max()

    return arr / mx



def planck_func(l):

    e = 2.71828

    z = l / (2897.8 / (T + 273.15))

    return (e**4.965 - 1) * (z**-5) / (e**(4.965 / z) - 1)
df['\u03c6'] = [planck_func(l*1e-4) for l in df['\u03bb, \u00c5']]



df
plt.plot(df['\u03bb, \u00c5'], df['\u03c6'], '--', label='інтенсивність випромінювання джерела')

plt.grid()

plt.xlabel('\u03bb, \u00c5')

#plt.ylabel('I (дж.)')

#plt.legend()

plt.title('Інтенсивність випромінювання джерела')

plt.savefig('lab6_planck.png', dpi=300)
plt.plot(df['\u03bb, \u00c5'], normalize(df['I']), '-o', label='спектр люмінісценції')

plt.grid()

plt.xlabel('\u03bb, \u00c5')

plt.ylabel('I')

#plt.legend()

plt.title('Дослідження спектру люмінісценції')

plt.savefig('lab6_lumen.png', dpi=300)
df['sensativity'] = df['I (підс.)'] / df['\u03c6']



plt.plot(df['\u03bb, \u00c5'], normalize(df['sensativity']), '-o', label='спектральная чутливість ФЕП')

plt.grid()

plt.xlabel('\u03bb, \u00c5')

plt.ylabel('R(\u03c6)')

#plt.legend()

plt.title('спектральная чутливість ФЕП')

plt.savefig('lab6_sens.png', dpi=300)
df['I (погл.)'] = df['I (дист. вода)'] / df['I (барв.)']



#plt.plot(df['\u03bb, \u00c5'], df['I (барв.)'], '-o', label='барвник')

#plt.plot(df['\u03bb, \u00c5'], df['I (дист. вода)'], '-o', label='дистильована вода')

#plt.plot(df['\u03bb, \u00c5'], normalize(df['I (дист. вода)'] - df['I (барв.)']), '-o', label='1')

plt.plot(df['\u03bb, \u00c5'], normalize(df['I (погл.)'] / df['\u03c6']), '-o', label='2')

#plt.plot(df['\u03bb, \u00c5'], normalize(() / df['sensativity']), '-o', label='3')

plt.grid()

plt.xlabel('\u03bb, \u00c5')

plt.ylabel('I')

#plt.legend()

plt.title('Дослідження спектру поглинання Родаміна')

plt.savefig('lab6_rodam.png', dpi=300)
def levshin(df):

    phi = 1/df['\u03bb, \u00c5'] * 1e4

    phi_max = phi.max()

    

    

    absorb = (np.log(df['I (погл.)']) * phi_max) / (np.log(df['I (погл.)'].max()) * phi)

    lumen = (df['I'] * (phi_max**4)) / (df['I'].max() * (phi**4))

    return phi, absorb, lumen



x, absorb, lumen = levshin(df)

             

plt.plot(x, normalize(absorb), '--', label='поглиннання')

plt.plot(x, normalize(lumen), '--', label='люмінісценція')

plt.ylim(0, 1)

plt.grid()

plt.xlabel('\u03bd, МГц')

plt.ylabel('I')

plt.legend()

plt.title('Дослідження правила Левшина')

plt.savefig('lab6_levshi.png', dpi=300)