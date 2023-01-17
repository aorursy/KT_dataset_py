import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import warnings

warnings.filterwarnings("ignore")



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

fig0 = plt.figure()

fig0.add_subplot(1,2,1)

train_df['spacegroup'].value_counts().plot(kind='bar')

fig0.add_subplot(1,2,2)

test_df['spacegroup'].value_counts().plot(kind='bar',color='r')
fig1 = plt.figure()

fig1.add_subplot(1,2,1)

train_df['number_of_total_atoms'].value_counts().plot(kind='bar')

fig1.add_subplot(1,2,2)

test_df['number_of_total_atoms'].value_counts().plot(kind='bar',color='r')
train_df[train_df['spacegroup']==227]['lattice_angle_gamma_degree'].describe()


fig2 = plt.figure()

fig2.add_subplot(2,3,1)

train_df[train_df['spacegroup']==12]['lattice_angle_alpha_degree'].hist(bins= 40)

fig2.add_subplot(2,3,2)

train_df[train_df['spacegroup']==33]['lattice_angle_alpha_degree'].hist(bins= 40)

fig2.add_subplot(2,3,3)

train_df[train_df['spacegroup']==167]['lattice_angle_alpha_degree'].hist(bins= 40)

fig2.add_subplot(2,3,4)

train_df[train_df['spacegroup']==194]['lattice_angle_alpha_degree'].hist(bins= 40)

fig2.add_subplot(2,3,5)

train_df[train_df['spacegroup']==206]['lattice_angle_alpha_degree'].hist(bins= 40)

fig2.add_subplot(2,3,6)

train_df[train_df['spacegroup']==227]['lattice_angle_alpha_degree'].hist(bins= 40,color= 'r')
fig3 = plt.figure()

fig3.add_subplot(2,3,1)

train_df[train_df['spacegroup']==12]['lattice_angle_beta_degree'].hist(bins= 40)

fig3.add_subplot(2,3,2)

train_df[train_df['spacegroup']==33]['lattice_angle_beta_degree'].hist(bins= 40)

fig3.add_subplot(2,3,3)

train_df[train_df['spacegroup']==167]['lattice_angle_beta_degree'].hist(bins= 40)

fig3.add_subplot(2,3,4)

train_df[train_df['spacegroup']==194]['lattice_angle_beta_degree'].hist(bins= 40)

fig3.add_subplot(2,3,5)

train_df[train_df['spacegroup']==206]['lattice_angle_beta_degree'].hist(bins= 40)

fig3.add_subplot(2,3,6)

train_df[train_df['spacegroup']==227]['lattice_angle_beta_degree'].hist(bins= 40,color = 'r')
fig4 = plt.figure()

fig4.add_subplot(2,3,1)

train_df[train_df['spacegroup']==12]['lattice_angle_gamma_degree'].hist(bins= 40)

fig4.add_subplot(2,3,2)

train_df[train_df['spacegroup']==33]['lattice_angle_gamma_degree'].hist(bins= 40)

fig4.add_subplot(2,3,3)

train_df[train_df['spacegroup']==167]['lattice_angle_gamma_degree'].hist(bins= 40)

fig4.add_subplot(2,3,4)

train_df[train_df['spacegroup']==194]['lattice_angle_gamma_degree'].hist(bins= 40)

fig4.add_subplot(2,3,5)

train_df[train_df['spacegroup']==206]['lattice_angle_gamma_degree'].hist(bins= 40)

fig4.add_subplot(2,3,6)

train_df[train_df['spacegroup']==227]['lattice_angle_gamma_degree'].hist(bins= 40,color='r')
fig5 = plt.figure()

fig5.add_subplot(2,3,1)

train_df[train_df['spacegroup']==227][train_df['number_of_total_atoms']==80]['lattice_angle_gamma_degree'].hist(bins=20)

fig5.add_subplot(2,3,2)

train_df[train_df['spacegroup']==227][train_df['number_of_total_atoms']==60]['lattice_angle_gamma_degree'].hist(bins=20)

fig5.add_subplot(2,3,3)

train_df[train_df['spacegroup']==227][train_df['number_of_total_atoms']==40]['lattice_angle_gamma_degree'].hist(bins=20)

fig5.add_subplot(2,3,4)

train_df[train_df['spacegroup']==227][train_df['number_of_total_atoms']==30]['lattice_angle_gamma_degree'].hist(bins=20)

fig5.add_subplot(2,3,5)

train_df[train_df['spacegroup']==227][train_df['number_of_total_atoms']==20]['lattice_angle_gamma_degree'].hist(bins=20)

fig5.add_subplot(2,3,6)

train_df[train_df['spacegroup']==227][train_df['number_of_total_atoms']==10]['lattice_angle_gamma_degree'].hist(bins=20)