from subprocess import check_output

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/dataset.csv')

df['Orbital Period (days)'] = df['Orbital Period (days)'].astype('float64')
sns.set_style('ticks')

sns.lmplot(data=df, x='Orbit Semi-Major Axis (AU)', y='Orbital Period (days)',

          hue='Planet Radius (Jupiter radii)', palette='Blues',fit_reg=False)

plt.title('Distance vs Orbit')
sns.barplot(data=df,y='Planet Radius (Jupiter radii)', x='Letter', 

            palette='muted',ci=df['Planet Radius Upper Unc. (Jupiter radii)'])
sns.lmplot(data=df, x='Orbit Semi-Major Axis (AU)',  y='Planet Radius (Jupiter radii)',

           fit_reg=False)
sns.barplot(data=df,y='Planet Mass (Jupiter mass)', x='Letter', 

            palette='muted')
sns.barplot(data=df,y='Planet Density (g/cm**3)', x='Letter', 

            palette='muted',ci=df['Planet Density Upper Unc. (g/cm**3)'])
sns.pairplot(data=df, x_vars=['Orbit Semi-Major Axis (AU)'], y_vars=['Planet Mass (Jupiter mass)',

             'Planet Radius (Jupiter radii)','Planet Density (g/cm**3)'],hue='Letter',palette='muted')