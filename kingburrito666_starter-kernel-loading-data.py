import pandas as pd # to dataframes

import matplotlib.pyplot as plt #to define plot parameters

import seaborn as sns #to graphical plots

import numpy as np #to math 



plt.style.use('ggplot') # to plot graphs with gggplot2 style



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

projects_cooking = pd.read_csv('/kaggle/input/instructables-diy-all-projects/projects_cooking.csv')

projects_craft = pd.read_csv('/kaggle/input/instructables-diy-all-projects/projects_craft.csv')

projects_workshop = pd.read_csv('/kaggle/input/instructables-diy-all-projects/projects_workshop.csv')

projects_living = pd.read_csv('/kaggle/input/instructables-diy-all-projects/projects_living.csv')

projects_outside = pd.read_csv('/kaggle/input/instructables-diy-all-projects/projects_outside.csv')

projects_circuits = pd.read_csv('/kaggle/input/instructables-diy-all-projects/projects_circuits.csv')

tupple_array = [(projects_cooking,"Projects -- Cooking"),

                  (projects_craft, "Projects -- Craft"),

                  (projects_workshop,"Projects -- Workshop"),

                  (projects_living, "Projects -- Living"),

                  (projects_outside, "Projects -- Outside"),

                  (projects_circuits, "Projects -- Circuits")]

for entry in tupple_array:

    print(entry[1], entry[0].describe(), "\n\n")