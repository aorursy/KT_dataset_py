import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

import os

if not os.path.exists("../input/cancer_b.csv"):

    os.symlink("../input/data-for-datavis/cancer_b.csv", "../input/cancer_b.csv")

    os.symlink("../input/data-for-datavis/cancer_m.csv", "../input/cancer_m.csv")

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex5 import *

print("Setup Complete")
# paths of the files to read

cancer_b_filepath = "../input/cancer_b.csv"

cancer_m_filepath = "../input/cancer_m.csv"



# read the (benign) file into a variable cancer_b_data

cancer_b_data = pd.read_csv(cancer_b_filepath, index_col="Id")



# read the (malignant) file into a variable cancer_m_data

cancer_m_data = pd.read_csv(cancer_m_filepath, index_col="Id")



# run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# lines below will give you a hint or solution code

# step_1.hint()

# step_1.solution()
# Print the first five rows of the (benign) data

cancer_b_data.head()
# Print the first five rows of the (malignant) data

cancer_m_data.head()
# in the first five rows of the data for benign (B) tumors, what is the largest value for 'Perimeter (mean)'?

max_perim = cancer_b_data['Perimeter (mean)'].iloc[:5].max()





# what is the value for 'Radius (mean)' for the tumor with Id 842517?

mean_radius = cancer_m_data['Radius (mean)'].loc[842517]



# check your answers

step_2.check()
# Lines below will give you a hint or solution code

# step_2.hint()

# step_2.solution()
# histograms for benign and maligant tumors

sns.distplot(a=cancer_b_data['Area (mean)'], label="Bengin", kde=False)

sns.distplot(a=cancer_m_data['Area (mean)'], label="Maligant", kde=False)

plt.legend()



# Check your answer

step_3.a.check()
# lines below will give you a hint or solution code

# step_3.a.hint()

# step_3.a.solution_plot()
# step_3.b.hint()
# check your answer (Run this code cell to receive credit!)

step_3.b.solution()
# KDE plots for benign and malignant tumors

sns.kdeplot(data=cancer_b_data['Radius (worst)'], label="Bengin", shade=True)

sns.kdeplot(data=cancer_m_data['Radius (worst)'], label="Maligant", shade=True)

plt.legend()



# check your answer

step_4.a.check()
# lines below will give you a hint or solution code

# step_4.a.hint()

# step_4.a.solution_plot()
# step_4.b.hint()
# check your answer (Run this code cell to receive credit!)

step_4.b.solution()