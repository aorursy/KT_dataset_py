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
# Paths of the files to read

cancer_b_filepath = "../input/cancer_b.csv"

cancer_m_filepath = "../input/cancer_m.csv"



# Fill in the line below to read the (benign) file into a variable cancer_b_data

cancer_b_data = pd.read_csv(cancer_b_filepath,index_col="Id")



# Fill in the line below to read the (malignant) file into a variable cancer_m_data

cancer_m_data = pd.read_csv(cancer_m_filepath,index_col="Id")



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
cancer_b_data.head()
# Print the first five rows of the (malignant) data

cancer_m_data.head()
# Fill in the line below: In the first five rows of the data for benign tumors, what is the

# largest value for 'Perimeter (mean)'?

max_perim = 87.46



# Fill in the line below: What is the value for 'Radius (mean)' for the tumor with Id 842517?

mean_radius = 20.57



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Histoams for benign and maligant tumors

sns.distplot(a=cancer_b_data['Area (mean)'],kde=False) # Your code here (benign tumors)

sns.distplot(a=cancer_m_data['Area (mean)'],kde=False) # Your code here (malignant tumors)



# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
#step_3.b.hint()
# Check your answer (Run this code cell to receive credit!)

step_3.b.solution()
# KDE plots for benign and malignant tumors

sns.kdeplot(data=cancer_b_data['Radius (worst)'],label="benign",shade=True) # Your code here (benign tumors)

sns.kdeplot(data=cancer_m_data['Radius (worst)'],label="malignant",shade=True) # Your code here (malignant tumors)





# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()
# Check your answer (Run this code cell to receive credit!)

step_4.b.solution()