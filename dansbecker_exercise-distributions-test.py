!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git@master
import sys

sys.path.append('/kaggle/working')
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_easy.ex5 import *

print("Setup Complete")
# Paths of the files to read

cancer_b_filepath = "../input/cancer_b.csv"

cancer_m_filepath = "../input/cancer_m.csv"



# Fill in the line below to read the (benign) file into a variable cancer_b_data

cancer_b_data = ____



# Fill in the line below to read the (malignant) file into a variable cancer_m_data

cancer_m_data = ____



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the first five rows of the (benign) data

____ # Your code here
# Print the first five rows of the (malignant) data

____ # Your code here
# Fill in the line below: In the first five rows of the data for benign tumors, what is the

# largest value for 'Perimeter (mean)'?

max_perim = ____



# Fill in the line below: What is the value for 'Radius (mean)' for the tumor with Id 842517?

mean_radius = ____



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Histograms for benign and maligant tumors

____ # Your code here (benign tumors)

____ # Your code here (malignant tumors)



# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
#step_3.b.hint()
#step_3.b.solution()
# KDE plots for benign and malignant tumors

____ # Your code here (benign tumors)

____ # Your code here (malignant tumors)



# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()
#step_4.b.solution()