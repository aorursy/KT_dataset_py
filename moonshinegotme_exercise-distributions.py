import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex5 import *

print("Setup Complete")
# Paths of the files to read

cancer_b_filepath = "../input/cancer_b.csv"

cancer_m_filepath = "../input/cancer_m.csv"



# Fill in the line below to read the (benign) file into a variable cancer_b_data

cancer_b_data = pd.read_csv(cancer_b_filepath, index_col = "Id")



# Fill in the line below to read the (malignant) file into a variable cancer_m_data

cancer_m_data = pd.read_csv(cancer_m_filepath, index_col = "Id")



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the first five rows of the (benign) data

cancer_b_data.shape # Your code here

cancer_b_data.head()
# Print the first five rows of the (malignant) data

cancer_m_data.head() # Your code here
# Fill in the line below: In the first five rows of the data for benign tumors, what is the

# largest value for 'Perimeter (mean)'?

max_perim = cancer_b_data.iloc[0:6, 3].max()

print(max_perim)



# Fill in the line below: What is the value for 'Radius (mean)' for the tumor with Id 842517?

mean_radius = cancer_m_data.iloc[1, 1]

print(mean_radius)



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Histograms for benign and maligant tumors

plt.figure(figsize = (14, 8)) # Your code here (benign tumors)

sns.distplot(a = cancer_b_data['Area (mean)'], kde = False, label = 'Benign tumors') # Your code here (malignant tumors)

sns.distplot(a = cancer_m_data['Area (mean)'], kde = False, label = 'Malignant tumors')

plt.legend()



# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
#step_3.b.hint()
#step_3.b.solution()
# KDE plots for benign and malignant tumors

plt.figure(figsize = (14,8))

sns.kdeplot(data = cancer_b_data['Radius (worst)'], label = 'Benign', shade = True) # Your code here (benign tumors)

sns.kdeplot(data = cancer_m_data['Radius (worst)'], label = 'Malignant', shade = True) # Your code here (malignant tumors)



plt.xlabel('Radius (worst)', fontsize = 14)

plt.ylabel('KDE', fontsize = 14)

plt.legend()



# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()
step_4.b.solution()
plt.figure(figsize = (14, 8))

graph = sns.jointplot(x = cancer_b_data['Radius (mean)'], y = cancer_b_data['Texture (mean)'])

plt.suptitle("Malignant and benign tumors", fontsize = 14)





"""plt.figure(figsize = (14, 8))

sns.jointplot(x = cancer_m_data['Area (mean)'], y = cancer_m_data['Smoothness (mean)'], kind = 'kde')

plt.suptitle("Malignant tumors", fontsize = 24)"""



graph.x = cancer_m_data['Radius (mean)']

graph.y = cancer_m_data['Texture (mean)']

graph.plot_joint(sns.kdeplot)