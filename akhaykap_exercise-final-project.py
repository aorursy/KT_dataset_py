import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex7 import *

print("Setup Complete")
# Check for a dataset with a CSV file

step_1.check()
# Fill in the line below: Specify the path of the CSV file to read

my_filepath = '../input/graduate-admissions/Admission_Predict.csv'



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
# Create a plot

____ # Your code here

#plt.figure(figsize = (10,10))

plt.title("Chances of admit, general plot")

# plotting a KDE plot for Chance of admit , in general

step_4.check()

sns.kdeplot(data = my_data['Chance of Admit '], shade=True)
plt.title("Chances of admit, general plot")

step_4.check()

#plotting a histogram for Chance of admit , in general

sns.distplot(a = my_data['Chance of Admit '], kde=True)
# lets do a density plot for reasearch

plt.title("Research plot")

#plotting a histogram for Chance of admit , in general

sns.distplot(a = my_data['Research'], kde=False)
#lets check the university rating distribution , the best thing so far to use the density plots

plt.title("University Rating density distribution")

sns.distplot(a = my_data['University Rating'], kde=False)
#lets check the university rating distribution on KDE plots

plt.title("University Rating density distribution")

sns.kdeplot(data= my_data['University Rating'], shade = True)
#lets check the LOR distribution 

plt.title("LOR Density distribution")

sns.distplot(a = my_data['LOR '], kde=False)
#lets check the SOP distribution 

plt.title("SOP Density distribution")

sns.distplot(a = my_data['SOP'], kde=False)
#lets start with co-relations

# I believe students with good GRE score also perform good at TOEFL 

plt.title("TOEFL VS GRE  score")

sns.regplot(x=my_data['GRE Score'],y = my_data['TOEFL Score'])
# lets check relationship between GRE and CGPA

plt.title(" CGPA VS GRE score")

sns.regplot(x=my_data['GRE Score'],y = my_data['CGPA'])
#lets check do professors like high CGPA students and hence give better LORs

plt.title(" LOR VS CGPA")

#sns.regplot(x=my_data['CGPA'],y = my_data['LOR '])

sns.swarmplot(x=my_data['LOR '],y = my_data['CGPA'])
#lets see, do high GPA Students have better SOPs ?

plt.title(" CGPA vs SOP")

sns.swarmplot(x=my_data['SOP'],y = my_data['CGPA'])
#Lets check if high rated universities are difficult to get into ?

plt.title(" University rating vs Chances of Admit")

sns.swarmplot(x=my_data['University Rating'],y = my_data['Chance of Admit '])
plt.title(" University rating vs Chances of Admit")

sns.scatterplot(x=my_data['University Rating'],y = my_data['Chance of Admit '])
# lets check relationship between CGPA And Reaserch work

plt.title(" CGPA vs Research work")

sns.swarmplot(x=my_data['Research'],y = my_data['CGPA'])
#lets check relationship between research and admit chances

plt.title(" Chances of admit vs Research work")

sns.swarmplot(x=my_data['Research'],y = my_data['Chance of Admit '])

# time to make scatterplot with 3 variables

plt.figure(figsize =(20,60))

sns.lmplot(x='CGPA',y = 'Chance of Admit ', hue = 'SOP', data = my_data)
# time to make scatterplot with 3 variables

plt.figure(figsize =(20,60))

sns.lmplot(x='CGPA',y = 'Chance of Admit ', hue = 'Research', data = my_data)
# time to make scatterplot with 3 variables

plt.figure(figsize =(20,100))

sns.lmplot(x='CGPA',y = 'Chance of Admit ', hue = 'University Rating', data = my_data)