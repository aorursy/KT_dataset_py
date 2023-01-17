#Introduction - The dataset used refers to presence of heart disease in patients. This dataset belongs to Cleveland city of Ohio. 

#The dataset is interesting to me because it gives me information about patients age, gender, maximum heart rate achieved, etc.

#I want to explore the trend of heart disease in male, female and how it is distribute along age. 





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as mplot





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


heartdisease = pd.read_csv("../input/heartdisease.csv")

heartdisease.head(5) #showing first five rows of dataframe
heartdisease.age.value_counts()
#creating a barplot of age and 

sb.barplot(x=heartdisease.age.value_counts().index,y=heartdisease.age.value_counts().values)

mplot.xlabel('Age')

mplot.ylabel('Age Counts')

mplot.title('Counts of age affected by heart disease')

mplot.show()
def age_count(Age):

    """

    Defines brackets for age groups:

    Age: age of the patient

    output: the age bracket Young, Middle or Old

    """

    if Age<=40:

        return "Young"

    elif Age>40 and Age<=60:

        return "Middle"

    elif Age>60:

        return "Old"







# creating a pie chart to analyze young, medium and old age group people affected by heart disease.



colors=['mediumseagreen','blue','red']

heartdisease['age'].apply(age_count).value_counts().plot(kind='pie', colors=colors) 



# creating a bar chart to which age group (young, medium and old) is more affected by heart disease.



sb.barplot(x=['young ages','middle ages','old ages'],y=heartdisease['age'].apply(age_count).value_counts())

mplot.xlabel('Age Range')

mplot.ylabel('Age Counts')

mplot.title('Age group affected by heart disease')

mplot.show()
# creating a bar chart to which age group (young, medium and old) is more affected by heart disease.

# creating the same graph using different approach

heartdisease['age'].apply(age_count).value_counts().plot(kind='bar', colors=colors)
heartdisease[["sex", "thalach"]] # subset of dataframe (sex and thalach: maximum heart rate achieved)
male_rate.describe() 
#creating male_rate and female_rate series for creating seperate dataframe both for male and female

#The aim is to explore maximum heart rate achieved trend in both male and female.



male_rate = heartdisease[(heartdisease['sex'] == 1)].thalach

female_rate = heartdisease[(heartdisease['sex'] == 0)].thalach
#converting series to dataframe



male_rate["thalach"]= heartdisease[(heartdisease['sex'] == 1)].thalach

female_rate["thalach"] = heartdisease[(heartdisease['sex'] == 0)].thalach
#Ploting trend of maximum heart rate achieved 

heartdisease['thalach'].plot()
#Ploting trend of maximum heart rate achieved for both male and female

male_rate["thalach"].plot()

female_rate["thalach"].plot()
#scatter plot of maximum heart rate achieved for both male and female.

groups = heartdisease.groupby('sex')

x = heartdisease.thalach

# Plot

fig, ax = mplot.subplots()

ax.margins(0.05)

for name, group in groups:

    ax.plot(group.sex, group.thalach, marker='o', linestyle='', ms=5, label=name)

ax.legend()



mplot.show()