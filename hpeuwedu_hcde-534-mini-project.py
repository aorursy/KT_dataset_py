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


import matplotlib.pyplot as plt

import pandas as pd


#These are variables for my data frames. college_data is from the US data.gov 

#WAEnrollment is data from urban institute which gathers it's data from US department of education



college_data = pd.read_csv("/kaggle/input/college-enrollment/college-enrollment-rates-2016.csv", index_col = "Year")



WAEnrollment = pd.read_csv("/kaggle/input/wa4yr2yrenrollment/WAInstitutionEnrollment.csv")

#I want to check what college_data df looks like. 

college_data
#Checking what WAEnrollment looks like as well.

WAEnrollment
#creating variables and condition for removing rows that have "graduate" and "First professional" in it. 

grad_index = WAEnrollment[WAEnrollment["level_of_study"] == "Graduate"].index

first_prof = WAEnrollment[WAEnrollment["level_of_study"] == "First professional"].index                          

                          

#this i deleting rows that have "Graduate" & First professional in it because I only want to look at undergrad enrollment

WAEnrollment.drop(grad_index, inplace=True)

WAEnrollment.drop(first_prof, inplace=True)



#now dataframe is only showing undergradute enrollment 

WAEnrollment
#creating a new data frame that will output only the year, institution name, level of study, and fall enrollment

#this will make it easier for me to review only year, institution name, level of study, and fall enrollment

WAEnrollment[["year","inst_name", "enrollment_fall", "level_of_study"]]
#new data frame variable to make it easier to create vizuals

new_WAEnrollment_df = WAEnrollment[["year","inst_name", "enrollment_fall", "level_of_study"]]



#plotting a graph of fall_enrollment based on year which will show me what enrollment looked like each year in WA

#I'm not sure why the output is two separate graphs. 

ax.set_xlabel("Year")

ax.set_ylabel("Enrollment Count")

new_WAEnrollment_df.hist(bins = 30,)
#creating a pivot table that tells what the overall percentage is of students enrolled based on race/ethnicity 

college_data.pivot_table(columns = "Race/ethnicity", values ="Percentage" )



#plotting a graph to show the percentage of students 

college_data.plot()
Two_or_more = college_data["Race/ethnicity"] == "Two or more races, non-Hispanic"

Two_or_more.sum()