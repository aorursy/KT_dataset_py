import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
mult_choice = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv")
# Check if there are missing values in the column 'Q1'

mult_choice['Q1'].isnull().any()

# Group the data based on the responses for 'Q1' and count the responses in each group

age_group = mult_choice.groupby('Q1')['Q1'].count() #There are no missing values

# Remove the group 'What is your age (# years)?' after grouping

age_group.drop(index='What is your age (# years)?',inplace=True)
# Plot a bar graph

ax = age_group.plot(kind="bar", figsize=(10,8), color="green", fontsize=12);

ax.set_xlabel('Age group', fontsize=14)

totals = []



for i in ax.patches:

    totals.append(i.get_height())



total = sum(totals)



for i in ax.patches:

    # get_x pulls left or right; get_height pushes up or down

    ax.text(i.get_x() - .05, i.get_height()+.5, str(round((i.get_height()/total)*100, 2))+'%')
mult_choice['Q2'].isnull().any()

gender = mult_choice.groupby('Q2')['Q2'].count().sort_values(ascending=False)

gender.drop(index='What is your gender? - Selected Choice',inplace=True)
def my_autopct(pct):

    return ('%1.1f%%' % pct) if pct > 1 else ''



df = pd.DataFrame(gender, index=['Male', 'Female', 'Prefer not to say', 'Prefer to self-describe'])

df.plot.pie(subplots=True, colors=['c', 'y', 'b', 'g'], autopct = my_autopct, fontsize=14, figsize=(6, 6))
# Let's count the rows in 'Q2' whose value is 'Male' and set normalize=True to get the proportion

income_of_males = mult_choice['Q10'][mult_choice['Q2'] == 'Male'].value_counts(normalize=True)

# Let's reset the index to put the salary range in its proper order

income_of_males = income_of_males.reindex(index = ["$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

                                                           "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

                                                           "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

                                                           "300,000-500,000", "> $500,000"])

# We repeat for the females

income_of_females = mult_choice['Q10'][mult_choice['Q2'] == 'Female'].value_counts(normalize=True)

income_of_females = income_of_females.reindex(index = ["$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

                                                           "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

                                                           "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

                                                           "300,000-500,000", "> $500,000"])
# We plot double horizontal bar graph to ease the comparison

ax = plt.subplots(figsize=(10,8)) 

width = .4 

ind = np.arange(24)

plt.style.use('ggplot')

plt.barh(ind, income_of_males, width, color='blue', label='Male')

plt.barh(ind + width, income_of_females, width, color='red', label='Female')



plt.xlabel('Proportion')

plt.ylabel('Current Yearly Compensation')

plt.title('Income distribution by gender')



plt.yticks(ind + width, ("$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

            "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

            "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

            "300,000-500,000", "> $500,000"))

plt.legend()

plt.show()

mult_choice['Q3'].isnull().any()

residence_country = mult_choice.groupby('Q3')['Q3'].count().sort_values(ascending=True)

residence_country.drop(index=['In which country do you currently reside?'],inplace=True)
ax = residence_country.plot(kind="barh", figsize=(24,20), color="green", fontsize=16);

ax.set_ylabel("Residence country", fontsize=18)

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y(), str(round((i.get_width()/total)*100, 2))+'%', fontsize=16)
income_in_US = mult_choice['Q10'][mult_choice['Q3'] == 'United States of America'].value_counts(normalize=True)

income_in_US = income_in_US.reindex(index = ["$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

                                                           "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

                                                           "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

                                                           "300,000-500,000", "> $500,000"])

income_in_India = mult_choice['Q10'][mult_choice['Q3'] == 'India'].value_counts(normalize=True)

income_in_India = income_in_India.reindex(index = ["$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

                                                           "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

                                                           "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

                                                           "300,000-500,000", "> $500,000"])
ax = plt.subplots(figsize=(10,8)) 

width = .4 

ind = np.arange(24)

plt.style.use('ggplot')

plt.barh(ind, income_in_US, width, color='blue', label='US')

plt.barh(ind + width, income_in_India, width, color='red', label='India')



plt.xlabel('Proportion')

plt.ylabel('Current Yearly Compensation')

plt.title('Income distribution by country')



plt.yticks(ind + width, ("$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

            "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

            "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

            "300,000-500,000", "> $500,000"))

plt.legend()

plt.show()
mult_choice['Q4'].isnull().any()

edu_level = mult_choice.groupby('Q4')['Q4'].count().sort_values(ascending=True)

edu_level.drop(index=['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'],inplace=True)
ax = edu_level.plot(kind="barh", figsize=(20,10), color="green", fontsize=16);

ax.set_ylabel("Educational Level", fontsize=20)

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=16)
income_of_doctors = mult_choice['Q10'][mult_choice['Q4'] == "Doctoral degree"].value_counts(normalize=True)

income_of_masters = mult_choice['Q10'][mult_choice['Q4'] == "Master’s degree"].value_counts(normalize=True)

income_of_no_edu = mult_choice['Q10'][mult_choice['Q4'] == "No formal education past high school"].value_counts(normalize=True)
idx = ["$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

        "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

        "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

        "300,000-500,000", "> $500,000"]

income_of_doctors = income_of_doctors.reindex(index = idx)

income_of_masters = income_of_masters.reindex(index = idx)

income_of_no_edu = income_of_no_edu.reindex(index = idx)
ax = plt.subplots(figsize=(10,8)) 

width = .4 

ind = np.arange(24)

plt.style.use('ggplot')

plt.barh(ind, income_of_no_edu, width, color='blue', label='No education past high school')

plt.barh(ind + width, income_of_masters, width, color='red', label='Master\'s degree')

plt.barh(ind + 2*width, income_of_doctors, width, color='green', label='Doctoral degree')



plt.xlabel('Proportion')

plt.ylabel('Current Yearly Compensation')

plt.title('Income distribution by educational level')



plt.yticks(ind + width, ("$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

            "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

            "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

            "300,000-500,000", "> $500,000"))

plt.legend()

plt.show()
# Check if there are missing values

mult_choice['Q5'].isnull().any()

# Drop missing rows with missing values

job_filtered = mult_choice['Q5'].dropna()

# Check the size of the cleaned data

job_filtered.shape

job_title = mult_choice.groupby('Q5')['Q5'].count().sort_values(ascending=True)

job_title.drop(index=['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'],inplace=True)
ax = job_title.plot(kind="barh", figsize=(16,6), color="green", fontsize=14);

ax.set_ylabel("Job title", fontsize=16)

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .1, str(round((i.get_width()/total)*100, 2))+'%', fontsize=14)
income_DataScientist = mult_choice['Q10'][mult_choice['Q5'] == 'Data Scientist'].value_counts(normalize=True)

income_SoftwareEngineer = mult_choice['Q10'][mult_choice['Q5'] == 'Software Engineer'].value_counts(normalize=True)

income_DataAnalyst = mult_choice['Q10'][mult_choice['Q5'] == 'Data Analyst'].value_counts(normalize=True)
idx = ["$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

        "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

        "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

        "300,000-500,000", "> $500,000"]

income_DataScientist = income_DataScientist.reindex(index = idx)

income_SoftwareEngineer = income_SoftwareEngineer.reindex(index = idx)

income_DataAnalyst = income_DataAnalyst.reindex(index = idx)
ax = plt.subplots(figsize=(10,8)) 

width = .4 

ind = np.arange(24)

plt.style.use('ggplot')

plt.barh(ind, income_DataScientist, width, color='red', label='Data Scientist')

plt.barh(ind + width, income_SoftwareEngineer, width, color='blue', label='Software Engineer')

plt.barh(ind + 2*width, income_DataAnalyst, width, color='green', label='Data Analyst')



plt.xlabel('Proportion')

plt.ylabel('Current Yearly Compensation')

plt.title('Income distribution by job title')



plt.yticks(ind + width, ("$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

            "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

            "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

            "300,000-500,000", "> $500,000"))

plt.legend()

plt.show()
mult_choice['Q6'].isnull().any()

company_filtered = mult_choice['Q6'].dropna()

company_size = mult_choice.groupby('Q6')['Q6'].count()

company_size.drop(index=['What is the size of the company where you are employed?'],inplace=True)

company_size = company_size.reindex(index = ["0-49 employees", "50-249 employees", "250-999 employees", "1000-9,999 employees", "> 10,000 employees"])
ax = company_size.plot(kind="barh", figsize=(14,4), color="green", fontsize=12);

ax.set_ylabel("Employees Number", fontsize=12)

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12)
mult_choice['Q7'].isnull().any()

no_of_employees_filtered = mult_choice['Q7'].dropna()

no_of_employees = mult_choice.groupby('Q7')['Q7'].count()

no_of_employees.drop(index=['Approximately how many individuals are responsible for data science workloads at your place of business?'],inplace=True)

no_of_employees = no_of_employees.reindex(index = ["0", "1-2","3-4","5-9","10-14", "15-19", "20+"])
ax = no_of_employees.plot(kind="barh", figsize=(20,4), color="green", fontsize=14);

ax.set_ylabel("No of employees with Data Science roles", fontsize=14)

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=14)
mult_choice['Q8'].isnull().any()

ML_incorporated = mult_choice['Q8'].dropna()

ML_incorporated = mult_choice.groupby('Q8')['Q8'].count()

ML_incorporated.drop(index=['Does your current employer incorporate machine learning methods into their business?'],inplace=True)
ax = ML_incorporated.plot(kind="barh", figsize=(20,8), color="green", fontsize=18);

ax.set_ylabel("Is ML incorporated?", fontsize=20)

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=16)
# Merge the columns for the different categories of 'Q9' separating the responces by a semicolon delimiter

mult_choice['Q9'] = mult_choice[mult_choice.columns[11:20]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

# Split back the column into different columns using the delimiter  

job_roles = mult_choice['Q9'].str.split(';', expand = True)

# Remove the rows with value -1

job_roles = job_roles[job_roles != '-1']

# Use .stack() to slice the dataframe apart and stack the columns on top of one another

job_roles = job_roles.stack().value_counts().nlargest(8).sort_values(ascending=True)
ax = job_roles.plot(kind="barh", figsize=(24,14), color="green", fontsize=20);

ax.set_ylabel("Job roles", fontsize=24)

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=18)
mult_choice['Q10'].isnull().any()

yearly_compensation = mult_choice['Q10'].dropna()

yearly_compensation.shape

yearly_compensation = mult_choice.groupby('Q10')['Q10'].count()

yearly_compensation.drop(index=['What is your current yearly compensation (approximate $USD)?'],inplace=True)

yearly_compensation = yearly_compensation.reindex(index = ["$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

                                                           "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

                                                           "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

                                                           "300,000-500,000", "> $500,000"])
ax = yearly_compensation.plot(kind="barh", figsize=(24,14), color="green", fontsize=20);

ax.set_ylabel("Current yearly compensation", fontsize=20)

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=18)
mult_choice['Q11'].isnull().any()

money_spent = mult_choice['Q11'].dropna()

money_spent = mult_choice.groupby('Q11')['Q11'].count()

money_spent.drop(index=['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?'],inplace=True)

money_spent = money_spent.reindex(index = ["$0 (USD)", "$1-$99", "$100-$999", "$1000-$9,999", "$10,000-$99,999", "> $100,000 ($USD)"])
ax = money_spent.plot(kind="barh", figsize=(16,3), color="green", fontsize=12);

ax.set_ylabel("Money spent on ML and cloud computing", fontsize=10)

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12)
mult_choice['Q12'] = mult_choice[mult_choice.columns[22:35]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

favorite_media = mult_choice['Q12'].str.split(';', expand = True)

favorite_media = favorite_media[favorite_media != '-1']

favorite_media = favorite_media.stack().value_counts().nlargest(11).sort_values(ascending=True)
ax = favorite_media.plot(kind="barh", figsize=(24,12), color="green", fontsize=20);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=18)
mult_choice['Q13'] = mult_choice[mult_choice.columns[35:48]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

course_platform = mult_choice['Q13'].str.split(';', expand = True)

course_platform = course_platform[course_platform != '-1']

course_platform = course_platform.stack().value_counts().nlargest(12).sort_values(ascending=True)
ax = course_platform.plot(kind="barh", figsize=(24,14), color="green", fontsize=20);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=18)
income_coursera = mult_choice['Q10'][mult_choice['Q13_Part_2'] == 'Coursera'].value_counts(normalize=True)

income_Kaggle = mult_choice['Q10'][mult_choice['Q13_Part_6'] == 'Kaggle Courses (i.e. Kaggle Learn)'].value_counts(normalize=True)

income_Udemy = mult_choice['Q10'][mult_choice['Q13_Part_8'] == 'Udemy'].value_counts(normalize=True)

idx = ["$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

        "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

        "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

        "300,000-500,000", "> $500,000"]

income_coursera = income_coursera.reindex(index = idx)

income_Kaggle = income_Kaggle.reindex(index = idx)

income_Udemy = income_Udemy.reindex(index = idx)
ax = plt.subplots(figsize=(10,8)) 

width = .4 

ind = np.arange(24)

plt.style.use('ggplot')

plt.barh(ind, income_coursera, width, color='red', label='Coursera')

plt.barh(ind + width, income_Kaggle, width, color='blue', label='Kaggle')

plt.barh(ind + 2*width, income_Udemy, width, color='green', label='Udemy')



plt.xlabel('Proportion')

plt.ylabel('Current Yearly Compensation')

plt.title('Income distribution by platform on which course began or completed')



plt.yticks(ind + width, ("$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

            "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

            "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

            "300,000-500,000", "> $500,000"))

plt.legend()

plt.show()
mult_choice['Primary tool'] = mult_choice[mult_choice.columns[48:55]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

primary_tool = mult_choice['Primary tool'].str.split(';', expand = True)

# Remove rows that contain values -1, 1, 0 and 14

primary_tool = primary_tool[~((primary_tool != '-1') ^ (primary_tool != '1') ^ (primary_tool != '0') ^ (primary_tool != '14'))]

primary_tool = primary_tool.stack().value_counts().nlargest(6).sort_values(ascending=True)
ax = primary_tool.plot(kind="barh", figsize=(24,7), color="green", fontsize=20);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=18)
mult_choice['Q15'].isnull().any()

exprience_length = mult_choice['Q15'].dropna()

exprience_length.shape

exprience_length = mult_choice.groupby('Q15')['Q15'].count()

exprience_length.drop(index=['How long have you been writing code to analyze data (at work or at school)?'],inplace=True)

exprience_length = exprience_length.reindex(index = ["I have never written code", "< 1 years", "1-2 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"])
ax = exprience_length.plot(kind="barh", figsize=(16,4), color="green", fontsize=12);

ax.set_ylabel("Length of code writing experience", fontsize=12)

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12)
mult_choice['Q16'] = mult_choice[mult_choice.columns[56:69]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

ide_used = mult_choice['Q16'].str.split(';', expand = True)

ide_used = ide_used[ide_used != '-1']

ide_used = ide_used.stack().value_counts().nlargest(11).sort_values(ascending=True)
ax = ide_used.plot(kind="barh", figsize=(24,10), color="green", fontsize=20);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=18)
mult_choice['Q17'] = mult_choice[mult_choice.columns[69:82]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

notebook_used = mult_choice['Q17'].str.split(';', expand = True)

notebook_used = notebook_used[notebook_used != '-1']

notebook_used = notebook_used.stack().value_counts().nlargest(12).sort_values(ascending=True)
ax = notebook_used.plot(kind="barh", figsize=(24,12), color="green", fontsize=20);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=18)
mult_choice['Q18'] = mult_choice[mult_choice.columns[82:95]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

language_used = mult_choice['Q18'].str.split(';', expand = True)

language_used = language_used[language_used != '-1']

language_used = language_used.stack().value_counts().nlargest(11).sort_values(ascending=True)
ax = language_used.plot(kind="barh", figsize=(24,8), color="green", fontsize=16);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=16)
income_Python = mult_choice['Q10'][mult_choice['Q18_Part_1'] == 'Python'].value_counts(normalize=True)

income_SQL = mult_choice['Q10'][mult_choice['Q18_Part_3'] == 'SQL'].value_counts(normalize=True)

income_R = mult_choice['Q10'][mult_choice['Q18_Part_2'] == 'R'].value_counts(normalize=True)
idx = ["$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

        "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

        "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

        "300,000-500,000", "> $500,000"]

income_Python = income_Python.reindex(index = idx)

income_SQL = income_SQL.reindex(index = idx)

income_R = income_R.reindex(index = idx)
ax = plt.subplots(figsize=(10,8)) 

width = .4 

ind = np.arange(24)

plt.style.use('ggplot')

plt.barh(ind, income_Python, width, color='red', label='Python')

plt.barh(ind + width, income_SQL, width, color='blue', label='SQL')

plt.barh(ind + 2*width, income_R, width, color='green', label='R')



plt.xlabel('Proportion')

plt.ylabel('Current Yearly Compensation')

plt.title('Income distribution by programming language')



plt.yticks(ind + width, ("$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", 

            "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", 

            "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "250,000-299,999", 

            "300,000-500,000", "> $500,000"))

plt.legend()

plt.show()
mult_choice['Q19'].isnull().any()

lang_recommended = mult_choice['Q19'].dropna()

lang_recommended = mult_choice.groupby('Q19')['Q19'].count().sort_values(ascending=True)

lang_recommended.drop(index=['What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'],inplace=True)
ax = lang_recommended.plot(kind="barh", figsize=(16,6), color="green", fontsize=12);

ax.set_ylabel("Language recommended to learn first", fontsize=14)

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12)
mult_choice['Q20'] = mult_choice[mult_choice.columns[97:110]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

visualization_library = mult_choice['Q20'].str.split(';', expand = True)

visualization_library = visualization_library[visualization_library != '-1']

visualization_library = visualization_library.stack().value_counts().nlargest(12).sort_values(ascending=True)
ax = visualization_library.plot(kind="barh", figsize=(24,10), color="green", fontsize=20);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=18)
mult_choice['Q21'] = mult_choice[mult_choice.columns[110:116]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

hardware_used = mult_choice['Q21'].str.split(';', expand = True)

hardware_used = hardware_used[hardware_used != '-1']

hardware_used = hardware_used.stack().value_counts().nlargest(5).sort_values(ascending=True)
ax = hardware_used.plot(kind="barh", figsize=(12,4), color="green", fontsize=14);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=14)
mult_choice['Q22'].isnull().any()

tpu_usage = mult_choice['Q22'].dropna()

tpu_usage.shape

tpu_usage = mult_choice.groupby('Q22')['Q22'].count()

tpu_usage.drop(index=['Have you ever used a TPU (tensor processing unit)?'],inplace=True)

tpu_usage = tpu_usage.reindex(index = ["Never", "Once", "2-5 times", "6-24 times", "> 25 times"])
ax = tpu_usage.plot(kind="barh", figsize=(14,3), color="green", fontsize=12);

ax.set_ylabel("Frequency of TPU usage")

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12)
mult_choice['Q23'].isnull().any()

ML_usage = mult_choice['Q23'].dropna()

ML_usage = mult_choice.groupby('Q23')['Q23'].count()

ML_usage.drop(index=['For how many years have you used machine learning methods?'],inplace=True)

ML_usage = ML_usage.reindex(index = ["< 1 years", "1-2 years", "2-3 years", "3-4 years", "4-5 years", "5-10 years", "10-15 years", "20+ years"])
ax = ML_usage.plot(kind="barh", figsize=(12,5), color="green", fontsize=12);

ax.set_ylabel("Length of ML methods usage", fontsize=12)

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12)
mult_choice['Q24'] = mult_choice[mult_choice.columns[118:131]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

ML_algorithm = mult_choice['Q24'].str.split(';', expand = True)

ML_algorithm = ML_algorithm[ML_algorithm != '-1']

ML_algorithm = ML_algorithm.stack().value_counts().nlargest(12).sort_values(ascending=True)
ax = ML_algorithm.plot(kind="barh", figsize=(16,8), color="green", fontsize=14);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12)
mult_choice['Q25'] = mult_choice[mult_choice.columns[131:140]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

ML_tools = mult_choice['Q25'].str.split(';', expand = True)

ML_tools = ML_tools[ML_tools != '-1']

ML_tools = ML_tools.stack().value_counts().nlargest(8).sort_values(ascending=True)
ax = ML_tools.plot(kind="barh", figsize=(16,8), color="green", fontsize=14);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12)
mult_choice['Q26'] = mult_choice[mult_choice.columns[140:148]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

vision_methods = mult_choice['Q26'].str.split(';', expand = True)

vision_methods = vision_methods[vision_methods != '-1']

vision_methods = vision_methods.stack().value_counts().nlargest(7).sort_values(ascending=True)
ax = vision_methods.plot(kind="barh", figsize=(20,12), color="green", fontsize=24);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=20)
mult_choice['Q27'] = mult_choice[mult_choice.columns[148:155]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

NLP_method = mult_choice['Q27'].str.split(';', expand = True)

NLP_method = NLP_method[NLP_method != '-1']

NLP_method = NLP_method.stack().value_counts().nlargest(6).sort_values(ascending=True)
ax = NLP_method.plot(kind="barh", figsize=(16,6), color="green", fontsize=20);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=18)
mult_choice['Q28'] = mult_choice[mult_choice.columns[155:168]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

ML_frameworks = mult_choice['Q28'].str.split(';', expand = True)

ML_frameworks = ML_frameworks[ML_frameworks != '-1']

ML_frameworks = ML_frameworks.stack().value_counts().nlargest(12).sort_values(ascending=True)
ax = ML_frameworks.plot(kind="barh", figsize=(20,10), color="green", fontsize=16);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=16)
mult_choice['Q29'] = mult_choice[mult_choice.columns[168:181]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

cloud_computing = mult_choice['Q29'].str.split(';', expand = True)

cloud_computing = cloud_computing[cloud_computing != '-1']

cloud_computing = cloud_computing.stack().value_counts().nlargest(12).sort_values(ascending=True)
ax = cloud_computing.plot(kind="barh", figsize=(20,10), color="green", fontsize=16);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=16)
mult_choice['Q30'] = mult_choice[mult_choice.columns[181:194]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

cloud_product = mult_choice['Q30'].str.split(';', expand = True)

cloud_product = cloud_product[cloud_product != '-1']

cloud_product = cloud_product.stack().value_counts().nlargest(12).sort_values(ascending=True)
ax = cloud_product.plot(kind="barh", figsize=(20,10), color="green", fontsize=16);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=16)
mult_choice['Q31'] = mult_choice[mult_choice.columns[194:207]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

bigData_product = mult_choice['Q31'].str.split(';', expand = True)

bigData_product = bigData_product[bigData_product != '-1']

bigData_product = bigData_product.stack().value_counts().nlargest(12).sort_values(ascending=True)
ax = bigData_product.plot(kind="barh", figsize=(20,10), color="green", fontsize=16);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=16)
mult_choice['Q32'] = mult_choice[mult_choice.columns[207:220]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

ML_product = mult_choice['Q32'].str.split(';', expand = True)

ML_product = ML_product[ML_product != '-1']

ML_product = ML_product.stack().value_counts().nlargest(12).sort_values(ascending=True)
ax = ML_product.plot(kind="barh", figsize=(20,10), color="green", fontsize=16);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=16)
mult_choice['Q33'] = mult_choice[mult_choice.columns[220:233]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

autoML_tools = mult_choice['Q33'].str.split(';', expand = True)

autoML_tools = autoML_tools[autoML_tools != '-1']

autoML_tools = autoML_tools.stack().value_counts().nlargest(12).sort_values(ascending=True)
ax = autoML_tools.plot(kind="barh", figsize=(14,6), color="green", fontsize=12);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12)
mult_choice['Q34'] = mult_choice[mult_choice.columns[233:246]].apply(lambda x: ';'.join(x.dropna().astype(str)), axis=1)

Relational_DB = mult_choice['Q34'].str.split(';', expand = True)

Relational_DB = Relational_DB[Relational_DB != '-1']

Relational_DB = Relational_DB.stack().value_counts().nlargest(12).sort_values(ascending=True)
ax = Relational_DB.plot(kind="barh", figsize=(20,10), color="green", fontsize=16);

totals = []



for i in ax.patches:

    totals.append(i.get_width())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_width() + 1, i.get_y() + .2, str(round((i.get_width()/total)*100, 2))+'%', fontsize=16)