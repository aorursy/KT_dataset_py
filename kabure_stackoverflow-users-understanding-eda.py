import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_stack = pd.read_csv('../input/survey_results_public.csv', low_memory=False)
nulls = df_stack.isnull().sum() / len(df_stack) * 100

print("Percentual of Null Values: ")
print(nulls.sort_values(ascending=False))
df_stack.nunique()
plt.figure(figsize = (16,6))

plt.subplot(1,2,1)
g = sns.distplot(np.log(df_stack['ConvertedSalary'].dropna() + 1))
g.set_xlabel('Converted Salary Declared', fontsize=15)
g.set_title("Histogram of Converted Salary", fontsize=19)

plt.subplot(1,2,2)
plt.scatter(range(df_stack.shape[0]), np.sort(df_stack['ConvertedSalary'].values))
plt.xlabel('Range of dataset', fontsize=15)
plt.title("Distribution of Converted Salary", fontsize=19)

plt.show()
# calculating summary statistics
data_mean, data_std = np.mean(df_stack['ConvertedSalary']), np.std(df_stack['ConvertedSalary'])

# identify outliers
cut = data_std * 3
lower, upper = data_mean - cut, data_mean + cut

# identify outliers
outliers = [x for x in df_stack['ConvertedSalary'] if x < lower or x > upper]

# remove outliers
outliers_removed = [x for x in df_stack['ConvertedSalary'] if x > lower and x < upper]
print('Identified outliers: %d' % len(outliers))
print('Non-outlier observations: %d' % len(outliers_removed))
print("Percentiles of Amount: ")
print(df_stack['ConvertedSalary'].quantile([.05,.25,.5,.75,.95]))
df_stack['ConvertedSalary_log'] = np.log(df_stack['ConvertedSalary']+1)
plt.figure(figsize=(18,10))

plt.subplot(2,2,1)
g = sns.countplot(x='Hobby', data=df_stack)
g.set_title("Code as Hobby Distribuition", fontsize=20)
g.set_xlabel("Code as Hobby?", fontsize=16)
g.set_ylabel("Count", fontsize=16)

plt.subplot(2,2,2)
g = sns.boxplot(x='Hobby', y='ConvertedSalary_log',data=df_stack)
g.set_title("Hobby Coder Converted Salary Distribuition", fontsize=20)
g.set_xlabel("Code as Hobby?", fontsize=16)
g.set_ylabel("Converted Salary (Log)", fontsize=16)

plt.subplot(2,2,3)
g1 = sns.countplot(x='OpenSource', data=df_stack)
g1.set_title("Open Source Contributor Distribuition", fontsize=20)
g1.set_xlabel("Is Open Source Contribuitor?", fontsize=16)
g1.set_ylabel("Count", fontsize=16)

plt.subplot(2,2,4)
g1 = sns.boxplot(x='OpenSource', y='ConvertedSalary_log', data=df_stack)
g1.set_title("Open Source Contributor Conv-Salary Distribuition", fontsize=20)
g1.set_xlabel("Is Open Source Contribuitor?", fontsize=16)
g1.set_ylabel("Converted Salary (Log)", fontsize=16)

plt.subplots_adjust(hspace = 0.5, wspace = 0.1, top = 0.8)

plt.show()

print("Percentual of Code as a Hobby:")
print(round(df_stack['Hobby'].value_counts() / len(df_stack['Hobby']) *100, 2))
print("")
print("Percentual of Contribuition to OpenSource:")
print(round(df_stack['OpenSource'].value_counts() / len(df_stack['OpenSource']) *100, 2))
plt.figure(figsize=(8,5))

g = sns.countplot(x='Hobby', data=df_stack, hue='OpenSource')
g.set_title("Hobby Coder by Open Source Contributor", fontsize=20)
g.set_xlabel("Coder as a Hobby?", fontsize=16)
g.set_ylabel("Count", fontsize=16)

plt.show()

print("Open Source Contributor x Hobby: ")
print("")
print(round(pd.crosstab(df_stack['OpenSource'], df_stack['Hobby'], normalize='columns') * 100, 2))
countrys = df_stack['Country'].value_counts()
print("Description percentual of Countrys")
print(countrys[:8] / len(df_stack['Country']) * 100)

plt.figure(figsize=(12,10))

plt.subplot(2,1,1)
g = sns.countplot(x='Country', 
                  data=df_stack[df_stack.Country.isin(countrys[:15].index.values)], 
                  order=countrys[:15].index.values)
g.set_title("TOP 15 Countrys of the Survey Respondets", fontsize=20)
g.set_xlabel("", fontsize=16)
g.set_ylabel("Count", fontsize=16)
g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.subplot(2,1,2)
g1 = sns.boxplot(x='Country', y='ConvertedSalary_log',
                  data=df_stack[df_stack.Country.isin(countrys[:15].index.values)], 
                  order=countrys[:15].index.values)
g1.set_title("TOP 15 Countrys of the Survey Respondets", fontsize=20)
g1.set_xlabel("Country Name", fontsize=16)
g1.set_ylabel("Count", fontsize=16)
g1.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.subplots_adjust(hspace = 0.8, top = 0.8)

plt.show()

plt.figure(figsize=(12,5))

g = sns.countplot(x='Country', 
                  data=df_stack[df_stack.Country.isin(countrys[:15].index.values)], 
                  hue='OpenSource', order=countrys[:15].index.values)
g.set_title("TOP 15 Countrys by Open Source Contributors", fontsize=20)
g.set_xlabel("Country Names", fontsize=16)
g.set_ylabel("Count Distribuition", fontsize=16)
g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.show()
df_stack.groupby(["Country",'OpenSource'])['OpenSource'].count().nlargest(8) / len(df_stack['Country']) * 100
df_stack.loc[df_stack['FormalEducation'] == 'Bachelor’s degree (BA, BS, B.Eng., etc.)', 'FormalEducation'] = 'Bachelor’s degree'
df_stack.loc[df_stack['FormalEducation'] == 'Master’s degree (MA, MS, M.Eng., MBA, etc.)', 'FormalEducation'] = "Master's Degree"
df_stack.loc[df_stack['FormalEducation'] == 'Some college/university study without earning a degree', 'FormalEducation'] = 'College/Univer without degree'
df_stack.loc[df_stack['FormalEducation'] == 'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)', 'FormalEducation'] = 'Secondary School'
df_stack.loc[df_stack['FormalEducation'] == 'Other doctoral degree (Ph.D, Ed.D., etc.)', 'FormalEducation'] = 'Doctoral Degree' 
df_stack.loc[df_stack['FormalEducation'] == 'Professional degree (JD, MD, etc.)', 'FormalEducation'] = 'Professional Degree'
df_stack.loc[df_stack['FormalEducation'] == 'I never completed any formal education', 'FormalEducation'] = "Never completed Formal Educ"

df_stack.loc[df_stack['Employment'] == 'Independent contractor, freelancer, or self-employed', 'Employment'] = "Independent Emp"
df_stack.loc[df_stack['Employment'] == 'Not employed, but looking for work', 'Employment'] = "NotWork, LookingFor"
df_stack.loc[df_stack['Employment'] == 'Not employed, and not looking for work', 'Employment'] = "NotWork, NotLooking"

student = df_stack['Student'].value_counts() / len(df_stack['Student']) * 100
formal_educ = df_stack['FormalEducation'].value_counts() / len(df_stack['FormalEducation']) * 100
employment = df_stack['Employment'].value_counts() / len(df_stack['Employment']) * 100
years_cod_prof = df_stack['YearsCodingProf'].value_counts() / len(df_stack['YearsCodingProf']) * 100
years_coding = df_stack['YearsCoding'].value_counts() / len(df_stack['YearsCoding']) * 100
ad_block = df_stack['AdBlocker'].value_counts() / len(df_stack['AdBlocker']) * 100
hours_in_comp = df_stack['HoursComputer'].value_counts() / len(df_stack['HoursComputer']) * 100
hours_out_comp = df_stack['HoursOutside'].value_counts() / len(df_stack['HoursOutside']) * 100 
plt.figure(figsize=(12,10))

plt.subplot(211)
g = sns.barplot(x=employment.index, 
                y=employment.values)
g.set_title("Employment in % of total", fontsize=20)
g.set_xlabel("", fontsize=16)
g.set_ylabel("Count", fontsize=16)
g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.subplot(212)
g1 = sns.violinplot(x='Employment', y='ConvertedSalary_log', order=employment.index, data=df_stack)
g1.set_title("Employment Salary Distribuition", fontsize=20)
g1.set_xlabel("Employment type", fontsize=16)
g1.set_ylabel("Conv Salary(Log)", fontsize=16)
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)

plt.subplots_adjust(hspace = 0.8, top = 0.9)

plt.show()

print("Description percentual of Countrys")
print(round(employment,2))

df_stack['YearsCodingProf'] = df_stack['YearsCodingProf'].replace({'0-2 years':'0-2', '3-5 years':'3-5', '6-8 years':'6-8', '9-11 years':'9-11', 
                                                           '12-14 years':'12-14', '15-17 years':'15-17', '18-20 years':'18-20', 
                                                           '21-23 years':'21-23', '30 or more years':'30 or more', '24-26 years':'24-26', 
                                                           '27-29 years': '27-29'})
plt.figure(figsize=(13,9))

plt.subplot(211)
g = sns.countplot(x='YearsCodingProf', data=df_stack, 
              order=['0-2', '3-5', '6-8', '9-11', '12-14',
                     '15-17', '18-20', '21-23','24-26',  '27-29', '30 or more'])
g.set_title("Years Coding as Profissional ", fontsize=20)
g.set_xlabel("Years of Experience", fontsize=16)
g.set_ylabel("Count", fontsize=16)

plt.subplot(212)
g1 = sns.boxplot(x='YearsCodingProf', y='ConvertedSalary_log', data=df_stack, 
                   order=['0-2', '3-5', '6-8', '9-11', '12-14',
                          '15-17', '18-20', '21-23','24-26',  '27-29', '30 or more'])
g1.set_title("Years Coding as Profissional ", fontsize=20)
g1.set_xlabel("Years of Experience", fontsize=16)
g1.set_ylabel("Converted Salary (Log)", fontsize=16)

plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()
date_int = ['Employment',"YearsCodingProf"]

cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_stack[date_int[0]], df_stack[date_int[1]], normalize='columns').style.background_gradient(cmap = cm)
df_stack['CompanySize'] = df_stack['CompanySize'].replace({'Fewer than 10 employees': 'Fewer than 10', 
                                                           '10 to 19 employees': '10 to 19', '20 to 99 employees': '20 to 99', 
                                                           '100 to 499 employees': '100 to 499', '500 to 999 employees':'500 to 999',
                                                           '1,000 to 4,999 employees':'1,000 to 4,999', '5,000 to 9,999 employees':'5,000 to 9,999',
                                                           '10,000 or more employees': '10,000 or more'})   
plt.figure(figsize=(12,14))
plt.subplot(211)
g = sns.countplot(x='CompanySize',
                  data=df_stack, order=['Fewer than 10','10 to 19','20 to 99',
                    '100 to 499','500 to 999','1,000 to 4,999', 
                    '5,000 to 9,999', '10,000 or more'])
g.set_title("Company Size Count", fontsize=20)
g.set_xlabel("Company Size", fontsize=16)
g.set_ylabel("Count", fontsize=16)

plt.subplot(212)
g1 = sns.boxplot(x='CompanySize',y='ConvertedSalary_log',
                  data=df_stack, order=['Fewer than 10','10 to 19','20 to 99',
                    '100 to 499','500 to 999','1,000 to 4,999',
                    '5,000 to 9,999', '10,000 or more'])
g1.set_title("Company Size Salary Distribuitions", fontsize=20)
g1.set_xlabel("Company Size", fontsize=16)
g1.set_ylabel("Count", fontsize=16)
g1.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplots_adjust(hspace = 0.5, top = 0.8)

plt.show()
date_int = ['CompanySize',"YearsCodingProf"]

cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_stack[date_int[0]], df_stack[date_int[1]], normalize='columns').style.background_gradient(cmap = cm)
print("Percentual of Form Education Distribuition: ")
print(round(formal_educ, 2))

plt.figure(figsize=(12,14))
plt.subplot(211)
g = sns.countplot(x='FormalEducation', hue='OpenSource',
                  data=df_stack, order=formal_educ.index)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Formal Education Count by Open Source Contributors", fontsize=20)
g.set_xlabel("", fontsize=16)
g.set_ylabel("Frequency", fontsize=16)

plt.subplot(212)
g1 = sns.boxplot(x='FormalEducation',y='ConvertedSalary_log', hue='OpenSource',
                  data=df_stack, order=formal_educ.index)
g1.set_title("Formal Education Converted Income divided by Open Source Contributors", fontsize=20)
g1.set_xlabel("Formal Education Name", fontsize=16)
g1.set_ylabel("Converted Salary(Log) Dist", fontsize=16)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()
date_int = ['FormalEducation',"YearsCodingProf"]

cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_stack[date_int[0]], df_stack[date_int[1]], normalize='index').style.background_gradient(cmap = cm)
plt.figure(figsize=(12,10))

plt.subplot(211)
g = sns.countplot(x='CareerSatisfaction', hue='OpenSource',data=df_stack,
                  order=['Extremely satisfied', 'Moderately satisfied',
                         'Slightly satisfied','Neither satisfied nor dissatisfied',
                         'Slightly dissatisfied','Moderately dissatisfied','Extremely dissatisfied'])
g.set_title("Carrer Satisfaction Dist by OpenSource Contributors", fontsize=20)
g.set_xlabel("", fontsize=16)
g.set_ylabel("Frequency", fontsize=16)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplot(2,1,2)
g1 = sns.boxplot(x='CareerSatisfaction', y='ConvertedSalary_log', hue='OpenSource', data=df_stack,
                 order=['Extremely satisfied', 'Moderately satisfied',
                         'Slightly satisfied','Neither satisfied nor dissatisfied',
                         'Slightly dissatisfied','Moderately dissatisfied','Extremely dissatisfied'])
g1.set_title("Carrer Satisfaction Level Salary by Open Source Contributors", fontsize=20)
g1.set_xlabel("Carrer Satisfaction Level", fontsize=16)
g1.set_ylabel("Converted Salary(Log)", fontsize=16)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.legend(loc=4)

plt.subplots_adjust(hspace = 0.9, top = 0.8)

plt.show()
date_int = ['FormalEducation',"CareerSatisfaction"]

cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_stack[date_int[0]], df_stack[date_int[1]], normalize='index').style.background_gradient(cmap = cm)
date_int = ['YearsCodingProf',"CareerSatisfaction"]

cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_stack[date_int[0]], df_stack[date_int[1]], normalize='index').style.background_gradient(cmap = cm)
print("The proportion of each declared Orientation: ")
print(orientation / len(df_stack['SexualOrientation'].dropna()) * 100 )
orientation = df_stack['SexualOrientation'].value_counts()[:4]

plt.figure(figsize=(12,14))
plt.subplot(211)
g = sns.countplot(x='SexualOrientation', hue='OpenSource',
                  data=df_stack[df_stack.SexualOrientation.isin(orientation.index.values)])

g.set_title("Users Sexual Orientation Count", fontsize=20)
g.set_xlabel("", fontsize=16)
g.set_ylabel("Frequency", fontsize=16)

plt.subplot(212)
g1 = sns.boxplot(x='SexualOrientation',y='ConvertedSalary_log', hue='OpenSource',
                  data=df_stack[df_stack.SexualOrientation.isin(orientation.index.values)])
g1.set_title("Users SexualOrientation Declared Salary", fontsize=20)
g1.set_xlabel("Sexual Orientation", fontsize=16)
g1.set_ylabel("Salary(Log)", fontsize=16)

g1.legend(loc=5)
plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()
age_order = ['Under 18 years old','18 - 24 years old','25 - 34 years old',
             '35 - 44 years old','45 - 54 years old','55 - 64 years old', 
             '65 years or older']

plt.figure(figsize=(12,14))
plt.subplot(211)
g = sns.countplot(x='Age', hue='Hobby',
                  data=df_stack, order=age_order)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Age Users Count", fontsize=20)
g.set_xlabel("", fontsize=16)
g.set_ylabel("Frequency", fontsize=16)

plt.subplot(212)
g1 = sns.boxplot(x='Age',y='ConvertedSalary_log', hue="Hobby",
                  data=df_stack, order=age_order)
g1.set_title("Age and Declared Salary", fontsize=20)
g1.set_xlabel("Age Distribution", fontsize=16)
g1.set_ylabel("Salary(Log)", fontsize=16)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.legend(loc=5)
plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()
plt.figure(figsize=(12,14))
plt.subplot(211)
g = sns.countplot(x='Age', hue='OpenSource',
                  data=df_stack, order=age_order)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Age Users Count", fontsize=20)
g.set_xlabel("", fontsize=16)
g.set_ylabel("Frequency", fontsize=16)

plt.subplot(212)
g1 = sns.boxplot(x='Age',y='ConvertedSalary_log', hue="Hobby",
                  data=df_stack, order=age_order)
g1.set_title("Age and Declared Salary", fontsize=20)
g1.set_xlabel("Age Distribution", fontsize=16)
g1.set_ylabel("Salary(Log)", fontsize=16)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.legend(loc=5)
plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()
plt.figure(figsize=(12,14))
plt.subplot(211)
g = sns.countplot(x='Age', hue='OpenSource',
                  data=df_stack, order=age_order)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Age Users Count", fontsize=20)
g.set_xlabel("", fontsize=16)
g.set_ylabel("Frequency", fontsize=16)

plt.subplot(212)
g1 = sns.boxplot(x='Age',y='ConvertedSalary_log', hue="Open Source",
                  data=df_stack, order=age_order)
g1.set_title("Age and Declared Salary", fontsize=20)
g1.set_xlabel("Age Distribution", fontsize=16)
g1.set_ylabel("Salary(Log)", fontsize=16)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.legend(loc=5)
plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()