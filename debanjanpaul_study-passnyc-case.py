import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale
%matplotlib inline
### Read : School data 2016
school_data = pd.read_csv(r'../input/data-science-for-good/2016 School Explorer.csv')

### Reda : D5 SHSAT
d5_shsat_data = pd.read_csv(r'../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv')

### Read : NYC school meals income levels [Source : Socrata]
nyc_school_meals_income_levels = pd.read_csv(r'../input/nyc-school-meals-income-levels/nyc-school-meals-income-levels.csv')

### Read : School ELA Results 2013-2017 (Public)

school_results = pd.read_csv(r'../input/school-ela-results-20132017-public/School ELA Results 2013-2017 (Public)_consolidated.csv')

### Used datasets
print('School data 2016 : ',school_data.shape, '\nDistrict 5 SHSAT data : ', d5_shsat_data.shape,'\nNYC school meals income levels : ', nyc_school_meals_income_levels.shape, '\nNYC School ELA Results(2013-2017) : ',school_results.shape)
############### Pre-processing ###############

### Grade Low/Grade High - conversion to type string
school_data['Grade Low'] = school_data['Grade Low'].astype(str)
school_data['Grade High'] = school_data['Grade High'].astype(str)

### String conversion to decimal
def string2decimal(s):
    if isinstance(s, str):
        s = s.strip()
        s = s.replace(",", "")
        s = s.replace("$", "")
        s = s.replace("%", "")
        return float(s)
    else:
        return s

school_data['School Income Estimate'] = school_data['School Income Estimate'].apply(lambda x: string2decimal(x))
school_data['Percent ELL'] = school_data['Percent ELL'].apply(lambda x: string2decimal(x)/100)
school_data['Percent Asian'] = school_data['Percent Asian'].apply(lambda x: string2decimal(x)/100)
school_data['Percent Black'] = school_data['Percent Black'].apply(lambda x: string2decimal(x)/100)
school_data['Percent Hispanic'] = school_data['Percent Hispanic'].apply(lambda x: string2decimal(x)/100)
school_data['Percent Black / Hispanic'] = school_data['Percent Black / Hispanic'].apply(lambda x: string2decimal(x)/100)
school_data['Percent White'] = school_data['Percent White'].apply(lambda x: string2decimal(x)/100)
school_data['Student Attendance Rate'] = school_data['Student Attendance Rate'].apply(lambda x: string2decimal(x)/100)
school_data['Percent of Students Chronically Absent'] = school_data['Percent of Students Chronically Absent'].apply(lambda x: string2decimal(x)/100)
school_data['Rigorous Instruction %'] = school_data['Rigorous Instruction %'].apply(lambda x: string2decimal(x)/100)
school_data['Collaborative Teachers %'] = school_data['Collaborative Teachers %'].apply(lambda x: string2decimal(x)/100)
school_data['Supportive Environment %'] = school_data['Supportive Environment %'].apply(lambda x: string2decimal(x)/100)
school_data['Effective School Leadership %'] = school_data['Effective School Leadership %'].apply(lambda x: string2decimal(x)/100)
school_data['Strong Family-Community Ties %'] = school_data['Strong Family-Community Ties %'].apply(lambda x: string2decimal(x)/100)
school_data['Trust %'] = school_data['Trust %'].apply(lambda x: string2decimal(x)/100)




school_results['Mean Scale Score'] = school_results['Mean Scale Score'].apply(lambda x: int(x) if x != 's' else 's')
school_results['Level 1%'] = school_results['Level 1%'].apply(lambda x: float(x) if x != 's' else 0.1)
school_results['Level 2%'] = school_results['Level 2%'].apply(lambda x: float(x) if x != 's' else 0.1)
school_results['Level 3%'] = school_results['Level 3%'].apply(lambda x: float(x) if x != 's' else 0.1)
school_results['Level 4%'] = school_results['Level 4%'].apply(lambda x: float(x) if x != 's' else 0.1)
school_results['Level 3+4%'] = school_results['Level 3+4%'].apply(lambda x: float(x) if x != 's' else 0.1)

school_results[['Level 1%', 'Level 2%', 'Level 3%', 'Level 4%', 'Level 3+4%']] = school_results[['Level 1%', 'Level 2%', 'Level 3%', 'Level 4%', 'Level 3+4%']].replace(0, 0.1)

plt.figure(figsize=(10,7))
service_grade_categories = school_data[['SED Code', 'Grade Low', 'Grade High']].pivot_table(values = 'SED Code'
                                        , index = 'Grade Low', columns = 'Grade High', aggfunc = np.size, fill_value = 0)
plt.title('Field value represents # of schools\n', fontsize=16)
sns.heatmap(service_grade_categories, annot = True, fmt = 'd', linewidths=.2, cmap="YlGnBu", )
school_data['Early_Childhood'] = 0
school_data['Elementary'] = 0
school_data['Middle'] = 0
school_data['High'] = 0

school_data['Early_Childhood']  = np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['0K', '02', '03', '04']), 1, \
                                           np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['05', '06']), 1, \
                                                    np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['08']), 1, \
                                                             np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['12']), 1, 0))))

school_data['Elementary'] = np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['05', '06']), 1, \
                                    np.where(school_data['Grade Low'].isin(['01', '02', '03', '04']) & school_data['Grade High'].isin(['05', '06']), 1, \
                                            np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['08']), 1, \
                                                    np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['12']), 1, 0))))

school_data['Middle'] = np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['08']), 1, \
                                 np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['12']), 1, \
                                          np.where(school_data['Grade Low'].isin(['04', '05', '06']) & school_data['Grade High'].isin(['08']), 1, \
                                                   np.where(school_data['Grade Low'].isin(['05', '06', '07']) & school_data['Grade High'].isin(['12']), 1, 0))))

school_data['High'] = np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['12']), 1, \
                                np.where(school_data['Grade Low'].isin(['05', '06', '07']) & school_data['Grade High'].isin(['10', '12']), 1,\
                                         np.where(school_data['Grade Low'].isin(['09']) & school_data['Grade High'].isin(['12']), 1, 0)))
nyc_school_meals_income_levels.head()
print("Median ENI (%) : ", 100*school_data['Economic Need Index'].median())
print("Mean ENI (%) : ", 100*school_data['Economic Need Index'].mean())
print("Median School Income Estimate ($) : ", school_data['School Income Estimate'].median())
print("Mean School Income Estimate ($) : ", school_data['School Income Estimate'].mean())
f,axes=plt.subplots(1,3,figsize=(22,5))

sns.kdeplot(school_data['Economic Need Index'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[0])
axes[0].set_title('Density Plot : Economic Need Index')
sns.kdeplot(school_data['School Income Estimate'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[1])
axes[1].set_title('Density Plot : School Income Estimate')
sns.regplot(x="Economic Need Index", y="School Income Estimate", data=school_data[['Economic Need Index', 'School Income Estimate']], ax = axes[2])
axes[2].set_title('Correlation : Economic Need Index Vs. School Income Estimate')
f,axes=plt.subplots(1,2,figsize=(18,5),sharex=True)
sns.regplot(x="Economic Need Index", y="Average ELA Proficiency", data=school_data[['Economic Need Index', 'Average ELA Proficiency']], ax = axes[0])
axes[0].axvline(x = 0.7, color = 'r')
axes[0].axhline(y = 3.0, color = 'r')
axes[0].set_title('Correlation : Economic Need Index Vs. Average ELA Proficiency')
sns.regplot(x="Economic Need Index", y="Average Math Proficiency", data=school_data[['Economic Need Index', 'Average Math Proficiency']], ax = axes[1])
axes[1].axvline(x = 0.7, color = 'r')
axes[1].axhline(y = 3.0, color = 'r')
axes[1].set_title('Correlation : Economic Need Index Vs. Average Math Proficiency')
### Ignore the missing part in the distribution
f,axes=plt.subplots(1,2,figsize=(18,5),sharex=True)

sns.kdeplot(school_data['Average ELA Proficiency'].fillna(0), shade  = True, color = 'r', ax = axes[0])
axes[0].set_title('Density Plot : Average ELA proficiency')

sns.kdeplot(school_data['Average Math Proficiency'].fillna(0), shade  = True, color = 'b', ax = axes[1])
axes[1].set_title('Density Plot : Average Math proficiency')
school_data_high_eni_ela_proficient = school_data[(school_data['Economic Need Index'] >= 0.65) & (school_data['Average ELA Proficiency'] >= 3)]
school_data_low_eni_ela_proficient = school_data[(school_data['Economic Need Index'] < 0.65) & (school_data['Average ELA Proficiency'] >= 3)]

school_data_high_eni_math_proficient = school_data[(school_data['Economic Need Index'] >= 0.65) & (school_data['Average Math Proficiency'] >= 3)]
school_data_low_eni_math_proficient = school_data[(school_data['Economic Need Index'] < 0.65) & (school_data['Average Math Proficiency'] >= 3)]
f,axes=plt.subplots(1,2,figsize=(18,5),sharex=True)

sns.kdeplot(school_data_high_eni_ela_proficient['Average ELA Proficiency'].fillna(0), shade  = True, color = 'b', ax = axes[0])
sns.kdeplot(school_data_low_eni_ela_proficient['Average ELA Proficiency'].fillna(0), shade  = True, color = 'y', ax = axes[0])
axes[0].set_title('Density Plot Comparision : Average ELA Proficiency ≥ 3\nBlue : High ENI   Yellow : Low ENI')

sns.kdeplot(school_data_high_eni_math_proficient['Average Math Proficiency'].fillna(0), shade  = True, color = 'b', ax = axes[1])
sns.kdeplot(school_data_low_eni_math_proficient['Average Math Proficiency'].fillna(0), shade  = True, color = 'y', ax = axes[1])
axes[1].set_title('Density Plot Comparision : Average Math Proficiency ≥ 3\nBlue : High ENI   Yellow : Low ENI')
school_data_high_eni_ela_below_proficient = school_data[(school_data['Economic Need Index'] >= 0.65) & (school_data['Average ELA Proficiency'] < 3)]
school_data_low_eni_ela_below_proficient = school_data[(school_data['Economic Need Index'] < 0.65) & (school_data['Average ELA Proficiency'] < 3)]

school_data_high_eni_math_below_proficient = school_data[(school_data['Economic Need Index'] >= 0.65) & (school_data['Average Math Proficiency'] < 3)]
school_data_low_eni_math_below_proficient = school_data[(school_data['Economic Need Index'] < 0.65) & (school_data['Average Math Proficiency'] < 3)]
f,axes=plt.subplots(1,2,figsize=(18,5),sharex=True)

sns.kdeplot(school_data_high_eni_ela_below_proficient['Average ELA Proficiency'].fillna(0), shade  = True, color = 'b', ax = axes[0])
sns.kdeplot(school_data_low_eni_ela_below_proficient['Average ELA Proficiency'].fillna(0), shade  = True, color = 'y', ax = axes[0])
axes[0].set_title('Density Plot Comparision : Average ELA Proficiency ≤ 3\nBlue : High ENI   Yellow : Low ENI')

sns.kdeplot(school_data_high_eni_math_below_proficient['Average Math Proficiency'].fillna(0), shade  = True, color = 'b', ax = axes[1])
sns.kdeplot(school_data_low_eni_math_below_proficient['Average Math Proficiency'].fillna(0), shade  = True, color = 'y', ax = axes[1])
axes[1].set_title('Density Plot Comparision : Average Math Proficiency ≤ 3\nBlue : High ENI   Yellow : Low ENI')
school_data['proficiency'] = np.where((school_data['Average ELA Proficiency'] >= 3) & (school_data['Average Math Proficiency'] >= 3), 1,
                                np.where((school_data['Average ELA Proficiency'] >= 3) & (school_data['Average Math Proficiency'] < 3), 2,
                                    np.where((school_data['Average ELA Proficiency'] < 3) & (school_data['Average Math Proficiency'] >= 3), 3, 4)))
f,axes=plt.subplots(4,2,figsize=(24,22))

school_data['proficiency'].hist(ax = axes[0,0])
axes[0,0].set_title('Distribution of Proficiency Levels (Discrete)\n1: Proficient in both    2: Proficient only in ELA    3: Proficient only in Math    4: Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Percent ELL'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[0,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Percent ELL'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[0,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Percent ELL'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[0,1])
axes[0,1].set_title('\nDensity Plot : Percent ELL\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Percent White'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[1,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Percent White'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[1,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Percent White'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[1,0])
axes[1,0].set_title('\nDensity Plot : Percent White\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Percent Black'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[1,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Percent Black'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[1,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Percent Black'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[1,1])
axes[1,1].set_title('\nDensity Plot : Percent Black\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Percent Asian'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[2,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Percent Asian'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[2,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Percent Asian'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[2,0])
axes[2,0].set_title('\nDensity Plot : Percent Asian\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Percent Hispanic'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[2,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Percent Hispanic'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[2,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Percent Hispanic'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[2,1])
axes[2,1].set_title('\nDensity Plot : Percent Hispanic\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Percent Black / Hispanic'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[3,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Percent Black / Hispanic'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[3,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Percent Black / Hispanic'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[3,0])
axes[3,0].set_title('\nDensity Plot : Percent Black/Hispanic\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Economic Need Index'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[3,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Economic Need Index'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[3,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Economic Need Index'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[3,1])
axes[3,1].set_title('\nDensity Plot : Percent Economic Need Index\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

school_data[school_data['Community School?'] == 'Yes']['proficiency'].value_counts()
f,axes=plt.subplots(4,2,figsize=(24,20))

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Rigorous Instruction %'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[0,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Rigorous Instruction %'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[0,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Rigorous Instruction %'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[0,0])
axes[0,0].set_title('\nDensity Plot : Survey Response - Rigorous Instruction %\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Collaborative Teachers %'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[0,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Collaborative Teachers %'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[0,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Collaborative Teachers %'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[0,1])
axes[0,1].set_title('\nDensity Plot : Survey Response - Collaborative Teachers %\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Supportive Environment %'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[1,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Supportive Environment %'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[1,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Supportive Environment %'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[1,0])
axes[1,0].set_title('\nDensity Plot : Survey Response - Supportive Environment %\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Effective School Leadership %'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[1,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Effective School Leadership %'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[1,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Effective School Leadership %'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[1,1])
axes[1,1].set_title('\nDensity Plot : Survey Response - Effective School Leadership %\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Strong Family-Community Ties %'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[2,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Strong Family-Community Ties %'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[2,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Strong Family-Community Ties %'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[2,0])
axes[2,0].set_title('\nDensity Plot : Survey Response - Strong Family-Community Ties %\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Trust %'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[2,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Trust %'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[2,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Trust %'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[2,1])
axes[2,1].set_title('\nDensity Plot : Survey Response - Trust %\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['Economic Need Index'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[3,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['Economic Need Index'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[3,0])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['Economic Need Index'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[3,0])
axes[3,0].set_title('\nDensity Plot : Economic Need Index\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

sns.kdeplot(100*school_data[school_data['proficiency'] == 1]['School Income Estimate'].fillna(0), cut = 0, shade  = True, color = 'b', ax = axes[3,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 3]['School Income Estimate'].fillna(0), cut = 0, shade  = True, color = 'g', ax = axes[3,1])
sns.kdeplot(100*school_data[school_data['proficiency'] == 4]['School Income Estimate'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[3,1])
axes[3,1].set_title('\nDensity Plot : School Income Estimate\nBlue : Proficient in both    Green : Proficient only in Math   Red : Proficient in None')

######################### Pre-processing #########################

##### Drop missing values #####
##### Dropped because missing in most of the columns #####
school_data_with_missing_values = pd.DataFrame()
school_data_with_missing_values = school_data_with_missing_values.append([school_data[~pd.notnull(school_data['Economic Need Index'])]])
school_data = school_data[pd.notnull(school_data['Economic Need Index'])]


##### Impute missing values #####
##### Strings with "missing", Numerical with mean #####
school_data['Student Achievement Rating'] = school_data['Student Achievement Rating'].fillna('missing')
school_data['Supportive Environment Rating'] = school_data['Supportive Environment Rating'].fillna('missing')
school_data['Collaborative Teachers Rating'] = school_data['Collaborative Teachers Rating'].fillna('missing')
school_data['Rigorous Instruction Rating'] = school_data['Rigorous Instruction Rating'].fillna('missing')
school_data['Trust Rating'] = school_data['Trust Rating'].fillna('missing')
school_data['Strong Family-Community Ties Rating'] = school_data['Strong Family-Community Ties Rating'].fillna('missing')
school_data['Effective School Leadership Rating'] = school_data['Effective School Leadership Rating'].fillna('missing')
school_data['Average Math Proficiency'] = school_data['Average Math Proficiency'].fillna(school_data['Average Math Proficiency'].mean())
school_data['Average ELA Proficiency'] = school_data['Average ELA Proficiency'].fillna(school_data['Average ELA Proficiency'].mean())


##### Reset index school_data
school_data.reset_index(drop = True, inplace=True)
school_data.shape

school_data['Community'] = np.where(school_data['Community School?'] == 'Yes', 1, 0)
school_data.shape
### categorical variable encoding module

def categorical_variable_encoding(df):
    return pd.get_dummies(df, columns=list(df))

for field in ['Rigorous Instruction Rating', 'Collaborative Teachers Rating', 'Supportive Environment Rating', 'Effective School Leadership Rating', 'Strong Family-Community Ties Rating', 'Trust Rating', 'Student Achievement Rating']:
    df = categorical_variable_encoding(school_data[field])
    df.columns = [field + ' ' + s for s in list(df)]
    school_data = school_data.join(df)
school_data.shape
##### Split school_data into 3 groups #####
group1 = school_data[school_data['proficiency'] == 1].copy()
group2 = school_data[school_data['proficiency'] == 3].copy()
group3 = school_data[school_data['proficiency'] == 4].copy()

group1.reset_index(drop=True, inplace=True)
group2.reset_index(drop=True, inplace=True)
group3.reset_index(drop=True, inplace=True)

##### Convert to array #####

X_group1 = np.array(group1[[    'Early_Childhood',
                                'Elementary',
                                'Middle',
                                'High',
                                'proficiency',
                                'Community',
                                'Economic Need Index',
                                'Percent ELL',
                                'Percent Asian',
                                'Percent Black',
                                'Percent Hispanic',
                                'Percent Black / Hispanic',
                                'Percent White',
                                'Student Attendance Rate',
                                'Percent of Students Chronically Absent',
                                'Rigorous Instruction %',
                                'Collaborative Teachers %',
                                'Supportive Environment %',
                                'Effective School Leadership %',
                                'Strong Family-Community Ties %',
                                'Trust %',
                                'Rigorous Instruction Rating Approaching Target',
                                'Rigorous Instruction Rating Exceeding Target',
                                'Rigorous Instruction Rating Meeting Target',
                                'Rigorous Instruction Rating Not Meeting Target',
                                'Rigorous Instruction Rating missing',
                                'Collaborative Teachers Rating Approaching Target',
                                'Collaborative Teachers Rating Exceeding Target',
                                'Collaborative Teachers Rating Meeting Target',
                                'Collaborative Teachers Rating Not Meeting Target',
                                'Collaborative Teachers Rating missing',
                                'Supportive Environment Rating Approaching Target',
                                'Supportive Environment Rating Exceeding Target',
                                'Supportive Environment Rating Meeting Target',
                                'Supportive Environment Rating Not Meeting Target',
                                'Supportive Environment Rating missing',
                                'Effective School Leadership Rating Approaching Target',
                                'Effective School Leadership Rating Exceeding Target',
                                'Effective School Leadership Rating Meeting Target',
                                'Effective School Leadership Rating Not Meeting Target',
                                'Effective School Leadership Rating missing',
                                'Strong Family-Community Ties Rating Approaching Target',
                                'Strong Family-Community Ties Rating Exceeding Target',
                                'Strong Family-Community Ties Rating Meeting Target',
                                'Strong Family-Community Ties Rating Not Meeting Target',
                                'Strong Family-Community Ties Rating missing',
                                'Trust Rating Approaching Target',
                                'Trust Rating Exceeding Target',
                                'Trust Rating Meeting Target',
                                'Trust Rating Not Meeting Target',
                                'Trust Rating missing',
                                'Student Achievement Rating Approaching Target',
                                'Student Achievement Rating Exceeding Target',
                                'Student Achievement Rating Meeting Target',
                                'Student Achievement Rating Not Meeting Target',
                                'Student Achievement Rating missing',
                                'Average ELA Proficiency',
                                'Average Math Proficiency']])

X_group2 = np.array(group2[[    'Early_Childhood', 
                                'Elementary', 
                                'Middle', 
                                'High',
                                'proficiency',
                                'Community',
                                'Economic Need Index',
                                'Percent ELL',
                                'Percent Asian',
                                'Percent Black',
                                'Percent Hispanic',
                                'Percent Black / Hispanic',
                                'Percent White',
                                'Student Attendance Rate',
                                'Percent of Students Chronically Absent',
                                'Rigorous Instruction %',
                                'Collaborative Teachers %',
                                'Supportive Environment %',
                                'Effective School Leadership %',
                                'Strong Family-Community Ties %',
                                'Trust %',
                                'Rigorous Instruction Rating Approaching Target',
                                'Rigorous Instruction Rating Exceeding Target',
                                'Rigorous Instruction Rating Meeting Target',
                                'Rigorous Instruction Rating Not Meeting Target',
                                'Rigorous Instruction Rating missing',
                                'Collaborative Teachers Rating Approaching Target',
                                'Collaborative Teachers Rating Exceeding Target',
                                'Collaborative Teachers Rating Meeting Target',
                                'Collaborative Teachers Rating Not Meeting Target',
                                'Collaborative Teachers Rating missing',
                                'Supportive Environment Rating Approaching Target',
                                'Supportive Environment Rating Exceeding Target',
                                'Supportive Environment Rating Meeting Target',
                                'Supportive Environment Rating Not Meeting Target',
                                'Supportive Environment Rating missing',
                                'Effective School Leadership Rating Approaching Target',
                                'Effective School Leadership Rating Exceeding Target',
                                'Effective School Leadership Rating Meeting Target',
                                'Effective School Leadership Rating Not Meeting Target',
                                'Effective School Leadership Rating missing',
                                'Strong Family-Community Ties Rating Approaching Target',
                                'Strong Family-Community Ties Rating Exceeding Target',
                                'Strong Family-Community Ties Rating Meeting Target',
                                'Strong Family-Community Ties Rating Not Meeting Target',
                                'Strong Family-Community Ties Rating missing',
                                'Trust Rating Approaching Target',
                                'Trust Rating Exceeding Target',
                                'Trust Rating Meeting Target',
                                'Trust Rating Not Meeting Target',
                                'Trust Rating missing',
                                'Student Achievement Rating Approaching Target',
                                'Student Achievement Rating Exceeding Target',
                                'Student Achievement Rating Meeting Target',
                                'Student Achievement Rating Not Meeting Target',
                                'Student Achievement Rating missing',
                                'Average ELA Proficiency',
                                'Average Math Proficiency']])

X_group3 = np.array(group3[[    'Early_Childhood', 
                                'Elementary', 
                                'Middle', 
                                'High',
                                'proficiency',
                                'Community',
                                'Economic Need Index',
                                'Percent ELL',
                                'Percent Asian',
                                'Percent Black',
                                'Percent Hispanic',
                                'Percent Black / Hispanic',
                                'Percent White',
                                'Student Attendance Rate',
                                'Percent of Students Chronically Absent',
                                'Rigorous Instruction %',
                                'Collaborative Teachers %',
                                'Supportive Environment %',
                                'Effective School Leadership %',
                                'Strong Family-Community Ties %',
                                'Trust %',
                                'Rigorous Instruction Rating Approaching Target',
                                'Rigorous Instruction Rating Exceeding Target',
                                'Rigorous Instruction Rating Meeting Target',
                                'Rigorous Instruction Rating Not Meeting Target',
                                'Rigorous Instruction Rating missing',
                                'Collaborative Teachers Rating Approaching Target',
                                'Collaborative Teachers Rating Exceeding Target',
                                'Collaborative Teachers Rating Meeting Target',
                                'Collaborative Teachers Rating Not Meeting Target',
                                'Collaborative Teachers Rating missing',
                                'Supportive Environment Rating Approaching Target',
                                'Supportive Environment Rating Exceeding Target',
                                'Supportive Environment Rating Meeting Target',
                                'Supportive Environment Rating Not Meeting Target',
                                'Supportive Environment Rating missing',
                                'Effective School Leadership Rating Approaching Target',
                                'Effective School Leadership Rating Exceeding Target',
                                'Effective School Leadership Rating Meeting Target',
                                'Effective School Leadership Rating Not Meeting Target',
                                'Effective School Leadership Rating missing',
                                'Strong Family-Community Ties Rating Approaching Target',
                                'Strong Family-Community Ties Rating Exceeding Target',
                                'Strong Family-Community Ties Rating Meeting Target',
                                'Strong Family-Community Ties Rating Not Meeting Target',
                                'Strong Family-Community Ties Rating missing',
                                'Trust Rating Approaching Target',
                                'Trust Rating Exceeding Target',
                                'Trust Rating Meeting Target',
                                'Trust Rating Not Meeting Target',
                                'Trust Rating missing',
                                'Student Achievement Rating Approaching Target',
                                'Student Achievement Rating Exceeding Target',
                                'Student Achievement Rating Meeting Target',
                                'Student Achievement Rating Not Meeting Target',
                                'Student Achievement Rating missing',
                                'Average ELA Proficiency',
                                'Average Math Proficiency']])
##### Find out 3 nearest(considering above dimensions) school
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X_group2)
kneighbour_matrix = nbrs.kneighbors_graph(X_group1).toarray()
kneighbour_matrix.shape
group1_school_index = 46
group1.iloc[[group1_school_index]]
i, = np.where(kneighbour_matrix[group1_school_index] == 1)
group2.iloc[i]
kneighbour_matrix.sum(axis = 0)
score = list(set(kneighbour_matrix.sum(axis = 0)))
score
group2_ranked = pd.DataFrame()
priority = 0
group2_ranked['priority'] = priority
for s in score:
    temp_df = pd.DataFrame()
    r, = np.where(kneighbour_matrix.sum(axis = 0) == s)
    temp_df = temp_df.append(group2.iloc[r])
    temp_df['priority'] = priority
    priority += 1
    group2_ranked = group2_ranked.append(temp_df)
group2_ranked.reset_index(drop = True, inplace = True)
group2_ranked.sort_values(by = 'priority', ascending = False, inplace = True)
group2_ranked.head(5)
group2_school_count_by_priority = pd.DataFrame(group2_ranked['priority'].value_counts()).reset_index(drop = False)
group2_school_count_by_priority.columns = ['priority', 'Number of schools']
group2_school_count_by_priority.sort_values(by = 'priority', inplace = True)
group2_school_count_by_priority
parameters ={
    'Economic Need Index' : 'mean',
    'Percent ELL' : 'mean', 
    'Percent Hispanic' : 'mean', 
    'Percent Black' : 'mean',
    'Percent White' : 'mean',
    'Percent Asian' : 'mean',
    'Percent Black / Hispanic' : 'mean'
    
}

group2_ranked[['priority', 'Economic Need Index', 'Percent ELL', 'Percent Hispanic', 'Percent Black', 'Percent White', 'Percent Asian', 'Percent Black / Hispanic']].groupby('priority').agg(parameters)
def calculate_cagr(ser):
    n = len(ser)
    cagr = ((ser.iloc[-1]/ser.iloc[0])**(1/n)) - 1
    return cagr*100
parameters = {
    'Enrollment on 10/31' : calculate_cagr,
    'Number of students who registered for the SHSAT' : calculate_cagr,
    'Number of students who took the SHSAT' : calculate_cagr
    }

d5_shsat_data[['Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']] = d5_shsat_data[['Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']].replace(0, 1)
cagr_d5_shsat = d5_shsat_data[['DBN', 'Grade level', 'Year of SHST', 'Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']].groupby(['DBN', 'Grade level']).agg(parameters).reset_index(drop = False)

#cagr_d5_shsat[['Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']] = cagr_d5_shsat[['Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']]*100
cagr_d5_shsat_grade8 = cagr_d5_shsat[cagr_d5_shsat['Grade level'] == 8]
cagr_d5_shsat_grade8.reset_index(drop = True, inplace = True)
cagr_d5_shsat_grade8
cagr_d5_shsat_grade9 = cagr_d5_shsat[cagr_d5_shsat['Grade level'] == 9]
cagr_d5_shsat_grade9.reset_index(drop = True, inplace = True)
cagr_d5_shsat_grade9
from sklearn.preprocessing import StandardScaler
scaler8 = StandardScaler()
scaler9 = StandardScaler()
cagr_d5_shsat_scaled8 = pd.DataFrame(scaler8.fit_transform(cagr_d5_shsat_grade8[['Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']]), columns=['Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT'])
cagr_d5_shsat_scaled9 = pd.DataFrame(scaler9.fit_transform(cagr_d5_shsat_grade9[['Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']]), columns=['Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT'])
f,axes=plt.subplots(1,2,figsize=(22,8))
sns.heatmap(cagr_d5_shsat_scaled8, annot = cagr_d5_shsat_grade8[['Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']], cmap = 'coolwarm_r', ax = axes[0])
axes[0].set_title('\nHeat Map : Grade 8 - CAGR Enrollment, Registration and Appearence \nMore Blue : Stronger +ve growth   More Red : Stronger -ve growth')
sns.heatmap(cagr_d5_shsat_scaled9, annot = cagr_d5_shsat_grade9[['Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']], cmap = 'coolwarm_r', ax = axes[1])
axes[1].set_title('\nHeat Map : Grade 9 - CAGR Enrollment, Registration and Appearence \nMore Blue : Stronger -ve growth   More Red : Stronger -ve growth')
#### Enter District Borough Number (dbn) below to check any specific school's figure
dbn = '84M341'
d5_shsat_data[d5_shsat_data['DBN'] == dbn]
#school_data_d5 = school_data[school_data['Location Code'].isin(d5_shsat_data['DBN'].unique())].copy()
school_data_d5 = school_data[school_data['District'] == 5].copy()
school_data_d5.shape
f,axes=plt.subplots(4,2,figsize=(24,22))

school_data_d5['proficiency'].hist(ax = axes[0,0])
sns.kdeplot(school_data_d5['Economic Need Index'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[0,1])

sns.kdeplot(school_data_d5['Percent ELL'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[1,0])
sns.kdeplot(school_data_d5['Percent Black'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[1,1])

sns.kdeplot(school_data_d5['Percent White'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[2,0])
sns.kdeplot(school_data_d5['Percent Black / Hispanic'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[2,1])
sns.kdeplot(school_data_d5['Percent Asian'].fillna(0), cut = 0, shade  = True, color = 'r', ax = axes[3,0])
parameters = {
    'Number Tested' : calculate_cagr,
    'Level 1%' : calculate_cagr,
    'Level 2%' : calculate_cagr,
    'Level 3%' : calculate_cagr,
    'Level 4%' : calculate_cagr,
    'Level 3+4%' : calculate_cagr
    }

cagr_proficiency_shift = school_results[['Category', 'DBN', 'Grade', 'Number Tested', 'Level 1%', 'Level 2%', 'Level 3%', 'Level 4%', 'Level 3+4%']].groupby(['Category','DBN', 'Grade']).agg(parameters).reset_index(drop = False)

grade = '8' ## Select from 3 to 8
category = 'All Students' ##  Select from 'All Students', 'Asian', 'Black', 'Hispanic', 'White',
              ## 'Econ Disadv', 'Not Econ Disadv', 'Female', 'Male', 'Ever ELL',
              ## 'Never ELL', 'Current ELL', 'Not SWD', 'SWD'
temp_df = cagr_proficiency_shift[(cagr_proficiency_shift['Grade'] == grade) & (cagr_proficiency_shift['Category'] == category)].copy()
temp_df.reset_index(drop = True, inplace = True)

def scale_rows(df):
    return scale(df, axis = 1)

heatmap_input = pd.DataFrame(scale_rows(temp_df[['Level 1%', 'Level 2%', 'Level 3%', 'Level 4%']]), columns = ['Level 1%', 'Level 2%', 'Level 3%', 'Level 4%'])
heatmap_input = heatmap_input.sort_values(by = 'Level 1%')
f,axes=plt.subplots(figsize=(14,4))
sns.heatmap(heatmap_input, cmap = 'coolwarm_r')