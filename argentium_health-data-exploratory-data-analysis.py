import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy

import math

import seaborn as sns

import os



import matplotlib.pyplot as plt

from matplotlib_venn import venn3, venn3_circles



import warnings

warnings.filterwarnings("ignore")
df_cause = pd.read_csv('../input/ChildMortalityCause.csv')
df_cause['AgeLabel'] = df_cause.apply(lambda row: 

                                      str(row['AgeStart']) + '-' + str(row['AgeEnd']),

                                      axis=1)



def get_range(row):

    if row['AgeEnd']==4:

        return 1

    if row['AgeEnd']==9:

        return 2

    return 3

df_cause['AgeRange'] = df_cause.apply(lambda row: 

                                      get_range(row),

                                      axis=1)



# df_cause_melt = pd.melt(df_cause, id_vars=['Year', 'AgeStart', 'AgeEnd', 'AgeRange', 'AgeLabel', 'Cause'], value_vars=['Male', 'Female'])

df_cause.describe()
df_cause['Total'] = df_cause['Male'] + df_cause['Female']

df_cause_abnormal = df_cause[df_cause['Total'] != df_cause['Both Sexes']]



ax = sns.scatterplot(x="Both Sexes", y="Total",

                     data=df_cause)

ax.set_title('Both Sexes vs Total Count')



df_cause_abnormal.head()
ax = sns.scatterplot(y="Total", x="Rate",

                     data=df_cause)

ax.set_title('Rate vs Total Count')
df_cause_highrates = df_cause[df_cause['Rate'] == 223.300000]

df_cause_highrates.head()
df_cause_normalrates = df_cause[df_cause['Rate'] != 223.300000]

df_cause_pneumonia = df_cause_normalrates[(df_cause_normalrates['Cause'] == 'Pneumonia') &

                                        (df_cause['AgeRange'] == 1)]

df_group = df_cause_pneumonia.groupby(['Year', 'AgeRange'])['Rate'].sum().reset_index()

df_group.describe()
df_cause['Population'] = 100000*df_cause['Total']/df_cause['Rate']

df_cause_2010 = df_cause[df_cause['Year']==2010]

df_cause_2010_age4 = df_cause_2010[df_cause_2010['AgeEnd']==4]

df_cause_2010_age4 = df_cause_2010_age4[df_cause_2010_age4['Rate'] != 223.3]



ax = sns.swarmplot(x="Population",

                     data=df_cause_2010_age4)

df_cause_2010_age4.describe()
print(100000*df_cause_highrates['Total'].iloc[0]/df_cause_2010_age4['Population'].mean())
df_cause['Rate'] = df_cause.apply(lambda row: 23.33 if row['Rate']==223.3

                                   else row['Rate'], axis=1)
g = sns.FacetGrid(df_cause, col="AgeLabel", col_wrap=1,

                  hue='Cause', legend_out=True, margin_titles=True)

g = (g.map(plt.scatter, "Year", "Rank").add_legend())

g = (g.map(sns.lineplot, "Year", "Rank").add_legend())
df_cause_total = df_cause.groupby(['Cause'])['Total'].sum().reset_index()

df_cause_total = df_cause_total.sort_values(by=['Total'], ascending = False)



ax = sns.barplot(x="Total", 

                 y='Cause',

                 data=df_cause_total)

ax.set_title("Total Mortality for ages 1-14 in 2001-2010\n(per 100,000)")



df_cause_total.head(40)
cause_categories = {

    'Infection': ['Pneumonia', 

                  'Diarrhea and gastroenteritis',

                 'Meningitis',

                 'Septicemia',

                 'Tuberculosis, all forms',

                 'Dengue',

                 'Measles'],

    'Abnormal Disease': ['Congenital anomalies', 'Malignant neoplasms', 'Leukemia'],

    'Accident': ['Accidental drowning and submersion', 

                 'Accidents', 

                 'Other accidents',

                'Transport accidents'],

    'Organ-specific': ['Other diseases of the nervous system',

                      'Nephritis, nephrotic syndrome and nephrosis',

                      'Diseases of the heart',

                      'Endocrine, nutritional & metabolic diseases',

                      'Diseases of the muscoskeletal system and connective tissue',

                      'Other protein-calorie malnutrition',

                      'Chronic obstructive pulmonary diseases', 

                         'Chronic lower respiratory diseases',

                        'Chronic rheumatic heart disease']

}



def set_category(row):

    for category in cause_categories.keys():

        for cause in cause_categories[category]:

            if row['Cause'] == cause:

                return category

    if row['Cause'] == 'not elsewhere classified' or row['Cause']=='Event of undetermined intent':

        return 'Unknown'

    return 'Others'



df_cause['Category'] = df_cause.apply(lambda row: set_category(row), axis=1)



df_cause_total = df_cause.groupby(['Category'])['Total'].sum().reset_index()

df_cause_total = df_cause_total.sort_values(by=['Total'], ascending = False)



ax = sns.barplot(x="Total", 

                 y='Category',

                 data=df_cause_total)

ax.set_title('Child Mortality Cases\nAccording to Cause Type')



df_cause_total.head()
df_cause_infection = df_cause[df_cause['Category']=='Infection']
df_cause_total = df_cause_infection.groupby(['Cause'])['Total'].sum().reset_index()

df_cause_total = df_cause_total.sort_values(by=['Total'], ascending = False)



ax = sns.barplot(x="Total", 

                 y='Cause',

                 data=df_cause_total)

ax.set_title('Child Mortality\nper Infection Type')
g = sns.FacetGrid(df_cause_infection, col="AgeEnd", row='Cause', margin_titles=True) #, hue="Cause") #,  row="smoker")

g = g.map(plt.bar, "Year", "Rate")

g.set_xticklabels(['2000','2002', '2004', '2006', '2008', '2010'])
df_cause_accident = df_cause[(df_cause['Category']=='Accident')]

df_cause_total = df_cause_accident.groupby(['Cause'])['Total'].sum().reset_index()

df_cause_total = df_cause_total.sort_values(by=['Total'], ascending = False)



ax = sns.barplot(x="Total", 

                 y='Cause',

                 data=df_cause_total)

ax.set_title('Child Mortality due to Accidents')
df_cause_organ = df_cause[df_cause['Category']=='Organ-specific']



def set_organ(row):

    if row['Cause'] == 'Chronic lower respiratory diseases':

        return 'Lungs'

    if row['Cause'] == 'Chronic obstructive pulmonary diseases':

        return 'Lungs'

    if row['Cause'] =='Other protein-calorie malnutrition':

        return 'Endocrine'

    if row['Cause'] =='Endocrine, nutritional & metabolic diseases':

        return 'Endocrine'

    if row['Cause'] =='Other diseases of the nervous system':

        return 'Nervous System'

    if row['Cause'] =='Diseases of the heart':

        return 'Heart'

    if row['Cause'] =='Chronic rheumatic heart disease':

        return 'Heart'

    if row['Cause'] =='Nephritis, nephrotic syndrome and nephrosis':

        return 'Kidney'

    if row['Cause'] =='Diseases of the muscoskeletal system and connective tissue':

        return 'Muscoskeletal System'

    return 'Others'



df_cause_organ['Organ'] = df_cause_organ.apply(lambda row: set_organ(row), axis=1)

df_cause_total = df_cause_organ.groupby(['Organ'])['Total'].sum().reset_index()

df_cause_total = df_cause_total.sort_values(by=['Total'], ascending = False)



ax = sns.barplot(x="Total", 

                 y='Organ',

                 data=df_cause_total)

ax.set_title('Child Mortality due to Non-infectious Organ Diseases')
g = sns.FacetGrid(df_cause_organ, col="AgeLabel", row='Organ', 

                  hue='Cause', legend_out=True, margin_titles=True)

g = (g.map(plt.bar, "Year", "Rate", edgecolor="w").add_legend())
df_age_group = df_cause.groupby(['Cause', 'AgeEnd'])['Rate'].sum().reset_index()

df_values = df_age_group['Cause'].value_counts().reset_index()

df_values.columns = ['Cause', 'Count']



# Select only diseases present in all age groups

df_values_all_ages = df_values[df_values['Count']==3]



for cause in df_values_all_ages['Cause']:

    print(cause + ':')

    df_cause_specific = df_age_group[df_age_group['Cause']==cause]

    df_cause_specific = df_cause_specific.dropna()

    

    # Pearson for linear correlation

    corr, p_value = scipy.stats.pearsonr(df_cause_specific['Rate'], 

                        df_cause_specific['AgeEnd'])

    print("Pearson's Correlation: " + str(corr) + ', p-value:' + str(p_value))

    

    # Spearman for non-linear correlation

    corr, p_value = scipy.stats.spearmanr(df_cause_specific['Rate'], 

                        df_cause_specific['AgeEnd'])

    print("Spearman R Correlation: " + str(corr) + ', p-value:' + str(p_value))

    print()



df_year_group = df_cause.groupby(['Cause', 'Year'])['Rate'].sum().reset_index()

df_values = df_year_group['Cause'].value_counts().reset_index()

df_values.columns = ['Cause', 'Count']



# Select only diseases present in all age groups

df_values_all_years = df_values[df_values['Count']>=9]



for cause in df_values_all_years['Cause']:

    print(cause + ':')

    df_cause_specific = df_year_group[df_year_group['Cause']==cause]

    df_cause_specific = df_cause_specific.dropna()

    

    # Pearson for linear correlation

    corr, p_value = scipy.stats.pearsonr(df_cause_specific['Rate'], 

                        df_cause_specific['Year'])

    print("Pearson's Correlation: " + str(corr) + ', p-value:' + str(p_value))

        

    # Spearman for non-linear correlation

    corr, p_value = scipy.stats.spearmanr(df_cause_specific['Rate'], 

                        df_cause_specific['Year'])

    print("Spearman R Correlation: " + str(corr) + ', p-value:' + str(p_value))

    print()
df_cause_abnormal = df_cause[(df_cause['Category']=='Abnormal Disease')]



df_cause_total = df_cause_abnormal.groupby(['Cause'])['Total'].sum().reset_index()

df_cause_total = df_cause_total.sort_values(by=['Total'], ascending = False)



ax = sns.barplot(x="Total", 

                 y='Cause',

                 data=df_cause_total)

ax.set_title('Child Mortality due to Abnormal Diseases')
df_year_group = df_cause_abnormal.groupby(['Cause', 'Year'])['Total'].sum().reset_index()



g = sns.FacetGrid(df_year_group, 

                  col='Cause', col_wrap=3,

                  margin_titles=True

                  )

g = (g.map(plt.bar, "Year", "Total", edgecolor="w").add_legend())
specialties = {

    'Pulmonology': ['Pneumonia', 

                      'Tuberculosis, all forms',

                   'Chronic obstructive pulmonary diseases', 

                         'Chronic lower respiratory diseases'],

    'Neurology': ['Meningitis', 'Other diseases of the nervous system'],

    'Infectious Disease': ['Diarrhea and gastroenteritis',

                 'Septicemia', 'Measles',

                 'Dengue fever and dengue hemmorhagic fever'],

    'Medical Genetics': ['Congenital anomalies'],

    'Emergency Physician': ['Accidental drowning and submersion', 

                 'Accidents', 

                 'Other accidents',

                'Transport accidents'],

    'Oncology': ['Malignant neoplasms', 'Leukemia'],

    'Cardiology': ['Chronic rheumatic heart disease',

                  'Diseases of the heart'],

    'Nephrology': ['Nephritis, nephrotic syndrome and nephrosis'],

    'Endocrinology': ['Endocrine, nutritional & metabolic diseases'],

    'Orthopedics': ['Diseases of the muscoskeletal system and connective tissue']

}



def set_specialty(row):

    for specialty in specialties.keys():

        for cause in specialties[specialty]:

            if row['Cause'] == cause:

                return specialty

    return 'Unknown'



df_cause['Specialty'] = df_cause.apply(lambda row: set_specialty(row), axis=1)
df_cause_total = df_cause.groupby(['Specialty'])['Total'].sum().reset_index()

df_cause_total = df_cause_total.sort_values(by=['Total'], ascending = False)



ax = sns.barplot(x="Total", 

                 y='Specialty',

                 data=df_cause_total)

ax.set_title('Specialist for each Child Mortality Case')
df_cause_total = df_cause.groupby(['Cause'])['Total'].sum().reset_index()

df_cause_total = df_cause_total.sort_values(by=['Total'], ascending = False)



df_specialty_total = df_cause.groupby(['Specialty'])['Total'].sum().reset_index()

df_specialty_total = df_cause_total.sort_values(by=['Total'], ascending = False)



pneumonia = df_cause_total[df_cause_total['Cause'] == 'Pneumonia']

tuberculosis = df_cause_total[df_cause_total['Cause'] == 'Tuberculosis, all forms']

meningitis = df_cause_total[df_cause_total['Cause'] == 'Meningitis']

dengue = df_cause_total[df_cause_total['Cause'] == 'Dengue']

measles = df_cause_total[df_cause_total['Cause'] == 'Measles']

copd = df_cause_total[df_cause_total['Cause'] == 'Chronic obstructive pulmonary diseases']

clpd = df_cause_total[df_cause_total['Cause'] == 'Chronic lower respiratory diseases']

nervous = df_cause_total[df_cause_total['Cause'] == 'Other diseases of the nervous system']

gastroenteritis = df_cause_total[df_cause_total['Cause'] == 'Diarrhea and gastroenteritis']

septicemia = df_cause_total[df_cause_total['Cause'] == 'Septicemia']



pulmonology_only = copd['Total'].iloc[0] + clpd['Total'].iloc[0]

neurology_only = nervous['Total'].iloc[0]

infectious_only = measles['Total'].iloc[0] + dengue['Total'].iloc[0] + gastroenteritis['Total'].iloc[0]  + septicemia['Total'].iloc[0]

infectious_pulmonology = pneumonia['Total'].iloc[0] + tuberculosis['Total'].iloc[0]

infectious_nervous = meningitis['Total'].iloc[0]



# Abc, aBc, ABc, abC, 

v=venn3(subsets = (pulmonology_only, infectious_only, infectious_pulmonology, 

                   neurology_only, 0, infectious_nervous, 

                   0), 

        set_labels = ('Pulmonology', 'Infectious Disease', 'Neurology'))

c=venn3_circles(subsets = (3673, 23599, 32396, 8555, 0, 5732, 0), linewidth=1, color="grey")
cause_categories = {

    'Infection': ['Pneumonia', 

                  'Diarrhea and gastroenteritis',

                 'Meningitis',

                 'Septicemia',

                 'Tuberculosis, all forms',

                 'Dengue',

                 'Measles'],

    'Abnormal Disease': ['Congenital anomalies', 'Malignant neoplasms', 'Leukemia'],

    'Accident': ['Accidental drowning and submersion', 

                 'Accidents', 

                 'Other accidents',

                'Transport accidents'],

    'Organ-specific': ['Other diseases of the nervous system',

                      'Nephritis, nephrotic syndrome and nephrosis',

                      'Diseases of the heart',

                      'Endocrine, nutritional & metabolic diseases',

                      'Diseases of the muscoskeletal system and connective tissue',

                      'Other protein-calorie malnutrition',

                      'Chronic obstructive pulmonary diseases', 

                         'Chronic lower respiratory diseases',

                        'Chronic rheumatic heart disease']

}



def set_category(row):

    for category in cause_categories.keys():

        for cause in cause_categories[category]:

            if row['Cause'] == cause:

                return category

    if row['Cause'] == 'not elsewhere classified' or row['Cause']=='Event of undetermined intent':

        return 'Unknown'

    return 'Others'



df_cause['Category'] = df_cause.apply(lambda row: set_category(row), axis=1)
def set_subcategory(row):

    if row['Category'] == 'Organ':

        if row['Cause'] == 'Chronic lower respiratory diseases':

            return 'Lungs'

        if row['Cause'] == 'Chronic obstructive pulmonary diseases':

            return 'Lungs'

        if row['Cause'] =='Other protein-calorie malnutrition':

            return 'Endocrine'

        if row['Cause'] =='Endocrine, nutritional & metabolic diseases':

            return 'Endocrine'

        if row['Cause'] =='Other diseases of the nervous system':

            return 'Nervous System'

        if row['Cause'] =='Diseases of the heart':

            return 'Heart'

        if row['Cause'] =='Chronic rheumatic heart disease':

            return 'Heart'

        if row['Cause'] =='Nephritis, nephrotic syndrome and nephrosis':

            return 'Kidney'

        if row['Cause'] =='Diseases of the muscoskeletal system and connective tissue':

            return 'Muscoskeletal System'

        return 'Others'

    if row['Category'] == 'Accident':

        if row['Cause'] == 'Accidental drowning and submersion':

            return 'Drowning'

        if row['Cause'] == 'Transport accidents':

            return 'Transport'

        return 'Others'

    if row['Category'] == 'Abnormal Disease':

        if row['Cause'] == 'Congenital anomalies':

            return 'Congenital anomalies'

        return 'Oncology'

    if row['Category'] == 'Unknown':

        return 'Unknown'



    return row['Cause']



df_cause['Subcategory'] = df_cause.apply(lambda row: set_subcategory(row), axis=1)

df_cause.to_csv('ProcessedData.csv')

df_cause.head()
df_cause_melt = pd.melt(df_cause, id_vars=['Year', 'AgeStart', 'AgeEnd', 'Total', 'Cause'], value_vars=['Male', 'Female'])



df_year_group = df_cause_melt.groupby(['Cause', 'value', 'variable'])['Total'].sum().reset_index()

sns.set(rc={'figure.figsize':(12, 20)})

ax = sns.violinplot(y="Cause", x="value", hue="variable", data=df_year_group, palette="Set3", 

                    split=True,

                   inner="quartile",

                   scale="count")
import statsmodels.api as sm

from statsmodels.formula.api import ols



# Select only diseases present in all age groups

df_values_all_years = df_values[df_values['Count']>=8]



for cause in df_values_all_years['Cause']:

    print(cause +':')

    data = df_year_group[df_year_group['Cause']==cause]

    mod = ols('value ~ variable',

                    data=data).fit()                

    aov_table = sm.stats.anova_lm(mod, typ=2)

    print(aov_table)

    print()
df_cause['x'] = df_cause.apply(lambda row: str(row['AgeEnd']) + str(row['Year']), axis=1)

df_cause['x'].fillna(0, inplace = True)

df_cause['x'] = df_cause['x'].astype(int)

df_pivot = df_cause.pivot("Cause", "x", "Rate")

ax = sns.heatmap(df_pivot, cbar=False, cmap=sns.light_palette("red"), annot=True) #, cmap="YlGnBu"