#numeric
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
import folium

from IPython.display import display

plt.style.use('bmh')
%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.titlepad'] = 25
sns.set_color_codes('pastel')

#Pandas warnings
import warnings
warnings.filterwarnings('ignore')

#system
import os
import re
phil_inc = pd.read_csv('../input/family-income-and-expenditure/Family Income and Expenditure.csv')

region_mapping_phil_inc = {'CAR' : 'car',\
                           'Caraga' : 'caraga',\
                           'VI - Western Visayas' : 'western visayas',\
                           'V - Bicol Region' : 'bicol',\
                           ' ARMM' : 'armm',\
                           'III - Central Luzon' : 'central luzon',\
                           'II - Cagayan Valley' : 'cagayan valley',\
                           'IVA - CALABARZON' : 'calabarzon',\
                           'VII - Central Visayas' : 'central visayas',\
                           'X - Northern Mindanao' : 'northern mindanao',\
                           'XI - Davao Region' : 'davao',\
                           'VIII - Eastern Visayas' : 'eastern visayas',\
                           'I - Ilocos Region' : 'ilocos',\
                           'NCR' : 'ncr',\
                           'IVB - MIMAROPA' : 'mimaropa',\
                           'XII - SOCCSKSARGEN' : 'soccsksargen',\
                           'IX - Zasmboanga Peninsula' : 'zamboanga'}

phil_inc.Region = phil_inc.Region.map(region_mapping_phil_inc)

phil_inc['Main Source of Income'] = phil_inc['Main Source of Income']\
.map({'Wage/Salaries' : 'main_inc_wage',\
      'Other sources of Income' : 'main_inc_other',\
      'Enterpreneurial Activities' : 'main_inc_entrepreneur'})

phil_inc_extract = phil_inc.join(pd.get_dummies(phil_inc['Main Source of Income']))

phil_inc_extract.drop(['Main Source of Income', 'Bread and Cereals Expenditure',\
                       'Total Rice Expenditure', 'Meat Expenditure',\
                       'Total Fish and  marine products Expenditure',\
                       'Fruit Expenditure', 'Vegetables Expenditure'],\
                      axis = 1,\
                      inplace = True)

phil_inc_extract['non_essential_expenses'] = phil_inc_extract['Restaurant and hotels Expenditure'] +\
                                             phil_inc_extract['Alcoholic Beverages Expenditure'] +\
                                             phil_inc_extract['Tobacco Expenditure']

phil_inc_extract.drop(['Restaurant and hotels Expenditure',\
                       'Alcoholic Beverages Expenditure',\
                       'Tobacco Expenditure'],\
                      axis = 1,\
                      inplace = True)

phil_inc_extract['Household Head Sex'] = phil_inc_extract['Household Head Sex']\
.map({'Female' : 1,\
      'Male' : 0})

phil_inc_extract.rename(columns = {'Household Head Sex' : 'house_head_sex_f'}, inplace = True)

single_civil_statuses = ['Single', 'Widowed', 'Divorced/Separated', 'Annulled', 'Unknown']

phil_inc_extract['Household Head Marital Status'] = ['house_head_single'\
                                                     if s in single_civil_statuses\
                                                     else 'house_head_partner'\
                                                     for s\
                                                     in phil_inc_extract['Household Head Marital Status']]

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Household Head Marital Status']))

phil_inc_extract.drop(['Household Head Marital Status'],\
                      axis = 1,\
                      inplace = True)

illiterate = ['No Grade Completed', 'Preschool']
primary_ed = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Elementary Graduate']
secondary_ed = ['First Year High School', 'Second Year High School', 'Third Year High School', 'High School Graduate']

def get_education_level(ed_string):
    if ed_string in illiterate:
        return 'house_head_illiterate'
    elif ed_string in primary_ed:
        return 'house_head_primary_ed'
    elif ed_string in secondary_ed:
        return 'house_head_secondary_ed'
    else:
        return 'house_head_tertiary_ed'

phil_inc_extract['Household Head Highest Grade Completed'] = [get_education_level(e)\
                                                              for e\
                                                              in phil_inc_extract['Household Head Highest Grade Completed']]

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Household Head Highest Grade Completed']))

phil_inc_extract.drop(['Household Head Highest Grade Completed'],\
                      axis = 1,\
                      inplace = True)

phil_inc_extract.rename(columns = {'Household Head Job or Business Indicator' : 'house_head_empl'}, inplace = True)
phil_inc_extract.house_head_empl = phil_inc_extract.house_head_empl.map({'With Job/Business' : 1,\
                                                                         'No Job/Business' : 0})

phil_inc_extract['Household Head Occupation'].fillna('', inplace = True)

unque_occupations = phil_inc_extract['Household Head Occupation'].unique().tolist()

farmer_flag = re.compile(r'farm', re.I)

phil_inc_extract['house_head_farmer'] = [1 if re.findall(farmer_flag, w) else 0 for w in phil_inc_extract['Household Head Occupation']]

phil_inc_extract.drop(['Household Head Occupation', 'Household Head Class of Worker'], axis = 1, inplace = True)

phil_inc_extract['Type of Household'] = phil_inc_extract['Type of Household']\
.map({'Extended Family' : 'house_ext_family',
      'Single Family' : 'house_singl_family',
      'Two or More Nonrelated Persons/Members' : 'house_mult_family'})

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Type of Household']))

phil_inc_extract.drop(['Type of Household'], axis = 1, inplace = True)

phil_inc_extract.rename(columns = {'Total Number of Family members' : 'num_family_members',
                                   'Members with age less than 5 year old' : 'num_children_younger_5',
                                   'Members with age 5 - 17 years old' : 'num_children_older_5',
                                   'Total number of family members employed' : 'num_family_members_employed'},
                        inplace = True)

phil_inc_extract['Type of Building/House'] = phil_inc_extract['Type of Building/House']\
.map({'Single house' : 'house',
      'Duplex' : 'duplex',
      'Commercial/industrial/agricultural building' : 'living_at_workplace',
      'Multi-unit residential' : 'residential_block',
      'Institutional living quarter' : 'institutional_housing',
      'Other building unit (e.g. cave, boat)' : 'other_housing'})

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Type of Building/House']))
phil_inc_extract.drop(['Type of Building/House'], axis = 1, inplace = True)

phil_inc_extract['Type of Roof'] = phil_inc_extract['Type of Roof']\
.map({'Strong material(galvanized,iron,al,tile,concrete,brick,stone,asbestos)' : 'roof_material_strong',
      'Light material (cogon,nipa,anahaw)' : 'roof_material_light',
      'Mixed but predominantly strong materials' : 'roof_material_mostly_strong',
      'Mixed but predominantly light materials' : 'roof_material_mostly_light',
      'Salvaged/makeshift materials' : 'roof_material_makeshift',
      'Mixed but predominantly salvaged materials' : 'roof_material_mostly_makeshift',
      'Not Applicable' : 'no_roof'})

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Type of Roof']))
phil_inc_extract.drop(['Type of Roof'], axis = 1, inplace = True)

phil_inc_extract['Type of Walls'] = phil_inc_extract['Type of Walls']\
.map({'Strong' : 'wall_material_strong',
      'Light' : 'wall_material_light',
      'Quite Strong' : 'wall_material_quite_strong',
      'Very Light' : 'wall_material_quite_light',
      'Salvaged' : 'wall_material_salvaged',
      'NOt applicable' : 'no_walls'})

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Type of Walls']))
phil_inc_extract.drop(['Type of Walls'], axis = 1, inplace = True)

phil_inc_extract.rename(columns = {'House Floor Area' : 'house_area',
                                   'House Age' : 'house_age',
                                   'Number of bedrooms' : 'num_bedrooms'},
                        inplace = True)

phil_inc_extract['Toilet Facilities'] = phil_inc_extract['Toilet Facilities']\
.map({'Water-sealed, sewer septic tank, used exclusively by household' : 'ws_septic_toiled',
      'Water-sealed, sewer septic tank, shared with other household' : 'ws_septic_toiled',
      'Closed pit' : 'septic_toiled',
      'Water-sealed, other depository, used exclusively by household' : 'ws_other_toilet',
      'Open pit' : 'septic_toiled',
      'Water-sealed, other depository, shared with other household' : 'ws_other_toilet',
      'None' : 'no_toilet',
      'Others' : 'other_toilet'})

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Toilet Facilities']))
phil_inc_extract.drop(['Toilet Facilities', 'Tenure Status'], axis = 1, inplace = True)

running_water = ['Own use, faucet, community water system', 'Shared, faucet, community water system']

phil_inc_extract['running_water'] = [1 if i in running_water else 0 for i in phil_inc_extract['Main Source of Water Supply']]

phil_inc_extract.drop(['Main Source of Water Supply'], axis = 1, inplace = True)

phil_inc_extract['num_electronics'] = phil_inc_extract['Number of Television'] +\
phil_inc_extract['Number of CD/VCD/DVD'] +\
phil_inc_extract['Number of Component/Stereo set'] +\
phil_inc_extract['Number of Personal Computer']

phil_inc_extract.drop(['Number of Television',\
                       'Number of CD/VCD/DVD',\
                       'Number of Component/Stereo set',\
                       'Number of Personal Computer'],\
                      axis = 1,\
                      inplace = True)

phil_inc_extract['num_comm_devices'] = phil_inc_extract['Number of Landline/wireless telephones'] +\
phil_inc_extract['Number of Cellular phone']

phil_inc_extract.drop(['Number of Landline/wireless telephones',\
                       'Number of Cellular phone'],\
                      axis = 1,\
                      inplace = True)

phil_inc_extract['num_vehicles'] = phil_inc_extract['Number of Car, Jeep, Van'] +\
phil_inc_extract['Number of Motorized Banca'] +\
phil_inc_extract['Number of Motorcycle/Tricycle']

phil_inc_extract.drop(['Number of Car, Jeep, Van',\
                       'Number of Motorized Banca',\
                       'Number of Motorcycle/Tricycle'],\
                      axis = 1,\
                      inplace = True)

phil_inc_extract.rename(columns = {'Total Household Income' : 'household_income',
                                   'Region' : 'region',
                                   'Total Food Expenditure' : 'food_expenses',
                                   'Agricultural Household indicator' : 'agricultural_household',
                                   'Clothing, Footwear and Other Wear Expenditure' : 'clothing_expenses',
                                   'Housing and water Expenditure' : 'house_and_water_expenses',
                                   'Imputed House Rental Value' : 'house_rental_value',
                                   'Medical Care Expenditure' : 'medical_expenses',
                                   'Transportation Expenditure' : 'transport_expenses',
                                   'Communication Expenditure' : 'comm_expenses',
                                   'Education Expenditure' : 'education_expenses',
                                   'Miscellaneous Goods and Services Expenditure' : 'misc_expenses',
                                   'Special Occasions Expenditure' : 'special_occasion_expenses',
                                   'Crop Farming and Gardening expenses' : 'farming_gardening_expenses',
                                   'Total Income from Entrepreneurial Acitivites' : 'income_from_entrepreneur_activities',
                                   'Household Head Age' : 'house_head_age',
                                   'Electricity' : 'electricity',
                                   'Number of Refrigerator/Freezer' : 'num_refrigerator',
                                   'Number of Washing Machine' : 'num_wash_machine',
                                   'Number of Airconditioner' : 'num_ac',
                                   'Number of Stove with Oven/Gas Range' : 'num_stove'},
                        inplace = True)

phil_inc_extract.loc[0]
phil_inc_extract.to_csv('philippines_census_data_cl.csv')
phil_inc_extract['est_family_employment'] = [fam_mem_empl / (fam_mem - child_younger_5 - child_older_5)\
                                             if (fam_mem - child_younger_5 - child_older_5) != 0\
                                             else 0\
                                             for fam_mem_empl, fam_mem, child_younger_5, child_older_5\
                                             in zip(phil_inc_extract.num_family_members_employed,\
                                                    phil_inc_extract.num_family_members,\
                                                    phil_inc_extract.num_children_younger_5,\
                                                    phil_inc_extract.num_children_older_5)]

phil_inc_extract['no_electricity'] = 1 - phil_inc_extract.electricity
phil_inc_extract['no_running_water'] = 1 - phil_inc_extract.running_water
phil_inc_grouped_r_s = phil_inc_extract.groupby(by = ['region', 'house_head_sex_f']).mean().reset_index()
phil_inc_grouped_r = phil_inc_extract.groupby(by = ['region']).mean().reset_index()
sns.barplot(x = 'region', y = 'est_family_employment', data = phil_inc_grouped_r_s, color = 'b')
plt.title('Estimated Average Family Employment by Region')
plt.xticks(rotation = 45)
x_ticks = np.arange(len(phil_inc_grouped_r.region))
plt.xticks(x_ticks, phil_inc_grouped_r.region, rotation = 45)
for f, i in zip(('no_toilet', 'no_running_water', 'no_electricity'), (0, 1, 2)):
    plt.bar(x_ticks + (0.25 * i), phil_inc_grouped_r[f], align = 'center', label = f, width = 0.25)
plt.legend()
plt.xlabel('region')
plt.ylabel('percentage')
plt.title('Various Poverty Indicators by Region')
sns.barplot(x = 'region', y = 'household_income', data = phil_inc_grouped_r_s, color = 'b')
plt.title('Household Income by Region')
plt.xticks(rotation = 45)
fig = plt.figure(figsize = (18, 10), dpi = 120)
fig.subplots_adjust(hspace = 0.3, wspace = 0.35)

ax1 = fig.add_subplot(2, 3, 1)
sns.distplot(phil_inc_extract.household_income[phil_inc_extract.household_income < 1000000])

ax2 = fig.add_subplot(2, 3, 2)
sns.distplot(phil_inc_extract.food_expenses[phil_inc_extract.food_expenses < 400000])
plt.scatter(phil_inc_extract.household_income[\
                                              (phil_inc_extract.household_income < 1000000) &\
                                              (phil_inc_extract.food_expenses < 400000)],\
            phil_inc_extract.food_expenses[\
                                              (phil_inc_extract.household_income < 1000000) &\
                                              (phil_inc_extract.food_expenses < 400000)])

plt.xlabel('household income')
plt.ylabel('food expenses')
plt.title('Household Income by Food Expenses')
sns.barplot(x = 'num_comm_devices', y = 'household_income', data = phil_inc_extract, color = 'b')
plt.title('Household Income by Number of Communication Devices')