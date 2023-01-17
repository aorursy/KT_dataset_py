# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Press "shift + Enter" to run the cell in .ipynb file



import warnings

warnings.simplefilter(action='ignore') # stop warnings 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
firms = pd.read_csv("../input/base_etablissement_par_tranche_effectif.csv")

geography = pd.read_csv("../input/name_geographic_information.csv")

salary = pd.read_csv("../input/net_salary_per_town_categories.csv")

population = pd.read_csv("../input/population.csv")
firms.head()
firms.info()
geography.head()
geography.info()
salary.head()
salary.info()
population.head()
population.info()
geography.drop_duplicates(subset=["code_insee"], keep="first", inplace=True)
firms["CODGEO"] = firms["CODGEO"].astype(str)

geography['code_insee'] = geography['code_insee'].astype(str)

salary["CODGEO"] = salary["CODGEO"].astype(str)

population["CODGEO"] = population["CODGEO"].astype(str)

merge1 = firms.merge(geography,left_on= "CODGEO",right_on="code_insee")

merge2 = merge1.merge(salary, on= "CODGEO")

merge3 = merge2.merge(population, on= "CODGEO")
merge3.values.shape
merge3.columns
data = merge3.drop(columns=['LIBGEO_x', 'REG', 'DEP','EU_circo', 'code_région', 'nom_région', 'chef.lieu_région',

'numéro_département', 'nom_département', 'préfecture','numéro_circonscription','codes_postaux', 'code_insee',

                            'éloignement', 'LIBGEO_y','NIVGEO','LIBGEO'])
data.columns
groupby_columns = list(data.columns[:-4])

data2 = data.reset_index().groupby(groupby_columns, as_index= False).sum()
data2 = data2.drop(columns = ['index', 'MOCO', 'AGEQ80_17','SEXE'])
data2.columns
data2.head()
data2.values.shape
data2['Micro_firms'] = data2['E14TS1'] + data2['E14TS6']

data2['Small_firms'] = data2['E14TS10'] + data2['E14TS20']

data2['Medium_firms'] = data2['E14TS50'] + data2['E14TS100']

data2['Large_firms'] = data2['E14TS200'] + data2['E14TS500']

data2 = data2.drop(columns=['E14TS0ND', 'E14TS1', 'E14TS6', 'E14TS10','E14TS20', 'E14TS50', 'E14TS100', 'E14TS200', 'E14TS500','E14TS0ND'])

data2 = data2.rename(columns = {'E14TST':'total_firms','NB':'total_population','nom_commune':'Town'})
data2.columns
data2=data2.rename(columns = {'SNHM14':'mean_salary',

                              'SNHMC14':'mean_executive_salary',

                              'SNHMP14':'mean_middle_manager_salary',

                              'SNHME14':'mean_employee_salary',

                              'SNHMO14':'mean_worker_salary',

                              'SNHMF14':'mean_female_salary',

                              'SNHMFC14':'mean_female_executive_salary',

                              'SNHMFP14':'mean_female_middle_manager_salary',

                              'SNHMFE14':'mean_female_employee_salary',

                              'SNHMFO14':'mean_female_worker_salary',

                              'SNHMH14':'mean_male_salary',

                              'SNHMHC14':'mean_male_executive_salary',

                              'SNHMHP14':'mean_male_middle_manager_salary',

                              'SNHMHE14':'mean_male_employee_salary',

                              'SNHMHO14':'mean_male_worker_salary',

                              'SNHM1814':'mean_young_age_salary',

                              'SNHM2614':'mean_medium_age_salary',

                              'SNHM5014':'mean_old_age_salary',

                              'SNHMF1814':'mean_young_female_salary',

                              'SNHMF2614':'mean_medium_female_salary',

                              'SNHMF5014':'mean_old_female_salary',

                              'SNHMH1814':'mean_young_male_salary',

                              'SNHMH2614':'mean_medium_male_salary',

                              'SNHMH5014':'mean_old_male_salary',

                              })
data2.columns
data2.to_csv("Final_Data.csv")