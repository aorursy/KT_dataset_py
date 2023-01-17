#for ViSenze

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd #geopandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
firmsize = pd.read_csv('../input/base_etablissement_par_tranche_effectif.csv')
net_salary = pd.read_csv('../input/net_salary_per_town_categories.csv')
geography = pd.read_csv('../input/name_geographic_information.csv')
population = pd.read_csv('../input/population.csv')
geography_map = gpd.read_file('../input/departements.geojson')
home_map = gpd.read_file('../input/communes.geojson')
salary_copy = salary.copy()
salary = net_salary[net_salary["CODGEO"].apply(lambda x: str(x).isdigit())]
salary["CODGEO"] = salary["CODGEO"].astype(int)
salary = salary[['CODGEO', 'SNHM14', 'SNHMH14', 'SNHMF14']]

salary = salary.rename(columns = {"CODGEO": "area_code",
                                  "SNHM14": "mean_salary",
                                  "SNHMH14": "mean_salary_male",
                                  "SNHMF14": "mean_salary_female",
                                                         }
                                              )
firmsize = firmsize.drop(["REG", "DEP", "E14TS0ND", "E14TS500"], axis = 1)
firmsize = firmsize.rename(columns = {"CODGEO": "area_code",
                                                          "LIBGEO": "location",
                                                          "E14TST": "total_size"
                                                     }
                                          )                                                        
firmsize = firmsize[firmsize["area_code"].apply(lambda x: str(x).isdigit())]
firmsize["area_code"] = firmsize["area_code"].astype(int)

salary_firmsize = salary.merge(firmsize, how="inner", on="area_code")
salary_firmsize_dupe = salary_firmsize.copy()
salary_firmsize_dupe["total_size"] = salary_firmsize_dupe["E14TS1"] +salary_firmsize_dupe["E14TS6"] + salary_firmsize_dupe["E14TS10"] + salary_firmsize_dupe["E14TS20"] + salary_firmsize_dupe["E14TS50"] + salary_firmsize_dupe["E14TS100"] + salary_firmsize_dupe["E14TS200"]
salary_firmsize_dupe.head(5)
salary_firmsize["E14TS1"] = salary_firmsize["E14TS1"] * 3
salary_firmsize["E14TS6"] = salary_firmsize["E14TS6"] * 8
salary_firmsize["E14TS10"] = salary_firmsize["E14TS10"] * 15
salary_firmsize["E14TS20"] = salary_firmsize["E14TS20"] * 35
salary_firmsize["E14TS50"] = salary_firmsize["E14TS50"] * 75
salary_firmsize["E14TS100"] = salary_firmsize["E14TS100"] * 150
salary_firmsize["E14TS200"] = salary_firmsize["E14TS200"] * 350
salary_firmsize["total_size"] = salary_firmsize["E14TS1"] +salary_firmsize["E14TS6"] + salary_firmsize["E14TS10"] + salary_firmsize["E14TS20"] + salary_firmsize["E14TS50"] + salary_firmsize["E14TS100"] + salary_firmsize["E14TS200"]
salary_firmsize["total_size"] = salary_firmsize["total_size"] / salary_firmsize_dupe["total_size"]
salary_firmsize.head(5)
salary_firmsize = salary_firmsize.drop(["E14TS1", "E14TS6", "E14TS10", "E14TS20", "E14TS50", "E14TS100", "E14TS200"], axis = 1)
salary_firmsize = salary_firmsize.rename(columns = {"total_size": "avg_firm_size"})
salary_firmsize.head(3)
sns.lmplot(x="avg_firm_size", y="mean_salary", data=salary_firmsize);
sns.lmplot(x="avg_firm_size", y="mean_salary_male", data=salary_firmsize);
sns.lmplot(x="avg_firm_size", y="mean_salary_female", data=salary_firmsize);
