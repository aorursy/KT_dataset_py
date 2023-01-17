import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import os


kiva_loans = pd.read_csv("../input/kiva_loans.csv")
loan_theme_ids = pd.read_csv("../input/loan_theme_ids.csv")
kiva_region_location = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan_themes_by_region = pd.read_csv("../input/loan_themes_by_region.csv")
kiva_loans.head(1)
loan_theme_ids.head(1)
kiva_region_location.head(1)
loan_themes_by_region.head(1)
map = Basemap()

lat = kiva_region_location["lat"].tolist()
lon = kiva_region_location["lon"].tolist()

x,y = map(lon,lat)

plt.figure(figsize=(15,8))
map.plot(x,y,"go",color ="orange",markersize =6,alpha=.6)
map.shadedrelief()
kiva_region_location.world_region.value_counts()
plt.figure(figsize=(13,9))
sectors = kiva_loans['sector'].value_counts()
sns.barplot(y=sectors.index, x=sectors.values)
plt.xlabel('Number of loans', fontsize=20)
plt.ylabel("Sectors", fontsize=20)
plt.title("Number of loans per sector", size=30)
plt.show()
import numpy
plt.figure(figsize=(13,9))

sectors = kiva_loans['country'].value_counts()
sec = sectors[:10]
sns.barplot(y=sec.index, x=sec.values)
plt.xlabel('USD', fontsize=20)
plt.ylabel("Conutry", fontsize=20)
plt.title("Total loans value by country", size=30)
plt.show()
kiva_loans["loan_amount"].min()

kiva_loans["loan_amount"].max()
gender = kiva_loans[['borrower_genders']]
gender = gender[gender["borrower_genders"].notnull()]
gender = gender["borrower_genders"].str.upper()
gender = gender.str.replace("FEMALE","F")
gender = gender.str.replace("MALE","M")
gender = pd.DataFrame(gender)
gender["F"] = gender["borrower_genders"].str.count("F")
gender["M"] = gender["borrower_genders"].str.count("M")
gender_list = [sum(gender["M"]),sum(gender["F"])]
gender_label = ["MALE","FEMALE"]
plt.figure(figsize=(7,7))
plt.pie(gender_list,labels=gender_label,shadow=True,colors = ["lightgrey","orange"],autopct="%1.0f%%",
        explode=[0,.1],wedgeprops={"linewidth":2,"edgecolor":"k"})
plt.title("GENDER DISTRIBUTION")
