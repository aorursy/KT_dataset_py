import pandas as pd

from matplotlib import pyplot as plt

import numpy as np

import seaborn as sns
file_name = "COVID19_2020_open_line_list.xlsx"

root_folder = "/kaggle/input/covid19-open-access-data/"



data_ex = pd.read_excel(

    root_folder + file_name,

    sheet_name="outside_Hubei",

    index_col="ID",

    usecols=range(33) # there are a couple of empty columns in the back

)



data_in = pd.read_excel(

    root_folder + file_name,

    sheet_name="Hubei",

    index_col="ID",

    usecols=range(32) # there are a couple of empty columns in the back

)
bounding_box = (

    np.floor(data_ex.longitude.min()),

    np.ceil(data_ex.longitude.max()),

    np.floor(data_ex.latitude.min()),

    np.ceil(data_ex.latitude.max())

)



bounding_box = (-180, 180, -90, 90)
plt.figure(figsize=(20,16))



plt.scatter(

    data_ex.longitude,

    data_ex.latitude,

    marker="o",

    zorder=1,

    c="r",

    alpha=0.6,

    s=50

)



plt.xlim(bounding_box[0], bounding_box[1])

plt.ylim(bounding_box[2], bounding_box[3])



plt.imshow(

    plt.imread('/kaggle/input/covid19-open-access-data/map.png'),

    zorder=0,

    extent=bounding_box,

    aspect='equal',

    alpha=0.5

)



plt.title('Plotting Spatial data about COVID-19')

plt.legend(loc=0)



plt.show()
country = data_ex.country.value_counts()

lat = data_ex.groupby(by="country")["latitude"].mean()

lon = data_ex.groupby(by="country")["longitude"].mean()

data_agg_countra = pd.DataFrame(country).merge(lat, left_index=True, right_index=True).merge(lon,  left_index=True, right_index=True)
plt.figure(figsize=(20,16))

                                                                                                                                                          

plt.scatter(

    data_agg_countra.longitude,

    data_agg_countra.latitude,

    marker="o",

    zorder=1,

    c="r",

    alpha=0.6,

    s=data_agg_countra.country

)



plt.xlim(bounding_box[0], bounding_box[1])

plt.ylim(bounding_box[2], bounding_box[3])



plt.imshow(

    plt.imread('/kaggle/input/covid19-open-access-data/map.png'),

    zorder=0,

    extent=bounding_box,

    aspect='equal',

    alpha=0.5

)



plt.title('Plotting Spatial data about COVID-19')



plt.show()
plt.figure(figsize=(20,8))



n = 20



plt.barh(

    width=data_ex.country.value_counts().head(n).sort_values(ascending=True),

    y=data_ex.country.value_counts().head(n).sort_values(ascending=True).index,

    color="k"

)

plt.title(f"Top {n} countries in COVID-19 infections")

plt.xlabel("number of infections")

plt.ylabel("country")

plt.show()