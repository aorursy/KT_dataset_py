import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.tools as tls



%matplotlib inline
df = pd.read_csv("../input/environment-impact-of-food-production/Food_Production.csv")

print(df.shape)



df.head()
from collections import Counter



rows = df.shape[0]

columns = df.shape[1]

print("The train dataset contains {0} rows and {1} columns".format(rows, columns))



Counter(df.dtypes.values)
df.info()
df_info= pd.DataFrame({"Dtype": df.dtypes, "Unique": df.nunique(), "Missing%": (df.isnull().sum()/df.shape[0])*100})

df_info
plt.figure(figsize=(8,5))

plt.scatter(range(df.shape[0]), np.sort(df.Total_emissions.values), s= 50)

plt.xlabel('index', fontsize=12)

plt.ylabel('Total Emissions', fontsize=12)

plt.show()
food_df= df.groupby("Food product")['Total_emissions'].sum().reset_index()



trace = go.Scatter(

    y = food_df.Total_emissions,

    x = food_df["Food product"],

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = food_df.Total_emissions*2,

        color = food_df.Total_emissions,

        colorscale='Portland',

        showscale=True

    )

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Total Emissions by Foods',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Total Emissions',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatterplot')
temp_df= df.sort_values(by= "Total_emissions", ascending= True).iloc[:,:8]





fig, ax = plt.subplots(figsize=(15,20))

sns.set()

temp_df.set_index('Food product').plot(kind='barh', stacked=True, ax= ax)

plt.xlabel("Greenhouse gas Emissions")

plt.show()
land_df= df.dropna().sort_values(by= 'Land use per 1000kcal (m² per 1000kcal)', ascending= True)[['Food product','Land use per 1000kcal (m² per 1000kcal)']]



fig, ax = plt.subplots(figsize=(15,10))

sns.set()

land_df.set_index('Food product').plot(kind='barh', stacked=True, ax= ax, color= "sienna")

plt.xlabel("Land Use per 100 Kcal")

plt.title("Land Use by Foods per 1000 Kcal\n", size= 20)

plt.show()
land_df= df.dropna().sort_values(by= 'Land use per kilogram (m² per kilogram)', ascending= True)[['Food product',

       'Land use per kilogram (m² per kilogram)']]



fig, ax = plt.subplots(figsize=(15,10))

sns.set()

land_df.set_index('Food product').plot(kind='barh', stacked=True, ax= ax, color= "sienna")

plt.xlabel("Land Use per Kg")

plt.title("Land Use by Foods per Kg \n", size= 20)

plt.show()
water_df= df.dropna().sort_values(by= 'Freshwater withdrawals per 1000kcal (liters per 1000kcal)', ascending= True)[['Food product',

       'Freshwater withdrawals per 1000kcal (liters per 1000kcal)']]



fig, ax = plt.subplots(figsize=(15,10))

sns.set()

water_df.set_index('Food product').plot(kind='barh', stacked=True, ax= ax, color= "lightblue")

plt.xlabel("Fresh Water Use litres per 1000kcal")

plt.title("Fresh Water Use by Foods per 1000Kcal \n", size= 20)

plt.show()
water_df= df.dropna().sort_values(by= 'Freshwater withdrawals per kilogram (liters per kilogram)', ascending= True)[['Food product',

       'Freshwater withdrawals per kilogram (liters per kilogram)']]



fig, ax = plt.subplots(figsize=(15,10))

sns.set()

water_df.set_index('Food product').plot(kind='barh', stacked=True, ax= ax, color= "lightblue")

plt.xlabel("Fresh Water Use litres per Kg")

plt.title("Fresh Water Use by Foods per Kg \n", size= 20)

plt.show()
emission_df= df.dropna().sort_values(by= 'Greenhouse gas emissions per 1000kcal (kgCO₂eq per 1000kcal)', ascending= True)[['Food product',

       'Greenhouse gas emissions per 1000kcal (kgCO₂eq per 1000kcal)']]



fig, ax = plt.subplots(figsize=(15,10))

sns.set()

emission_df.set_index('Food product').plot(kind='barh', stacked=True, ax= ax, color= "darkgreen")

plt.xlabel("Greenhouse gas emission per 1000kcal")

plt.title("Greenhouse Gas Emission Per 1000kcal\n", size= 20)

plt.show()
emission_df= df.dropna().sort_values(by= 'Greenhouse gas emissions per 100g protein (kgCO₂eq per 100g protein)', ascending= True)[['Food product',

       'Greenhouse gas emissions per 100g protein (kgCO₂eq per 100g protein)']]



fig, ax = plt.subplots(figsize=(15,10))

sns.set()

emission_df.set_index('Food product').plot(kind='barh', stacked=True, ax= ax, color= "darkgreen")

plt.xlabel("Greenhouse gas emission per 100g protein")

plt.title("Greenhouse Gas Emission Per 100g Protein\n", size= 20)

plt.show()
# comparing different foods by scarcity-weighted water required to produce 1 kg food



scarcity_df= df.dropna().sort_values(by= 'Scarcity-weighted water use per kilogram (liters per kilogram)', ascending= True)[['Food product',

       'Scarcity-weighted water use per kilogram (liters per kilogram)']]



fig, ax = plt.subplots(figsize=(15,10))

sns.set()

scarcity_df.set_index('Food product').plot(kind='barh', stacked=True, ax= ax)

plt.xlabel('Scarcity-weighted water use per kg')

plt.title('Scarcity-Weighted Water Use Per Kg', size= 20)

plt.show()
# comparing different foods by scarcity-weighted water in terms of nutritional values



scarcity_df= df.dropna().sort_values(by= 'Scarcity-weighted water use per 100g protein (liters per 100g protein)', ascending= True)[['Food product',

       'Scarcity-weighted water use per 100g protein (liters per 100g protein)']]



fig, ax = plt.subplots(figsize=(15,10))

sns.set()

scarcity_df.set_index('Food product').plot(kind='barh', stacked=True, ax= ax)

plt.xlabel('Scarcity-weighted water use per 100g protein')

plt.title('Scarcity-weighted water use per 100g protein', size= 20)

plt.show()
#comparing eutrophication emissions per 1000kcal



eutrophication_df= df.dropna().sort_values(by= 'Eutrophying emissions per 1000kcal (gPO₄eq per 1000kcal)', ascending= True)[['Food product',

       'Eutrophying emissions per 1000kcal (gPO₄eq per 1000kcal)']]



fig, ax = plt.subplots(figsize=(15,10))

sns.set()

eutrophication_df.set_index('Food product').plot(kind='barh', stacked=True, ax= ax, color= "black")

plt.xlabel('Eutrophication emissions Per 1000kcal')

plt.title('Eutrophication Emissions Per 1000kcal \n', size= 20)

plt.show()
#comparing eutrophication emissions of different foods required to produce 1 kg food



eutrophication_df= df.dropna().sort_values(by= 'Eutrophying emissions per kilogram (gPO₄eq per kilogram)', ascending= True)[['Food product',

       'Eutrophying emissions per kilogram (gPO₄eq per kilogram)']]



fig, ax = plt.subplots(figsize=(15,10))

sns.set()

eutrophication_df.set_index('Food product').plot(kind='barh', stacked=True, ax= ax, color= "black")

plt.xlabel('Eutrophication emissions Per Kg')

plt.title('Eutrophication Emissions Per Kg \n', size= 20)

plt.show()
plt.figure(figsize=(10,10))

temp_series = df.groupby('Food product')['Land use change'].sum()

labels = (np.array(temp_series.index))

sizes = (np.array((temp_series / temp_series.sum())*100))

plt.pie(sizes, labels=labels, 

        autopct='%1.1f%%', startangle=200)

plt.title("Food distribution by emissions via Transport", fontsize=20)

plt.show()
plt.figure(figsize=(10,10))

temp_series = df.groupby('Food product')['Transport'].sum()

labels = (np.array(temp_series.index))

sizes = (np.array((temp_series / temp_series.sum())*100))

plt.pie(sizes, labels=labels, 

        autopct='%1.1f%%', startangle=200)

plt.title("Food distribution by emissions via Transport", fontsize=20)

plt.show()
# To check the relation among different attributes of foods



corrmat = df.corr(method='pearson')

f, ax = plt.subplots(figsize=(20, 15))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=1., square=True, annot= True,cmap= "viridis")

plt.title("Correlation between variables \n", fontsize=20)

plt.show()