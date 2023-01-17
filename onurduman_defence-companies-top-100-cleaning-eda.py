import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import plotly.graph_objects as go

import plotly.express as px



import pycountry
defence_companies = pd.read_csv("/kaggle/input/defence-companies-top-100-for-each-year-from-2005/defence_companies_from_2005.csv")

df = defence_companies.copy()

df.head()
df.info()
df.Country.unique()
a = df.Company.str.replace("\d","") # To Remove Numbers

a = a.str.replace("[^\w\s]","") # To Remove Punctuations

a = a.str.strip() # To Remove Space Characters



df["Company"] = a
df["Country"] = df["Country"].str.replace("So Africa","South Africa").replace("S. Africa","South Africa")

df["Country"] = df["Country"].str.replace("U.S.","USA").replace("US","USA").replace("U.S","USA")

df["Country"] = df["Country"].str.replace("U.K.","United Kingdom").replace("UK","United Kingdom")

df["Country"] = df["Country"].str.replace("So. Korea","South Korea").replace("Korea","South Korea")

df["Country"] = df["Country"].str.replace("canada","Canada").replace("Swiss","Switzerland")

b = df[df.Country == "Netherlands/France"]

b.Country = ["France"] * 4 

df = df.append(b, ignore_index=True)

df.Country = df.Country.apply(lambda x: x.split("/")[0] if "/" in x else x)

df["Defense_Revenue_From_A_Year_Ago(in millions)"] = df["Defense_Revenue_From_A_Year_Ago(in millions)"].str.replace(",","")

df["Defense_Revenue_From_A_Year_Ago(in millions)"] = df["Defense_Revenue_From_A_Year_Ago(in millions)"].str.replace("$","")

df.iloc[1072,5] = "778.09"

df["Defense_Revenue_From_A_Year_Ago(in millions)"] = df["Defense_Revenue_From_A_Year_Ago(in millions)"].astype(np.float32)

df["Defense_Revenue_From_Two_Years_Ago(in millions)"] = df["Defense_Revenue_From_Two_Years_Ago(in millions)"].str.replace(",","")

df["Defense_Revenue_From_Two_Years_Ago(in millions)"] = df["Defense_Revenue_From_Two_Years_Ago(in millions)"].str.replace("$","")

df["Defense_Revenue_From_Two_Years_Ago(in millions)"] = df["Defense_Revenue_From_Two_Years_Ago(in millions)"].apply(lambda x: np.nan if (x=="~" or x=="-" or x=="NR") else x)

df["Defense_Revenue_From_Two_Years_Ago(in millions)"] = df["Defense_Revenue_From_Two_Years_Ago(in millions)"].astype(np.float32)

df["Total Revenue(in millions)"] = df["Total Revenue(in millions)"].str.replace(",","")

df["Total Revenue(in millions)"] = df["Total Revenue(in millions)"].str.replace("$","")

df["Total Revenue(in millions)"] = df["Total Revenue(in millions)"].apply(lambda x: np.nan if (x=="~" or x=="-" or x=="NR") else x)

df["Total Revenue(in millions)"] = df["Total Revenue(in millions)"].astype(np.float32)
a = ((df["Defense_Revenue_From_A_Year_Ago(in millions)"]-df["Defense_Revenue_From_Two_Years_Ago(in millions)"])/df["Defense_Revenue_From_Two_Years_Ago(in millions)"])*100

df["%Defense Revenue Change"] = a
### Updating %of Revenue from Defence Column From Revenues Which Total and A Year Ago
df["%of Revenue from Defence"] = df["Defense_Revenue_From_A_Year_Ago(in millions)"] / df["Total Revenue(in millions)"] * 100
pd.options.display.float_format = "{:,.2f}".format
df.head()
df.info()
df["IsoAlpha3"] = df.Country.apply(lambda x: pycountry.countries.search_fuzzy(x)[0].alpha_3 if x != "South Korea" else pycountry.countries.search_fuzzy("Korea")[0].alpha_3)
df["Defense_Revenue_From_A_Year_Ago(in millions)"] = df["Defense_Revenue_From_A_Year_Ago(in millions)"] * 1000000

df["Defense_Revenue_From_Two_Years_Ago(in millions)"] = df["Defense_Revenue_From_Two_Years_Ago(in millions)"] * 1000000

df["Total Revenue(in millions)"] = df["Total Revenue(in millions)"] * 1000000
pd.options.display.float_format = '{:,.2f}'.format
df2020 = df[df.Year == 2020]

df2020 = df2020.sort_values("Defense_Revenue_From_A_Year_Ago(in millions)", ascending = False)
fig = px.bar(df2020[:15], x="Company", y="Defense_Revenue_From_A_Year_Ago(in millions)", 

             hover_name= "Company", text="Country", color="Defense_Revenue_From_A_Year_Ago(in millions)")

fig.update_layout(

    title='Top 15 Companies by Revenue',

    yaxis=dict(

        title='Revenue ($)'

    )

)

fig.show()
prepared_df2020 = df2020.groupby("Country")["Company"].count().reset_index()
fig = px.pie(prepared_df2020, values='Company', names='Country', labels="Country",

             color_discrete_sequence=px.colors.sequential.Electric, title="Distribution of Countries")

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
prepared_df2020 = df2020.groupby(["IsoAlpha3"])["Company"].count().reset_index()
fig = px.scatter_geo(prepared_df2020, locations="IsoAlpha3", size="Company", color="IsoAlpha3")

fig.show()
fig = px.bar(df2020[:15], x="Company", y="%Defense Revenue Change", hover_name= "Company", 

             title="Top 15 Countries Defence Revenue Change %", color="Country")

fig.show()
fig = px.bar(df2020.sort_values("%Defense Revenue Change", ascending=False)[:15], 

             x="Company", y="%Defense Revenue Change", hover_name= "Country")

fig.update_traces(marker_color='green')

fig.show()
fig = px.bar(df2020.sort_values("%Defense Revenue Change", ascending=True)[:15], 

             x="Company", y="%Defense Revenue Change", hover_name= "Country")

fig.update_traces(marker_color='red')

fig.show()
prepared_df2020 = df2020.groupby("Country")["%Defense Revenue Change"].mean().reset_index()

prepared_df2020 = prepared_df2020.sort_values("%Defense Revenue Change", ascending=False)

fig = px.bar(prepared_df2020, x="%Defense Revenue Change", y="Country", hover_name= "Country", 

             title=" % Defence Revenue Change of Countries AVG", color="%Defense Revenue Change", orientation='h')

fig.show()
a = ((df2020["Defense_Revenue_From_A_Year_Ago(in millions)"].sum()-df2020["Defense_Revenue_From_Two_Years_Ago(in millions)"].sum())/df2020["Defense_Revenue_From_Two_Years_Ago(in millions)"].sum())*100

print("Top 100 AVG % Defence Revenue Change: {}%".format("%.2f" % a))
fig = px.bar(df2020[:15], x="Company", y="%of Revenue from Defence", hover_name= "Company", 

             title="Top 15 Companies % of Revenue from Defence", color="Country")

fig.show()
fig = px.bar(df2020.sort_values("%of Revenue from Defence", ascending=False)[:15], 

             x="Company", y="%of Revenue from Defence", hover_name= "Country")

fig.update_traces(marker_color='green')

fig.show()
fig = px.bar(df2020.sort_values("%of Revenue from Defence", ascending=True)[:15], 

             x="Company", y="%of Revenue from Defence", hover_name= "Country")

fig.update_traces(marker_color='red')

fig.show()
prepared_df2020 = df2020.groupby("Country")["%of Revenue from Defence"].mean().reset_index()

prepared_df2020 = prepared_df2020.sort_values("%of Revenue from Defence", ascending=False)
fig = px.bar(prepared_df2020, x="%of Revenue from Defence", y="Country", hover_name= "Country", 

             title="Defence Companies Revenue From Defence of Countries AVG", color="%of Revenue from Defence", orientation='h')

fig.show()
prepared_df = pd.DataFrame(df.groupby(["Year","Country"])["Company"].count()).reset_index()
fig = px.line(prepared_df, x="Year", y="Company", color="Country")

fig.show()
prepared_df = pd.DataFrame(df.groupby(["Year"])["Country"].unique()).reset_index()

prepared_df["Number"] = prepared_df["Country"].apply(lambda x: len(x))
fig = px.line(prepared_df, x="Year", y="Number")

fig.show()