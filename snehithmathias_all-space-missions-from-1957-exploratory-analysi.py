import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools




data = pd.read_csv(r"../input/all-space-missions-from-1957/Space_Corrected.csv")
data.head()
data.info()
# Company Wise Data
company = data['Company Name'].value_counts()
company_name = list(company.index)
print(company_name)
counts = list((company/company.sum())*100)


figure = go.Pie(labels=company_name, values=counts)
layout = go.Layout(
    title= " % Distribution rockets launched by individual company",
    height=600,
    width=600)
fig = go.Figure(data=figure, layout=layout)
fig.show()
# Country Wise Data
country = data["Location"]
data["Location"] = [i.split(".")[0].split(",")[-1].strip() for i in country]

#print(data["Location"])

country = data['Location'].value_counts()
country_names = list(country.index)

block = go.Bar(
    x = list(country_names),
    y = country,
    name='Rocket launched by individual countries')
layout = go.Layout(
    title = "rockets launch by each country",
    barmode='group')
fig = go.Figure(data=block, layout=layout)
fig.show()

df = pd.DataFrame(columns=['Country', 'Success'])


for country_name in country_names:
    data_1 = []
    data_1.append(country_name)
    data_1.append(len(data.loc(data['Status Mission']=='Success') & (data["Location"]==country_name), 'name'))
print(data_1)

             