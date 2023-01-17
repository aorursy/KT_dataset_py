from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True) 



import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_rows',500)

pd.set_option('max_columns',800)
df = pd.read_csv('../input/us-summary/us_salary_summary.csv')

group_title_size = df.groupby(['Title','Company_Size'])['Count'].agg(['sum'])

group_title_size['% within title'] = 100*group_title_size['sum']/group_title_size['sum'].sum(level='Title')

group_title_size['scale_plot'] = 25*group_title_size['sum']/group_title_size['sum'].sum(level='Title')



!pip install pywaffle

from pywaffle import Waffle
data = group_title_size.scale_plot

names_array = ['Business Analyst','Data Analyst', 'Data Scientist', 'Product/Project Manager', 'Research Scientist']

fig = plt.figure(

    FigureClass=Waffle, 

    plots={

    '511':{

        'values': data[names_array[0]],

        'colors': ('#921ECE','#721ECE','#521ECE','#321ECE','#121ECE'),

        'labels': ["{0}".format(k, v) for k, v in data[names_array[0]].items()],

        'legend': {'loc': 'upper left','bbox_to_anchor': (1.05, 1),'fontsize': 8},

         'title': {'label': names_array[0],'loc': 'left'}

    },

    '512':{

        'values': data[names_array[1]],

        'colors': ('#FF9E4A','#FF7E4A','#FF5E4A','#FF3E4A','#FF1E4A'),

        'labels': ["{0}".format(k, v) for k, v in data[names_array[1]].items()],

        'legend': {'loc': 'upper left','bbox_to_anchor': (1.05, 1),'fontsize': 8},

         'title': {'label': names_array[1],'loc': 'left'}

    },

    '513':{

        'values': data[names_array[2]],

        'colors': ('#99BF5C','#79BF5C','#59BF5C','#39BF5C','#19BF5C'),

        'labels': ["{0}".format(k, v) for k, v in data[names_array[2]].items()],

        'legend': {'loc': 'upper left','bbox_to_anchor': (1.05, 1),'fontsize': 8},

         'title': {'label': names_array[2],'loc': 'left'}

    },

    '514':{

        'values': data[names_array[3]],

        'colors': ('#ED969D','#ED767D','#ED565D','#ED363D','#ED161D'),

        'labels': ["{0}".format(k, v) for k, v in data[names_array[3]].items()],

        'legend': {'loc': 'upper left','bbox_to_anchor': (1.05, 1),'fontsize': 8},

         'title': {'label': names_array[3],'loc': 'left'}

    },    

    '515':{

        'values': data[names_array[4]],

        'colors': ('#AD9BC9','#AD7BC9','#AD5BC9','#AD5BC9','#AD1BC9'),

        'labels': ["{0}".format(k, v) for k, v in data[names_array[4]].items()],

        'legend': {'loc': 'upper left','bbox_to_anchor': (1.05, 1),'fontsize': 8},

         'title': {'label': names_array[4],'loc': 'left'}}

        

},

    rows=5, 

    #icon_size=40,

    #legend={'loc': 'lower center', 'bbox_to_anchor': (0,0)},

    tight=False,

    figsize=(20, 12)

)

fig.set_facecolor('#EEEEEE')

plt.show()
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_response = pd.read_csv('../input/kagglesurvey2019/multiple_choice_responses.csv')

df_question = pd.read_csv('../input/kagglesurvey2019/questions_only.csv')

df_wdidata = pd.read_csv("../input/wdidatacsv/WDIData.csv")
df_data = df_response.loc[:,["Q1", "Q2", "Q3", "Q4", "Q5", "Q10", "Q11"]]

df_data.columns = ["Age", "Gender", "Country", "Education", "Title", "Salary", "Spending"]

df_data["Country"] = df_data["Country"].str.replace("United Kingdom of Great Britain and Northern Ireland","United Kingdom", regex=False)

df_data["Country"] = df_data["Country"].str.replace("United States of America","United States", regex=False)

df_data["Country"] = df_data["Country"].str.replace("Iran, Islamic Republic of...","Iran", regex=False)

df_data["Country"] = df_data["Country"].str.replace("Republic of Korea","South Korea", regex=False)

df_data["Country"] = df_data["Country"].str.replace("Viet Nam","Vietnam", regex=False)

df_data_ds = df_data[df_data.Title == "Data Scientist"]
df_salary = df_data.dropna()

df_salary["SalaryTemp"] = df_salary["Salary"].str.replace("$0-999","0-999", regex=False)

df_salary["SalaryTemp"] = df_salary["SalaryTemp"].str.replace("> $500,000","500,000-499,999", regex=False)

df_salary["SalaryTemp"] = df_salary["SalaryTemp"].str.replace("300,000-500,000","300,000-499,999", regex=False)

df_salary["SalaryTemp"] = df_salary["SalaryTemp"].str.replace(",","", regex=False)

df_salary["SalaryLowLimit"] = df_salary["SalaryTemp"].str.split('-', expand=True)[0]

df_salary["SalaryLowLimit"] = pd.to_numeric(df_salary["SalaryLowLimit"], errors='coerce')

df_salary["SalaryUpLimit"] = df_salary["SalaryTemp"].str.split('-', expand=True)[1]

df_salary["SalaryUpLimit"] = pd.to_numeric(df_salary["SalaryUpLimit"], errors='coerce')+1

df_salary["SalaryMean"] = (df_salary["SalaryLowLimit"]+df_salary["SalaryUpLimit"])/2



df_ds_salary = df_salary[df_salary.Title == "Data Scientist"]

df_pm_salary = df_salary[df_salary.Title == 'Product/Project Manager']



df_ds_salary_by_country  = df_ds_salary.loc[:,["SalaryMean","Country"]].groupby(["Country"]).mean().reset_index()

df_ds_salary_by_country["SalaryMean"] = df_ds_salary_by_country["SalaryMean"].round(2)

df_pm_salary_by_country  = df_pm_salary.loc[:,["SalaryMean","Country"]].groupby(["Country"]).mean().reset_index()

df_pm_salary_by_country["SalaryMean"] = df_pm_salary_by_country["SalaryMean"].round(2)



df_countrygdp = df_wdidata[df_wdidata["Indicator Name"]=="GDP per capita (current US$)"].loc[:,["Country Name", "Country Code", "2017"]]

df_countrygdp.columns = ["Country", "CountryCode", "GDP"]

df_countrygdp["GDP"] = df_countrygdp["GDP"].round(2)

df_countrygdp["Country"] = df_countrygdp["Country"].str.replace("Egypt, Arab Rep.","Egypt", regex=False)

df_countrygdp["Country"] = df_countrygdp["Country"].str.replace("Hong Kong SAR, China","Hong Kong (S.A.R.)", regex=False)

df_countrygdp["Country"] = df_countrygdp["Country"].str.replace("Iran, Islamic Rep.","Iran", regex=False)

df_countrygdp["Country"] = df_countrygdp["Country"].str.replace("Korea, Rep.","South Korea", regex=False)

df_countrygdp["Country"] = df_countrygdp["Country"].str.replace("Russian Federation","Russia", regex=False)

df_ds_salary_by_country = pd.merge(df_ds_salary_by_country, df_countrygdp, on='Country')

df_pm_salary_by_country = pd.merge(df_pm_salary_by_country, df_countrygdp, on='Country')
df_salary_spent = df_salary.loc[:,["Spending", "Salary", "SalaryMean"]].groupby(["Spending", "Salary", "SalaryMean"]).size().reset_index().sort_values(["Spending","SalaryMean"]).dropna()

df_salary_spent.columns = ["Spending", "Salary", "SalaryMean", "Count"]



df_salary_spent["Spent"] = df_salary_spent["Spending"]

df_salary_spent["Spent"] = df_salary_spent["Spent"].str.replace("$0 (USD)","$ 0", regex=False)

df_salary_spent["Spent"] = df_salary_spent["Spent"].str.replace("$1-$99","$ 1-99", regex=False)

df_salary_spent["Spent"] = df_salary_spent["Spent"].str.replace("$100-$999","$ 100-999", regex=False)

df_salary_spent["Spent"] = df_salary_spent["Spent"].str.replace("$1000-$9,999","$ 1000-9999", regex=False)

df_salary_spent["Spent"] = df_salary_spent["Spent"].str.replace("$10,000-$99,999","$ 10000-99999", regex=False)

df_salary_spent["Spent"] = df_salary_spent["Spent"].str.replace("> $100,000 ($USD)","$ 100000 and more", regex=False)
df_s_usa = df_salary[df_salary["Country"]=="United States"].loc[:,["Title", "SalaryMean"]].groupby(["Title"], as_index=False).mean().sort_values("SalaryMean", ascending=False)

df_s_canada = df_salary[df_salary["Country"]=="Canada"].loc[:,["Title", "SalaryMean"]].groupby(["Title"], as_index=False).mean().sort_values("SalaryMean", ascending=False)

df_s_germany = df_salary[df_salary["Country"]=="Germany"].loc[:,["Title", "SalaryMean"]].groupby(["Title"], as_index=False).mean().sort_values("SalaryMean", ascending=False)

df_s_uk = df_salary[df_salary["Country"]=="United Kingdom"].loc[:,["Title", "SalaryMean"]].groupby(["Title"], as_index=False).mean().sort_values("SalaryMean", ascending=False)



mc_c1 = ["gold",]*10

mc_c2 = ["gold",]*10

mc_c3 = ["gold",]*10

mc_c4 = ["gold",]*10

mc_c1[0] = mc_c2[3] = mc_c3[4] = mc_c4[4] ="lime"

mc_c1[1] = mc_c2[4] = mc_c3[1] = mc_c4[5] ="purple"





fig = make_subplots(rows=3, cols=2, specs=[[{'type':'xy'}, {'type':'xy'}],[{'type':'xy'}, {'type':'xy'}],[{'type':'xy'}, {'type':'xy'}]], subplot_titles=('United States', 'Canada','Germany', 'United Kingdom'))



fig.add_trace(

    go.Bar(x=df_s_usa["Title"], y=df_s_usa["SalaryMean"], name="United States", marker_color=mc_c1),

    row=1, col=1)



fig.add_trace(

    go.Bar(x=df_s_canada["Title"], y=df_s_canada["SalaryMean"], name="Canada", marker_color=mc_c2),

    row=1, col=2)



fig.add_trace(

    go.Bar(x=df_s_germany["Title"], y=df_s_germany["SalaryMean"], name="Germany", marker_color=mc_c3),

    row=2, col=1)



fig.add_trace(

    go.Bar(x=df_s_uk["Title"], y=df_s_uk["SalaryMean"], name="United Kingdom", marker_color=mc_c4),

    row=2, col=2)



fig.update_layout(

    title_text="Salary Variation by Countries ",

    height=850, width=850, showlegend=False)



fig.update_traces(marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6) 

                 

fig.show()
df_ds_salary_by_country_top25 = df_ds_salary_by_country.sort_values("SalaryMean", ascending=False)[:25]





fig = go.Figure()

fig.add_trace(go.Bar(x=df_ds_salary_by_country_top25["Country"],

                y=df_ds_salary_by_country_top25["SalaryMean"],

                marker_color=df_ds_salary_by_country_top25["SalaryMean"],

                marker_line_color='rgb(8,48,107)',

                marker_line_width=1.5, 

                opacity=0.8))



fig.update_layout(

    title_text='Top 25 Countries with the Highest Average Data Scientist Salary',

    height=550, width=700,

    showlegend=False)



fig.show()
df_pm_salary_by_country_top25 = df_pm_salary_by_country.sort_values("SalaryMean", ascending=False)[:25]





fig = go.Figure()

fig.add_trace(go.Bar(x=df_pm_salary_by_country_top25["Country"],

                y=df_pm_salary_by_country_top25["SalaryMean"],

                marker_color=df_pm_salary_by_country_top25["SalaryMean"],

                marker_line_color='rgb(8,48,107)',

                marker_line_width=1.5, 

                opacity=0.8))



fig.update_layout(

    title_text='Top 25 Countries with the Highest Average Product/Project Manager Salary',

    height=550, width=700,

    showlegend=False)



fig.show()
fig = px.scatter_geo(df_ds_salary_by_country, locations="CountryCode", color="GDP",

                     hover_name="Country", size="SalaryMean", 

                     projection="natural earth") 



fig.update_layout(title="GDP Correlation with Average Data Scientists Salaries Around the World")



fig.show()