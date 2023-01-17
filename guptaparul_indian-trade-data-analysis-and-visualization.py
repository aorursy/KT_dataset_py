import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 

# charts
import seaborn as sns 
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

%matplotlib inline

#ignore warning 
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_import = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv")
df_export = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_export.csv")
df_export.head()
df_import.head()
df_export.describe()
df_import.describe()
df_export.info()
df_import.info()
df_export.isnull().sum()
df_export[df_export.value==0].count()
country_list=list(df_export.country.unique())
country_list
print("Duplicate exports : "+str(df_export.duplicated().sum()))
df_import.isnull().sum()
df_import[df_import.value==0].count()
country_list1=list(df_import.country.unique())
country_list1
print("Duplicate imports : "+str(df_import.duplicated().sum()))
def cleanup(df_data):
    df_data['country']= df_data['country'].apply(lambda x : np.NaN if x == "UNSPECIFIED" else x)
    df_data.dropna(inplace=True)
    df_data = df_data[df_data.value!=0] 
    df_data.drop_duplicates(keep="first",inplace=True)
    df_data = df_data.reset_index(drop=True)
    return df_data
df_export = cleanup(df_export)
df_import = cleanup(df_import)
df_import.isnull().sum()
df_import[df_import.value==0].count()
print("Count of Commodities Exported: "+ str(len(df_export['Commodity'].unique())))
print("Count of Commodities Imported: "+ str(len(df_import['Commodity'].unique())))
df_import_temp = df_import.copy(deep=True)
df_export_temp = df_export.copy(deep=True)
df_import_temp['commodity_sum'] = df_import_temp['value'].groupby(df_import_temp['Commodity']).transform('sum')
df_export_temp['commodity_sum'] = df_export_temp['value'].groupby(df_export_temp['Commodity']).transform('sum')
df_import_temp.drop(['value','country','year','HSCode'],axis=1,inplace=True)
df_export_temp.drop(['value','country','year','HSCode'],axis=1,inplace=True)

df_import_temp.sort_values(by='commodity_sum',inplace=True,ascending=False)
df_export_temp.sort_values(by='commodity_sum',inplace=True,ascending=False)

df_import_temp.drop_duplicates(inplace=True)
df_export_temp.drop_duplicates(inplace=True)
# Top 7 Goods exported as per their aggregate values
df_export_temp['Commodity'] = df_export_temp['Commodity'].apply(lambda x:x.split()[0])
px.bar(data_frame=df_export_temp.head(7),y='Commodity', x='commodity_sum', orientation='h',
       color='commodity_sum', title='Expensive Goods Exported from India Between 2010-2018 According to their Aggregate Value',
       labels={'commodity_sum':'Commoditiy Value in Million US $'})
pd.DataFrame(df_export.groupby(df_export['Commodity'])['value'].sum().sort_values(ascending=False).head(7))
exp_exports = pd.DataFrame(data=df_export[df_export.value>700])
px.box(x="HSCode", y="value", data_frame=exp_exports, title='Expensive Exports HSCodewise', 
            color='HSCode', hover_name='value', height=700, width = 1400)
# Top 7 Goods imported as per their aggergate values
df_import_temp['Commodity'] = df_import_temp['Commodity'].apply(lambda x:x.split()[0])

px.bar(data_frame=df_import_temp.head(7),y='Commodity', x='commodity_sum', orientation='h',
       color='commodity_sum', title='Expensive Goods Imported to India Between 2010-2018 According to their Aggregate Value',
       labels={'commodity_sum':'Commoditiy Value in Million USD'})
pd.DataFrame(df_import.groupby(df_import['Commodity'])['value'].sum().sort_values(ascending=False).head(7))
exp_imports = pd.DataFrame(data=df_import[df_import.value>1000])
px.box(x="HSCode", y="value", data_frame=exp_imports, title='Expensive Imports HSCodewise', 
            color='HSCode',  height=700, width = 1400)
df = pd.DataFrame(df_export['Commodity'].value_counts())
df.head(10)
exp_temp = df_export.copy()
exp_temp.drop(['HSCode', 'country'], axis=1, inplace=True)
exp_temp['Commodity'] = exp_temp['Commodity'].apply(lambda x:x.split(';')[0])
exp_temp.set_index('Commodity', inplace=True)
exp_temp
g= pd.DataFrame(exp_temp.loc[["ELECTRICAL MACHINERY AND EQUIPMENT AND PARTS THEREOF"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g1= pd.DataFrame(exp_temp.loc[["NUCLEAR REACTORS, BOILERS, MACHINERY AND MECHANICAL APPLIANCES"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g2= pd.DataFrame(exp_temp.loc[["OPTICAL, PHOTOGRAPHIC CINEMATOGRAPHIC MEASURING, CHECKING PRECISION, MEDICAL OR SURGICAL INST. AND APPARATUS PARTS AND ACCESSORIES THEREOF"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g3= pd.DataFrame(exp_temp.loc[["PHARMACEUTICAL PRODUCTS"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g4= pd.DataFrame(exp_temp.loc[["ARTICLES OF APPAREL AND CLOTHING ACCESSORIES, NOT KNITTED OR CROCHETED."]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()

# Initialize figure with subplots
fig = make_subplots(
    rows=5, cols=1, subplot_titles=("Trend for Electrical Machinery & Equipments and Parts",
                                    "Trend for Nuclear Reactors",
                                    "Trend for Medical or Surgical Apparatus & Equipments",
                                    "Trend for Pharmaceutical Products",
                                    "Trend for Apparel and Clothing Accessories"
                                   )
)

# Add traces
fig.add_trace(go.Scatter(x=g.year, y=g.value), row=1, col=1)
fig.add_trace(go.Scatter(x=g1.year, y=g1.value), row=2, col=1)
fig.add_trace(go.Scatter(x=g2.year, y=g2.value), row=3, col=1)
fig.add_trace(go.Scatter(x=g3.year, y=g3.value), row=4, col=1)
fig.add_trace(go.Scatter(x=g4.year, y=g4.value), row=5, col=1)


# Update xaxis properties
fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_xaxes(title_text="Year", row=2, col=1)
fig.update_xaxes(title_text="Year", row=3, col=1)
fig.update_xaxes(title_text="Year", row=4, col=1)
fig.update_xaxes(title_text="Year", row=5, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Million US $", row=1, col=1)
fig.update_yaxes(title_text="Million US $", row=2, col=1)
fig.update_yaxes(title_text="Million US $", row=3, col=1)
fig.update_yaxes(title_text="Million US $", row=4, col=1)
fig.update_yaxes(title_text="Million US $", row=5, col=1)

# Update title and height
fig.update_layout(title_text="Trade Trends for Some Popular Commodities Exported", showlegend=False, height = 1500 )
fig.show()
df1 = pd.DataFrame(df_import['Commodity'].value_counts())
df1.head(10)
imp_temp = df_import.copy()
imp_temp.drop(['HSCode', 'country'], axis=1, inplace=True)
imp_temp['Commodity'] = imp_temp['Commodity'].apply(lambda x:x.split(';')[0])
imp_temp.set_index('Commodity', inplace=True)
imp_temp
g= pd.DataFrame(imp_temp.loc[["ELECTRICAL MACHINERY AND EQUIPMENT AND PARTS THEREOF"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g1= pd.DataFrame(imp_temp.loc[["NUCLEAR REACTORS, BOILERS, MACHINERY AND MECHANICAL APPLIANCES"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g2= pd.DataFrame(imp_temp.loc[["MISCELLANEOUS GOODS."]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g3= pd.DataFrame(imp_temp.loc[["PLASTIC AND ARTICLES THEREOF."]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()
g4= pd.DataFrame(imp_temp.loc[["IRON AND STEEL"]].groupby(['year', 'Commodity'])['value'].sum()).reset_index()

# Initialize figure with subplots
fig = make_subplots(
    rows=5, cols=1, subplot_titles=("Trend for Electrical Machinery & Equipments and Parts",
                                    "Trend for Nuclear Reactors",
                                    "Trend for Miscellaneous Goods",
                                    "Trend for Plastic and Articles",
                                    "Trend for Iron and Steel"
                                   )
)

# Add traces
fig.add_trace(go.Scatter(x=g.year, y=g.value), row=1, col=1)
fig.add_trace(go.Scatter(x=g1.year, y=g1.value), row=2, col=1)
fig.add_trace(go.Scatter(x=g2.year, y=g2.value), row=3, col=1)
fig.add_trace(go.Scatter(x=g3.year, y=g3.value), row=4, col=1)
fig.add_trace(go.Scatter(x=g4.year, y=g4.value), row=5, col=1)


# Update xaxis properties
fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_xaxes(title_text="Year", row=2, col=1)
fig.update_xaxes(title_text="Year", row=3, col=1)
fig.update_xaxes(title_text="Year", row=4, col=1)
fig.update_xaxes(title_text="Year", row=5, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Million US $", row=1, col=1)
fig.update_yaxes(title_text="Million US $", row=2, col=1)
fig.update_yaxes(title_text="Million US $", row=3, col=1)
fig.update_yaxes(title_text="Million US $", row=4, col=1)
fig.update_yaxes(title_text="Million US $", row=5, col=1)

# Update title and height
fig.update_layout(title_text="Trade Trends for Some Popular Commodities Imported", showlegend=False, height = 1500 )
fig.show()
print("Number of Countries to whom we export comodities: " + str(df_export['country'].nunique()))
print("Number of Countries from whom we import comodities: " + str(df_import['country'].nunique()))
import warnings
warnings.filterwarnings("ignore")

exp_country = df_export.groupby('country').agg({'value':'sum'})
exp_country = exp_country.rename(columns={'value': 'Export'})
exp_country = exp_country.sort_values(by = 'Export', ascending = False)
exp_country = exp_country[:20]
exp_country_tmp = exp_country[:10]
px.bar(data_frame = exp_country_tmp, x=exp_country_tmp.index, y ='Export',
labels={'country':"Countries", 'Export': "Total Exports in Million US$" } , color='Export', width=1200)
imp_country = df_import.groupby('country').agg({'value':'sum'})
imp_country = imp_country.rename(columns={'value': 'Import'})
imp_country = imp_country.sort_values(by = 'Import', ascending = False)
imp_country = imp_country[:20]
imp_country_tmp = imp_country[:10]
px.bar(data_frame = imp_country_tmp, x=imp_country_tmp.index, y ='Import',
labels={'country':"Countries", 'Import': "Total Exports in Million US$" } , color='Import', width=1200 )
total_trade = pd.concat([exp_country, imp_country], axis = 1)
total_trade['Trade Deficit'] = exp_country.Export - imp_country.Import
total_trade = total_trade.sort_values(by = 'Trade Deficit', ascending = False)
total_trade = total_trade[:11]

print('Countrywise Trade Export/Import and Trade Balance of India')
display(total_trade)
px.bar(data_frame = total_trade, x=total_trade.index, y=['Import', 'Export', 'Trade Deficit'], barmode='group', labels={'index':'Countries', 'value':'Million US $'})
Import =df_import.groupby(['year']).agg({'value':'sum'}).reset_index()
Export =df_export.groupby(['year']).agg({'value':'sum'}).reset_index()
Import['Deficit'] = Export.value - Import.value
fig = go.Figure()

# Create and style traces
fig.add_trace(go.Scatter(x=Import.year, y=Import.value, name='Import',mode='lines+markers',
                         line=dict(color='blue', width=4)))
fig.add_trace(go.Scatter(x=Export.year, y=Export.value, name = 'Export',mode='lines+markers',
                         line=dict(color='green', width=4)))
fig.add_trace(go.Scatter(x=Import.year, y=Import.Deficit, name='Deficit',mode='lines+markers',
                         line=dict(color='red', width=4)))

fig.update_layout(
    title=go.layout.Title(
        text="Indian Trade Over The Years 2010-2018",
        xref="paper",
        x=0
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Year",
            font=dict(
                family="Times New",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Million US $",
            font=dict(
                family="Times New",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)

fig.show()

exp_country = df_export.copy()
exp_country.drop(['HSCode', 'Commodity'], axis=1, inplace=True)
exp_country = pd.DataFrame(exp_country.groupby(['country', 'year'])['value'].sum())
exp_country.reset_index('year', inplace=True)


imp_country = df_import.copy()
imp_country.drop(['HSCode', 'Commodity'], axis=1, inplace=True)
imp_country = pd.DataFrame(imp_country.groupby(['country', 'year'])['value'].sum())
imp_country.reset_index('year', inplace=True)
imp_country
# Initialize figure with subplots
fig = make_subplots(
    rows=4, cols=1, subplot_titles=("Chinese Trade with India Over The Years 2010-2018",
                                    "Saudi Arab Trade with India Over The Years 2010-2018",
                                    "USA Trade with India Over The Years 2010-2018",
                                    "United Arab Emts Trade with India Over The Years 2010-2018"
                                   )
)


# Create traces
g1 = pd.DataFrame(imp_country.loc[["CHINA P RP"]]).groupby(['year'])['value'].sum().reset_index()
g2 = pd.DataFrame(exp_country.loc[["CHINA P RP"]]).groupby(['year'])['value'].sum().reset_index()
g3 = pd.DataFrame(imp_country.loc[["SAUDI ARAB"]]).groupby(['year'])['value'].sum().reset_index()
g4 = pd.DataFrame(exp_country.loc[["SAUDI ARAB"]]).groupby(['year'])['value'].sum().reset_index()
g5 = pd.DataFrame(imp_country.loc[["U S A"]]).groupby(['year'])['value'].sum().reset_index()
g6 = pd.DataFrame(exp_country.loc[["U S A"]]).groupby(['year'])['value'].sum().reset_index()
g7 = pd.DataFrame(imp_country.loc[["U ARAB EMTS"]]).groupby(['year'])['value'].sum().reset_index()
g8 = pd.DataFrame(exp_country.loc[["U ARAB EMTS"]]).groupby(['year'])['value'].sum().reset_index()


# Add traces
fig.add_trace(go.Scatter(x=g1.year, y=g1.value, name='Import to India',mode='lines+markers',
                         line=dict(color='red', width=4)), row=1, col=1)
fig.add_trace(go.Scatter(x=g2.year, y=g2.value, name = 'Export to China',mode='lines+markers',
                         line=dict(color='blue', width=4)), row=1, col=1)

fig.add_trace(go.Scatter(x=g3.year, y=g3.value, name='Import to India',mode='lines+markers',
                         line=dict(color='orange', width=4)), row=2, col=1)
fig.add_trace(go.Scatter(x=g4.year, y=g4.value, name = 'Export to Saudi Arab',mode='lines+markers',
                         line=dict(color='green', width=4)), row=2, col=1)

fig.add_trace(go.Scatter(x=g5.year, y=g5.value, name='Import to India',mode='lines+markers',
                         line=dict(color='gold', width=4)), row=3, col=1)
fig.add_trace(go.Scatter(x=g6.year, y=g6.value, name = 'Export to USA',mode='lines+markers',
                         line=dict(color='purple', width=4)), row=3, col=1)

fig.add_trace(go.Scatter(x=g7.year, y=g7.value, name='Import to India',mode='lines+markers',
                         line=dict(color='olive', width=4)), row=4, col=1)
fig.add_trace(go.Scatter(x=g8.year, y=g8.value, name = 'Export to U Arab Emts',mode='lines+markers',
                         line=dict(color='yellow', width=4)), row=4, col=1)



# Update xaxis properties
fig.update_xaxes(title_text="Year", row=1, col=1)
fig.update_xaxes(title_text="Year", row=2, col=1)
fig.update_xaxes(title_text="Year", row=3, col=1)
fig.update_xaxes(title_text="Year", row=4, col=1)


# Update yaxis properties
fig.update_yaxes(title_text="Million US $", row=1, col=1)
fig.update_yaxes(title_text="Million US $", row=2, col=1)
fig.update_yaxes(title_text="Million US $", row=3, col=1)
fig.update_yaxes(title_text="Million US $", row=4, col=1)


# Update title and height
fig.update_layout(title_text="Trade Trends for Some Popular Commodities Imported", height = 1500 )

fig.show()
