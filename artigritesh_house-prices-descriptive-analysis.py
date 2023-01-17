#Import neccessary libraries
import pandas as pd
import numpy as np
import os
from datetime import date, datetime
# configuring plotting modeules
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.pylab as pylab
import plotly.express as px
from scipy.stats import mode, pearsonr, pointbiserialr
import plotly.graph_objects as go
import itertools
from plotly.subplots import make_subplots
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
style.use("ggplot")
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (20, 8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
input_data_dir = "../input/house-prices-advanced-regression-techniques"
train_df = pd.read_csv(os.path.join(input_data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(input_data_dir, "test.csv"))
full_df = pd.concat([train_df, test_df], sort=False)
print("Size of training dataset       : {}".format(train_df.shape))
print("Size of test dataset           : {}".format(test_df.shape))
full_df.describe().T

null_stats_df = full_df.isna().sum()
null_stats_df = pd.DataFrame({"null_values": null_stats_df})
null_stats_df["total_length"] = full_df.shape[0]
null_stats_df["null_%"] = (null_stats_df["null_values"]/null_stats_df["total_length"]) * 100

null_stats_df[null_stats_df['null_%'] > 1].sort_values(by="null_%", ascending=True)["null_%"].plot(kind='barh')
plt.xlabel("Percentage of Null values", size=16)
plt.ylabel("Columns", size=16)
plt.title("Null value stats", size=20, pad = 16)
x = train_df.SalePrice.sort_values().reset_index().index
y = train_df.SalePrice.sort_values().reset_index()["SalePrice"]
plt.scatter(x, y, color = "lightskyblue")
plt.xlabel("Index", size=16)
plt.ylabel("Sales Price", size=16)
plt.title("Distribution of target variable", size=16, pad=16)
def plot_histogram(x, density=False, kde=False):
    fig, ax = plt.subplots()

    # Plot
        # Plot histogram
    x.plot(kind = "hist", density = density, alpha = 0.65, bins = 30, color='lightskyblue') # change density to true, because KDE uses density
        # Plot KDE
    if kde:
        x.plot(kind = "kde")

        # Quantile lines
    quant_5, quant_25, quant_50, quant_75, quant_95 = x.quantile(0.05), x.quantile(0.25), x.quantile(0.5), x.quantile(0.75), x.quantile(0.95)
    quants = [[quant_5, 0.6, 0.16], [quant_25, 0.8, 0.26], [quant_50, 1, 0.36],  [quant_75, 0.8, 0.46], [quant_95, 0.6, 0.56]]
    for i in quants:
        ax.axvline(i[0], alpha = i[1], ymax = i[2], linestyle = ":")

    # X
    x_start, x_end = ax.get_xlim()
    ax.set_xlim(x_start, x_end)

# #     # Y
    if kde:
        ax.set_yticklabels([])
        ax.set_ylabel("")

#     # Annotations
    y_tick_max = ax.get_ylim()[1]
    x_tick_diff = (abs(ax.get_xticks()[1]) - abs(ax.get_xticks()[0]))/10
    ax.text(quant_5,(y_tick_max * 0.17 ), "5th", size=10, alpha = 0.80)
    ax.text(quant_25, (y_tick_max * 0.27), "25th", size = 11, alpha = 0.85)
    ax.text(quant_50, (y_tick_max * 0.37), "50th", size = 12, alpha = 1)
    ax.text(quant_75, (y_tick_max * 0.47), "75th", size = 11, alpha = 0.85)
    ax.text(quant_95, (y_tick_max * 0.57), "95th Percentile", size = 10, alpha =.8)


# #         # Remove ticks and spines
    if kde:
        ax.tick_params(left = False, bottom = False)
        for ax, spine in ax.spines.items():
            spine.set_visible(False)

    return ax

higher_quantile_range = train_df["SalePrice"].quantile(0.99)
sale_price = train_df["SalePrice"]
cleaned_sales_price = sale_price[sale_price<higher_quantile_range]
plot_histogram(cleaned_sales_price, kde=True, density=True)
f, a = plt.subplots(figsize=(16,12))
corr_df = train_df.corr()
sns.heatmap(corr_df, vmax=1., square=True, xticklabels = 1, yticklabels = 1)
corr_df.SalePrice.apply(lambda x: abs(x)).sort_values(ascending=False).iloc[1:15][::-1].plot(kind='barh',
                                                                                            color='lightskyblue')
plt.title("Top 15 highly correlated continuos features", size=16, pad=20)
plt.xlabel("Correlation coefficient")
plt.ylabel("Features")
sns.violinplot(train_df['OverallQual'], train_df['SalePrice'])
plt.xlabel("Quality of dwelling", size=16)
plt.ylabel("Sale Price", size=16)
plt.title("Price distribution w.r.t Quality", size=20, pad=16)
print("MSZoning abbreviation dictionary:")
print("C       : Commercial")
print("FV      : Floating Village Residential")
print("RH      : Residential High Density")
print("RL      : Residential Low Density")
print("RM      : Residential Medium Density")
pxfig = px.scatter(
    data_frame=train_df, 
    x="GrLivArea", 
    y="SalePrice", 
    color="MSZoning",
    size_max=60,
    hover_name="Id"
)
pxfig.update_layout(title="Sale-Price Vs Ground-Living-Area", xaxis=dict(title="Ground Living Area"))
pxfig.show()

train_df[["GrLivArea", "TotalBsmtSF", "1stFlrSF"]]
temp_df = train_df.groupby("FullBath")["Id"].count().reset_index()
temp_df.columns = ["FullBath", "counts"]
pxfig = px.bar(
    data_frame=temp_df, 
    x="FullBath", 
    y="counts", 
)
pxfig.update_layout(title="", xaxis=dict(title="Full Bathrooms"))
pxfig.show()

temp_df = train_df.groupby("BedroomAbvGr")["Id"].count().reset_index()
temp_df.columns = ["FullBath", "counts"]
pxfig = px.bar(
    data_frame=temp_df, 
    x="FullBath", 
    y="counts", 
)
pxfig.update_layout(title="", xaxis=dict(title="Full Bedrooms"))
pxfig.show()
temp_df = train_df.copy()
temp_df["PricePerSF"] = train_df["SalePrice"]/train_df["GrLivArea"]
sns.boxplot(temp_df['YearBuilt'], temp_df['PricePerSF'])
plt.xlabel("Year", size=16)
plt.xticks(rotation=90)
plt.ylabel("Price Per Square Meter($)", size=16)
plt.title("Price distribution w.r.t Year", size=20, pad=16)

sns.boxplot(temp_df['YearRemodAdd'], temp_df['PricePerSF'])
plt.xlabel("Year Modifier", size=16)
plt.xticks(rotation=90)
plt.ylabel("Price Per Square Meter ($)", size=16)
plt.title("Price distribution w.r.t Year Modified", size=20, pad=16)

sns.swarmplot(temp_df['YrSold'], temp_df['PricePerSF'])
plt.xlabel("Year Sold", size=16)
plt.xticks(rotation=90)
plt.ylabel("Price Per Square Meter($)", size=16)
plt.title("Price distribution w.r.t Year Sold", size=20, pad=16)
zone_mapper = {
               'C (all)':'Commercial',
               'FV' : 'Floating Village Residential',
               'RH' : 'Residential High Density',
               'RL' : 'Residential Low Density',
               'RM' : 'Residential Medium Density'
              }

temp_df = train_df[["MSZoning", "SalePrice"]]
temp_df['MSZoning'] = temp_df['MSZoning'].map(zone_mapper)

pxfig = px.box(
    data_frame=temp_df, 
    x="MSZoning", 
    y="SalePrice", 
)
pxfig.update_layout(title="Selling Price Vs Zone", xaxis=dict(title="ZONE"),
                   yaxis=dict(title="Selling price($)"))
pxfig.show()
shape_mapper = {
       "Reg":"Regular",
       "IR1":"Slightly irregular",
       "IR2":"Moderately Irregular",
       "IR3":"Irregular"
              }

temp_df = train_df[["LotShape", "SalePrice"]]
temp_df.LotShape = temp_df.LotShape.map(shape_mapper)

pxfig = px.box(
    data_frame=temp_df, 
    x="LotShape", 
    y="SalePrice", 
)
pxfig.update_layout(title="Selling Price Vs Shape of property", xaxis=dict(title="Shape of property"),
                   yaxis=dict(title="Selling price($)"))
pxfig.show()
coutour_mapper= {
       "Lvl": "Near Flat/Level",
       "Bnk": "Banked",
       "HLS": "Hillside",
       "Low": "Depression"
        }

temp_df = train_df[["LandContour", "SalePrice"]]
temp_df.LandContour = train_df.LandContour.map(coutour_mapper)
pxfig = px.box(
    data_frame=temp_df, 
    x="LandContour", 
    y="SalePrice", 
)
pxfig.update_layout(title="Selling Price Vs Flatness of property", xaxis=dict(title="Flatness"),
                   yaxis=dict(title="Selling price($)"))
pxfig.show()
pxfig = px.box(
    data_frame=train_df, 
    x="Neighborhood", 
    y="SalePrice", 
)
pxfig.update_layout(title="Selling Price Vs Neighborhood", xaxis=dict(title="Neighborhood"), yaxis=dict(title="Selling price($)"))
pxfig.show()
b_type_mapper=    {   
    "1Fam": "Single-family",
       "2fmCon":"Two-family Conversion",
       "Duplex":"Duplex",
       "TwnhsE":"Townhouse End Unit",
       "Twnhs":"Townhouse Inside Unit"
        }
temp_df = train_df[["BldgType", "SalePrice"]]
temp_df.BldgType = temp_df.BldgType.map(b_type_mapper)
        
pxfig = px.box(
    data_frame=temp_df, 
    x="BldgType", 
    y="SalePrice", 
)
pxfig.update_layout(title="Selling Price Vs Building Type", xaxis=dict(title="Building Type"), yaxis=dict(title="Selling price($)"))
pxfig.show()
h_style_mapper=    {   
       "1Story": "One story",
       "1.5Fin": "1.5 story: 2nd level finished",
       "1.5Unf": "1.5 story: 2nd level unfinished",
       "2Story": "Two story",
       "2.5Fin": "2.5 story: 2nd level finished",
       "2.5Unf": "2.5 story: 2nd level unfinished",
       "SFoyer": "Split Foyer",
       "SLvl": "Split Level"
        }

temp_df = train_df.groupby(["HouseStyle", "BldgType"])["Id"].count().reset_index()
temp_df.columns = ["HouseStyle", "BldgType", "Number of house"]
temp_df.BldgType = temp_df.BldgType.map(b_type_mapper)
temp_df.HouseStyle = temp_df.HouseStyle.map(h_style_mapper)

px.bar(temp_df, x="HouseStyle", y="Number of house", color="BldgType")
temp_df = train_df.groupby(["BldgType", "RoofStyle"])["Id"].count().reset_index()
temp_df.columns = ["BldgType", "RoofStyle", "Number of house"]

temp_df.BldgType = temp_df.BldgType.map(b_type_mapper)

pxfig = px.bar(temp_df, x="BldgType", y="Number of house", color="RoofStyle")

pxfig.update_layout(title="Selling Price Vs Roof Style", xaxis=dict(title="Building Type"), yaxis=dict(title="Number of house"))
pxfig.show()