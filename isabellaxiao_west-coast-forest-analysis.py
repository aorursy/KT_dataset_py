import numpy as np
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
usfs = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="usfs_fia")
bq_assistant = BigQueryHelper("bigquery-public-data", "usfs_fia")
bq_assistant.list_tables()
bq_assistant.table_schema("plot_tree")
queryt = """
SELECT
    DISTINCT measurement_year,
    COUNT(DISTINCT species_common_name) AS ctree
FROM
    `bigquery-public-data.usfs_fia.plot_tree`
WHERE
    plot_state_code = 53
    OR plot_state_code = 06
    OR plot_state_code = 41
    AND measurement_year in 
        (SELECT
            DISTINCT measurement_year
        FROM
            `bigquery-public-data.usfs_fia.plot_tree`
        WHERE
            plot_state_code = 53)
     AND measurement_year in 
        (SELECT
            DISTINCT measurement_year
        FROM
            `bigquery-public-data.usfs_fia.plot_tree`
        WHERE
            plot_state_code = 06)
     AND measurement_year in 
        (SELECT
            DISTINCT measurement_year
        FROM
            `bigquery-public-data.usfs_fia.plot_tree`
        WHERE
            plot_state_code = 41)

GROUP BY
    measurement_year
ORDER BY
    measurement_year
;
        """
responset = usfs.query_to_pandas_safe(queryt, max_gb_scanned=10)
responsett = responset.drop([0,1,2,3,4,5])

# different trees in Washington
queryw = """
SELECT
    DISTINCT measurement_year,
    COUNT(DISTINCT species_common_name) AS w
FROM
    `bigquery-public-data.usfs_fia.plot_tree`
WHERE
    plot_state_code = 53
GROUP BY
    measurement_year
ORDER BY
    measurement_year
;
        """
responsew = usfs.query_to_pandas_safe(queryw, max_gb_scanned=10)
responsew = responsew.drop([0,1])
# different trees in California
queryc = """
SELECT
    DISTINCT measurement_year,
    COUNT(DISTINCT species_common_name) AS c
FROM
    `bigquery-public-data.usfs_fia.plot_tree`
WHERE
    plot_state_code = 06
GROUP BY
    measurement_year
ORDER BY
    measurement_year
;
        """
responsec = usfs.query_to_pandas_safe(queryc, max_gb_scanned=10)
responsec = responsec.drop([0,1,2,3])
# different trees in California
queryo = """
SELECT
    DISTINCT measurement_year,
    COUNT(DISTINCT species_common_name) AS o
FROM
    `bigquery-public-data.usfs_fia.plot_tree`
WHERE
    plot_state_code = 41
GROUP BY
    measurement_year
ORDER BY
    measurement_year
;
        """
responseo = usfs.query_to_pandas_safe(queryo, max_gb_scanned=10)
responseo = responseo.drop([0,1,2,3,4])
import numpy as np
import matplotlib.pyplot as plt
x = responsett[['measurement_year']]
y = responsett[['ctree']]
yw = responsew[['w']]
yc = responsec[['c']]
yo = responseo[['o']]
plt.plot(x,y,label="West Coast",color="gold",linewidth=2)
plt.plot(x,yw,label="Washington",color="navy",linewidth=2)
plt.plot(x,yc,label="California",color="firebrick",linewidth=2)
plt.plot(x,yo,label="Oregan",color="olive",linewidth=2)
plt.xlabel("Year")
plt.ylabel("Variability")
plt.legend()
query1 = """
SELECT
    species_common_name,
    species_group_code,
    total_height,
    measurement_year,
    current_diameter,
    latitude,
    longitude,
    plot_state_code,
    plot_county_code
FROM
    `bigquery-public-data.usfs_fia.plot_tree`
WHERE
    plot_state_code = 53
    OR plot_state_code = 06
    OR plot_state_code = 41
    AND total_height > 0
;
        """
response1 = usfs.query_to_pandas_safe(query1, max_gb_scanned=10)
westtree1 = response1.dropna(axis=0,how='any')
#From 1991 - 2017, average tree height on U.S. west coast based on diferent species
#Group together height and diameter measurements for each species taken each year
treeh = westtree1.groupby(['species_common_name', 'measurement_year']).agg({'current_diameter': 'mean'})

# Remove tree types that have at least 14 measurements, and transpose so subplots groups species and not year
heighttree = treeh.unstack().dropna(axis='index', thresh=14)

# Rename columns to remove total_height label
heighttree.columns = heighttree.columns.map(lambda t: t[1])
h= heighttree.transpose().plot.bar(subplots=True, layout=(40,2), figsize=(20, 100))
from bokeh.models import ColumnDataSource
from bokeh.models import FactorRange
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.transform import factor_cmap
import matplotlib
import matplotlib.pyplot as plt
output_notebook()
westtree_info = westtree1[["species_common_name","plot_state_code","measurement_year","total_height"]]
westtree_info["plot_state_code"][westtree_info["plot_state_code"]==6]="California"
westtree_info["plot_state_code"][westtree_info["plot_state_code"]==41]="Oregon"
westtree_info["plot_state_code"][westtree_info["plot_state_code"]==53]="Washington"
tree = westtree_info.groupby(['species_common_name',"plot_state_code",'measurement_year']).mean()
tree = tree.reset_index()
tree_type = []
for i in np.unique(tree["species_common_name"]):
    tree_type.append(i)
print(tree_type)
def showTreeSpe(typeOfTree):
    state = ["California","Oregon","Washington"]
    year = []
    newyear = ""
    for i in np.unique(tree["measurement_year"][tree["species_common_name"]==typeOfTree]):
        newyear = str(i)
        year.append(newyear)

    data = {"year":year}
    height = []
    for i in state:
        height = []
        for j in year:
            k = tree[(tree["species_common_name"]==typeOfTree)&(tree["plot_state_code"]==i)&(tree["measurement_year"]==int(j))&(tree["total_height"]>0)].index
            height.append(tree["total_height"][k])
        data[i] = height
    x = [(y, s) for y in year for s in state]
    counts = sum(zip(data["California"],data["Oregon"],data["Washington"]),())
    # print(counts)
    source = ColumnDataSource(data=dict(x=x, counts=counts))
    p = figure(x_range=FactorRange(*x), plot_height=250,plot_width = 2500,title=typeOfTree)
    p.vbar(x='x', top='counts', width=0.9, source=source, line_color="white",
           fill_color=factor_cmap('x', palette=['firebrick', 'olive', 'navy'], factors=state, start=1, end=2))
    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    x1 = [(y,"California")for y in year]
    counts1 = sum(zip(data["California"]),())
    x2 = [(y,"Oregon")for y in year]
    counts2 = sum(zip(data["Oregon"]),())
    x3 = [(y,"Washington")for y in year]
    counts3 = sum(zip(data["Washington"]),())
    p.line(x=x1, y=counts1, color="firebrick", line_width=1,legend="California")
    p.circle(x=x1, y=counts1, line_color="firebrick", fill_color="white", size=4)
    p.line(x=x2, y=counts2, color="olive", line_width=1,legend="Oregon")
    p.circle(x=x2, y=counts2, line_color="olive", fill_color="white", size=4)
    p.line(x=x3, y=counts3, color="navy", line_width=1,legend ="Washington")
    p.circle(x=x3, y=counts3, line_color="navy", fill_color="white", size=4)
    return p

from bokeh.layouts import column
lay = []
for i in tree_type:
    p = showTreeSpe(typeOfTree = i)
    lay.append(p)
# p1 = showTreeSpe(typeOfTree = "Pacific dogwood")
# p2 = showTreeSpe(typeOfTree = "California live oak")
layout = column(lay)
show(layout)

from bokeh.io import show
from bokeh.models import ColumnDataSource, GMapOptions,HoverTool
from bokeh.plotting import gmap
from bokeh.plotting import output_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
output_notebook()

map_options = GMapOptions(lat=35.636, lng=-120.986, map_type="terrain", zoom=6)


for year in range(2001,2007):
    p = gmap("AIzaSyABt-rVE5FZtf1e5scSw8SzxFCWYft_ViM", map_options, title=str(year))
    source1 = ColumnDataSource(data=westtree1[(westtree1["species_common_name"]=="Pacific dogwood") & (westtree1["measurement_year"]==year)].to_dict(orient= 'list'))
    source2 = ColumnDataSource(data=westtree1[(westtree1["species_common_name"]=="California live oak") & (westtree1["measurement_year"]==year)].to_dict(orient= 'list'))
    source3 = ColumnDataSource(data=westtree1[(westtree1["species_common_name"]=="Douglas-fir") & (westtree1["measurement_year"]==year)].to_dict(orient= 'list'))
    source4 = ColumnDataSource(data=westtree1[(westtree1["species_common_name"]=="Alaska yellow-cedar") & (westtree1["measurement_year"]==year)].to_dict(orient= 'list'))
    source5 = ColumnDataSource(data=westtree1[(westtree1["species_common_name"]=="Coulter pine") & (westtree1["measurement_year"]==year)].to_dict(orient= 'list'))
    source6 = ColumnDataSource(data=westtree1[(westtree1["species_common_name"]=="Oregon white oak") & (westtree1["measurement_year"]==year)].to_dict(orient= 'list'))
    source7 = ColumnDataSource(data=westtree1[(westtree1["species_common_name"]=="willow spp.") & (westtree1["measurement_year"]==year)].to_dict(orient= 'list'))
    p.circle(x="longitude", y="latitude", size="current_diameter",line_color="black", fill_color="brown", fill_alpha=0.5, hover_line_color="orange", hover_fill_color="orange",source=source1,legend="Pacific dogwood")
    p.circle(x="longitude", y="latitude", size="current_diameter",line_color="black", fill_color="blue", fill_alpha=0.5,hover_line_color="orange", hover_fill_color="orange", source=source2,legend="California live oak")
    p.circle(x="longitude", y="latitude", size="current_diameter",line_color="black", fill_color="#00FF00", fill_alpha=0.5, hover_line_color="orange", hover_fill_color="orange",source=source3,legend="Douglas-fir")
    p.circle(x="longitude", y="latitude", size="current_diameter",line_color="black", fill_color="#FFFF00", fill_alpha=0.5,hover_line_color="orange", hover_fill_color="orange", source=source4,legend="Alaska yellow-cedar")
    p.circle(x="longitude", y="latitude", size="current_diameter",line_color="black", fill_color="#CCEEFF", fill_alpha=0.5, hover_line_color="orange", hover_fill_color="orange",source=source5,legend="Coulter pine")
    p.circle(x="longitude", y="latitude", size="current_diameter",line_color="black", fill_color="#FFE4E1", fill_alpha=0.5,hover_line_color="orange", hover_fill_color="orange", source=source6,legend="Oregon white oak")
    p.circle(x="longitude", y="latitude", size="current_diameter",line_color="black", fill_color="#CD1076", fill_alpha=0.5, hover_line_color="orange", hover_fill_color="orange",source=source7,legend="willow spp.")
    p.add_tools(HoverTool(tooltips=[("current_diameter", "@current_diameter"), ("total_height", "@total_height"),("plot_county_code","@plot_county_code")]))
    p.legend.location = "top_right"
    p.legend.click_policy="hide"

    show(p)

import seaborn as sns
import matplotlib.pyplot as plt 

# visualize the relationship between the features and the response using scatterplots

sns.pairplot(westtree1, x_vars=['measurement_year','current_diameter','species_group_code','plot_state_code'], y_vars='total_height', height=7, aspect=0.8,kind='reg')
plt.show()
new_examDf = westtree1[['current_diameter','species_group_code','plot_state_code','total_height']]
print(new_examDf.corr())
X = westtree1[['current_diameter']]
y = westtree1[['total_height']]
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
print(model.coef_)
print(model.intercept_)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

X_train,X_test, y_train, y_test = train_test_split(X, y, random_state=1)
y_train_pred = model.predict(X_train)
plt.plot(X_train, y_train_pred, color='blue', linewidth=2, label="best line")
plt.scatter(X_train, y_train, color="darkgreen", label="train data")
plt.scatter(X_test, y_test, color="red", label="test data")
plt.legend(loc=2)
plt.xlabel("diameter")
plt.ylabel("tree height")
plt.show()


query2 = """
SELECT
    plot_state_code,
    plot_inventory_year,
    species_common_name,
    uncompacted_live_crown_ratio,
    total_height,
    mortality_year,
    crown_light_exposure_code,
    crown_position_code,
    current_diameter
FROM
    `bigquery-public-data.usfs_fia.plot_tree`
WHERE
    plot_state_code = 53
    OR plot_state_code = 06
    OR plot_state_code = 41
;
        """
response2 = usfs.query_to_pandas_safe(query2, max_gb_scanned=10)
response2.head(10)
westtree2 = response2.dropna(axis=0,how='any')
westtree2.head(10)
query3 = """
SELECT
    plot_state_code,
    plot_inventory_year,
    measurement_year,
    species_common_name,
    species_group_code,
    total_height,
    mortality_year,
    current_diameter
FROM
    `bigquery-public-data.usfs_fia.plot_tree`
WHERE
    plot_state_code = 53
;
        """
response3 = usfs.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)
westtree3 = response3.dropna(axis=0,how='any')
import seaborn as sns
import matplotlib.pyplot as plt 

# visualize the relationship between the features and the response using scatterplots

sns.pairplot(westtree3, x_vars=['total_height','current_diameter','species_group_code'], y_vars='mortality_year', size=7, aspect=0.8,kind='reg')
plt.show()
new_examDf1 = westtree3[['total_height','current_diameter','species_group_code','mortality_year']]
print(new_examDf1.corr())