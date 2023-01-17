!pip install pandas-bokeh
import numpy as np
import pandas as pd
import pandas_bokeh
pandas_bokeh.output_notebook()
pd.set_option('plotting.backend', 'pandas_bokeh')
# Create Bokeh-Table with DataFrame:
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource
df = pd.read_csv('../input/state-wise-power-consumption-in-india/dataset_tk.csv')
df_long = pd.read_csv('../input/state-wise-power-consumption-in-india/long_data_.csv')
df["Date"]=df["Unnamed: 0"]
df['Date'] = pd.to_datetime(df.Date, dayfirst=True)
df = df.drop(["Unnamed: 0"], axis = 1) 
df.info()
df['NR'] = df['Punjab']+ df['Haryana']+ df['Rajasthan']+ df['Delhi']+df['UP']+df['Uttarakhand']+df['HP']+df['J&K']+df['Chandigarh']
df['WR'] = df['Chhattisgarh']+df['Gujarat']+df['MP']+df['Maharashtra']+df['Goa']+df['DNH']
df['SR'] = df['Andhra Pradesh']+df['Telangana']+df['Karnataka']+df['Kerala']+df['Tamil Nadu']+df['Pondy']
df['ER'] = df['Bihar']+df['Jharkhand']+ df['Odisha']+df['West Bengal']+df['Sikkim']
df['NER'] =df['Arunachal Pradesh']+df['Assam']+df['Manipur']+df['Meghalaya']+df['Mizoram']+df['Nagaland']+df['Tripura']
df_line = pd.DataFrame({"Northern Region": df["NR"].values,
                        "Southern Region": df["SR"].values,
                        "Eastern Region": df["ER"].values,
                        "Western Region": df["WR"].values,
                        "North Eastern Region": df["NER"].values},index=df.Date)

df_line.plot_bokeh(kind="line",title ="India - Power Consumption Regionwise",
                   figsize =(1000,800),
                   xlabel = "Date",
                   ylabel="MU(millions of units)"
                   )
df_line.plot_bokeh(kind="bar",title ="India - Power Consumption Regionwise",figsize =(1000,800),xlabel = "Date",ylabel="MU(millions of units)")
df_line.plot_bokeh(kind="point",title ="India - Power Consumption Regionwise",figsize =(1000,800),xlabel = "Date",ylabel="MU(millions of units)")
df_line.plot_bokeh(kind="hist",title ="India - Power Consumption Regionwise",
                   figsize =(1000,800),
                   xlabel = "Date",
                   ylabel="MU(millions of units)"
                )
df_line = pd.DataFrame({"Northern Region": df["NR"].values,
                        "Southern Region": df["SR"].values,
                        "Eastern Region": df["ER"].values,
                        "Western Region": df["WR"].values,
                        "North Eastern Region": df["NER"].values},index=df.Date)

df_line.plot_bokeh(kind="line",title ="India - Power Consumption Regionwise",
                   figsize =(1000,800),
                   xlabel = "Date",
                   ylabel="MU(millions of units)",rangetool=True
                   )
df_line.plot_bokeh.point(
    x=df.Date,
    xticks=range(0,1),
    size=5,
    colormap=["#009933", "#ff3399","#ae0399","#220111","#890300"],
    title=" Point Plot - India Power Consumption",
    fontsize_title=20,
    marker="x",figsize =(1000,800))
df_line.plot_bokeh.step(
    x=df.Date,
    xticks=range(-1, 1),
    colormap=["#009933", "#ff3399","#ae0399","#220111","#890300"],
    title="Step Plot - India Power Consumption",
    figsize=(1000,800),
    fontsize_title=20,
    fontsize_label=20,
    fontsize_ticks=20,
    fontsize_legend=8,
    )
df = pd.read_csv("../input/iris/Iris.csv")
df = df.sample(frac=1)
df.head()
data_table = DataTable(
    columns=[TableColumn(field=Ci, title=Ci) for Ci in df.columns],
    source=ColumnDataSource(df),
    height=300,
)

# Create Scatterplot:
p_scatter = df.plot_bokeh.scatter(
    x="PetalLengthCm",
    y="SepalWidthCm",
    category="Species",
    title="Iris DataSet Visualization",
    show_figure=False
)

# Combine Table and Scatterplot via grid layout:
pandas_bokeh.plot_grid([[data_table, p_scatter]], plot_width=400, plot_height=350)
data = {
    'Cars':
    ['Maruti Suzuki', 'Honda', 'Toyota', 'Hyundai', 'Benz', 'BMW'],
    '2018': [20000, 15722, 4340, 38000, 2890, 412],
    '2019': [19000, 13700, 340, 31200, 290, 234],
    '2020': [23456, 15891, 440, 36700, 890, 417]
}
df = pd.DataFrame(data).set_index("Cars")

p_bar = df.plot_bokeh.bar(
    ylabel="Price per Unit", 
    title="Car Units sold per Year", 
    alpha=0.6)
stacked_bar = df.plot_bokeh.bar(
    ylabel="Price per Unit", 
    title="Car Units sold per Year", 
    stacked=True,
    alpha=0.6)
#Reset index, such that "Cars" is now a column of the DataFrame:
df.reset_index(inplace=True)

#Create horizontal bar (via kind keyword):
p_hbar = df.plot_bokeh(
    kind="barh",
    x="Cars",
    ylabel="Price per Unit", 
    title="Car Units sold per Year", 
    alpha=0.6,
    legend = "bottom_right",
    show_figure=False)

#Create stacked horizontal bar (via barh accessor):
stacked_hbar = df.plot_bokeh.barh(
    x="Cars",
    stacked=True,
    ylabel="Price per Unit", 
    title="Car Units sold per Year", 
    alpha=0.6,
    legend = "bottom_right",
    show_figure=False)

#Plot all barplot examples in a grid:
pandas_bokeh.plot_grid([[p_bar, stacked_bar],
                        [p_hbar, stacked_hbar]], 
                       plot_width=450)
df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv",index_col='SalePrice')
numeric_features = df.select_dtypes(include=[np.number])
p_bar = numeric_features.plot_bokeh.bar(
    ylabel="Sale Price", 
    figsize=(1000,800),
    title="Housing Prices", 
    alpha=0.6)
!pip install pandas-bokeh
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# Import Bokeh Library for output
from bokeh.io import output_notebook
output_notebook()
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import LinearInterpolator,CategoricalColorMapper
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.palettes import Spectral8
data = pd.read_csv('../input/karnataka-state-education/Town-wise-education - Karnataka.csv')
data.info()
data.head()
data.tail()
data.shape
data.info()
data.isnull().sum()
data.describe(include = 'all')
categorical_features = data.select_dtypes(include=[np.object])
categorical_features.info()
for column_name in data.columns:
    if data[column_name].dtypes == 'object':
        data[column_name] = data[column_name].fillna(data[column_name].mode().iloc[0])
        unique_category = len(data[column_name].unique())
        print("Feature '{column_name}' has '{unique_category}' unique categories".format(column_name = column_name,
                                                                                         unique_category=unique_category))
data.drop('Table Name',axis =1,inplace = True)

data.drop('State Code',axis =1,inplace = True)

data.drop('Total/ Rural/ Urban',axis =1,inplace = True)

data.head()
data = data[data['Age-Group'] != 'All ages']
columns = [ 'Total - Persons',
           'Total - Males',
           'Total - Females',
           'Literate - Persons',
           'Literate - Males',
           'Literate - Females',
           'Educational Level - Middle Persons',
           'Educational Level - Middle Males',
           'Educational Level - Middle Females',
           'Educational Level - Matric/Secondary Persons',
           'Educational Level - Matric/Secondary Males',
           'Educational Level - Matric/Secondary Females',
           'Educational Level - Higher Secondary/Intermediate Pre-University/Senior Secondary Persons',
           'Educational Level - Higher Secondary/Intermediate Pre-University/Senior Secondary Males',
           'Educational Level - Higher Secondary/Intermediate Pre-University/Senior Secondary Females',
           'Educational Level - Non-technical Diploma or Certificate Not Equal to Degree Persons',
           'Educational Level - Non-technical Diploma or Certificate Not Equal to Degree Males',
           'Educational Level - Non-technical Diploma or Certificate Not Equal to Degree Females',
           'Educational Level - Technical Diploma or Certificate Not Equal to Degree Persons',
           'Educational Level - Technical Diploma or Certificate Not Equal to Degree Males',
           'Educational Level - Technical Diploma or Certificate Not Equal to Degree Females',
           'Educational Level - Graduate & Above Persons',
           'Educational Level - Graduate & Above Males',
           'Educational Level - Graduate & Above Females']                                                                            

data.drop(columns,axis =1,inplace = True)
source = ColumnDataSource(dict(
    x = data['District Code'],
    y = data['Illiterate - Persons'],
    area = data['Area Name'],
    illerate = data['Illiterate - Persons'],
    illerate_male = data['Illiterate - Males'],
    illerate_female = data['Illiterate - Females'],
    age = data['Age-Group']
)       
)

size_mapper = LinearInterpolator(
    x = [data['Illiterate - Persons'].min(),data['Illiterate - Persons'].max()],
    y = [2,100]
)

color_mapper = CategoricalColorMapper(
    factors = list(data['Area Name'].unique()),
    palette = Spectral8
)

PLOT_OPTS = dict(height = 800,width = 800,x_range = (1,30),y_range=(10,100000))

p = figure(title = 'Illiteracy District/Area Wise',
           toolbar_location = 'above',
           tools = [HoverTool(
               tooltips = [('Area ','@area'),
                           ('Illerate - Total Persons ','@illerate'),
                           ('Illerate - Total Males ','@illerate_male'),
                           ('Illerate - Total Females ','@illerate_female'),
                           ('Age Group ','@age'),
                        ],show_arrow = False)],
           x_axis_label = 'District Code',
           y_axis_label = 'No of Illiterates',
           **PLOT_OPTS)

p.circle(x='x',
         y='y', 
         size = {'field': 'illerate','transform':size_mapper},
         color = {'field': 'area','transform':color_mapper},
         alpha = 0.7,
         legend = 'area',
         source = source)
p.legend.location = (0,-50)
p.right.append(p.legend[0])
p.legend.border_line_color = None
show(p,notebook_handle=True)
source = ColumnDataSource(dict(
    x = data['District Code'],
    y = data['Educational Level - Literate without Educational Level Persons'],
    area = data['Area Name'],
    illerate = data['Educational Level - Literate without Educational Level Persons'],
    illerate_male = data['Educational Level - Literate without Educational Level Males'],
    illerate_female = data['Educational Level - Literate without Educational Level Females'],
    age = data['Age-Group']
)       
)

size_mapper = LinearInterpolator(
    x = [data['Educational Level - Literate without Educational Level Persons'].min(),
         data['Educational Level - Literate without Educational Level Persons'].max()],
    y = [2,100]
)

color_mapper = CategoricalColorMapper(
    factors = list(data['Area Name'].unique()),
    palette = Spectral8
)

PLOT_OPTS = dict(height = 800,width = 800,x_range = (1,30),y_range=(10,6000))

p = figure(title = 'Educational Level - Literate without Educational Level (District vs. Age Wise)',
           toolbar_location = 'above',
           tools = [HoverTool(
               tooltips = [('Area ','@area'),
                           ('Educational Level - Literate without Educational Level - Total Persons ','@illerate'),
                           ('Educational Level - Literate without Educational Level - Total Males ','@illerate_male'),
                           ('Educational Level - Literate without Educational Level - Total Females ','@illerate_female'),
                           ('Age Group ','@age'),
                        ],show_arrow = False)],
           x_axis_label = 'District Code',
           y_axis_label = 'No of Educational Level - Literate without Educational Level',
           **PLOT_OPTS)

p.circle(x='x',
         y='y', 
         size = {'field': 'illerate','transform':size_mapper},
         color = {'field': 'area','transform':color_mapper},
         alpha = 0.7,
         legend = 'area',
         source = source)
p.legend.location = (0,-50)
p.right.append(p.legend[0])
p.legend.border_line_color = None
show(p,notebook_handle=True)
source = ColumnDataSource(dict(
    x = data['District Code'],
    y = data['Educational Level - Below Primary Persons'],
    area = data['Area Name'],
    illerate = data['Educational Level - Below Primary Persons'],
    illerate_male = data['Educational Level - Below Primary Males'],
    illerate_female = data['Educational Level - Below Primary Females'],
    age = data['Age-Group']
)       
)

size_mapper = LinearInterpolator(
    x = [data['Educational Level - Below Primary Persons'].min(),
         data['Educational Level - Below Primary Persons'].max()],
    y = [2,50]
)

color_mapper = CategoricalColorMapper(
    factors = list(data['Area Name'].unique()),
    palette = Spectral8
)

PLOT_OPTS = dict(height = 800,width = 800,x_range = (1,30),y_range=(10,100000))

p = figure(title = 'Educational Level - Below Primary (District vs. Age Wise)',
           toolbar_location = 'above',
           tools = [HoverTool(
               tooltips = [('Area ','@area'),
                           ('Educational Level - Below Primary Total Persons ','@illerate'),
                            ('Educational Level - Below Primary Total Males ','@illerate_male'),
                           ('Educational Level -  Below Primary Total Females ','@illerate_female'),
                           ('Age Group ','@age'),
                        ],show_arrow = False)],
           x_axis_label = 'District Code',
           y_axis_label = 'No of Educational Level - Below Primary Persons',
           **PLOT_OPTS)

p.circle(x='x',
         y='y', 
         size = {'field': 'illerate','transform':size_mapper},
         color = {'field': 'area','transform':color_mapper},
         alpha = 0.7,
         legend = 'area',
         source = source)
p.legend.location = (0,-50)
p.right.append(p.legend[0])
p.legend.border_line_color = None
show(p,notebook_handle=True)
source = ColumnDataSource(dict(
    x = data['District Code'],
    y = data['Educational Level - Primary Persons'],
    area = data['Area Name'],
    illerate = data['Educational Level - Primary Persons'],
    illerate_male = data['Educational Level - Primary Males'],
    illerate_female = data['Educational Level - Primary Females'],
    age = data['Age-Group']
)       
)

size_mapper = LinearInterpolator(
    x = [data['Educational Level - Primary Persons'].min(),
         data['Educational Level - Primary Persons'].max()],
    y = [2,50]
)

color_mapper = CategoricalColorMapper(
    factors = list(data['Area Name'].unique()),
    palette = Spectral8
)

PLOT_OPTS = dict(height = 800,width = 800,x_range = (1,30),y_range=(10,100000))

p = figure(title = 'Educational Level - Primary (District vs. Age Wise)',
           toolbar_location = 'above',
           tools = [HoverTool(
               tooltips = [('Area ','@area'),
                           ('Educational Level -  Primary Total Persons ','@illerate'),
                           ('Educational Level -  Primary Total Male ','@illerate_male'),
                           ('Educational Level -  Primary Total Female ','@illerate_female'),
                           ('Age Group ','@age'),
                        ],show_arrow = False)],
           x_axis_label = 'District Code',
           y_axis_label = 'No of Educational Level - Primary Persons',
           **PLOT_OPTS)

p.circle(x='x',
         y='y', 
         size = {'field': 'illerate','transform':size_mapper},
         color = {'field': 'area','transform':color_mapper},
         alpha = 0.7,
         legend = 'area',
         source = source)
p.legend.location = (0,-50)
p.right.append(p.legend[0])
p.legend.border_line_color = None
show(p,notebook_handle=True)
source = ColumnDataSource(dict(
    x = data['District Code'],
    y = data['Unclassified - Persons'],
    area = data['Area Name'],
    illerate = data['Unclassified - Persons'],
    illerate_male = data['Unclassified - Males'],
    illerate_female = data['Unclassified - Females'],
    age = data['Age-Group']
)       
)

size_mapper = LinearInterpolator(
    x = [data['Unclassified - Persons'].min(),
         data['Unclassified - Persons'].max()],
    y = [1,100]
)

color_mapper = CategoricalColorMapper(
    factors = list(data['Area Name'].unique()),
    palette = Spectral8
)

PLOT_OPTS = dict(height = 800,width = 800,x_range = (1,30),y_range=(10,400))

p = figure(title = 'Unclassified -  (District vs. Age Wise)',
           toolbar_location = 'above',
           tools = [HoverTool(
               tooltips = [('Area ','@area'),
                           ('Unclassified - Total Persons ','@illerate'),
                           ('Unclassified - Total Males ','@illerate_male'),
                           ('Unclassified - Total Females ','@illerate_female'),
                           ('Age Group ','@age'),
                        ],show_arrow = False)],
           x_axis_label = 'District Code',
           y_axis_label = 'Unclassified - Persons',
           **PLOT_OPTS)

p.circle(x='x',
         y='y', 
         size = {'field': 'illerate','transform':size_mapper},
         color = {'field': 'area','transform':color_mapper},
         alpha = 0.7,
         legend = 'area',
         source = source)
p.legend.location = (0,-50)
p.right.append(p.legend[0])
p.legend.border_line_color = None

show(p,notebook_handle=True)
data['Total']=data['Illiterate - Persons']+data['Educational Level - Below Primary Persons']+data['Educational Level - Literate without Educational Level Persons']+data['Educational Level - Primary Persons']+data['Unclassified - Persons']
data['Total_Males']=data['Illiterate - Males']+data['Educational Level - Below Primary Males']+data['Educational Level - Literate without Educational Level Males']+data['Educational Level - Primary Males']+data['Unclassified - Males']
data['Total_Females']=data['Illiterate - Females']+data['Educational Level - Below Primary Females']+data['Educational Level - Literate without Educational Level Females']+data['Educational Level - Primary Females']+data['Unclassified - Females']

source = ColumnDataSource(dict(
    x = data['District Code'],
    y = data['Total'],
    area = data['Area Name'],
    illerate = data['Total'],
    illerate_male = data['Total_Males'],
    illerate_female = data['Total_Females'],
    age = data['Age-Group']
)       
)

size_mapper = LinearInterpolator(
    x = [data['Total'].min(),
         data['Total'].max()],
    y = [5,100]
)

color_mapper = CategoricalColorMapper(
    factors = list(data['Area Name'].unique()),
    palette = Spectral8
)

PLOT_OPTS = dict(height = 800,width = 800,x_range = (1,30),y_range=(10,120000))

p = figure(title = 'Summary -  (District vs. Age Wise)',
           toolbar_location = 'above',
           tools = [HoverTool(
               tooltips = [('Area ','@area'),
                           ('Summary','@illerate'),
                           ('Total Males ','@illerate_male'),
                           ('Total Females ','@illerate_female'),
                           ('Age Group ','@age'),
                        ],show_arrow = False)],
           x_axis_label = 'District Code',
           y_axis_label = 'Total Population needing Primary Education',
           **PLOT_OPTS)

p.circle(x='x',
         y='y', 
         size = {'field': 'illerate','transform':size_mapper},
         color = {'field': 'area','transform':color_mapper},
         alpha = 0.7,
         legend = 'area',
         source = source)
p.legend.location = (0,-50)
p.right.append(p.legend[0])
p.legend.border_line_color = None
show(p,notebook_handle=True)