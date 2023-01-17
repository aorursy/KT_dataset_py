import pandas as pd
from bokeh.plotting import figure, output_file, show,output_notebook
output_notebook()
    
#csv_path='../input/countries-of-the-world/countries of the world.csv'
#d1=pd.read_csv(csv_path)           
#d1.head()  
links={'GDP':'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/projects/coursera_project/clean_gdp.csv',\
       'unemployment':'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/projects/coursera_project/clean_unemployment.csv'}
df_GDP = pd.read_csv('../input/gdpdataset/clean_gdp.csv')
df_GDP.head()
csv_path='../input/unemployment-data/clean_unemployment.csv'
d1=pd.read_csv(csv_path)  
d1.head()
df_unemployment85 = d1[d1['unemployment']>8.5]
df_unemployment85.head()
def make_dashboard(x, gdp_change, unemployment, title, file_name):
    output_file(file_name)
    p = figure(title=title, x_axis_label='year', y_axis_label='%')
    p.line(x.squeeze(), gdp_change.squeeze(), color="firebrick", line_width=4, legend="% GDP change")
    p.line(x.squeeze(), unemployment.squeeze(), line_width=4, legend="% unemployed")
    show(p)

x = pd.DataFrame(df_GDP['date'])
gdp_change = pd.DataFrame(df_GDP['change-current'])
unemployment = pd.DataFrame(d1['unemployment'])
title = "DashBoard Title"
file_name = "index.html"
make_dashboard(x,gdp_change,unemployment,title,file_name)