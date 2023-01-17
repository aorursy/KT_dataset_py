import pandas as pd

import numpy as np

import os

import warnings

from math import pi

from bokeh.models import ColumnDataSource, Range1d

warnings.filterwarnings('ignore') # silence annoying warnings
import os

path = "../"

app_input="../input"

app_output="../output"

# Check current working directory.

print (os.getcwd())

print("\n")



# Now change the directory

os.chdir( app_input )



# Check current working directory.

print ("Directory changed successfully \n")

print (os.getcwd())



#print the list of csv files in the input folder

from subprocess import check_output

print ("\n")

print(check_output(["ls", "../input"]).decode("utf8"))
# aisles

aisles = pd.read_csv('../input/aisles.csv', engine='c')

print('Total aisles: {}'.format(aisles.shape[0]))

aisles.head()
# departments

departments = pd.read_csv('../input/departments.csv', engine='c')

print('Total departments: {}'.format(departments.shape[0]))

departments.head()
# products

products = pd.read_csv('../input/products.csv', engine='c')

print('Total products: {}'.format(products.shape[0]))

products.head(5)
# combine aisles, departments and products (left joined to products)

goods = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')

# to retain '-' and make product names more "standard"

goods.product_name = goods.product_name.str.replace(' ', '_').str.lower() 



goods.head()
"""Note the concept of reset_index is important here. Without reset_index groupby values 

becomes index instead of becoming a column.

Alternatively, use as_index=False"""

#df = pd.DataFrame(goods.groupby(['department']).count()['product_id','aisle_id'].sort_values(ascending=False)).reset_index()

gb = goods.groupby(['department']).agg({

    "product_id": pd.Series.nunique, "aisle_id": pd.Series.nunique}).rename(columns={

    'product_id': 'count_products_by_dept', 'aisle_id': 'count_aisle_by_dept'})

df = pd.DataFrame(gb).reset_index()

source = ColumnDataSource(df)

df.head()


from bokeh.layouts import gridplot

from bokeh.plotting import figure, show, output_notebook

from bokeh.palettes import all_palettes as alp

from bokeh.models import Range1d, PanTool, ResetTool, HoverTool



#x_range displays the labels for the x axis

p1 = figure(title="No of Products by Department",background_fill_color="#E8DDCB",x_range=list(df["department"]),

            tools="pan,lasso_select,reset", active_drag="lasso_select",)

p2 = figure(title="No of Aisles by Department",background_fill_color="#E8DDCB",x_range=list(df["department"]),

            tools="pan,lasso_select,reset", active_drag="lasso_select",)







#very imp concept here. We are placing the major ticks using the width list

width = [index+.5 for index,dep in enumerate(df.department)]

p1.quad(top="count_products_by_dept", bottom=0, left=width[:-1], right=width[1:],

        #fill_color="#036564", legend="department",source=source)

        fill_color="#036564", source=source)



p2.quad(top="count_aisle_by_dept", bottom=0, left=width[:-1], right=width[1:],

        #fill_color="#036564", legend="department",source=source)

        fill_color="#036564", source=source)







#style the axis

p1.xaxis.major_label_orientation = pi/2 #we want to display the vertical text using this 

p1.xaxis.axis_label = 'Departments'

p1.yaxis.axis_label = 'Number of Products'



p2.xaxis.major_label_orientation = pi/2 #we want to display the vertical text using this 

p2.xaxis.axis_label = 'Departments'

p2.yaxis.axis_label = 'Number of Aisle'





output_notebook()



show(gridplot(p1,p2, ncols=2, plot_width=400, plot_height=400),notebook_handle=True)