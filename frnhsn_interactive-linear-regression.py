# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# bokeh packages

from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure, output_file, show

from bokeh.models import ColumnDataSource,CustomJS,HoverTool,CategoricalColorMapper

from bokeh.layouts import row,column,gridplot

from bokeh.models.widgets import Tabs,Panel

output_notebook()
def generate_linear_dataset(slope, n, std_dev):

    # Generate x as an array of `n` samples which can take a value between 0 and 100

    x = np.random.random(size=n) * 100

    # Generate the random error of n samples, with a random value from a normal distribution, with a standard

    # deviation provided in the function argument

    e = np.random.randn(n) * std_dev

    # Calculate `y` according to the equation discussed

    y = x * slope + e

    return x, y
s2.data
output_file("callback.html")



x, y = generate_linear_dataset(2,100,20)





s1 = ColumnDataSource(data=dict(x=x, y=y))

p1 = figure(plot_width=400, plot_height=400, tools="lasso_select", title="Seleksi di sini")

p1.circle('x', 'y', source=s1, alpha=0.6)



s2 = ColumnDataSource(data=dict(x=[], y=[]))

p2 = figure(plot_width=400, plot_height=400, x_range=(min(x), max(x)), y_range=(min(y), max(y)),

            tools="", title="Regresi linear")

p2.circle('x', 'y', source=s2, alpha=0.6)





s3 = ColumnDataSource(data=dict(x=[0,10], y=[0,10]))

p2.line(x='x', y='y', color="orange", line_width=1, alpha=0.6, source=s3)





s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s2=s2, s3=s3), code="""

        var inds = cb_obj.indices;

        var d1 = s1.data;

        var d2 = s2.data;

        var d3 = s3.data;

        

        function linearRegression(y,x){

            var lr = {};

            var n = y.length;

            var sum_x = 0;

            var sum_y = 0;

            var sum_xy = 0;

            var sum_xx = 0;

            var sum_yy = 0;



            for (var i = 0; i < y.length; i++) {



                sum_x += x[i];

                sum_y += y[i];

                sum_xy += (x[i]*y[i]);

                sum_xx += (x[i]*x[i]);

                sum_yy += (y[i]*y[i]);

            } 



            lr['slope'] = (n * sum_xy - sum_x * sum_y) / (n*sum_xx - sum_x * sum_x);

            lr['intercept'] = (sum_y - lr.slope * sum_x)/n;

            lr['r2'] = Math.pow((n*sum_xy - sum_x*sum_y)/Math.sqrt((n*sum_xx-sum_x*sum_x)*(n*sum_yy-sum_y*sum_y)),2);



            return lr;

        }

        

        function makeArr(startValue, stopValue, cardinality) {

          var arr = [];

          var currValue = startValue;

          var step = (stopValue - startValue) / (cardinality - 1);

          for (var i = 0; i < cardinality; i++) {

            arr.push(currValue + (step * i));

          }

          return arr;

        }

        

        d2['x'] = []

        d2['y'] = []

        for (var i = 0; i < inds.length; i++) {

            d2['x'].push(d1['x'][inds[i]])

            d2['y'].push(d1['y'][inds[i]])

        }

        s2.change.emit();

        

        reg_result = linearRegression(d2['x'],d2['y']);

        

        d3['x'] = []

        d3['y'] = []

        d3['x'] = makeArr(Math.min.apply(null, d2['x']), Math.max.apply(null, d2['x']), 2)

        console.log(reg_result['slope'],reg_result['intercept'],reg_result['r2'])

        

        for (i in d3['x']) {

            d3['y'].push((d3['x'][i] - reg_result['intercept']) / reg_result['slope'])

        }

        

        s3.data = d3

        s3.change.emit();

        

    """)

)





layout = row(p1, p2)



show(layout)
