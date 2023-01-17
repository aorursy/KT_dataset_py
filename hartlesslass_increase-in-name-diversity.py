# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import bq_helper
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
usa_names=bq_helper.BigQueryHelper(active_project="bigquery-public-data", 
                              dataset_name="usa_names")
usa_names.list_tables()
usa_names.table_schema("usa_1910_current")
usa_names.head('usa_1910_current', selected_columns='name', num_rows=10)
query = """select a.year, a.gender,a.name
    from `bigquery-public-data.usa_names.usa_1910_current` a
    inner join
            (SELECT year,gender,max(number) number
            FROM `bigquery-public-data.usa_names.usa_1910_current`
            GROUP BY year,gender) b
    on a.year=b.year and a.gender=b.gender and a.number=b.number
    ORDER BY a.year,a.gender
            """

names = usa_names.query_to_pandas_safe(query)
names[(names.year>2000) & (names.gender=='M')]
query2 = """select year, gender,count(name) name_diversity, sum(number) births
    from `bigquery-public-data.usa_names.usa_1910_current`
    GROUP BY year, gender
    ORDER BY year,gender   
            """

#usa_names.estimate_query_size(query2)
diversity = usa_names.query_to_pandas_safe(query2)
diversity['ratio']=diversity.name_diversity/diversity.births
diversity
#diversity.head()
trace = go.Scatter(
    x = diversity.year[:10],
    y = diversity.name_diversity[:10]
)

diversity.year[:10]
import plotly.plotly as py
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

N = 500
random_x = np.linspace(0, 1, N)
random_y = np.random.randn(N)

# Create a trace
trace = go.Scatter(
    x = random_x,
    y = random_y
)

data = [trace]

py.iplot(data, filename='basic-line')