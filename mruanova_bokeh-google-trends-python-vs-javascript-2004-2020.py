import pandas as pd

from bokeh.plotting import figure, show, output_notebook

from bokeh.models import ColumnDataSource

df = pd.read_csv('../input/pythonvsjavascript/google-trends-2004-2020.csv', parse_dates=['Week'])

source = ColumnDataSource(df)

plot = figure(x_axis_type="datetime")

legend_python = 'Python (Programming Language) Google Trends 2004-2020 USA'

legend_javascript = 'JavaScript (Programming Language) Google Trends 2004-2020 USA'

plot.line(x='Week', y='Python', line_width=1, source=source, color='blue', legend_label=legend_python)

plot.line(x='Week', y='JavaScript', line_width=1, source=source, color='red', legend_label=legend_javascript)

output_notebook() # show the output in jupyter notebook

show(plot)