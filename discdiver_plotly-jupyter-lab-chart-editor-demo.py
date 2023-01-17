# import
import plotly.io as pio
import plotly.graph_objs as go
# make a figure
# go to the plotly docs for help: https://plot.ly/python/figurewidget/

fig = go.FigureWidget()
fig.add_scatter(y=[1,2,3])
fig
# Save a figure to .json format with 
pio.write_json(fig, 'my_file.json')
fig_after_work = pio.read_json('my_file_revised.json', output_type='FigureWidget')
fig_after_work
fig_after_work.layout.title