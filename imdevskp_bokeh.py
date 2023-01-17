from bokeh.io import output_file, show, output_notebook

from bokeh.plotting import figure
xs = [1,2,3,4,5]

ys = [8,6,5,2,3]
plot= figure(plot_width=400, tools='pan, box_zoom')

plot.circle(xs, ys)

output_file('circles.html')

output_notebook()

show(plot)
plot = figure()

plot.circle(x=10, y=[2,5,8,12], size=[10,20,30,40])

output_notebook()

show(plot)
plot = figure()

plot.square(x=10, y=[2,5,8,12], size=[10,20,30,40])

output_notebook()

show(plot)
# plot.line?
x = [1,2,3,4,5]

y = [8,6,5,2,3]

z = [7,5,4,3,2]
plot = figure()

plot.line(x, y, line_width=3, color='red', alpha=0.5)

output_notebook()

show(plot)
plot = figure()

plot.line(x, y, line_width=2)

plot.circle(x, y, fill_color='white', size=10)

output_notebook()

show(plot)
plot = figure()

plot.line(x, y, line_width=3, color='green', alpha=0.5, legend_label='a')

plot.line(x, z, line_width=3, color='purple', alpha=0.5, legend_label='b')

output_notebook()

show(plot)
xs = [[1,5,2.5]]

ys = [[1,1,5]]



plot = figure()

plot.patches(xs, ys, 

             fill_color=['blue'],

             line_color='white')

output_notebook()

show(plot)
xs = [[1,1,4,4]]

ys = [[1,4,4,1]]



plot = figure()

plot.patches(xs, ys, 

             fill_color=['red'],

             line_color='white')

output_notebook()

show(plot)
xs = [[1,2,3,4,3,2]]

ys = [[2,1,1,2,3,3]]



plot = figure()

plot.patches(xs, ys, 

             fill_color=['purple'],

             line_color='white')

output_notebook()

show(plot)
xs = [ [1,1,2,2], [2,2,4], [2,2,3,3] ]

ys = [ [2,5,5,2], [3,5,5], [2,3,4,2] ]



plot = figure()

plot.patches(xs, ys, 

             fill_color=['red', 'blue','green'],

             line_color='white')

output_notebook()

show(plot)