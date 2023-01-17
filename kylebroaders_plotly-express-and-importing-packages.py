import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",width=800, height=600)
fig.update_xaxes(title_text='Sepal Width (mm)')
fig.update_yaxes(title_text='Sepal Length (mm)')
fig.update_layout(title_text="Sepal Dimensions")
fig.show()
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 template="simple_white",width=800, height=600)
fig.update_xaxes(title_text='Sepal Width (mm)')
fig.update_yaxes(title_text='Sepal Length (mm)')
fig.update_layout(title_text="Sepal Dimensions")
fig.show()
df = px.data.iris()
fig = px.scatter(df, x="sepal_length", y="petal_length", color="species", 
                 marginal_y="box", marginal_x="box", trendline="ols", 
                 template="simple_white",width=800, height=600)
fig.update_xaxes(title_text='Sepal Length (mm)')
fig.update_yaxes(title_text='Petal Length (mm)')
fig.update_layout(title_text="Petal vs Sepal Length")
fig.show()
df = px.data.gapminder()
fig = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp", size="pop", color="continent",
           hover_name="country", log_x=True, size_max=60)
fig.show()
df.head()
df = px.data.gapminder()
fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=45, range_x=[100,100000], range_y=[25,90])
fig.show()
df = px.data.gapminder()
fig = px.line(df.query("continent=='Oceania' or country=='Malaysia' or country=='Indonesia' or country=='Singapore' or country=='Philippines'"), 
              x="year", y="lifeExp", color="country")
fig.show()
df = px.data.gapminder()
fig = px.area(df.query("continent=='Oceania' or country=='Malaysia' or country=='Indonesia' or country=='Singapore' or country=='Philippines'"), 
              x="year", y="pop", color="country")
fig.show()
df = px.data.iris()
fig = px.histogram(df, x="sepal_length", color="species", marginal="rug", hover_data=df.columns)
fig.show()
df = px.data.iris()
fig = px.violin(df, y="sepal_length", color="species", box=True, points="all", hover_data=df.columns)
fig.show()

# To make a self-contained HTML file, just use FIG.write_html("NAME.html") where FIG is a plotly object

fig.write_html("test.html")

# To make a HTML file that does not include the plotly library use the include_plotlyjs=False argument. This is useful for embedding in a website that already has access to plotly
fig.write_html("test-noplotly.html", include_plotlyjs=False)

# To make a HTML file that refers to an online library, use the 'cdn' option. This will work as long as your end user has internet access.
fig.write_html("test-cdn.html", include_plotlyjs='cdn')
!pip install python-ternary

import ternary

import matplotlib
## This was copied directly from the GitHub page: https://github.com/marcharper/python-ternary

matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['figure.figsize'] = (4, 4)

scale = 40
figure, tax = ternary.figure(scale=scale) # Makes the figure object

# Draw Boundary and Gridlines

tax.gridlines(color="grey", multiple=2.5) # Makes gridlines with set color and spacing of gridlines
tax.boundary(linewidth=2.0)               # Makes outline with set width

# Set Axis labels and Title
fontsize = 12    # Label size
offset = 0.14    # Distance to labels
tax.set_title("Various Lines\n", fontsize=fontsize)
tax.right_corner_label("X", fontsize=fontsize)
tax.top_corner_label("Y", fontsize=fontsize)
tax.left_corner_label("Z", fontsize=fontsize)
tax.left_axis_label("Left label $\\alpha^2$", fontsize=fontsize, offset=offset)
tax.right_axis_label("Right label $\\beta^2$", fontsize=fontsize, offset=offset)
tax.bottom_axis_label("Bottom label $\\Gamma - \\Omega$", fontsize=fontsize, offset=offset)

# Draw lines parallel to the axes
tax.horizontal_line(16)
tax.left_parallel_line(10, linewidth=2., color='red', linestyle="--")
tax.right_parallel_line(20, linewidth=3., color='blue')

# Draw an arbitrary line, ternary will project the points for you
p1 = (22, 8, 10) # Starting point
p2 = (2, 22, 16) # End point
tax.line(p1, p2, linewidth=3., marker='s', color='green', linestyle=":")

tax.ticks(axis='lbr', multiple=5, linewidth=1, offset=0.025) # Draw the tickmarks on left, bottom, and right
tax.get_axes().axis('off') # Gets rid of outline box left over from MatPlotLib
tax.show()
