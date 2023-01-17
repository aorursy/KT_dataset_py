import plotly.express as px
df = px.data.iris()

fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

fig.show()
df = px.data.iris()

fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", marginal_y="violin",

           marginal_x="box", trendline="ols", template="simple_white")

fig.show()
df = px.data.iris()

df["e"] = df["sepal_width"]/100

fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", error_x="e", error_y="e")

fig.show()
df = px.data.tips()

fig = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group")

fig.show()
df = px.data.tips()

fig = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group", facet_row="time", facet_col="day",

       category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]})

fig.show()
df = px.data.iris()

fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"], color="species")

fig.show()
df = px.data.iris()

fig = px.parallel_coordinates(df, color="species_id", labels={"species_id": "Species",

                  "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",

                  "petal_width": "Petal Width", "petal_length": "Petal Length", },

                    color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)

fig.show()
df = px.data.tips()

fig = px.parallel_categories(df, color="size", color_continuous_scale=px.colors.sequential.Inferno)

fig.show()
df = px.data.gapminder()

fig = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp", size="pop", color="continent",

           hover_name="country", log_x=True, size_max=60)

fig.show()
df = px.data.gapminder()

fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",

           size="pop", color="continent", hover_name="country", facet_col="continent",

           log_x=True, size_max=45, range_x=[100,100000], range_y=[25,90])

fig.show()
df = px.data.gapminder()

fig = px.line(df, x="year", y="lifeExp", color="continent", line_group="country", hover_name="country",

        line_shape="spline", render_mode="svg")

fig.show()
df = px.data.gapminder()

fig = px.area(df, x="year", y="pop", color="continent", line_group="country")

fig.show()
df = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")

df.loc[df['pop'] < 2.e6, 'country'] = 'Other countries' # Represent only large countries

fig = px.pie(df, values='pop', names='country', title='Population of European continent')

fig.show()
df = px.data.gapminder().query("year == 2007")

fig = px.sunburst(df, path=['continent', 'country'], values='pop',

                  color='lifeExp', hover_data=['iso_alpha'])

fig.show()
import numpy as np

df = px.data.gapminder().query("year == 2007")

fig = px.treemap(df, path=[px.Constant('world'), 'continent', 'country'], values='pop',

                  color='lifeExp', hover_data=['iso_alpha'])

fig.show()
df = px.data.tips()

fig = px.histogram(df, x="total_bill", y="tip", color="sex", marginal="rug", hover_data=df.columns)

fig.show()
df = px.data.tips()

fig = px.violin(df, y="tip", x="smoker", color="sex", box=True, points="all", hover_data=df.columns)

fig.show()
df = px.data.iris()

fig = px.density_heatmap(df, x="sepal_width", y="sepal_length", marginal_x="rug", marginal_y="histogram")

fig.show()
df = px.data.iris()

fig = px.density_contour(df, x="sepal_width", y="sepal_length", color="species", marginal_x="rug", marginal_y="histogram")

fig.show()
df = px.data.carshare()

fig = px.scatter_mapbox(df, lat="centroid_lat", lon="centroid_lon", color="peak_hour", size="car_hours",

                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,

                  mapbox_style="carto-positron")

fig.show()
df = px.data.election()

geojson = px.data.election_geojson()



fig = px.choropleth_mapbox(df, geojson=geojson, color="Bergeron",

                           locations="district", featureidkey="properties.district",

                           center={"lat": 45.5517, "lon": -73.7073},

                           mapbox_style="carto-positron", zoom=9)

fig.show()
df = px.data.gapminder()

fig = px.scatter_geo(df, locations="iso_alpha", color="continent", hover_name="country", size="pop",

               animation_frame="year", projection="natural earth")

fig.show()
df = px.data.wind()

fig = px.scatter_polar(df, r="frequency", theta="direction", color="strength", symbol="strength",

            color_discrete_sequence=px.colors.sequential.Plasma_r)

fig.show()
df = px.data.wind()

fig = px.line_polar(df, r="frequency", theta="direction", color="strength", line_close=True,

            color_discrete_sequence=px.colors.sequential.Plasma_r)

fig.show()
df = px.data.wind()

fig = px.bar_polar(df, r="frequency", theta="direction", color="strength", template="plotly_dark",

            color_discrete_sequence= px.colors.sequential.Plasma_r)

fig.show()