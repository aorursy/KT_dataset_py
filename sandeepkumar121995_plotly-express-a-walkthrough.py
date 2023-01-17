import plotly_express as px

gapminder = px.data.gapminder()

gapminder2007 = gapminder.query("year == 2007")



px.scatter(gapminder2007, x="gdpPercap", y="lifeExp")
px.scatter(gapminder2007, x="gdpPercap", y="lifeExp", color="continent")
px.scatter(gapminder2007, x="gdpPercap", y="lifeExp", color="continent", size="pop", size_max=60)
px.scatter(gapminder2007, x="gdpPercap", y="lifeExp", color="continent", size="pop", size_max=60, hover_name="country")
px.scatter(gapminder2007, x="gdpPercap", y="lifeExp", color="continent", size="pop", size_max=60,

          hover_name="country", facet_col="continent", log_x=True)
px.scatter(gapminder, x="gdpPercap", y="lifeExp",size="pop", size_max=60, color="continent", hover_name="country", 

           animation_frame="year", animation_group="country", log_x=True, range_x=[100,100000], range_y=[25,90],

           labels=dict(pop="Population", gdpPercap="GDP per Capita", lifeExp="Life Expectancy"))
px.choropleth(gapminder, locations="iso_alpha", color="lifeExp", hover_name="country", animation_frame="year",

              color_continuous_scale=px.colors.sequential.Plasma, projection="natural earth")
tips = px.data.tips()
px.histogram(tips, x="total_bill", y="tip", histfunc="sum", color="smoker")
px.box(tips, x="total_bill", y="day", orientation="h", color="smoker", notched=True,

       category_orders={"day": ["Thur", "Fri", "Sat", "Sun"]})
px.violin(tips, y="tip", x="smoker", color="sex", box=True, points="all")
px.scatter(tips, x="total_bill", y="tip", color="smoker", trendline="ols", marginal_x="violin", marginal_y="box")