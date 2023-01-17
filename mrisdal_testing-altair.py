!pip install altair vega_datasets
import altair as alt

alt.renderers.enable('default')

from vega_datasets import data

url = data.cars.url



alt.Chart(url).mark_point().encode(

    x='Horsepower:Q',

    y='Miles_per_Gallon:Q'

)