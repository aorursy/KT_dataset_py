# !pip install nb_black

# %load_ext nb_black

import pandas as pd

import numpy as np

import plotly.graph_objects as go

import plotly.io as pio

from ipywidgets import widgets



pio.templates.default = "plotly_white"



data_path = "../input/kaggle-survey-2019/"

multiple_choice_responses_19 = pd.read_csv(

    data_path + "multiple_choice_responses.csv", engine="c", low_memory=False

)

questions_only_19 = pd.read_csv(data_path + "questions_only.csv", engine="c")
# Table with countries and respondents, ordered descending.

# It's necessary later to make the data proportional to the number of the respondents of the country.

countries = (

    multiple_choice_responses_19["Q3"][1:]

    .value_counts()

    .to_frame(name="count")

    .rename_axis("country")

    .reset_index()

)
def where(question):

    """

    This function gets the data for the question, processes it and plot a Choropleth map

    

    :param question: Question to plot

    :return: DataFrame with country and percentage of each response of the question

    """

    # Skip the Question 3 'In which country do you currently reside?'    

    if question == "Q3":

        return



    # Plot title    

    title = f"<b>{question}</b>: {questions_only_19[question].values[0]}"

    

    # NORMAL QUESTION

    # ---------------    

    # If the question is single choice, it is in the multiple_choice_responses columns



    if question in multiple_choice_responses_19.columns:

        # Get the country and question #n columns of the multiple_choice_responses_19 table

        # and skip the first row        

        m = multiple_choice_responses_19[["Q3", question]][1:]

        # Create a count column        

        m["count"] = 1

        # Do the magic        

        p = (

            m.groupby(["Q3", question]) # Group the country - response combination

            .agg(np.sum) # Get the sum

            .reset_index() # Plot.ly needs normal columns

            .rename(columns={"Q3": "country"}) # Rename country column

        )



    # MULTIPLE CHOICE QUESTION

    # ------------------------

    # If the question is multiple choice, the question is followed by _Part_

    # Get all columns with _Part_

    

    else:

        m = multiple_choice_responses_19[1:].filter(

            like=f"{question}_Part_", axis="columns"

        )

        # Get the relation of the part number and the most frequent value       

        parts = {v + 1: k for v, k in enumerate(m.mode().values[0].tolist())}

        # Add the country column 

        m["country"] = multiple_choice_responses_19["Q3"][1:]

        # Group the table by country and do the counting

        g = m.groupby(["country"]).agg("count").reset_index()

        # Convert a wide table to a long table using _Part_ columns

        p = (

            pd.wide_to_long(g, stubnames=f"{question}_Part_", i="country", j="part")

            .reset_index()

            .rename(columns={f"{question}_Part_": "count"})

        )

        # Create a new column with the part most frequent value instead of the part number

        p[question] = p["part"].map(parts)



    # COMMON

    # ------

    # Instead of the value, get the percentage for each country

    

    p["percentage"] = p.apply(

        lambda x: x["count"]

        / countries[countries["country"] == x["country"]]["count"].values[0],

        axis="columns",

    )

    

    return p
# Create the options for the questions widget

# It should be a list of tuples with the index and value (text shown)

options = [

    (f"{index}: {value}", index)

    for index, value in questions_only_19.iloc[:, 1:].T[0].items()

]

del options[2]



# Question dropdown widget

question = widgets.Dropdown(

    options=options, value="Q1", description="Question:", disabled=False,

)



# Slider widget

slider = widgets.SelectionSlider(

    options=[""],

    value="",

    description="",

    disabled=False,

    continuous_update=False,

    orientation="horizontal",

    readout=True,

)



# Color widget

color = widgets.Dropdown(

    options=["aggrnyl", "agsunset", "algae", "amp", "armyrose", "balance", "blackbody", "bluered", "blues", "blugrn", "bluyl", "brbg", "brwnyl", "bugn", "bupu", "burg", "burgyl", "cividis", "curl", "darkmint", "deep", "delta", "dense", "earth", "edge", "electric", "emrld", "fall", "geyser", "gnbu", "gray", "greens", "greys", "haline", "hot", "hsv", "ice", "icefire", "inferno", "jet", "magenta", "magma", "matter", "mint", "mrybm", "mygbm", "oranges", "orrd", "oryel", "peach", "phase", "picnic", "pinkyl", "piyg", "plasma", "plotly3", "portland", "prgn", "pubu", "pubugn", "puor", "purd", "purp", "purples", "purpor", "rainbow", "rdbu", "rdgy", "rdpu", "rdylbu", "rdylgn", "redor", "reds", "solar", "spectral", "speed", "sunset", "sunsetdark", "teal", "tealgrn", "tealrose", "tempo", "temps", "thermal", "tropic", "turbid", "twilight", "viridis", "ylgn", "ylgnbu", "ylorbr", "ylorrd"],

    value="deep",

    description="Color:",

    disabled=False,

)



# Basic data part of the plot. Only common configuration.

data = go.Choropleth(

    locations=[],

    locationmode="country names",

    colorbar=go.choropleth.ColorBar(tickformat=",.0%"),

    hovertemplate="%{location}: %{z:.2%}",

)



# Basic layout part of the plot.

layout = go.Layout(sliders=[go.layout.Slider(active=0)], height=800)



# Figure

g = go.FigureWidget(data=data, layout=layout)



# Callback that the widgets call

def response(change):

    # Get the information of the question

    p = where(question.value)

    # Update values

    with g.batch_update():

        g.layout.title.text = (

            f"<b>{question.value}</b>: {questions_only_19[question.value].values[0]}"

        )

        slider.options = p[question.value].unique().tolist()

        slider.description = f"Responses: "

        g.data[0].locations = p[p[question.value] == slider.value]["country"]

        g.data[0].z = p[p[question.value] == slider.value]["percentage"]

        g.data[0].colorscale = color.value





# Set buttons' callback

question.observe(response, names="value")

slider.observe(response, names="value")

color.observe(response, names="value")



# Force refreshing at init

response("refresh")



# Plot components

control = widgets.HBox([question, color])

widgets.VBox([control, g, slider])