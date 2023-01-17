import pandas as pd
from PIL import Image


# Load data.
df = pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")

# Correct for Equirectangular map projection.
df.loc[:, 'Lat'] *= 2

# Combine Province/State with Country/Region.
def combine(row):
    ps, cr = row['Province/State'], row['Country/Region']
    return cr if pd.isnull(ps) else "{}, {}".format(ps, cr)
df['Country/Region'] = df.apply(combine, axis=1)

# Add new column representing the number of active cases.
def active(row):
    return row['Confirmed'] - row['Deaths'] - row['Recovered']
df = df.dropna(subset=['Confirmed', 'Deaths', 'Recovered'])
df['Active'] = df.apply(active, axis=1)
df = df[df['Active'] >= 0]

# Equirectangular map projection image.
img = Image.open("/kaggle/input/equirectangularmap/Equirectangular.jpg")
import plotly.express as px
import plotly.offline as py


def make_fig(data_frame, series, bg_img):
    """
    Create the COVID-19 interactive visualization from the given data.
    
    :param data_frame: the dataset
    :type data_frame: pandas DataFrame
    :param series: one of ['Confirmed', 'Deaths', 'Recovered', 'Active']
    :type series: string
    :param bg_img: Equirectangular map projection image
    :type bg_img: PIL Image
    :return: the resulting figure
    :rtype: plotly Figure
    """
    if series not in ['Confirmed', 'Deaths', 'Recovered', 'Active']:
        raise ValueError("'series' must be one of ['Confirmed', 'Deaths', 'Recovered', 'Active']")
    
    # Make interactive scatter plot figure.
    fig = px.scatter(
        data_frame,
        x = 'Long',
        y = 'Lat',
        animation_frame = 'Date',
        animation_group = 'Country/Region',
        size = series,
        hover_name = 'Country/Region',
        size_max = 40,
        range_x = [-180, 180],
        range_y = [-180, 180]
    )
    
    # Add the title and axes labels.
    count_type = "Current" if series == 'Active' else "Cumulative"
    series_type = series.lower() + ("" if series == 'Deaths' else " cases")
    start_date = data_frame['Date'].iloc[0]
    end_date = data_frame['Date'].iloc[-1]
    title = "{} number of {} from {} to {}".format(
        count_type, series_type, start_date, end_date)
    fig.update_layout(
        title = {
            'text': title,
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title = "Longitude",
        yaxis_title = "Latitude"
    )
    
    # Remove axes lines and zero lines.
    fig.update_layout(
        xaxis = dict(showgrid = False, zeroline = False),
        yaxis = dict(showgrid = False, zeroline = False),
    )

    # Add Equirectangular map projection as background image.
    fig.add_layout_image(
        dict(
            source = bg_img,
            xref = "x",
            yref = "y",
            x = -180,
            y = 180,
            sizex = 360,
            sizey = 360,
            sizing = "stretch",
            opacity = 1.0,
            layer = "below"
        )
    )
    fig.update_layout(template = "plotly_white")
    
    return fig

fig = make_fig(df, 'Confirmed', img)
fig.show()
fig = make_fig(df, 'Deaths', img)
fig.show()
fig = make_fig(df, 'Recovered', img)
fig.show()
fig = make_fig(df, 'Active', img)
fig.show()