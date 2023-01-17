import pandas as pd

import plotly_express as px  # The plotting library I will be showing as an example
data_dir = "../input/alsa/clinical_trials_cleaned.csv"

df = pd.read_csv(data_dir, index_col=0)



df.head()
# If the "Status" column contains a NaN, replace with TBD

df["Status"][pd.isnull(df["Status"])] = "TBD"

# If the "Completion Date" column contains a NaN, replace with TBD

df["Completion Date"][pd.isnull(df["Completion Date"])] = "TBD"
px.scatter_geo(

        df, 

        lon="LONGITUDE", 

        lat="LATITUDE", 

        color="Rank", 

        hover_name="LOCATION",

        hover_data=["Title", "Rank", "Start Date", "Completion Date", "Study Results"]

    )
px.scatter_geo(

        df, 

        lon="LONGITUDE", 

        lat="LATITUDE", 

        color="Rank",

        animation_frame="Completion Date",

        hover_name="LOCATION",

        hover_data=["Title", "Rank", "Start Date", "Completion Date", "Study Results"]

    )