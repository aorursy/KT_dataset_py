from IPython.display import HTML

import pandas as pd



df = pd.read_csv("https://www1.nseindia.com/content/indices/mir.csv", header=None)



caption = df.iloc[0, 0]

df.columns = ["Sector", "1m", "3m", "6m", "12m"]

df = df[3:]

df.set_index("Sector", inplace=True)

df["1m"] = df["1m"].astype(float) / 100

df["3m"] = df["3m"].astype(float) / 100

df["6m"] = df["6m"].astype(float) / 100

df["12m"] = df["12m"].astype(float) / 100

df["diminishing_returns"] = False



mask_diminishing_returns = (

    (df["12m"] > df["6m"]) & (df["6m"] > df["3m"]) & (df["3m"] > df["1m"])

)

df.loc[mask_diminishing_returns, "diminishing_returns"] = True

df = df.sort_values(

    by=["diminishing_returns", "12m", "6m", "3m", "1m"], ascending=False

)
def color_negative_red(val):

    color = "red" if val < 0 else "black"

    return "color: %s" % color





def hover(hover_color="#f0f0f0"):

    return dict(selector="tr:hover", props=[("background-color", "%s" % hover_color)])





styles = [

    hover(),

    dict(selector="th", props=[("font-size", "105%"), ("text-align", "left")]),

    dict(selector="caption", props=[("caption-side", "top")]),

]



format_dict = {

    "1m": "{:.2%}",

    "3m": "{:.2%}",

    "6m": "{:.2%}",

    "12m": "{:.2%}",

}



html = (

    df.style.format(format_dict)

    .set_table_styles(styles)

    .applymap(color_negative_red)

    .highlight_max(color="lightgreen")

    .set_caption(caption)

)
html