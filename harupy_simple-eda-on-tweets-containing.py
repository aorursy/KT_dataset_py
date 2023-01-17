import re

import warnings



import pandas as pd

import seaborn as sns

import matplotlib as mpl



from IPython.display import display, HTML



warnings.filterwarnings("ignore")

pd.set_option("display.max_colwidth", 500)

mpl.rcParams['figure.dpi'] = 100

sns.set()
train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

train = train.dropna()

train.head()
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")

test.head()
PATTERN = r"([\*]{2,})"





def format_style(style):

    """

    Convert a dict representing css style into string that an HTML element can accept.

    """

    return "; ".join([": ".join(item) for item in style.items()])





def highlight_xxxx(df):

    """

    Highlight "****".

    """

    def highlight(text):

        style = format_style({

            "color": "red",

            "font-weight": "bold",

        })

        return re.sub(PATTERN, r'<span style="{}">\1</span>'.format(style), text)



    if "selected_text" in df.columns:

        return df.assign(

            text=df["text"].map(highlight),

            selected_text=df["selected_text"].map(highlight),

        )

    else:

        return df.assign(text=df["text"].map(highlight))





def filter_xxxx(df):

    """

    Filter rows containing "****".

    """

    if "selected_text" in df.columns:

        return df[df["text"].str.contains(PATTERN) & df["selected_text"].str.contains(PATTERN)]

    else:

        return df[df["text"].str.contains(PATTERN)]





def color_rows_by_sentiment(row):

    """

    Color rows by sentiment value.

    """

    colors = {

        "positive": "#e3fbe3",

        "negative": "#fdf0f2",

        "neutral": "white",

    }



    style = format_style({

        "background-color": colors.get(row["sentiment"]),

        "border": "1px solid grey",

    })



    return [style] * len(row)





def render(df):

    """

    Render a dataframe as HTML.

    """

    if isinstance(df, pd.DataFrame):

        display(HTML(df.to_html(escape=False)))

    elif isinstance(df, pd.io.formats.style.Styler):

        display(HTML(df.hide_index().render()))

    else:

        raise TypeError("Invalid object type: {}.".format(type(df)))
render(

    train

    .drop("textID", axis=1)

    .pipe(filter_xxxx)

    .pipe(highlight_xxxx)

    .groupby("sentiment").head(10)  # grab 10 tweets from each sentiment.

    .reset_index(drop=True)

    .sort_values("sentiment")

    .style

    .set_properties(**{'text-align': 'left'})

    .apply(color_rows_by_sentiment, axis=1)

)
train.pipe(filter_xxxx).shape
train.pipe(filter_xxxx)["sentiment"].value_counts().plot.bar()
render(

    test

    .drop("textID", axis=1)

    .pipe(filter_xxxx)

    .pipe(highlight_xxxx)

    .groupby("sentiment").head(10)  # grab 10 tweets from each sentiment.

    .reset_index(drop=True)

    .sort_values("sentiment")

    .style

    .set_properties(**{'text-align': 'left'})

    .apply(color_rows_by_sentiment, axis=1)

)
test.pipe(filter_xxxx).shape
test.pipe(filter_xxxx)["sentiment"].value_counts().plot.bar()