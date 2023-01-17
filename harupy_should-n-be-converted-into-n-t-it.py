import re



import pandas as pd

from IPython.display import display, HTML





pd.set_option("display.max_colwidth", 300)
PATTERN = r"([^ ]+)(n`\*\*\*\*)"







def format_style(style):

    return "; ".join([": ".join(item) for item in style.items()])





def color_matched(text):

    style1 = format_style({

        "color": "blue",

        "font-weight": "bold",

    })



    style2 = format_style({

        "color": "red",

        "font-weight": "bold",

    })

    return re.sub(PATTERN, r'<span style="{}">\1</span><span style="{}">\2</span>'.format(style1, style2), text)





def highlight(df):

    if "selected_text" in df.columns:

        return df.assign(

            text=df["text"].map(color_matched),

            selected_text=df["selected_text"].map(color_matched),

        )

    else:

        return df.assign(text=df["text"].map(color_matched))





def filter_matched(df):

    return df[df["text"].str.contains(PATTERN)]





def render(df):

    if isinstance(df, pd.DataFrame):

        display(HTML(df.to_html(index=False, escape=False)))

    elif isinstance(df, pd.io.formats.style.Styler):

        display(HTML(df.hide_index().render()))

    else:

        raise TypeError("Invalid object type: {}.".format(type(df)))
train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv").dropna()



render(

    train

    .pipe(filter_matched)

    .pipe(highlight)

)
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")



render(

    test

    .pipe(filter_matched)

    .pipe(highlight)

)