import pandas as pd



pd.set_option("max_colwidth", 500)
train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

train = train.dropna()

train.head()
def highlight_selected_text(row):

    text = row["text"]

    selected_text = row["selected_text"]

    sentiment = row["sentiment"]



    color = {

        "positive": "green",

        "negative": "red",

        "neutral": "blue",

    }[sentiment]



    highlighted = f'<span style="color: {color}; font-weight: bold">{selected_text}</span>'

    return text.replace(selected_text, highlighted)
train["highlighted"] = train.apply(highlight_selected_text, axis=1)

train.head()
from IPython.display import display, HTML



display(HTML(train.sample(30).to_html(escape=False)))
display(HTML(train[train["sentiment"] == "positive"][["highlighted"]].sample(30).to_html(escape=False)))
display(HTML(train[train["sentiment"] == "negative"][["highlighted"]].sample(30).to_html(escape=False)))
display(HTML(train[train["sentiment"] == "neutral"][["highlighted"]].sample(30).to_html(escape=False)))