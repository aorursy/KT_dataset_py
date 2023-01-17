import altair as alt
import itertools
import keras
import math
import numpy as np
import pandas as pd
import re
import string
import spacy
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Altair cannot do word cloud so I use Matplotlib just to render the image
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
%matplotlib inline
%config InlineBackend.figure_format = "retina"
webmddf = pd.read_csv("../input/webmd-drug-reviews-dataset/webmd.csv")
webmddf.head(3)
df = webmddf[["Age", "Condition", "Drug", "DrugId", "Satisfaction", "Sex", "Reviews"]]
df.isnull().sum()
df = df.dropna()
for col in df.columns:
    if df[col].dtype.kind == "O":
        df[col] = df[col].str.strip()
data = [[col, df[col].nunique()] for col in df.columns.difference(["Reviews"])]
uniques = pd.DataFrame(data=data, columns=["columns", "num of unique values"])

bars = (alt.Chart()
           .mark_bar(size=25, 
                     color="#FFAA00",
                     strokeWidth=1,
                     stroke="white",
                     strokeOpacity=0.7)
           .encode(x=alt.X(shorthand="num of unique values:Q",
                           scale=alt.Scale(type="log"),
                           axis=alt.Axis(title="num of unique values, log scaled")),
                   y=alt.Y("columns:O", sort="-x"),
                   tooltip=("num of unique values:Q",
                            "columns:O",),
                   color=alt.Color("num of unique values",
                                   scale=alt.Scale(scheme="lightgreyteal",
                                                   type="log")))
           .properties(title='Unique Values'))

text = (alt.Chart()
           .mark_text(align="left",
                      baseline="middle",
                      dx=3)
           .encode(x=alt.X(shorthand="num of unique values:Q"),
                   y=alt.Y("columns:O",
                           axis=alt.Axis(title="columns",
                                         grid=False),
                           sort="-x"),
                   text="num of unique values:Q"))

chart = ((alt.layer(bars, text, data=uniques)
             .configure(background='#11043a')
             .configure_title(font="Arial",
                              fontSize=18,
                              color="#e6f3ff",
                              dy=-10)
             .configure_text(color="white")
             .configure_legend(titleFontSize=12,
                               titleColor="white",
                               tickCount=10,
                               titleOpacity=0.8,
                               labelColor="white",
                               labelOpacity=0.7,
                               titlePadding=10)
             .configure_axis(titleFontSize=13,
                             titlePadding=20,
                             titleColor="white",
                             titleOpacity=0.8,
                             labelColor="white",
                             labelOpacity=0.7,
                             labelFontSize=11,
                             tickOffset=0,
                             grid=True,
                             gridOpacity=0.15)
             .configure_view(strokeWidth=0)
             .properties(height=200, width=680)))

chart
def missing_values(df):
    """Returns a summary of missing values in df"""
    nrows = df.shape[0]
    data = []
    
    def pct(n, total):
        return round(n/total, 2)
    
    for col in df.columns:

        # string (Object) type columns
        if df[col].dtype.kind == "O":
            df[col] = df[col].str.strip()
            nulls = df[df[col] == ""][col].count()
            nulls += df[col].isnull().sum()

        # numerical (int) type columns
        elif df[col].dtype.kind == "i":
            nulls = df[col].isnull().sum()

        pctofnulls = pct(nulls, nrows)
        data.extend(
            [{"column": col, "pct": 1-pctofnulls, "num of records": nrows-nulls, "type": "not missing"},
             {"column": col, "pct": pctofnulls, "num of records": nulls, "type": "missing"}])
    
    return pd.DataFrame(data)

missing = missing_values(df)

bars = (alt.Chart()
           .mark_bar(size=25, 
                     strokeWidth=1,
                     stroke="white",
                     strokeOpacity=0.7,
                     )
           .encode(x=alt.X("sum(num of records)",
                           axis=alt.Axis(title="number of records",
                                         grid=True)), 
                   y=alt.Y("column:O",
                           axis=alt.Axis(title="columns")),
                   tooltip=("column", "type", "num of records:Q",
                            alt.Tooltip("pct:Q", format=".1%")),
                   color=alt.Color("type",
                                   scale=alt.Scale(range=["#11043a", "#648bce"])))
           .properties(title="Missing Values"))

text = (alt.Chart()
           .mark_text(align="right",
                      dx=-1)
           .encode(x=alt.X("sum(num of records)", 
                           stack="zero"),
                   y=alt.Y("column"),
                   color=alt.Color("type",
                                   legend=None,
                                   scale=alt.Scale(range=["white"])),
                   text=alt.Text("pct", format=".0%")))

(alt.layer(bars, text, data=missing)
    .configure(background='#11043a')
    .configure_title(font="Arial",
                     fontSize=18,
                     color="#e6f3ff",
                     dy=-10)
    .configure_text(color="white")
    .configure_legend(titleFontSize=12,
                      titleColor="white",
                      tickCount=10,
                      titleOpacity=0.8,
                      labelColor="white",
                      labelOpacity=0.7,
                      titlePadding=10)
    .configure_axis(titleFontSize=13,
                    titlePadding=20,
                    titleColor="white",
                    titleOpacity=0.8,
                    labelFontSize=11,
                    labelColor="white",
                    labelOpacity=0.7,
                    tickOffset=0,
                    grid=False,
                    gridOpacity=0.15)
    .configure_view(strokeWidth=0)
    .resolve_scale(color='independent')
    .properties(height=300, width=680))
for col in ["Age", "Condition", "Sex", "Reviews"]:
    df = df[(df[col].astype(bool) & df[col].notnull())]
print(df["Satisfaction"].value_counts())
df = df[df["Satisfaction"] <= 5]
def relabel(x):
    return 0 if x < 3 else 1 if x == 3 else 2

df["Satisfaction"] = df["Satisfaction"].apply(relabel)
print(df["Sex"].value_counts())
drugs = {}
for drugid, drug in df[["DrugId", "Drug"]].itertuples(index=False):
    drugs.setdefault(drugid, set()).add(drug)
drugs = {k:list(v) for k,v in drugs.items()}

drugs_with_more_names = {k:list(v) for k,v in drugs.items() if len(v) > 1}
for k,v in dict(itertools.islice(drugs_with_more_names.items(), 10)).items():
    print(f"{k:10}: {list(v)[:2]}")
value_count_per_condition = df["Condition"].value_counts()
value_count_per_condition_norm = df["Condition"].value_counts(normalize=True)
unique_drugs_per_condition = df.groupby("Condition")["DrugId"].apply(set).to_frame().reset_index()
unique_drugs_per_condition.columns = ["condition", "unique_drugs"]

tempdf = pd.DataFrame({"condition": value_count_per_condition.index, 
                       "condition_freq": value_count_per_condition.values,
                       "condition_freq_norm": value_count_per_condition_norm.values})

tempdf = pd.merge(tempdf, unique_drugs_per_condition, on="condition")
def mrange(*args, ceiling=True):
    """Returns money range generator, yields 1, 2, 5, 10, 20, 50..."""
    f = lambda x: (((x - 1) % 3)**2 + 1) * 10**((x-1)//3)
    if len(args) == 1:
        start, stop = 1, args[0]
    else:
        start, stop = max(1, args[0]), args[1]
    c = 1
    x = f(c)
    while x < start:
        c += 1
        x = f(c)
    while True:
        yield x
        c += 1
        x = f(c)
        if x > stop:
            break
    if ceiling:
        yield x

def roundup(x, nearest=1000):
    """Rounds x to the nearest 1000 or the optional argument."""
    return int(math.ceil(x / float(nearest))) * nearest

ceiling = roundup(value_count_per_condition[0]) + 1
bins = [0] + [x for x in mrange(20, ceiling)]
labels = [str(x) for x in bins[1:]]
binlabels = pd.cut(tempdf["condition_freq"], bins=bins, labels=labels)
conddf = tempdf.assign(bin=binlabels.values)
topN = 15

data = conddf[:topN][["condition", "condition_freq", "condition_freq_norm"]]

bars = (alt.Chart(title=f"Top {topN} Conditions")
           .mark_bar(size=20,
                     strokeWidth=1,
                     stroke="white",
                     strokeOpacity=0.7,
                     xOffset=-1)
           .encode(x=alt.X("condition", sort="-y"),
                   y=alt.Y("condition_freq:Q",
                           axis=alt.Axis(title="number of reviews",
                                         grid=True)), 
                   tooltip=("condition",
                            "condition_freq:Q",
                            alt.Tooltip("condition_freq_norm:Q", format=".1%")),
                   color=alt.Color("condition_freq:Q",
                                   scale=alt.Scale(scheme="lightgreyteal",
                                                   type="log"))))

text = (alt.Chart()
           .mark_text(align="center",
                      baseline="bottom",
                      dx=-1, dy=-3)
           .encode(x=alt.X("condition", sort="-y"),
                   y=alt.Y("condition_freq:Q"),
                   size = alt.SizeValue(9),
                   text=alt.Text("condition_freq_norm:Q", format=".1%")))

chart = (alt.layer(bars, text, data=data)
            .configure(background='#11043a')
            .configure_title(font="Arial",
                             fontSize=18,
                             color="#e6f3ff",
                             dy=-10)
            .configure_text(color="white")
            .configure_legend(title=None,
                              titleFontSize=12,
                              titleColor="white",
                              tickCount=5,
                              titleOpacity=0.8,
                              labelColor="white",
                              labelOpacity=0.7,
                              titlePadding=10)
            .configure_axis(titleFontSize=13,
                            titlePadding=20,
                            titleColor="white",
                            titleOpacity=0.8,
                            labelFontSize=11,
                            labelColor="white",
                            labelOpacity=0.7,
                            #labelAngle=45,
                            tickOffset=0,
                            grid=False,
                            gridOpacity=0.15)
            .configure_view(strokeWidth=0)
            .properties(height=300, width=700))
chart
# this aggregates the sets of unique_drugs which fall into the same bin and counts the number of elements
aggr_sets = lambda x: sum(1 for n in set.union(*x))

data = (conddf.groupby("bin")
              .agg({"condition": "count", "condition_freq": "sum",
                    "condition_freq_norm": "sum", "unique_drugs": aggr_sets})
              .reset_index())
data.columns = ["bin", "condition_count", "condition_freq_sum",
                "condition_freq_norm_sum", "unique_drugs_count"]

bars = (alt.Chart(title="Distribution Of Condition Frequency And Drug Use")
           .mark_bar(size=20,
                     strokeWidth=1,
                     stroke="white",
                     strokeOpacity=0.7,
                     xOffset=-1)
           .encode(x=alt.X(shorthand="bin:Q",
                           scale=alt.Scale(round=False, type="log"),
                           axis=alt.Axis(title="binned condition counts",
                                         grid=False,
                                         orient="bottom")),
                   y=alt.Y(shorthand="condition_freq_sum:Q",
                           scale=alt.Scale(type="log"),
                           axis=alt.Axis(title="sum of condition counts and unique drug use, log scaled")),
                   tooltip=("bin:Q",
                            "condition_count:Q",
                            "condition_freq_sum:Q", 
                            alt.Tooltip("condition_freq_norm_sum:Q", format=".1%"),
                                        "unique_drugs_count:Q"),
                   color=alt.Color("condition_count:Q",
                                   scale=alt.Scale(scheme="lightgreyteal",
                                                   type="log"))))

text = (alt.Chart()
           .mark_text(align="center",
                      baseline="bottom",
                      dx=-1, dy=-3)
           .encode(x=alt.X("bin:Q"),
                   y=alt.Y("condition_freq_sum:Q"),
                   size = alt.SizeValue(9),
                   text=alt.Text("condition_freq_norm_sum:Q", format=".1%")))

line = (alt.Chart()
           .mark_line(color="red",
                      xOffset=-1,
                      size=1)
           .encode(x=alt.X("bin:Q"),
                   y=alt.Y("unique_drugs_count:Q")))

point = (alt.Chart()
            .mark_point(color="red",
                        xOffset=-1,
                        size=15,
                        shape="diamond")
            .encode(x=alt.X("bin:Q"),
                    y=alt.Y("unique_drugs_count:Q"),
                    tooltip=("unique_drugs_count:Q")))

chart = (alt.layer(bars, line, text, point, data=data[data["condition_count"] > 0])
            .configure(background='#11043a')
            .configure_title(font="Arial",
                             fontSize=18,
                             color="#e6f3ff",
                             dy=-10)
            .configure_text(color="white")
            .configure_legend(title=None,
                              titleFontSize=12,
                              titleColor="white",
                              tickCount=10,
                              titleOpacity=0.8,
                              labelColor="white",
                              labelOpacity=0.7,
                              titlePadding=10)
            .configure_axis(titleFontSize=13,
                            titlePadding=20,
                            titleColor="white",
                            titleOpacity=0.8,
                            labelColor="white",
                            labelOpacity=0.7,
                            tickOffset=0,
                            grid=True,
                            gridOpacity=0.15)
            .configure_view(strokeWidth=0)
            .properties(height=300, width=700))

chart.resolve_scale(color="independent")
data = (df.groupby(["Age", "Satisfaction"])
          .agg({"Reviews": "count"})
          .reset_index()).sort_values(["Age", "Satisfaction"], ascending=True)
#data['Cumulative_Reviews'] = data.groupby(['Age'])['Reviews'].apply(lambda x: x.cumsum())

bars = (alt.Chart(data=data, title="Distribution of Reviews Over Age")
           .mark_bar(size=40,
                     strokeWidth=0.5,
                     stroke="white")
           .encode(x=alt.X('Age:O',
                           axis=alt.Axis(title="Age groups", grid=False)),
                   y=alt.Y('Reviews:Q', stack='zero',
                           scale=alt.Scale(type="linear"),
                           axis=alt.Axis(title="num of reviews")),
                   order=alt.Order('Satisfaction', sort='ascending'),
                   color=alt.Color("Satisfaction:Q",
                                   scale=alt.Scale(scheme="lightgreyteal",
                                                   bins=[0,1,2,3],
                                                   reverse=False))))

text = (alt.Chart(data=data[data["Reviews"] > 1500])
           .mark_text(align="center",
                      baseline="middle",
                      dx=0, dy=5)
           .encode(x=alt.X("Age:O"),
                   y=alt.Y("Reviews:Q", stack='zero'),
                   size = alt.SizeValue(9),
                   text="Reviews:Q",
                   color=alt.condition(alt.datum.Satisfaction > 1,
                                          alt.value("white"),
                                          alt.value("black"))))
    
chart = (alt.layer(bars, text)
            .configure(background="#11043a")
            .configure_title(font="Arial",
                             fontSize=18,
                             color="#e6f3ff",
                             dy=-10)
            .configure_text(color="white")
            .configure_legend(titleFontSize=12,
                              titleColor="white",
                              tickCount=10,
                              titleOpacity=0.8,
                              labelColor="white",
                              labelOpacity=0.7,
                              titlePadding=10)
            .configure_axis(titleFontSize=13,
                            titlePadding=20,
                            titleColor="white",
                            titleOpacity=0.8,
                            labelColor="white",
                            labelOpacity=0.7,
                            labelAngle=0,
                            tickOffset=0,
                            grid=True,
                            gridOpacity=0.15)
            .configure_view(strokeWidth=0)
            .properties(height=300, width=700)
)
chart
data = (df.groupby(["DrugId"])
          .agg({"Reviews": "count", "Satisfaction": "mean"})
          .reset_index()
          .sort_values(["Reviews"], ascending=False))
data["Drug"] = data["DrugId"].map(drugs)

alt.data_transformers.disable_max_rows()
scatter = (alt.Chart(title="Distribution Of Reviews Over Satisfaction")
            .mark_point(color="#648bce")
            .encode(x=alt.X('Satisfaction:Q',
                            axis=alt.Axis(title="Mean Satisfaction",
                                          grid=False)),
                    y=alt.Y('Reviews:Q',
                             scale=alt.Scale(type="log"),
                             axis=alt.Axis(title="Number of Reviews, log scaled")),
                    size='Reviews:Q',
                    color=alt.Color("Satisfaction:Q",
                                   scale=alt.Scale(scheme="lightgreyteal",
                                                   type="linear")),
                    tooltip=['DrugId', 'Drug', 'Reviews',
                              alt.Tooltip("Satisfaction", format=".3")])
            .interactive())

chart = (alt.layer(scatter, data=data[data["Reviews"] > 20])
            .configure(background='#11043a')
            .configure_title(font="Arial",
                             fontSize=18,
                             color="#e6f3ff",
                             dy=-10)
            .configure_text(color="white")
            .configure_legend(titleFontSize=12,
                              titleColor="white",
                              tickCount=6,
                              titleOpacity=0.8,
                              labelColor="white",
                              labelOpacity=0.7,
                              titlePadding=10)
            .configure_axis(titleFontSize=13,
                            titlePadding=20,
                            titleColor="white",
                            titleOpacity=0.8,
                            labelFontSize=11,
                            labelColor="white",
                            labelOpacity=0.7,
                            labelAngle=0,
                            tickOffset=0,
                            grid=False,
                            gridOpacity=0.15)
            .configure_view(strokeWidth=0)
            .properties(height=300, width=700)
)
chart
indexes = np.random.randint(df.shape[0], size=3)
print(" ".join(df["Reviews"].iloc[indexes].tolist()))
%%time

nlp = spacy.load("en", disable=["ner", "parser"])
STOPWORDS = set(ENGLISH_STOP_WORDS).union(set(stopwords.words("english")))

def clean_review(text, STOPWORDS=STOPWORDS, nlp=nlp):
    """Cleans up text"""
    
    def rep_emo(text, placeholder_pos=' happyemoticon ', placeholder_neg=' sademoticon '):
        """Replace emoticons"""
        # Credit https://github.com/shaheen-syed/Twitter-Sentiment-Analysis/blob/master/helper_functions.py
        emoticons_pos = [":)", ":-)", ":p", ":-p", ":P", ":-P", ":D",":-D", ":]", ":-]", ";)", ";-)",
                         ";p", ";-p", ";P", ";-P", ";D", ";-D", ";]", ";-]", "=)", "=-)", "<3"]
        emoticons_neg = [":o", ":-o", ":O", ":-O", ":(", ":-(", ":c", ":-c", ":C", ":-C", ":[", ":-[",
                         ":/", ":-/", ":\\", ":-\\", ":n", ":-n", ":u", ":-u", "=(", "=-(", ":$", ":-$"]

        for e in emoticons_pos:
            text = text.replace(e, placeholder_pos)

        for e in emoticons_neg:
            text = text.replace(e, placeholder_neg)   
        return text

    def rep_punct(text):
        """Replace all punctuation with space"""
        for c in string.punctuation:
            text = text.replace(c, " ")
        return text

    def rem_stop_num(text):
        """Remove stop words and anything starting with number"""
        return " ".join(word for word in text.split() if word not in STOPWORDS and not word[0].isdigit())

    def lemmatize(text):
        """Return lemmas of tokens in text"""
        return " ".join(tok.lemma_.lower().strip() for tok in nlp(text) if tok.lemma_ != "-PRON-")  

    return lemmatize(rem_stop_num(rep_punct(rep_emo(text))))

mldf = df[["Satisfaction", "Reviews"]]
mldf["Reviews"] = mldf["Reviews"].apply(clean_review)

# remove any rows with new empty strings following the clean-up
mldf["Reviews"].replace("", np.nan, inplace=True)
mldf.dropna(inplace=True)
# adding indexes as "index" column for later use to recreate same splits 
mldf.reset_index(inplace=True)
print(" ".join(mldf["Reviews"].iloc[indexes].tolist()))
del df
negdf = mldf[mldf["Satisfaction"] == 0]
negatives = []
for review in negdf["Reviews"]:
    negatives.append(review)
negatives = pd.Series(negatives).str.cat(sep=" ")

wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(negatives)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
posdf = mldf[mldf["Satisfaction"] == 2]
positives = []
for review in posdf["Reviews"]:
    positives.append(review)
positives = pd.Series(positives).str.cat(sep=" ")

wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(positives)
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
train_set, test_set = train_test_split(mldf, test_size=0.25, random_state=0, stratify=mldf["Satisfaction"])
train_index = train_set.index
test_index = test_set.index
print(train_set.shape)
print(test_set.shape)
def confusion_matrix_altair(labels, predictions):
    """Returns Altair heatmap as confusion matrix"""
    
    alt.data_transformers.disable_max_rows()
    source = pd.DataFrame([labels, predictions]).T
    source.columns=["True", "Predicted"]

    # Configure base chart
    base = (alt.Chart(source, title="Confusion Matrix")
               .transform_aggregate(count="count()",
                                    groupby=["True", "Predicted"])
               .transform_joinaggregate(total="sum(count)")
               .transform_calculate(pct="datum.count / datum.total")
               .encode(x=alt.X("Predicted:O", scale=alt.Scale(paddingInner=0)),
                       y=alt.Y("True:O", scale=alt.Scale(paddingInner=0)),
                       tooltip=(alt.Tooltip("pct:Q", format=".1%"))))
    # Configure heatmap
    heatmap = (base.mark_rect()
                   .encode(color=alt.Color("count:Q",
                           scale=alt.Scale(scheme="blues"),
                           legend=alt.Legend(direction="vertical"))))
    # Configure text
    text = (base.mark_text(baseline="middle")
                .encode(text="count:Q",
                        color=alt.condition(alt.datum.count > 10000,
                                            alt.value("white"),
                                            alt.value("black"))))
    # Draw the chart
    chart = ((heatmap + text)
                .configure(background="#11043a")
                .configure_title(fontSize=18,
                                 color="#e6f3ff",
                                 dy=-20)
                .configure_text(color="white",
                                fontSize=14)
                .configure_legend(titleFontSize=12,
                                  titleColor="white",
                                  titleOpacity=0.8,
                                  labelColor="white",
                                  labelOpacity=0.7,
                                  titlePadding=10)
                .configure_axis(titleFontSize=14,
                                titlePadding=20,
                                titleColor="white",
                                titleOpacity=0.8,
                                labelFontSize=13,
                                labelColor="white",
                                labelOpacity=0.7,
                                labelAngle=0)
                .configure_view(strokeWidth=0)
                .properties(height=400, width=400)
            )
    return chart
%%time
vectorizer = CountVectorizer(max_features=2500, min_df=10, max_df=0.8)
X_train = vectorizer.fit_transform(train_set["Reviews"]).toarray()
X_test = vectorizer.transform(test_set["Reviews"]).toarray()
y_train = train_set["Satisfaction"].values
y_test = test_set["Satisfaction"].values
%%time
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
acc_train = accuracy_score(y_train, model.predict(X_train))
print(f"\nAccuracy in train set: {acc_train:.2}")
predictions = model.predict(X_test)
acc_test = accuracy_score(y_test, predictions)
print(f"\nAccuracy in test  set: {acc_test:.2}\n")
print(classification_report(y_test, predictions))
confusion_matrix_altair(y_test, predictions)
%%time
train_set = train_set[train_set["Satisfaction"] != 1]
test_set = test_set[test_set["Satisfaction"] != 1]
print(train_set.shape)
print(test_set.shape)

vectorizer = CountVectorizer(max_features=2500, min_df=10, max_df=0.8)
X_train = vectorizer.fit_transform(train_set["Reviews"]).toarray()
X_test = vectorizer.transform(test_set["Reviews"]).toarray()
y_train = train_set["Satisfaction"].values
y_test = test_set["Satisfaction"].values

model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

acc_train = accuracy_score(y_train, model.predict(X_train))
print(f"\nAccuracy in train set: {acc_train:.2}")
predictions = model.predict(X_test)
acc_test = accuracy_score(y_test, predictions)
print(f"\nAccuracy in test  set: {acc_test:.2}\n")
print(classification_report(y_test, predictions))
confusion_matrix_altair(y_test, predictions)
%%time
train_set = mldf.loc[train_index]
test_set  = mldf.loc[test_index]
print(train_set.shape)
print(test_set.shape)

vectorizer = TfidfVectorizer(max_features=2500, min_df=10, max_df=0.8)
X_train = vectorizer.fit_transform(train_set["Reviews"]).toarray()
X_test = vectorizer.transform(test_set["Reviews"]).toarray()
y_train = train_set["Satisfaction"].values
y_test = test_set["Satisfaction"].values
%%time
model = RandomForestClassifier(min_samples_split=6, random_state=0)
model.fit(X_train, y_train)
acc_train = accuracy_score(y_train, model.predict(X_train))
print(f"\nAccuracy in train set: {acc_train:.2}")
predictions = model.predict(X_test)
acc_test = accuracy_score(y_test, predictions)
print(f"\nAccuracy in test  set: {acc_test:.2}\n")
print(classification_report(y_test, predictions))
confusion_matrix_altair(y_test, predictions)
%%time
train_set = train_set[train_set["Satisfaction"] != 1]
test_set = test_set[test_set["Satisfaction"] != 1]
print(train_set.shape)
print(test_set.shape)

vectorizer = TfidfVectorizer(max_features=2500, min_df=10, max_df=0.8)
X_train = vectorizer.fit_transform(train_set["Reviews"]).toarray()
X_test = vectorizer.transform(test_set["Reviews"]).toarray()
y_train = train_set["Satisfaction"].values
y_test = test_set["Satisfaction"].values

model = RandomForestClassifier(min_samples_split=6, random_state=0)
model.fit(X_train, y_train)
acc_train = accuracy_score(y_train, model.predict(X_train))
print(f"\nAccuracy in train set: {acc_train:.2}")
predictions = model.predict(X_test)
acc_test = accuracy_score(y_test, predictions)
print(f"\nAccuracy in test  set: {acc_test:.2}\n")
print(classification_report(y_test, predictions))
confusion_matrix_altair(y_test, predictions)
%%time
X_train = mldf.loc[train_index]
X_test = mldf.loc[test_index]
y_train = mldf.loc[train_index]
y_test = mldf.loc[test_index]

X_train = X_train[X_train["Satisfaction"] != 1]["Reviews"].values
X_test = X_test[X_test["Satisfaction"] != 1]["Reviews"].values
y_train = y_train[y_train["Satisfaction"] != 1]["Satisfaction"].values
y_test = y_test[y_test["Satisfaction"] != 1]["Satisfaction"].values

num_words = 2500
maxlen = 200

tokenizer = Tokenizer(num_words=num_words, split=" ", lower=False)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values
%%time
embedding_vector_length = 100

model = Sequential()
model.add(Embedding(num_words, embedding_vector_length, input_length=maxlen))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(embedding_vector_length))
model.add(Dense(2, activation="softmax"))
model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])
print(model.summary())
%%time
epochs = 15
batch_size = 64
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=2, min_delta=0.0001)])
accr = model.evaluate(X_test, y_test)
print("Loss in test set: {:0.3f}\nAccuracy in test set: {:0.3f}\n".format(accr[0], accr[1]))
predictions = model.predict_classes(X_test, batch_size = batch_size)
labels = np.argmax(y_test, axis=1)
print(classification_report(labels, predictions))
confusion_matrix_altair(labels, predictions)
def predict_sentiment(text):
    cleaned_text = tokenizer.texts_to_sequences([clean_review(text)])
    padded_text = pad_sequences(cleaned_text, maxlen=maxlen)
    return "Positive" if model.predict_classes(padded_text)[0] else "Negative"

predict_sentiment("The drug is expensive but it is worth every cent.")