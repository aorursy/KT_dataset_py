import math
import numpy as np
import pandas as pd
pd.set_option("max_rows", 12)
pd.set_option("max_columns", None)
pd.set_option("max_colwidth", 1024)
import plotly.tools as pt
import plotly.offline as po
import plotly.graph_objs as pg
po.init_notebook_mode(connected=True)
import gensim
from sklearn import (
    manifold,
    preprocessing
)
multiple_choice = (pd.read_csv("../input/multipleChoiceResponses.csv", low_memory=False)
     .pipe(lambda frame: frame.set_axis(
         pd.MultiIndex.from_tuples(
             [tuple(entry.split(" - ", 1)) for entry in frame.loc[0, :]]),
         axis=1, inplace=False
     ))
     .drop(0, axis=0)
     .reset_index(drop=True)
)
free_form = (pd.read_csv("../input/freeFormResponses.csv", low_memory=False)
     .pipe(lambda frame: frame.set_axis(
         pd.MultiIndex.from_tuples(
             [tuple(entry.split(" - ", 1)) for entry in frame.loc[0, :]]),
         axis=1, inplace=False
     ))
     .drop(0, axis=0)
     .reset_index(drop=True)
)
pd.concat([
    series
        .rename(series.name[0])
        .reset_index(drop=True)
    for series in [
        free_form
            .iloc[:, 0]
            .loc[[10217, 10, 18255]],
        free_form
            .iloc[:, 12]
            .loc[[4686, 12144, 5970]],
        free_form
            .iloc[:, 14]
            .loc[[20085, 1022, 9231]],
        free_form
            .iloc[:, 33]
            .loc[[6144, 12144, 23038]]
    ]
], axis=1)
# Define a toolbox
data = (multiple_choice
     # Drop students, not employed and other, out of scope of the discussion.  Sorry :)
     .pipe(lambda frame: frame.loc[
         frame
             .loc[:, pd.IndexSlice[
                 frame.columns.get_level_values(0).str.contains("title most similar to your current role"), :
             ]]
             .iloc[:, 0]
             .apply(lambda title: title != "Student"
                                  and title != "Not employed"
                                  and title != "Other"),
         pd.IndexSlice[:, :]
     ])
     # Assign a toolbox
     .assign(
         toolbox=lambda frame: frame
             # Focus on tools-related questions
             .loc[
                 :,
                 pd.IndexSlice[
                     frame.columns.levels[0][
                         frame.columns.levels[0].str.contains("IDE's")
                         | frame.columns.levels[0].str.contains("hosted notebooks")
                         #| frame.columns.levels[0].str.contains("cloud computing services")
                         | frame.columns.levels[0].str.contains("programming languages")
                         | frame.columns.levels[0].str.contains("machine learning frameworks")
                         | frame.columns.levels[0].str.contains("data visualization libraries or tools")
                         | frame.columns.levels[0].str.contains("cloud computing products")
                         | frame.columns.levels[0].str.contains("machine learning products")
                         | frame.columns.levels[0].str.contains("relational database products")
                         | frame.columns.levels[0].str.contains("big data and analytics products")
                     ],
                     frame.columns.levels[1][
                         ~(frame.columns.levels[1].str.contains("Other - Text")
                           | frame.columns.levels[1].str.contains("Choice - None")
                           | frame.columns.levels[1].str.contains("Choice - I have not used any cloud providers"))
                     ]
                 ]
             ]
             # And produce a list of tools from every row
             .apply(lambda row: row.dropna().tolist(), axis=1)
     )
     # And drop the samples with empty toolbox
     .pipe(lambda frame: frame.loc[frame.toolbox.apply(len) > 0])
)
data.loc[:, ["toolbox"]].sample(3, random_state=1)
# Define a simple word2vec model (or actually tool2vec in this particular case :D )
model = (data
     .pipe(lambda frame: gensim.models.Word2Vec(
         sentences=frame.toolbox.tolist(),
         size=48, # Pretty arbitrary, but seems to make sense
         window=frame.toolbox.apply(len).max(), # Why to truncate any context?  Extend boundaries to fit the biggest
         min_count=1, # Even the rarely used tools should be represented
         seed=0, # To make the algo deterministic
         workers=1, # Not that it is a very computationally-heavy task :D
         sg=1, # Skip-gram makes more sense to me, at least in this particular case
         iter=10, # Pretty arbitrary too, but again, seems to make sense
     ))
)
tools = pd.DataFrame.from_dict(
    {word: model.wv[word] for word in model.wv.vocab},
    orient="index"
)
tools.sample(3, random_state=0)
(tools
     # Reduce to 2D
     .pipe(lambda frame: pd.DataFrame(manifold.TSNE(
         n_components=2, perplexity=50.0, random_state=0
     ).fit_transform(frame.values), index=frame.index))
     # And plot :)
     .pipe(lambda frame: po.iplot(pg.Figure(
         data=[pg.Scatter(
             x=frame[1], y=frame[0],
             mode="markers+text",
             text=frame.index,
             marker=pg.scatter.Marker(
                 size=pd.Series(data.toolbox.sum())
                     .value_counts(normalize=True)
                     .pow(0.25).mul(100),
                 color=tools.apply(lambda row: 1 if "Amazon" in row.name
                                                 else 1 if "AWS" in row.name
                                                 else 2 if "Microsoft" in row.name
                                                 else 2 if "Azure" in row.name
                                                 else 3 if "Google" in row.name
                                                 else 4 if "IBM" in row.name
                                                 else 5, axis=1),
                 colorscale="Rainbow"
             )
         )],
         layout=pg.Layout(title="Our tools reduced to 2D")
     ), link_text=''))
)
(tools
     # Reduce to 3D
     .pipe(lambda frame: pd.DataFrame(manifold.TSNE(
         n_components=3, perplexity=50.0, random_state=0
     ).fit_transform(frame.values), index=frame.index))
     # And plot this beautiful 3D scatterplot :)
     .pipe(lambda frame: po.iplot(pg.Figure(
         data=[pg.Scatter3d(
             x=frame[0], y=frame[1], z=frame[2],
             mode="markers+text",
             text=frame.index,
             marker=pg.scatter3d.Marker(
                 size=pd.Series(data.toolbox.sum())
                     .value_counts(normalize=True)
                     .pow(0.25).mul(50),
             )
         )],
         layout=pg.Layout(title="Our tools reduced to 3D")
     ), link_text=''))
)
data = (data
     # Make a DataFrame of vectors and join to our data.  Just to keep it together
     .pipe(lambda frame: frame.join(
         frame.toolbox
             # Compute a mean of every toolbox -> Series of ND arrays
             .apply(lambda toolbox: np.mean(
                 np.vstack([model.wv[tool] for tool in toolbox]),
                 axis=0
             ))
             # Convert Series of ND arrays to the DataFrame of shape (len(Series), ND)
             .pipe(lambda series: pd.DataFrame(
                 np.vstack(series.values),
                 index=frame.index
             ))
             # And enhance the second axis (to ease the manipulation)
             .pipe(lambda frame: frame.set_axis(
                 pd.MultiIndex.from_product([["toolbox_vector"], frame.columns]), axis=1, inplace=False
             ))
     ))
)
titles = (data
     .set_index(("Select the title most similar to your current role (or most recent title if retired):", "Selected Choice"))
     .rename_axis("title")
     .pipe(lambda frame: pd.concat((
         frame
             .loc[:, "toolbox_vector"]
         ,
     ), axis=1))
)
titles.sample(3, random_state=0)
(titles
     .sample(frac=0.1, random_state=0)
     # Reduce to 2D
     .pipe(lambda frame: pd.DataFrame(manifold.TSNE(
         n_components=2, perplexity=50.0, random_state=0
     ).fit_transform(frame.values), index=frame.index))
     # And draw a plot once again
     .pipe(lambda frame: po.iplot(pg.Figure(
         data=[pg.Scatter(
             x=frame.loc[[title], 0],
             y=frame.loc[[title], 1],
             mode="markers+text",
             name=title
         ) for title in frame.index.unique()],
         layout=pg.Layout(title="(Subset of) our community.  You are probably somewhere here too! :)")
     ), link_text=''))
)