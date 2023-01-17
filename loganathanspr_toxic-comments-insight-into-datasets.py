# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# ------------------------ Standard Kaggle statements END --------------------------------------



import spacy

from spacy import __version__

#print("Going to use Spacy version - ", __version__)



from plotly import tools

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

import plotly.figure_factory as ff

#print("Going to use Plotly version - ", __version__)

init_notebook_mode(connected=True)



import sklearn

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



from nltk.corpus import stopwords



# Let's load Spacy

nlp = spacy.load("en")



# Plotly definitions 

# ------------------



# Plot background color

paper_bgcolor = "rgb(240, 240, 240)"

plot_bgcolor = "rgb(240, 240, 240)"



# Red, blue, green (used by plotly by default)

rgb_def = ['rgb(228,26,28)', 'rgb(77,175,74)', 'rgb(55,126,184)']



# Contrasting 2 qualities, highlighting one

contra_2_cols = ["rgb(150,150,150)", "rgb(55,126,184)"]



# Barchart axis templates

# template 1

bchart_xaxis_temp1 = dict(

    zeroline=False,

    showline=False, 

    showgrid=False, 

    showticklabels=False,    

    tickfont=dict(

        size=9,

        color="grey"

    )      

)



bchart_yaxis_temp1=dict(

    tickfont=dict(

        size=9,

        color="grey"

    )        

)



# template 2

bchart_xaxis_temp2 = dict(

    zeroline=False,

    showline=False, 

    showgrid=False, 

    showticklabels=False,    

    tickfont=dict(

        size=10,

        color="grey"

    )      

)



bchart_yaxis_temp2=dict(

    tickfont=dict(

        size=10,

        color="grey"

    )        

)



# Heatmap templates

heatmap_axis_temp1 = dict(

    zeroline=False,

    showline=False,

    showgrid=False, 

    showticklabels=False,  

    ticks=''                

)   
def concat_label_columns(row):

    multiheads = []

    for col in list(train_orig.columns)[2:]:

        if row[col]:

            multiheads.append(col) 

    if len(multiheads) == 0:

        return "non_toxic"

    else:

        return ":".join(multiheads)

    

def get_reshaped_array(one_d_nparray, fc):

    """Given an 1d array of an arbitrary size, make the shape of the 

    1d array divisible by factor "fc" by appending np.nan values when it is not divisible evenly. 

    For ex: If the array 

    dimension is (114, ), and the factor "fc" is 100,  

    then the np.nan 1d array of shape 86 will 

    be appended to the original 1d array to become array size 200. 

    Array shape 200 evenly divides by 100.

    """    

    remainder = one_d_nparray.shape[0] % fc

    if remainder > 0:

        cells_to_fill = fc - remainder

        nan_array = np.full(cells_to_fill, np.nan)

        one_d_nparray = np.append(one_d_nparray, nan_array)

    num_cols_heatmap = int(one_d_nparray.shape[0] / fc)

    num_rows_heatmap = fc

    return one_d_nparray.reshape((num_rows_heatmap, num_cols_heatmap))
train_orig = pd.read_csv("../input/train.csv")

test_orig = pd.read_csv("../input/test.csv")



# copy of the datasets 

train = train_orig.copy()

test = test_orig.copy()



# concatenate the labels into comma separated one label and save it in a new column 

train["concatenated_label"] = train.apply(concat_label_columns, axis=1)



comment_types = list(train_orig.columns)[2:]

comment_types_incl = ["non_toxic"] + comment_types



multi_comment_types = list(train["concatenated_label"].unique())
print(train_orig.shape)
print(comment_types)
train_orig[0:5]
# Fingerprint of training data per comment type

fig_coords = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

axes_names = [("x1", "y1"), ("x2", "y2"), ("x3", "y3"), ("x4", "y4"), ("x5", "y5"), ("x6", "y6")]

axes_lo_names = [("xaxis1", "yaxis1"), ("xaxis2", "yaxis2"), ("xaxis3", "yaxis3"), ("xaxis4", "yaxis4"), ("xaxis5", "yaxis5"), ("xaxis6", "yaxis6")]

fig = tools.make_subplots(

    rows=2, 

    cols=3, 

    horizontal_spacing=0.1,

    vertical_spacing=0.1,

    subplot_titles=(comment_types[0], comment_types[1], comment_types[2], comment_types[3], comment_types[4], comment_types[5])

)

for c_type, fig_coord, ax in zip(comment_types, fig_coords, axes_names):

    reshaped_labels = get_reshaped_array(train_orig[c_type].as_matrix(), 75)

    trace = go.Heatmap(

        z=reshaped_labels, 

        colorscale = 'YlGnBu', 

        zmin=0, 

        zmax=1, 

        xaxis=ax[0], 

        yaxis=ax[1],

        name=c_type

    )

    fig.append_trace(trace, fig_coord[0], fig_coord[1])



fig["layout"].update(

    title = "<b>Fingerprint of training data by comment types</b>",

    xaxis1=heatmap_axis_temp1,

    yaxis1=heatmap_axis_temp1,   

    xaxis2=heatmap_axis_temp1,

    yaxis2=heatmap_axis_temp1, 

    xaxis3=heatmap_axis_temp1,

    yaxis3=heatmap_axis_temp1, 

    xaxis4=heatmap_axis_temp1,

    yaxis4=heatmap_axis_temp1,  

    xaxis5=heatmap_axis_temp1,

    yaxis5=heatmap_axis_temp1, 

    xaxis6=heatmap_axis_temp1,

    yaxis6=heatmap_axis_temp1,  

    margin=go.Margin(

        l=100,

        r=150,

        t=150,

        b=25

    ),

    autosize=False,

    width=900,

    height=600,

)

iplot(fig)
individual_cmnt_type_counts = []

for cmnt_type in comment_types_incl:

    if cmnt_type == "non_toxic":

        individual_cmnt_type_counts.append(train[train["concatenated_label"] == cmnt_type].shape[0])

    else:

        individual_cmnt_type_counts.append(train[cmnt_type].sum())

individual_cmnt_type_counts = pd.Series(individual_cmnt_type_counts, index=comment_types_incl)

individual_cmnt_type_counts = individual_cmnt_type_counts.sort_values()

bar_colors = [contra_2_cols[1]] * len(comment_types_incl)

# set color for "non_toxic" category

bar_colors[list(individual_cmnt_type_counts.index).index("non_toxic")] = contra_2_cols[0]



data = []

trace1 = go.Bar(

    x=individual_cmnt_type_counts, 

    y=list(individual_cmnt_type_counts.index), 

    orientation="h",

    marker=dict(

        color=bar_colors

    ),

    name="Comment type"

)

trace2 = go.Scatter(

    x=np.full(len(individual_cmnt_type_counts), train.shape[0]), 

    y=list(individual_cmnt_type_counts.index), 

    mode="lines",

    line = dict(

        color=(rgb_def[0]),

        width = 7,

        dash = 'dashdot',

    ),

    name="training data size"

)

data.append(trace1)

data.append(trace2)

layout=go.Layout(

    title="<b>Distribution of individual comment types in training data</b>",

    xaxis=dict(

        title="Count of comment types in training data",

        type='log',

        autorange=True,

        tickfont=dict(

            color="grey"

        )        

    ),

    yaxis=dict(

        title="Comment type",

        tickfont=dict(

            color="grey"

        )

    ),

    showlegend=False,

    annotations=[

        dict(

            x=5.1,

            y=5.75,

            xref='x',

            yref='y',

            text='Training data size',

            showarrow=True,

            arrowhead=4,

            ax=20,

            ay=-50            

        )

    ],   

    autosize=False,    

    width=900,

    height=600,    

    margin=go.Margin(

        l=150,

        r=150,

        b=25,

        t=100,

    ),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
multi_cmnt_type_counts = []

for cmnt_type in multi_comment_types:

    multi_cmnt_type_counts.append(train[train["concatenated_label"] == cmnt_type].shape[0])    

multi_cmnt_type_counts = pd.Series(multi_cmnt_type_counts, index=multi_comment_types)

multi_cmnt_type_counts = multi_cmnt_type_counts.sort_values()

data = []

bar_colors = [contra_2_cols[1]] * len(multi_comment_types)

# set color for "non_toxic" category

bar_colors[list(multi_cmnt_type_counts.index).index("non_toxic")] = contra_2_cols[0]



trace1 = go.Bar(

    x=multi_cmnt_type_counts, 

    y=list(multi_cmnt_type_counts.index), 

    orientation="h",

    marker=dict(

        color=bar_colors

    ),

    name="Comment type"

)

trace2 = go.Scatter(

    x=np.full(len(multi_cmnt_type_counts), train.shape[0]), 

    y=list(multi_cmnt_type_counts.index), 

    mode="lines",

    line = dict(

        color=(rgb_def[0]),

        width = 7,

        dash = 'dashdot',

    ),

    name="training data size"

)

data.append(trace1)

data.append(trace2)

layout=go.Layout(

    title="<b>Distribution of multilabel comment types in the training data</b>",

    xaxis=dict(

        title="Count of comment types in training data",

        type='log',

        autorange=True,

        tickfont=dict(

            color="grey",

        )        

    ),

    yaxis=dict(

        title="Comment type",

        tickfont=dict(

            color="grey",

            size=8            

        )

    ),

    showlegend=False,

    annotations=[

        dict(

            x=4.9,

            y=5,

            xref='x',

            yref='y',

            text='Training data size',

            showarrow=True,

            arrowhead=4,

            ax=-70,

            ay=-40

        )

    ],   

    autosize=False,    

    width=900,

    height=900,    

    margin=go.Margin(

        l=200,

        r=100,

        b=25,

        t=100,

    ),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
cmnt_count_matrix = []

for cmnt_type1 in comment_types:

    cmnt_type_frame = train[train[cmnt_type1] == 1]

    cmnt_type2_count = []

    for cmnt_type2 in comment_types:

        cmnt_type2_count.append(cmnt_type_frame[cmnt_type2].sum())

    cmnt_count_matrix.append(cmnt_type2_count)

cmnt_count_matrix = np.array(cmnt_count_matrix)



fig_coords = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

axes_names = [("x1", "y1"), ("x2", "y2"), ("x3", "y3"), ("x4", "y4"), ("x5", "y5"), ("x6", "y6")]

axes_lo_names = [("xaxis1", "yaxis1"), ("xaxis2", "yaxis2"), ("xaxis3", "yaxis3"), ("xaxis4", "yaxis4"), ("xaxis5", "yaxis5"), ("xaxis6", "yaxis6")]

fig = tools.make_subplots(

    rows=2, 

    cols=3, 

    horizontal_spacing=0.15, 

    vertical_spacing=0.25,

    subplot_titles=(comment_types[0], comment_types[1], comment_types[2], comment_types[3], comment_types[4], comment_types[5])

)

for i, c_type, fig_coord, ax in zip(range(len(comment_types)),comment_types, fig_coords, axes_names):

    inner_count = pd.Series(cmnt_count_matrix[i, :], index=comment_types)

    inner_count = inner_count.sort_values()

    trace = go.Bar(x=inner_count, y=list(inner_count.index), orientation = 'h')

    fig.append_trace(trace, fig_coord[0], fig_coord[1])



fig["layout"].update(

    showlegend=False,

    title="<b>Co-occurrence of comment types</b>",

    xaxis1=bchart_xaxis_temp2,

    yaxis1=bchart_yaxis_temp2,

    xaxis2=bchart_xaxis_temp2,

    yaxis2=bchart_yaxis_temp2,

    xaxis3=bchart_xaxis_temp2,

    yaxis3=bchart_yaxis_temp2,    

    xaxis4=bchart_xaxis_temp2,

    yaxis4=bchart_yaxis_temp2,

    xaxis5=bchart_xaxis_temp2,

    yaxis5=bchart_yaxis_temp2,

    xaxis6=bchart_xaxis_temp2,

    yaxis6=bchart_yaxis_temp2,



    margin=go.Margin(

        l=100,

        r=100,

        t=100,

        b=25,

    ),

    autosize=False,

    width=900,

    height=500,

)

iplot(fig)



# As a heatmap

fig = ff.create_annotated_heatmap(

    z=cmnt_count_matrix, 

    x=comment_types, 

    y=comment_types, 

    colorscale='YlGnBu', 

    zmin=1, 

    zmax=cmnt_count_matrix.max()

)

fig["layout"]["xaxis"].update(side="bottom")

fig["layout"].update(

    title="<b>Co-occurrence of comment types</b>",    

    xaxis=dict(

        title="Major comment category",

        tickfont=dict(

            color="grey"

        )        

    ),   

    yaxis=dict(

        title="Co-occurring comment category",

        tickfont=dict(

            color="grey"

        )        

    ),   

    

    margin=go.Margin(

        l=150,

        r=150,

        t=150,

        b=75

    ),

    autosize=False,

    width=900,

    height=450,

)

iplot(fig)
stop_words_new = list(sklearn.feature_extraction.text.ENGLISH_STOP_WORDS.union(stopwords.words("english")))

count_vect = CountVectorizer(min_df=2, stop_words=stop_words_new)

train_counts = count_vect.fit_transform(train["comment_text"])

tfidf_transformer = TfidfTransformer()

train_tfidf = tfidf_transformer.fit_transform(train_counts)

features_array = np.array(count_vect.get_feature_names())
fig_coords = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

axes_names = [("x1", "y1"), ("x2", "y2"), ("x3", "y3"), ("x4", "y4"), ("x5", "y5"), ("x6", "y6")]

axes_lo_names = [("xaxis1", "yaxis1"), ("xaxis2", "yaxis2"), ("xaxis3", "yaxis3"), ("xaxis4", "yaxis4"), ("xaxis5", "yaxis5"), ("xaxis6", "yaxis6")]

fig = tools.make_subplots(

    rows=2, 

    cols=3, 

    horizontal_spacing=0.01, 

    vertical_spacing=0.05,

    subplot_titles=(comment_types[0], comment_types[1], comment_types[2], comment_types[3], comment_types[4], comment_types[5])

)



num_top_words = 30



for i, cmnt_type, fig_coord, ax in zip(range(len(comment_types)),comment_types, fig_coords, axes_names):

    instances_of_cmnt_type_ind = list(train[train[cmnt_type] == 1].index)

    tfidf_cmnt_type = train_tfidf[instances_of_cmnt_type_ind].toarray()

    mean_tfidf_cmnt_type = tfidf_cmnt_type.mean(axis=0)

    top_words_vals = np.sort(mean_tfidf_cmnt_type)[::-1][0:num_top_words]

    top_words_ind = mean_tfidf_cmnt_type.argsort()[::-1][0:num_top_words]

    top_words = features_array[top_words_ind]

    trace = go.Bar(

        x=top_words_vals[::-1], 

        y=top_words[::-1], 

        orientation = 'h',

        name=cmnt_type

    )

    fig.append_trace(trace, fig_coord[0], fig_coord[1])

    

fig["layout"].update(

    showlegend=False,

    title="<b>Top 30 words for each comment type</b>",

    xaxis1=bchart_xaxis_temp1,

    yaxis1=bchart_yaxis_temp1,

    xaxis2=bchart_xaxis_temp1,

    yaxis2=bchart_yaxis_temp1,

    xaxis3=bchart_xaxis_temp1,

    yaxis3=bchart_yaxis_temp1,    

    xaxis4=bchart_xaxis_temp1,

    yaxis4=bchart_yaxis_temp1,

    xaxis5=bchart_xaxis_temp1,

    yaxis5=bchart_yaxis_temp1,

    xaxis6=bchart_xaxis_temp1,

    yaxis6=bchart_yaxis_temp1,

    margin=go.Margin(

        l=75,

        r=75,

        t=100,

        b=100,

    ),

    autosize=False,

    width=900,

    height=900,

)

iplot(fig)