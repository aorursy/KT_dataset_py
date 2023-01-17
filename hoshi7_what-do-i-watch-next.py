import pandas as pd

import numpy as np

import os

import seaborn as sns

from tqdm import tqdm, tqdm_notebook

import matplotlib.pyplot as plt

import plotly_express as px

import seaborn as sns

import plotly.offline as py

import plotly.tools as tls

from plotly.offline import init_notebook_mode

import plotly.graph_objs as go

import palettable

from IPython.display import HTML

import json

from  altair.vega import v5

import plotly.figure_factory as ff

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import altair as alt

from collections import Counter
stop =["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]

df = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
#Taken From: https://www.kaggle.com/shivamb/netflix-shows-and-movies-exploratory-analysis



df['date_added'] = pd.to_datetime(df.date_added)

df['year_added'] = df.date_added.dt.year

df['month_added'] = df.date_added.dt.month



df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)

df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)

df.head()
#Finding Number of rows and columns

print("Dataset contains {} rows and {} columns".format(df.shape[0], df.shape[1]))

df.isnull().sum()
##-----------------------------------------------------------

# This whole section 

vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v5.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {}

}});

"""



#------------------------------------------------ Defs for future rendering

def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

            

@add_autoincrement

def render(chart, id="vega-chart"):

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )







HTML("".join((

    "<script>",

    workaround.format(json.dumps(paths)),

    "</script>")))
# Necessary Functions: 

def pie_plot(labels, values, colors, title):

    fig = {

      "data": [

        {

          "values": values,

          "labels": labels,

          "domain": {"x": [0, .48]},

          "name": "Job Type",

          "sort": False,

          "marker": {'colors': colors},

          "textinfo":"percent+label+value",

          "textfont": {'color': '#FFFFFF', 'size': 10},

          "hole": .6,

          "type": "pie"

        } ],

        "layout": {

            "title":title,

            "annotations": [

                {

                    "font": {

                        "size": 25,



                    },

                    "showarrow": False,

                    "text": ""



                }

            ]

        }

    }

    return fig





def word_cloud(df, pixwidth=6000, pixheight=350, column="index", counts="count"):

    data= [dict(name="dataset", values=df.to_dict(orient="records"))]

    wordcloud = {

        "$schema": "https://vega.github.io/schema/vega/v5.json",

        "width": pixwidth,

        "height": pixheight,

        "padding": 0,

        "title": "Hover to see number of occureances from all the sequences",

        "data": data

    }

    scale = dict(

        name="color",

        type="ordinal",

        range=["cadetblue", "royalblue", "steelblue", "navy", "teal"]

    )

    mark = {

        "type":"text",

        "from":dict(data="dataset"),

        "encode":dict(

            enter=dict(

                text=dict(field=column),

                align=dict(value="center"),

                baseline=dict(value="alphabetic"),

                fill=dict(scale="color", field=column),

                tooltip=dict(signal="datum.count + ' occurrances'")

            )

        ),

        "transform": [{

            "type": "wordcloud",

            "text": dict(field=column),

            "size": [pixwidth, pixheight],

            "font": "Helvetica Neue, Arial",

            "fontSize": dict(field="datum.{}".format(counts)),

            "fontSizeRange": [10, 60],

            "padding": 2

        }]

    }

    wordcloud["scales"] = [scale]

    wordcloud["marks"] = [mark]

    

    return wordcloud



from collections import defaultdict



def wordcloud_create(df):

    ult = {}

    corpus = df.description.values.tolist()

    final = defaultdict(int) #Declaring an empty dictionary for count (Saves ram usage)

    for words in corpus:

        for word in words.split():

             final[word]+=1

    temp = Counter(final)

    for k, v in  temp.most_common(200):

        ult[k] = v

    corpus = pd.Series(ult) #Creating a dataframe from the final default dict

    return render(word_cloud(corpus.to_frame(name="count").reset_index(), pixheight=600, pixwidth=900))



value_counts = df['type'].value_counts()

labels = value_counts.index.tolist()

py.iplot(pie_plot(labels, value_counts,['#1B9E77', '#7570B3'], "Type Distribution"))
top_work_unitdf = df['country'].value_counts().rename_axis('Country').reset_index(name='counts')[:10]



fig = px.bar(top_work_unitdf, y="Country", x='counts', orientation='h', title = "Country with the most number of titles",color=  "counts", color_continuous_scale=px.colors.qualitative.Prism).update_yaxes(categoryorder="total ascending")



fig.show()
top_months = df['month_added'].value_counts().rename_axis('Month_Added').reset_index(name='counts')



fig = px.bar(top_months, y="counts", x='Month_Added', title = "Country with the most number of titles",color=  "counts", color_continuous_scale=px.colors.qualitative.D3).update_yaxes(categoryorder="total ascending")



fig.show()
mov = df[df.type =='Movie']

mov_dur = mov['duration'].fillna(0.0).astype(float)

ff.create_distplot([mov_dur], ['y'], bin_size=0.5, colors=['#1B9E77']).show()
tv = df[df["type"] == "TV Show"]

mov = df[df["type"] == "Movie"]



col = "year_added"



df1 = tv[col].value_counts().reset_index()

df1 = df1.rename(columns = {col : "count", "index" : col})

df1 = df1.sort_values(col)



df2 = mov[col].value_counts().reset_index()

df2 = df2.rename(columns = {col : "count", "index" : col})

df2 = df2.sort_values(col)



trace1 = go.Scatter(x=df1[col], y=df1["count"], name="TV Shows", marker=dict(color="#1B9E77"), )

trace2 = go.Scatter(x=df2[col], y=df2["count"], name="Movies", marker=dict(color="#7570B3"))

data = [trace1, trace2]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.8, y=1.2, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
wordcloud_create(df)
df['description']= df['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
wordcloud_create(df)


#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')



#Replace NaN with an empty string

df['description'] = df['description'].fillna('')



#Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(df['description'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
names = pd.Series(df.index, index=df['title']).drop_duplicates()
def get_recommendations(title, number=10, cosine_sim=cosine_sim):

    # Get the index of the movie that matches the title

    mov = names[title]



    # Get the pairwsie similarity scores of all movies with that movie

    score = list(enumerate(cosine_sim[mov]))



    # Sort the movies based on the similarity scores

    score = sorted(score, key=lambda x: x[1], reverse=True)



    # Get the scores of the n most similar movies

    score = score[1:number]



    # Get the movie indices

    movie_indices = [i[0] for i in score]



    # Return the top n most similar movies

    return df['title'].iloc[movie_indices]
get_recommendations('MINDHUNTER', 7)
get_recommendations('3 Idiots', 10)