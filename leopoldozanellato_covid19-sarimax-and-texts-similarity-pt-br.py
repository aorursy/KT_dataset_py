!pip install scispacy
!pip install pmdarima
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz
!python -m spacy download en
import glob, json, zipfile, en_core_sci_md
"""from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap
from bokeh.io import output_file, show
from bokeh.transform import transform
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import RadioButtonGroup
from bokeh.models import TextInput
from bokeh.layouts import gridplot
from bokeh.models import Div
from bokeh.models import Paragraph
from bokeh.layouts import column, widgetbox"""

import numpy as np
import pandas as pd
import seaborn as sns
import spacy, scispacy, operator
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.core.display import HTML
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tools.eval_measures import rmse, mse
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.statespace.tools import diff
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
"""from google.colab import drive
drive.mount("/content/gdrive")""";
"""import zipfile
path = '/content/gdrive/My Drive/CORD-19-research-challenge.zip'
zip_object = zipfile.ZipFile(file = path, mode = 'r')
zip_object.extractall('./')
zip_object.close()""";
corona_features = {'paper_id': [], 'title': [],
                   'abstract': [], 'text': []}
corona_df = pd.DataFrame.from_dict(corona_features)
json_filenames = glob.glob(f'{"./"}//**/*.json', recursive = True)
def return_corona_df(json_filenames, df):
    "Função para ler os arquivos json"
    for file_name in json_filenames:
        row = {'paper_id': None, 'title': None,
           'abstract': None,'text': None}
    
        with open(file_name) as json_data:
            try:
                if file_name == "./sample_data/anscombe.json":
                    continue
      
                data = json.load(json_data)

                row['paper_id'] = data['paper_id'].strip()
                row['title'] = data['metadata']['title'].strip()

                abstract_list = [abstract['text'] for abstract in data['abstract']]
                abstract = '\n '.join (abstract_list)
                row['abstract'] = abstract.strip()

                text_list = [text['text'] for text in data['body_text']]
                text = '\n '.join(text_list)
                row['text'] = text.strip()

                df = df.append(row, ignore_index = True)
            except:
                pass 
    return df

corona_df = return_corona_df(json_filenames, corona_df)
corona_df.head()
corona_df = corona_df[corona_df['title']!= ""]
corona_df = corona_df[corona_df['abstract']!= ""]
corona_df.drop_duplicates(['abstract','text','title'], inplace=True)
corona_df.shape
corona_df.to_csv("pre_processado.csv")
nlp = en_core_sci_md.load(disable = ['tagger', 'parser', 'ner'])
nlp.max_length = 2000000
new_stop_words = ['et', 'al','doi','cppyright','http', 
                  'https', 'fig','table','result','show']
for word in new_stop_words:
    nlp.vocab[word].is_stop = True
def spacy_tokenizer(sentence):
    sentence = sentence.lower()
    lista = []
    lista = [word.lemma_ for word in nlp(sentence) if not (word.is_stop or
                                                         word.like_num or
                                                         word.is_punct or 
                                                         word.is_space or
                                                         len(word)==1)]
    lista = ' '.join([str(element) for element in lista])
    return lista
corona_df['text'] = corona_df['text'].apply(spacy_tokenizer)
corona_df.to_csv("pos_processamento.csv", encoding = "UTF-8")
corona_df_completo = pd.read_csv("../input/coronavirus/pos_processamento.csv", index_col = 0)
#corona_df_completo = pd.read_csv("../input/CORD-19-research-challenge/metadata.csv", index_col=0)
corona_df_completo.head()
print(corona_df_completo.iloc[0]["text"])
corona_df_completo.dropna(inplace=True)
corona_df_completo.shape
corona_df_completo.head()
dataset_texts = corona_df_completo['text'].tolist()
len(dataset_texts)
tfidf = TfidfVectorizer(max_features=2**12) #utilização desse vetor para limitação do tamanho da matriz esparsa
vectorized = tfidf.fit_transform(dataset_texts)
vectorized

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(vectorized.toarray())
components = pca.explained_variance_ratio_
components
wcss = []
for i in range(1,21):
    kmeans = MiniBatchKMeans(n_clusters = i, random_state = 0)
    kmeans.fit(vectorized)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,21), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel("WCSS")
k = 5
kmeans = MiniBatchKMeans(n_clusters=k, random_state = 16)
y_pred = kmeans.fit_predict(vectorized)
np.unique(y_pred)
# Baseado em: https://www.kaggle.com/maksimeren/covid-19-literature-clustering


output_notebook()
y_labels = y_pred

# data sources
source = ColumnDataSource(data=dict(
    x= X_pca[:,0], 
    y= X_pca[:,1],
    x_backup = X_pca[:,0],
    y_backup = X_pca[:,1],
    desc= y_labels, 
    titles= corona_df_completo['title'],
    abstract = corona_df_completo['abstract'],
    labels = ["C-" + str(x) for x in y_labels]
    ))

# hover over information
hover = HoverTool(tooltips=[
    ("Title", "@titles{safe}"),
    ("Abstract", "@abstract{safe}"),
],
                 point_policy="follow_mouse")

# map colors
mapper = linear_cmap(field_name='desc', 
                     palette=Category20[9],
                     low=min(y_labels) ,high=max(y_labels))

# prepare the figure
p = figure(plot_width=800, plot_height=800, 
           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'], 
           title="Covid-19 Papers", 
           toolbar_location="right")

# plot
p.scatter('x', 'y', size=5, 
          source=source,
          fill_color=mapper,
          line_alpha=0.3,
          line_color="black",
          legend = 'labels')

# add callback to control 
callback = CustomJS(args=dict(p=p, source=source), code="""
            
            var radio_value = cb_obj.active;
            var data = source.data; 
            
            x = data['x'];
            y = data['y'];
            
            x_backup = data['x_backup'];
            y_backup = data['y_backup'];
            
            labels = data['desc'];
            
            if (radio_value == '5') {
                for (i = 0; i < x.length; i++) {
                    x[i] = x_backup[i];
                    y[i] = y_backup[i];
                }
            }
            else {
                for (i = 0; i < x.length; i++) {
                    if(labels[i] == radio_value) {
                        x[i] = x_backup[i];
                        y[i] = y_backup[i];
                    } else {
                        x[i] = undefined;
                        y[i] = undefined;
                    }
                }
            }


        source.change.emit();
        """)


# option
option = RadioButtonGroup(labels=["C-0", "C-1", "C-2",
                                  "C-3", "C-5",
                                   "All"], 
                          active=5, callback=callback)

#header
header = Div(text="""<h1>Covid-19 Papers</h1>""")

# show
show(column(header, widgetbox(option),p))
from IPython.display import Image
Image(filename='../input/coronavirus/imagem3.jpg', width=500, height=500)
plt.figure(figsize=(10,8))
sns.scatterplot(X_pca[:,0], X_pca[:,1], hue=y_pred,palette="bright")
plt.title("Covid-19 Papers")
corona_df = corona_df_completo.copy()
corona_df = corona_df.sample(n = 500, random_state = 0)
corona_df
!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
text = str(corona_df['text'][25534][:1000])

nlp_ent = spacy.load("en_core_web_sm")
nlp_ent.max_length = 2000000
doc = nlp_ent(text)
from spacy import displacy
displacy.render(doc,style = 'ent', jupyter = True)
gpe = []
for index, row in corona_df.iterrows():
    text = row['text']
    doc = nlp_ent(text)
    for entity in doc.ents:
        if entity.label_ =='GPE':
            gpe.append(str(entity.text))
values_gpe,counts_gpe = np.unique(np.array(gpe), return_counts = True)
gpe_df = pd.DataFrame({'value': values_gpe, 'counts': counts_gpe})
gpe_df = gpe_df.sort_values(by='counts', ascending=False).head(8)
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(float(rect.get_height()),2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
fig, ax = plt.subplots(figsize=(10,8))
rect1 = ax.bar(x=gpe_df["value"], height=gpe_df["counts"])
plt.title("Citações dos países nos 500 textos")
plt.ylabel('Citações de cada País / localização')
plt.xlabel('País/Localização')
autolabel(rect1)
def find_all_texts(input_str, search_str, number_of_words):
    text_list = []
    index = 0
    number_of_words = number_of_words
    while index < len(input_str):
        i = input_str.find(search_str, index)
        if i == -1:
            return text_list
    
        if input_str[i-number_of_words:i] == '':
            start = 0
        else:
            start = i - number_of_words
    
        text_list.append(input_str[start:i] + input_str[i:i + number_of_words])
        index = i + i
    return text_list
nlp = en_core_sci_md.load(disable = ['tagger', 'parser', 'ner'])
nlp.max_length = 2000000
def spacy_tokenizer(sentence):
    sentence = sentence.lower()
    lista = []
    lista = [word.lemma_ for word in nlp(sentence) if not (word.is_stop or
                                                         word.like_num or
                                                         word.is_punct or 
                                                         word.is_space or
                                                         len(word)==1)]
    lista = ' '.join([str(element) for element in lista])
    return lista
search_strings = ["traveler"]
tokens_list = [nlp(spacy_tokenizer(item)) for item in search_strings]
tokens_list
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)
matcher.add("SEARCH", None, *tokens_list)
number_of_words = 50
corona_df_200 = corona_df.copy()
corona_df_200 = corona_df_200[:200]

for index, row in corona_df_200.iterrows():
    marked_text = ""
    doc = nlp(row["text"])
    paper_id = row["paper_id"]
    title=row['title']
    matches = matcher(doc)
    if matches == []:
        continue
        
    print(f"\n \n \nWords: {search_strings}\n")
    print(f"Title: {title}\n")
    print(f"Paper ID: {paper_id}\n")
    print(f"Matches: {len(matches)}\n")
    
    
    for i in matches:
        start=i[1] - number_of_words
        if start<0:
            start=0
        for j in range(len(tokens_list)):
            if doc[i[1]:i[2]].similarity(tokens_list[j]) ==1:
                search_text = str(tokens_list[j])
                market_text = str(doc[start:i[2] + number_of_words]).replace(search_text, search_text)
                print(f"TEXTO: {market_text}")

texts = corona_df['text'].tolist()
tfidf = TfidfVectorizer()
vectorized = tfidf.fit_transform(texts)
search_string = 'Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases. Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments'
search_string = spacy_tokenizer(search_string)
print(search_string)
search_string_vectorized = tfidf.transform([search_string])
similarity = cosine_similarity(search_string_vectorized, vectorized)
similarity
scores_dict = {}
for i in range(len(similarity[0])):
    scores_dict[i] = similarity[0][i]
sorted_scores = sorted(scores_dict.items(), key=operator.itemgetter(1), reverse = True)
for i in sorted_scores[:5]:
    df=corona_df.iloc[i[0]]
    print(f"Title: {df['title']}")
    print(f"Paper ID: {df['paper_id']}")
    print(f"Score: {i[1]}")
    print(f"Abstract: {str(df['abstract'])[0:500]}")
    print("--------------------------------------------------------------------------------------------------------------------------------------")
df_brasil = pd.read_csv("../input/coronavirus/brazil_covid19_macro.csv")
brasil = df_brasil[['date', 'deaths']].groupby('date').sum().reset_index()
brasil = brasil[brasil['deaths'] >0]
brasil['date'] = pd.to_datetime(brasil['date'])
brasil.set_index('date', inplace=True)
brasil.index.freq = "D"
brasil
plt.figure(figsize=(8,5))
plt.title("Covid-19 no Brasil")
plt.plot(brasil.values,label="nº Mortes")
plt.xlabel("Dias")
plt.legend()
plot_acf(brasil['deaths'], lags=40);
plot_pacf(brasil['deaths'], lags=40);
seasonal = seasonal_decompose(brasil['deaths'], model='aditive');
seasonal.plot();
plt.figure(figsize=(12,5))
seasonal.seasonal.plot();
plt.title("Seasonal")
plt.xlabel('Date')
plt.figure(figsize=(12,5))
seasonal.trend.plot()
plt.title("Tendência")
plt.xlabel('Date')
plt.figure(figsize=(12,5))
seasonal.resid.plot()
plt.title("Residual")
plt.xlabel('Date')
def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
adf_test(brasil["deaths"])
df1 = brasil.copy()
plt.figure(figsize=(12,5))
df1['d2'] = diff(brasil['deaths'],k_diff=2)
df1['d2'][2:].plot();
plt.title("Stacionary timeseries")
plt.ylabel("Date")
adf_test(df1['d2'])
train_set = brasil[:90]
test_set = brasil[90:]
sarima = auto_arima(brasil['deaths'],start_p=0, start_q=0,seasonal=True,trace=True, m=7)
sarima
sarima.summary()
start = len(train_set)
end = len(train_set) + len(test_set) - 1
model = SARIMAX(train_set['deaths'], order=(2, 1, 3),seasonal_order=(0, 1, 1, 7)).fit()
predictions = model.predict(start,end,typ="levels").rename("SARIMAX(2, 1, 3)x(0, 1, 1, 7)")
predictions
test_set['deaths'].plot(label="test set", legend=True)
train_set['deaths'].plot(legend=True, label="train set")
predictions.plot(label="prediction", legend=True)
test_set['deaths'].plot(label="test set", legend=True)
predictions.plot(label="prediction", legend=True)
rmse(test_set['deaths'], predictions)
start = len(train_set)
end = len(train_set) + len(test_set) + 6
predictions = model.predict(start,end,typ="levels").rename("SARIMAX(2, 1, 3)x(0, 1, 1, 7)")
predictions
brasil_covid = pd.read_csv("../input/coronavirus/brasil_covid.csv", index_col=1, parse_dates=True,dayfirst=True, sep=";")
comparacao = brasil_covid[-21:]
plt.figure(figsize=(8,5))
train_set['deaths'].plot(legend=True, label="Dataset de Treino")
comparacao["obitosAcumulado"].plot(legend=True, label="Dataset de Test")
predictions.plot(legend=True, label="Predictions")
plt.title("Predictions")
plt.ylabel("Mortes")
plt.xlabel("Data")
plt.figure(figsize=(8,5))
comparacao["obitosAcumulado"].plot(legend=True, label="Dataset de Test")
predictions.plot(legend=True, label="Predictions")
plt.title("Predictions")
plt.ylabel("Mortes")
plt.xlabel("Data")
dataframe = pd.DataFrame({"predictions": predictions.values,
                          "mortes":comparacao['obitosAcumulado']})
dataframe['diferença'] = dataframe['predictions'] - dataframe['mortes']
dataframe['diferença'] = dataframe['diferença'].astype(int)
dataframe