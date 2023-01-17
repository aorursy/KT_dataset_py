import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import json



import matplotlib.pyplot as plt

plt.style.use('ggplot')
!ls /kaggle/input/CORD-19-research-challenge/
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
meta_df.info()
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

first_row = FileReader(all_json[0])

print(first_row)
def get_breaks(content, length):

    data = ""

    words = content.split(' ')

    total_chars = 0



    # add break every length characters

    for i in range(len(words)):

        total_chars += len(words[i])

        if total_chars > length:

            data = data + "<br>" + words[i]

            total_chars = 0

        else:

            data = data + " " + words[i]

    return data
dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    content = FileReader(entry)

    

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    # no metadata, skip this paper

    if len(meta_data) == 0:

        continue

    

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

    

    # also create a column for the summary of abstract to be used in a plot

    if len(content.abstract) == 0: 

        # no abstract provided

        dict_['abstract_summary'].append("Not provided.")

    elif len(content.abstract.split(' ')) > 100:

        # abstract provided is too long for plot, take first 300 words append with ...

        info = content.abstract.split(' ')[:100]

        summary = get_breaks(' '.join(info), 40)

        dict_['abstract_summary'].append(summary + "...")

    else:

        # abstract is short enough

        summary = get_breaks(content.abstract, 40)

        dict_['abstract_summary'].append(summary)

        

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    

    try:

        # if more than one author

        authors = meta_data['authors'].values[0].split(';')

        if len(authors) > 2:

            # more than 2 authors, may be problem when plotting, so take first 2 append with ...

            dict_['authors'].append(". ".join(authors[:2]) + "...")

        else:

            # authors will fit in plot

            dict_['authors'].append(". ".join(authors))

    except Exception as e:

        # if only one author - or Null valie

        dict_['authors'].append(meta_data['authors'].values[0])

    

    # add the title information, add breaks when needed

    try:

        title = get_breaks(meta_data['title'].values[0], 40)

        dict_['title'].append(title)

    # if title was not provided

    except Exception as e:

        dict_['title'].append(meta_data['title'].values[0])

    

    # add the journal information

    dict_['journal'].append(meta_data['journal'].values[0])

    

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])

df_covid.head()
dict_ = None
df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))

df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))

df_covid.head()
df_covid.info()
df_covid['abstract'].describe(include='all')
df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)

df_covid['abstract'].describe(include='all')
df_covid['body_text'].describe(include='all')
df_covid.head()
df_covid.describe()
df_covid.dropna(inplace=True)

df_covid.info()
import re



df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
def lower_case(input_str):

    input_str = input_str.lower()

    return input_str



df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: lower_case(x))
df_covid.head(4)
token_columns = df_covid.drop(["abstract_word_count", "body_word_count",], axis=1)
token_columns.head()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import MinMaxScaler





maxlen = 4096 #only use this number of most frequent words

training_samples = 8000

validation_samples = 4500

max_words = 100000



tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(df_covid["body_text"]) # generates word index

sequences = tokenizer.texts_to_sequences(df_covid["body_text"]) # transforms strings in list of intergers

word_index = tokenizer.word_index # calculated word index

print(f"{len(word_index)} unique tokens found")



data = pad_sequences(sequences, maxlen=maxlen) #transforms integer lists into 2D tensor
scaler = MinMaxScaler() 

data_1 = scaler.fit_transform(data) # the values of all features are rescaled into the range of [0, 1]
x = data_1
from time import time

import keras.backend as K

from keras.engine.topology import Layer, InputSpec

from keras.layers import Dense, Input, Embedding

from keras.models import Model

from keras.optimizers import SGD

from keras import callbacks

from keras.initializers import VarianceScaling

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt



%matplotlib inline
def autoencoder(dims, act='relu', init='glorot_uniform'):

    """

    Fully connected symmetric auto-encoder model.

  

    dims: list of the sizes of layers of encoder like [500, 500, 2000, 10]. 

          dims[0] is input dim, dims[-1] is size of the latent hidden layer.



    act: activation function

    

    return:

        (autoencoder_model, encoder_model): Model of autoencoder and model of encoder

    """

    n_stacks = len(dims) - 1

    

    input_data = Input(shape=(dims[0],), name='input')

    x = input_data

    

    # internal layers of encoder

    for i in range(n_stacks-1):

        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)



    # latent hidden layer

    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)



    x = encoded

    # internal layers of decoder

    for i in range(n_stacks-1, 0, -1):

        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)



    # decoder output

    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)

    

    decoded = x

    

    autoencoder_model = Model(inputs=input_data, outputs=decoded, name='autoencoder')

    encoder_model     = Model(inputs=input_data, outputs=encoded, name='encoder')

    

    return autoencoder_model, encoder_model
n_clusters = 20 

n_epochs   = 15

batch_size = 128
dims = [x.shape[-1], 500, 500, 2000, 10] 

init = VarianceScaling(scale=1. / 3., mode='fan_in',

                           distribution='uniform')

pretrain_optimizer = SGD(lr=1, momentum=0.9)

pretrain_epochs = n_epochs

batch_size = batch_size

save_dir = 'kaggle/working'
dims
class ClusteringLayer(Layer):

    '''

    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the

    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    '''



    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:

            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(ClusteringLayer, self).__init__(**kwargs)

        self.n_clusters = n_clusters

        self.alpha = alpha

        self.initial_weights = weights

        self.input_spec = InputSpec(ndim=2)



    def build(self, input_shape):

        assert len(input_shape) == 2

        input_dim = input_shape[1]

        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.clusters = self.add_weight(name='clusters', shape=(self.n_clusters, input_dim), initializer='glorot_uniform') 

        

        if self.initial_weights is not None:

            self.set_weights(self.initial_weights)

            del self.initial_weights

        self.built = True



    def call(self, inputs, **kwargs):

        ''' 

        student t-distribution, as used in t-SNE algorithm.

        It measures the similarity between embedded point z_i and centroid µ_j.

                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.

                 q_ij can be interpreted as the probability of assigning sample i to cluster j.

                 (i.e., a soft assignment)

       

        inputs: the variable containing data, shape=(n_samples, n_features)

        

        Return: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)

        '''

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))

        q **= (self.alpha + 1.0) / 2.0

        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure all of the values of each sample sum up to 1.

        

        return q



    def compute_output_shape(self, input_shape):

        assert input_shape and len(input_shape) == 2

        return input_shape[0], self.n_clusters



    def get_config(self):

        config = {'n_clusters': self.n_clusters}

        base_config = super(ClusteringLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
autoencoder, encoder = autoencoder(dims, init=init)
from keras.utils import plot_model

plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)

from IPython.display import Image

Image(filename='autoencoder.png')
from keras.utils import plot_model

plot_model(encoder, to_file='encoder.png', show_shapes=True)

from IPython.display import Image

Image(filename='encoder.png')
autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')

autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)

#autoencoder.save_weights(save_dir + '/ae_weights.h5')
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)

model = Model(inputs=encoder.input, outputs=clustering_layer)
from keras.utils import plot_model

plot_model(model, to_file='model.png', show_shapes=True)

from IPython.display import Image

Image(filename='model.png')
model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
kmeans = KMeans(n_clusters=n_clusters, n_init=20)

y_pred = kmeans.fit_predict(encoder.predict(x))
y_pred_last = np.copy(y_pred)
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
# computing an auxiliary target distribution

def target_distribution(q):

    weight = q ** 2 / q.sum(0)

    return (weight.T / weight.sum(1)).T
loss = 0

index = 0

maxiter = 1000 # 8000

update_interval = 100 # 140

index_array = np.arange(x.shape[0])
tol = 0.001 # tolerance threshold to stop training
for ite in range(int(maxiter)):

    if ite % update_interval == 0:

        q = model.predict(x, verbose=0)

        p = target_distribution(q)  # update the auxiliary target distribution p



    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]

    loss = model.train_on_batch(x=x[idx], y=p[idx])

    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0



#model.save_weights(save_dir + '/DEC_model_final.h5')
# Eval.

q = model.predict(x, verbose=0)

p = target_distribution(q)  # update the auxiliary target distribution p



# evaluate the clustering performance

y_pred = q.argmax(1)
data_all = df_covid.copy()
data_all['cluster'] = y_pred

data_all.head()
data_all['cluster'].value_counts()
import numpy as np

from sklearn.manifold import TSNE



x_embedded = TSNE(n_components=2).fit_transform(x)



x_embedded.shape
from matplotlib import pyplot as plt

import seaborn as sns



# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", len(set(y_pred)))



# plot

sns.scatterplot(x_embedded[:,0], x_embedded[:,1], hue=y_pred, legend='full', palette=palette)

plt.title("t-SNE Covid-19 Articles, Clustered(Autoencoder and custem Keras Layer), Tf-idf with Plain Text")

# plt.savefig("plots/t-sne_covid19_label_TFID.png")

plt.show()




from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS

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

from bokeh.layouts import column, widgetbox



output_notebook()

y_labels = y_pred



# data sources

source = ColumnDataSource(data=dict(

    x= x_embedded[:,0], 

    y= x_embedded[:,1],

    x_backup = x_embedded[:,0],

    y_backup = x_embedded[:,1],

    desc= y_labels, 

    titles= df_covid['title'],

    authors = df_covid['authors'],

    journal = df_covid['journal'],

    abstract = df_covid['abstract_summary'],

    labels = ["C-" + str(x) for x in y_labels]

    ))



# hover over information

hover = HoverTool(tooltips=[

    ("Title", "@titles{safe}"),

    ("Author(s)", "@authors"),

    ("Journal", "@journal"),

    ("Abstract", "@abstract{safe}"),

],

                 point_policy="follow_mouse")



# map colors

mapper = linear_cmap(field_name='desc', 

                     palette=Category20[20],

                     low=min(y_labels) ,high=max(y_labels))



# prepare the figure

p = figure(plot_width=800, plot_height=800, 

           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'], 

           title="t-SNE Covid-19 Articles, Clustered(Autoencoder and custem Keras Layer), Tf-idf with Plain Text", 

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

            

            if (radio_value == '20') {

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



# callback for searchbar

keyword_callback = CustomJS(args=dict(p=p, source=source), code="""

            

            var text_value = cb_obj.value;

            var data = source.data; 

            

            x = data['x'];

            y = data['y'];

            

            x_backup = data['x_backup'];

            y_backup = data['y_backup'];

            

            abstract = data['abstract'];

            titles = data['titles'];

            authors = data['authors'];

            journal = data['journal'];



            for (i = 0; i < x.length; i++) {

                if(abstract[i].includes(text_value) || 

                   titles[i].includes(text_value) || 

                   authors[i].includes(text_value) || 

                   journal[i].includes(text_value)) {

                    x[i] = x_backup[i];

                    y[i] = y_backup[i];

                } else {

                    x[i] = undefined;

                    y[i] = undefined;

                }

            }

            





        source.change.emit();

        """)



# option

option = RadioButtonGroup(labels=["C-0", "C-1", "C-2",

                                  "C-3", "C-4", "C-5",

                                  "C-6", "C-7", "C-8",

                                  "C-9", "C-10", "C-11",

                                  "C-12", "C-13", "C-14",

                                  "C-15", "C-16", "C-17",

                                  "C-18", "C-19", "All"], 

                          active=20, callback=callback)



# search box

keyword = TextInput(title="Search:", callback=keyword_callback)



#header

header = Div(text="""<h1>COVID-19 Literature Cluster</h1>""")



# show

show(column(header, widgetbox(option, keyword),p))