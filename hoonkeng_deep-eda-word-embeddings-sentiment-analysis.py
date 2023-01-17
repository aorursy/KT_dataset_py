#Some fundamental libraries
import numpy as np
import pandas as pd
import sys
import os
import re
import collections
import csv
import gc

#Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
import colorlover as cl

#NLP libraries
from six.moves import xrange 
import tensorflow as tf
from nltk.corpus import stopwords
list_data = ['reuters-newswire-2011.csv','reuters-newswire-2012.csv','reuters-newswire-2013.csv','reuters-newswire-2014.csv','reuters-newswire-2015.csv','reuters-newswire-2016.csv','reuters-newswire-2017.csv']
dfs=list()
for csv in list_data:
    df = pd.read_csv('../input/reuters-news-wire-archive/'+csv)
    dfs.append(df)
data = pd.concat(dfs).reset_index()
print(len(data))
data.head()
data.tail()
texts=[]
for index, row in data.iterrows():
    if not pd.isnull(row['headline_text']):
        text = re.sub('\W',' ',row['headline_text'])
        text = text.split()
        for word in text:
            texts.append(word.lower())
            
print('Data size', len(texts))
del(texts)
gc.collect()
'''
def read_data():
    stopWords = set(stopwords.words('english'))
    list_data = ['reuters-newswire-2011.csv','reuters-newswire-2012.csv','reuters-newswire-2013.csv','reuters-newswire-2014.csv','reuters-newswire-2015.csv','reuters-newswire-2016.csv','reuters-newswire-2017.csv']
    dfs=list()
    texts = []
    for csv in list_data:
        df = pd.read_csv('../input/reuters-news-wire-archive/'+csv)
        dfs.append(df)
    data = pd.concat(dfs)
    length = len(data)
    for index, row in data.iterrows():
        sys.stdout.write('\rProcessing----%d/%d'%(index+1,length))
        sys.stdout.flush()
        if not pd.isnull(row['headline_text']):
            text = re.sub('\W',' ',row['headline_text'])
            text = text.split()
            for word in text:
                #if word not in stopWords:
                texts.append(word.lower())
        #gc.collect()
    return texts

vocabulary = read_data()
print('Data size', len(vocabulary))
#print(vocabulary)

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.
#print(reverse_dictionary)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  #print(buffer)
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      for word in data[:span]:
        buffer.append(word)
      data_index = span
    else:
      buffer.append(data[data_index])
      #print(buffer)
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64      # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 50000     # Random set of words to evaluate similarity on.
valid_window = 50000  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  show_emb = tf.Print(valid_embeddings, [valid_embeddings])
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()
num_steps = 50000001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    #print(len(batch_inputs))
    #print(len(batch_labels))
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val
    vocab_words = []
    vocab_code = []
    nearests = []
    if step % 100000 == 0:
      if step > 0:
        average_loss /= 10000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000000 == 0:
      sim = similarity.eval()
      show = show_emb.eval()
      p_embeddings = []
      for s in show:
        p_embeddings.append(s)
      #for s in show:
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[i]
        vocab_words.append(valid_word)
      
      rows = zip(vocab_words,p_embeddings)
      with open('Full_Embedding/5_epochs_v1/product_emb_epoch%d.csv'%(step/10000000), "w") as f:
        writer = csv.writer(f)
        for row in rows:
          writer.writerow(row)
      #PE = pd.DataFrame({"emb":p_embeddings})
      #PE.to_csv('product_emb_300d%d.csv'%(step/2300000), sep='\t', encoding='utf-8')
      vocab_words = []
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 50  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        #print(log_str)
        vocab_words.append(valid_word)
        nearests.append(log_str)
      w = pd.DataFrame({'stockcode':vocab_words})
      n = pd.DataFrame({'nearest':nearests})
      data_merged = pd.concat([w, n], axis=1)
      data_merged.to_csv('Nearest/v1/product_nearest_epoch%d.csv'%(step/10000001), sep='\t', encoding='utf-8')
      gc.collect()
  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, 'tsne.png') #os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
'''
def clean(text):
    text = text.split(': ')
    return text[1]

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def get_freq(keys,csv):
    stopWords = set(stopwords.words('english'))
    dfs=list()
    texts = []
    data = pd.read_csv(csv)
    #data = data[:10000]
    length = len(data)
    for index, row in data.iterrows():
        if index%100000 ==  0:
            sys.stdout.write('\rProcessing----%d/%d'%(index,length))
            sys.stdout.flush()
        if not pd.isnull(row['headline_text']):
            text = re.sub(r"\.", "", row['headline_text']) 
            text = re.sub('\W',' ',text)
            text = text.split()
            for word in text:
                word = word.lower()
                if word in keys:
                    texts.append(word)
    gc.collect()
    return texts

def get_y_data(key,NUM_KEY):
    y_data = []
    key = key[:NUM_KEY]
    print(key)
    for c, csv in enumerate(list_data):
        if c == 0:
            freq_key = get_freq(key,'../input/reuters-news-wire-archive/'+csv)
            data_freq, count_freq, dictionary_freq, reverse_dictionary_freq = build_dataset(freq_key,NUM_KEY+1)
            count_freq = dict(count_freq)
            for k in key:
                try: 
                    count_freq[k]
                except:
                    y_data.append([0])
                else:
                    y_data.append([count_freq[k]])

        else:
            freq_key = get_freq(key,'../input/reuters-news-wire-archive/'+csv)
            data_freq, count_freq, dictionary_freq, reverse_dictionary_freq = build_dataset(freq_key,NUM_KEY+1)
            count_freq = dict(count_freq)
            for index, k in enumerate(key):
                try: 
                    count_freq[k]
                except:
                    y_data[index].append(0)
                else:
                    y_data[index].append(count_freq[k])
    return(y_data)
def plot_nearest(_list,csv,title):
    word_output = []
    near = pd.read_csv(csv)
    near['nearest'] = near['nearest'].apply(clean)

    for layer1 in _list:
        layer2 = near.loc[near['stockcode']==layer1]['nearest'].values
        if layer2:
            layer2 = layer2[0].split(', ')
            for word_layer2 in layer2:
                word_layer2 = re.sub('\W','',word_layer2)
                word_output.append(word_layer2)
                layer3 = near.loc[near['stockcode']==word_layer2]['nearest'].values
                layer3 = layer3[0].split(', ')
                for word_layer3 in layer3:
                    word_layer3 = re.sub('\W','',word_layer3)
                    word_output.append(word_layer3)
                    
    data_list, count_list, dictionary_list, reverse_dictionary_list = build_dataset(word_output,50)
    count_list = dict(count_list)
    keys = list(dictionary_list.keys())
    value = list(dictionary_list.values())
    key = [ keys[i] for i in sorted(range(len(value)), key=lambda k: value[k])]
    value = [ count_list[k] for k in key ]
    key = key[1:]
    value = value[1:]

    n_phase = len(key)
    plot_width = 400

    # height of a section and difference between sections 
    section_h = 40
    section_d = 10

    # multiplication factor to calculate the width of other sections
    unit_width = plot_width / max(value)

    # width of each funnel section relative to the plot width
    phase_w = [int(v * unit_width) for v in value]

    # plot height based on the number of sections and the gap in between them
    height = section_h * n_phase + section_d * (n_phase - 1)

    # list containing all the plot shapes
    shapes = []

    # list containing the Y-axis location for each section's name and value text
    label_y = []

    for i in range(n_phase):
        if (i == n_phase-1):
                points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]
        else:
                points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]

        path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

        shape = {
                'type': 'path',
                'path': path,
                'fillcolor': 'rgb(32,155,160)',
                'line': {
                    'width': 1
                }
        }
        shapes.append(shape)
        
        # Y-axis location for this section's details (text)
        label_y.append(height - (section_h / 2))

        height = height - (section_h + section_d)

    # For phase names
    label_trace = go.Scatter(
        x=[-350]*n_phase,
        y=label_y,
        mode='text',
        text=key,
        textfont=dict(
            color='rgb(200,200,200)',
            size=15
        )
    )
    
    # For phase values\
    value_trace = go.Scatter(
        x=[350]*n_phase,
        y=label_y,
        mode='text',
        text=value,
        textfont=dict(
            color='rgb(200,200,200)',
            size=15
        )
    )

    data = [label_trace, value_trace]

    layout = go.Layout(
        title="<b>"+title+"</b>",
        titlefont=dict(
            size=20,
            color='rgb(203,203,203)'
        ),
        shapes=shapes,
        height=1000,
        width=800,
        showlegend=False,
        paper_bgcolor='rgba(44,58,71,1)',
        plot_bgcolor='rgba(44,58,71,1)',
        xaxis=dict(
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showticklabels=False,
            zeroline=False
        )
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
    return key
    
list_country = ['us','china','russia']
key = plot_nearest(list_country,"../input/embeddings/product_nearest_epoch1.csv","Country Names From 10M Data")
list_country = ['us','china','russia']
key_country = plot_nearest(list_country,"../input/embeddings/product_nearest_epoch10.csv","Country Names From 100M Data")
def plot_freq_years(key,NUM_KEY,y_data,title):
    title = title
    labels = key
    colors = cl.scales['12']['qual']['Paired']

    for i in range(int(NUM_KEY/12)+1):
        colors += colors

    colors = colors[:NUM_KEY]

    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
    x_data=[]
    for n in range(NUM_KEY):
        x_data.append(years)

    traces = []

    for i in range(0, NUM_KEY):
        traces.append(go.Scatter(
            x=x_data[i],
            y=y_data[i],
            mode='lines',
            name ='',
            text=key[i],
            line=dict(color=colors[i], width=2),
            connectgaps=True,
        ))

        traces.append(go.Scatter(
            x=[x_data[i][0], x_data[i][len(years)-1]],
            y=[y_data[i][0], y_data[i][len(years)-1]],
            mode='markers',
            marker=dict(color=colors[i], size=12)
        ))

    layout = go.Layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            autotick=False,
            ticks='outside',
            tickcolor='rgb(204, 204, 204)',
            tickwidth=2,
            ticklen=5,
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
        ),
        autosize=False,
        margin=dict(
            autoexpand=False,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=False,
    )

    annotations = []

# Adding labels
    for y_trace, label, color in zip(y_data, labels, colors):
        # labeling the left_side of the plot
        annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],
                                      xanchor='right', yanchor='middle',
                                      text=label + ' {}'.format(y_trace[0]),
                                      font=dict(family='Arial',
                                                size=10,
                                                color=colors,),
                                      showarrow=False))
        # labeling the right_side of the plot
        annotations.append(dict(xref='paper', x=0.95, y=y_trace[len(years)-1],
                                      xanchor='left', yanchor='middle',
                                      text='{}'.format(y_trace[len(years)-1]),
                                      font=dict(family='Arial',
                                                size=10,
                                                color=colors,),
                                      showarrow=False))
    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                  xanchor='left', yanchor='bottom',
                                  text=title,
                                  font=dict(family='Arial',
                                            size=30,
                                            color='rgb(37,37,37)'),
                                  showarrow=False))
    # Source
    annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.1,
                                  xanchor='center', yanchor='top',
                                  text='EDA Analysis',
                                  font=dict(family='Arial',
                                            size=12,
                                            color='rgb(150,150,150)'),
                                  showarrow=False))

    layout['annotations'] = annotations

    fig = go.Figure(data=traces, layout=layout)
    py.iplot(fig, filename='Reuter-Disasters')
extra_country = ['us','china']
key_country = extra_country + key_country
y_data = get_y_data(key_country,50)
plot_freq_years(key_country,50,y_data,'Keyword Appearance for countries')
list_disaster = ['famine','tornado','tsunami','landslide','flood','influenza']
key_disaster = plot_nearest(list_disaster,"../input/embeddings/product_nearest_epoch10.csv","Disaster From 100M Data")
y_data = get_y_data(key_disaster,25)
plot_freq_years(key_disaster,25,y_data,'Keyword Appearance for disaster')
list_company = ['apple', 'google', 'facebook', 'alibaba']
key_company = plot_nearest(list_company,"../input/embeddings/product_nearest_epoch10.csv","Companies From 100M Data")
y_data = get_y_data(key_company,40)
plot_freq_years(key_company,40,y_data,'Keyword Appearance for companies')
list_politicians = ['obama', 'trump', 'putin']
key_politicians = plot_nearest(list_politicians,"../input/embeddings/product_nearest_epoch10.csv","Politicians From 100M Data")
y_data = get_y_data(key_politicians,20)
plot_freq_years(key_politicians,20,y_data,'Keyword Appearance for politicians')
list_sportstar = ['ronaldo','messi','nadal','sharapova','phelps','lebron']
key_sportstar = plot_nearest(list_sportstar,"../input/embeddings/product_nearest_epoch10.csv","Sport stars From 100M Data")
y_data = get_y_data(key_sportstar,50)
plot_freq_years(key_sportstar,20,y_data,'Keyword Appearance for Sport Stars')
'''
def get_polarity(text):
    global gloindex
    sys.stdout.write('\rProcessing----%d'%gloindex)
    sys.stdout.flush()
    try:
        textblob = TextBlob(text)
        pol = textblob.sentiment.polarity
    except:
        pol = 0.0
    gloindex+=1
    return pol

def get_weekday(date):
    year = int(str(date)[0:4])
    month = int(str(date)[4:6])
    day = int(str(date)[4:6])
    return datetime.datetime(year,month,day).weekday()

list_data = ['reuters-newswire-2011.csv','reuters-newswire-2012.csv','reuters-newswire-2013.csv','reuters-newswire-2014.csv','reuters-newswire-2015.csv','reuters-newswire-2016.csv','reuters-newswire-2017.csv']
dfs=list()
texts = []
for csv in list_data:
    df = pd.read_csv(csv)
    dfs.append(df)
data = pd.concat(dfs)
print(data.head())
print(len(data))
data['word_count'] = data['headline_text'].apply(lambda x : len(str(x).split()))
data['year'] = data['publish_time'].apply(lambda x : str(x)[0:4])
data['month'] = data['publish_time'].apply(lambda x : str(x)[4:6])
data['date'] = data['publish_time'].apply(lambda x : str(x)[6:8])
data['hour'] = data['publish_time'].apply(lambda x : str(x)[8:10])
data['minute'] = data['publish_time'].apply(lambda x : str(x)[10:])
data['weekday'] = data['publish_time'].apply(get_weekday)
data['polarity'] = data['headline_text'].apply(get_polarity)
data.to_csv('reuter_processed.csv', sep='\t')
'''
data_processed = pd.read_csv('../input/private-data/reuter_processed.csv')
data_processed.head()
dji = pd.read_csv('../input/private-data/DJIA_122017.csv')
dji.head()
dji['Date'] = pd.to_datetime(dji['Date'],errors='coerce', format='%m/%d/%Y')
data = [go.Scatter(x=dji.Date, y=dji.Close)]

py.iplot(data)
def Normalize(value,previous):
    return(((value/previous)-1)*100)

for index, row in dji.iterrows():
    #print(row)
    close = row['Close']
    if index > 0:
        value_nor = Normalize(close,pre)
        dji['Close'][index] = value_nor
    pre = close
    
dji = dji[1:]
dji.head()
dji['Date'] = pd.to_datetime(dji['Date'],errors='coerce', format='%m/%d/%Y')
data = [go.Scatter(x=dji.Date, y=dji.Close)]

py.iplot(data)
data_processed = data_processed[data_processed['hour']<9]
pol_data = data_processed.groupby(['year','month','date'])['polarity'].mean().reset_index()
pol_data.tail()
Date = []
Polarity = []
Close = []
for index, row in dji.iterrows():
    Year = int(str(row['Date'].year))
    Month = int(str(row['Date'].month))
    Day = int(str(row['Date'].day))
    pol = pol_data.loc[(pol_data['year'] == Year) & (pol_data['month']== Month) & (pol_data['date']== Day)]['polarity']
    if pol.values:
        Date.append(str(row['Date']))
        Polarity.append(pol.values[0])
        Close.append(row['Close'])

Date = pd.DataFrame({'Date':Date})
Polarity = pd.DataFrame({'Polarity':Polarity})
Close = pd.DataFrame({'Close':Close})

result = pd.concat([Date,Polarity,Close], axis=1)
result.head()
result['Polarity'] = result['Polarity'].apply(lambda x: round(x,3))
res = result.groupby('Polarity')['Close'].agg('sum').reset_index()
res.head()
datas = [go.Bar(
            x=res['Polarity'],
            y=res['Close'],
            textposition = 'auto',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )]
layout =  go.Layout(
    xaxis=dict(
        title='Sentiment Polarity',
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='lightgrey'
        ),
        showticklabels=True,
        tickangle=45,
        tickfont=dict(
            family='Old Standard TT, serif',
            size=14,
            color='black'
        ),
        exponentformat='e',
        showexponent='All'
    ),
    yaxis=dict(
        title='Index Polarity',
        titlefont=dict(
            family='Arial, sans-serif',
            size=18,
            color='lightgrey'
        ),
        showticklabels=True,
        tickangle=45,
        tickfont=dict(
            family='Old Standard TT, serif',
            size=14,
            color='black'
        ),
        exponentformat='e',
        showexponent='All'
    )
)
fig = dict(data=datas, layout=layout)
py.iplot(fig, filename='bar-direct-labels')