import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Configs
pd.options.display.float_format = '{:,.4f}'.format
sns.set(style="whitegrid")
plt.style.use('seaborn')
seed = 42
np.random.seed(seed)
# df = pd.read_csv('dados/olist_order_reviews_dataset.csv')

file_path = '/kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv'
df = pd.read_csv(file_path)
print("DataSet = {:,d} rows and {} columns".format(df.shape[0], df.shape[1]))

print("\nAll Columns:\n=>", df.columns.tolist())

quantitative = [f for f in df.columns if df.dtypes[f] != 'object']
qualitative = [f for f in df.columns if df.dtypes[f] == 'object']

print("\nStrings Variables:\n=>", qualitative,
      "\n\nNumerics Variables:\n=>", quantitative)

df.head()
df = df[['review_score','review_comment_message']]
df.columns = ['score', 'comment']
df.head()
import time

def time_spent(time0):
    t = time.time() - time0
    t_int = int(t) // 60
    t_min = t % 60
    if(t_int != 0):
        return '{}min {:.3f}s'.format(t_int, t_min)
    else:
        return '{:.3f}s'.format(t_min)
from sklearn.metrics import confusion_matrix, classification_report

this_labels = ['Negative','Positive']

def class_report(y_real, y_my_preds, name="", labels=this_labels):
    if(name != ''):
        print(name,"\n")
    print(confusion_matrix(y_real, y_my_preds), '\n')
    print(classification_report(y_real, y_my_preds, target_names=labels))
def plot_words_distribution(mydf, target_column, title='Words distribution', x_axis='Words in column'):
    # adaptade of https://www.kaggle.com/alexcherniuk/imdb-review-word2vec-bilstm-99-acc
    # def statistics
    len_name = target_column +'_len'
    mydf[len_name] = np.array(list(map(len, mydf[target_column])))
    sw = mydf[len_name]
    median = sw.median()
    mean   = sw.mean()
    mode   = sw.mode()[0]
    # figure
    fig, ax = plt.subplots()
    sns.distplot(mydf[len_name], bins=mydf[len_name].max(),
                hist_kws={"alpha": 0.9, "color": "blue"}, ax=ax,
                kde_kws={"color": "black", 'linewidth': 3})
    ax.set_xlim(left=0, right=np.percentile(mydf[len_name], 95)) # Dont get outiliers
    ax.set_xlabel(x_axis)
    ymax = 0.020
    plt.ylim(0, ymax)
    # plot vertical lines for statistics
    ax.plot([mode, mode], [0, ymax], '--', label=f'mode = {mode:.2f}', linewidth=4)
    ax.plot([mean, mean], [0, ymax], '--', label=f'mean = {mean:.2f}', linewidth=4)
    ax.plot([median, median], [0, ymax], '--', label=f'median = {median:.2f}', linewidth=4)
    ax.set_title(title, fontsize=20)
    plt.legend()
    plt.show()
def eda_categ_feat_desc_plot(series_categorical, title = "", fix_labels=False, bar_format='{}'):
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True)
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    
    fig, ax = plt.subplots(figsize = (12,4), ncols=2, nrows=1) # figsize = (width, height)
    if(title != ""):
        fig.suptitle(title, fontsize=18)
        fig.subplots_adjust(top=0.8)

    s = sns.barplot(x=series_name, y='quantity', data=val_concat, ax=ax[0])
    if(fix_labels):
        val_concat = val_concat.sort_values(series_name).reset_index()
    
    for index, row in val_concat.iterrows():
        s.text(row.name, row['quantity'], bar_format.format(row['quantity']), color='black', ha="center")

    s2 = val_concat.plot.pie(y='percentage', autopct=lambda value: '{:.2f}%'.format(value),
                             labels=val_concat[series_name].tolist(), legend=None, ax=ax[1],
                             title="Percentage Plot")

    ax[1].set_ylabel('')
    ax[0].set_title('Quantity Plot')

    plt.show()
def plot_nn_loss_acc(history):
    fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
    # summarize history for accuracy
    axis1.plot(history.history['accuracy'], label='Train', linewidth=3)
    axis1.plot(history.history['val_accuracy'], label='Validation', linewidth=3)
    axis1.set_title('Model accuracy', fontsize=16)
    axis1.set_ylabel('accuracy')
    axis1.set_xlabel('epoch')
    axis1.legend(loc='upper left')
    # summarize history for loss
    axis2.plot(history.history['loss'], label='Train', linewidth=3)
    axis2.plot(history.history['val_loss'], label='Validation', linewidth=3)
    axis2.set_title('Model loss', fontsize=16)
    axis2.set_ylabel('loss')
    axis2.set_xlabel('epoch')
    axis2.legend(loc='upper right')
    plt.show()
def describe_y_by_x_cat_boxplot(dtf, x_feat, y_target, title='', figsize=(15,5), rotatioon_degree=0):
    the_title = title if title != '' else '{} by {}'.format(y_target, x_feat)
    fig, ax1 = plt.subplots(figsize = figsize)
    sns.boxplot(x=x_feat, y=y_target, data=dtf, ax=ax1)
    ax1.set_title(the_title, fontsize=18)
    plt.xticks(rotation=rotatioon_degree)
    plt.show()
from sklearn.feature_extraction.text import CountVectorizer

def ngrams_corpus_counter(corpus,ngram_range,n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(stop_words = 'english',ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    total_list=words_freq[:n]
    df=pd.DataFrame(total_list,columns=['text','count'])
    return df

def plot_ngrams_words(series_words, title='Top 10 words'):
    """Plot 3 graphs
    @series_words: a series where each row is a set of words
    """
    # Generate
    df_1_grams = ngrams_corpus_counter(series_words, (1,1), 10)
    df_2_grams = ngrams_corpus_counter(series_words, (2,2), 10)
    df_3_grams = ngrams_corpus_counter(series_words, (3,3), 10)

    fig, axes = plt.subplots(figsize = (18,4), ncols=3)
    fig.suptitle(title)

    sns.barplot(y=df_1_grams['text'], x=df_1_grams['count'],ax=axes[0])
    axes[0].set_title("1 grams")

    sns.barplot(y=df_2_grams['text'], x=df_2_grams['count'],ax=axes[1])
    axes[1].set_title("2 grams",)

    sns.barplot(y=df_3_grams['text'], x=df_3_grams['count'],ax=axes[2])
    axes[2].set_title("3 grams")

    plt.show()
    
# Example: plot_ngrams_words(df['comment'])
import random 

def compare_text_cleaning(mydf, column1, column2, rows=10):
    """Compare Text after text cleaning
    """
    max_values = len(mydf)
    for i in range(rows):
        anumber = random.randint(0, max_values)
        print('Before:', mydf[column1][anumber])
        print('After :',  mydf[column2][anumber], '\n')
before = df.shape[0]
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.show()
# df.duplicated().sum() # 4187 rows
# df = df.drop_duplicates()

after = df.shape[0]

df = df.dropna()
df = df.reset_index(drop=True)

print('Before {:,d} | After Clean Data {:,d} | was removed {:,d} rows, that is {:.2%} of data'.format(
    before, after, before - after, (before - after)/before))
eda_categ_feat_desc_plot(df['score'], fix_labels=True, bar_format='{:,.0f}', title='Distribution of score')
plot_words_distribution(df, 'comment', 'Words distribution of all reviews')
describe_y_by_x_cat_boxplot(df, 'score', 'comment_len', figsize=(10,5))
plot_words_distribution(df.query('score == 5'), 'comment', 'Distribution of words for the best evaluations')
plot_words_distribution(df.query('score == 1'), 'comment', 'Words distribution for the worst evaluations')
plot_ngrams_words(df['comment'], 'Top 10 words to all comments')
plot_ngrams_words(df.query('score == 1')['comment'], 'Top 10 words for the worst ratings')
plot_ngrams_words(df.query('score == 5')['comment'], 'top 10 words for the best reviews')
mispell_dict = {'hj': 'hoje', 'nao': 'não', 'MT': 'muito', 'pessima': 'péssima'}

def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x
'nao' in stops
import re
from string import punctuation

from nltk.corpus import stopwords
# Stop Words in Python. Is better search in a 'set' structure
stops = set(stopwords.words("portuguese"))  
stops.remove("não")

def clean_text(words):
    words = re.sub("[^a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÍÏÓÔÕÖÚÇÑ1234567890]", " ", words) 
    words = words.lower().split()
    words = [w for w in words if not w in stops]   
    # Join the words back into one string separated by space, 
    words = ' '.join([c for c in words if c not in punctuation]).replace('\r\n',' ')
    return words
df['clean_comment'] = df['comment'].apply(clean_text)

df['clean_comment'] = df['clean_comment'].apply(lambda x: correct_spelling(x, mispell_dict))
compare_text_cleaning(df, 'comment', 'clean_comment')
df.head()
from sklearn.model_selection import train_test_split

X = df['clean_comment'].values

y = df['score'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

# Tokenizer
maxlen = 130
max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(np.concatenate((x_train, x_test), axis=0))

# Convert x_train
list_tokenized_train = tokenizer.texts_to_sequences(x_train) # convert string to numbers, 
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen) # create a array of 130 spaces and put all words in end

## Convert x_test
X_tt = tokenizer.texts_to_sequences(x_test)
X_tt = pad_sequences(X_tt, maxlen=maxlen)
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 5
history = model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, validation_data=(X_tt, y_test))
plot_nn_loss_acc(history)
y_pred = model.predict_classes(X_tt)

class_report(y_test, y_pred, "Strategie 1: Keras NN")
from sklearn.feature_extraction.text import CountVectorizer

# instantiate the vectorizer
vect = CountVectorizer()
vect.fit(np.concatenate((x_train, x_test), axis=0))

X_train_dtm = vect.transform(x_train)
X_test_dtm = vect.transform(x_test)
x_full_dtm = vect.transform( np.concatenate((x_train, x_test), axis=0) )

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit( x_full_dtm )
X_train_dtm_tfft = tfidf_transformer.transform(X_train_dtm)
X_test_dtm_tfft  = tfidf_transformer.transform(X_test_dtm)
from sklearn.svm import LinearSVC

linear_svc = LinearSVC(C=0.5, random_state=42)
linear_svc.fit(X_train_dtm, y_train)
y_pred_class = linear_svc.predict(X_test_dtm)
class_report(y_test, y_pred_class, "Strategie 1: Keras NN CountVectorizer") # 0.89, 0.88

linear_svc = LinearSVC(C=0.5, random_state=42)
linear_svc.fit(X_train_dtm_tfft, y_train)
y_pred_class = linear_svc.predict(X_test_dtm_tfft)
class_report(y_test, y_pred_class, "Strategie 1: Keras NN TD-IDF") # 0.90, 0.89
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train_dtm, y_train)
y_pred_class = logreg.predict(X_test_dtm)
class_report(y_test, y_pred_class, "Strategie 2: Keras NN TD-IDF")

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train_dtm_tfft, y_train)
y_pred_class = logreg.predict(X_test_dtm_tfft)
class_report(y_test, y_pred_class, "Strategie 2: Keras NN TD-IDF")
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier() 
forest = forest.fit(X_train_dtm, y_train)
y_pred = forest.predict(X_test_dtm)

class_report(y_test, y_pred, 'Strategie 2: RandomForestClassifier VetCount:')
# plt.figure(figsize = (12,5)) # fom imbd-popcorn my kernel
# feat_importances = pd.Series(forest.feature_importances_)#, index=X.columns)
# feat_importances.nlargest(30).reset_index().replace(inv_map).set_index('index').plot(kind='barh', use_index=True)
# plt.show()

int_top = 30
inv_map = {v: k for k, v in vect.vocabulary_.items()}
feat_importances = pd.Series(forest.feature_importances_)
labels_top = [ inv_map[el] for el in feat_importances.nlargest(int_top).index.tolist()]
plt.figure(figsize = (12,5))
sns.barplot(y=labels_top, x=feat_importances.nlargest(int_top))
plt.show()
pause_here
from collections import Counter

## Constroi um dict que mapeia palavras para inteiros
counts = Counter(df['clean_comment'])
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

## Usa o dict para ccodificar cada comentario em reviews_split
## Armazenar os comentários em reviews_ints
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])
print('Palavras unicas: ', len((vocab_to_int)), '\n')  

#imprime primeiro comentario codificado
print('Comentário codificado: \n', reviews_ints[:2])
encoded_labels = [1 if c>3 else 0 for c in df['score']]
plt.hist(encoded_labels); # Qtd de positivos (1) e negativos (0) depois de mapear {1,2,3} => Zero| Negativo, resto Um|Positivo
# Comentarios  
review_lens = Counter([len(x) for x in reviews_ints])
print("Comentários de tamanho zero: {}".format(review_lens[0]))
print("Tamanho máximo de um comentário: {}".format( max(review_lens)))
# Comentários de tamanho zero: 132
# Tamanho máximo de um comentário: 45
## Remove quaisquer comentários / etiquetas com comprimento zero da lista reviews_ints.

# Obtem indices de comentarios com comprimento difrente de zero
non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

# Remove comntarios de tamanho zero
reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])
reviews_ints = reviews_ints[:41500]
encoded_labels = encoded_labels[:41500]

# punctuation é uma lista de strings de pontuação {'.', ',', ';', '!' ...}
from string import punctuation

# Remove pontuação e outros caracteres
reviews_split= []
for review in reviews:
    reviews_split.append(''.join([c for c in review if c not in punctuation]).replace('\r\n',' '))

all_text = ' '.join(reviews_split)

# Cria uma lista com as palavras
words = all_text.split()
words[:30]
reviews_split[:5]
dfe = pd.read_pickle('vocab_to_int')
dfe['o']
from collections import Counter

## Constroi um dict que mapeia palavras para inteiros
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

## Usa o dict para ccodificar cada comentario em reviews_split
## Armazenar os comentários em reviews_ints
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])
print('Palavras unicas: ', len((vocab_to_int)), '\n')  

#imprime primeiro comentario codificado
print('Comentário codificado: \n', reviews_ints[:1])
# Mapeamento de palavras em números
vocab_to_string = { ii: word for word, ii  in vocab_to_int.items()}
print(vocab_to_string)
encoded_labels = [1 if c>3 else 0 for c in labels]
plt.hist(encoded_labels); # Qtd de positivos (1) e negativos (0) depois de mapear {1,2,3} => Zero| Negativo, resto Um|Positivo
# Comentarios  
review_lens = Counter([len(x) for x in reviews_ints])
print("Comentários de tamanho zero: {}".format(review_lens[0]))
print("Tamanho máximo de um comentário: {}".format(max(review_lens)))
## Remove quaisquer comentários / etiquetas com comprimento zero da lista reviews_ints.

# Obtem indices de comentarios com comprimento difrente de zero
non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

# Remove comntarios de tamanho zero
reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])
reviews_ints = reviews_ints[:41500]
encoded_labels = encoded_labels[:41500]
def pad_features(reviews_ints, seq_length):

    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features
seq_length = 20 # cada comentario na rede será de size = 20

features = pad_features(reviews_ints, seq_length=seq_length)

# imprime os primeiros 10 valores dos primeiros 30 lotes
print(features[:30,:10])
split_frac = 0.8 # Dividir 80% para treino e 20% para teste

## Separa os dados em treino, validação e teste

split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
import torch
from torch.utils.data import TensorDataset, DataLoader

# cria Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# dataloaders
batch_size = 25

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
#Checa se tem GPU
train_on_gpu = torch.cuda.is_available()

if(train_on_gpu):
    print('Treina em GPU.')
else:
    print('GPU não disponivel, treina em CPU.')
import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_sizes = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_sizes, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
        
#  Instancia o modelo com os parâmetros
vocab_size = len(vocab_to_int)+1 
output_size = 1         # nó de saida
embedding_dim = 200     # cada palavra vai ser convertida em um array de 200
hidden_dim = 256        # 256 neuronios na hidden layer
n_layers = 2            # 2 camdas de hidden layer

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)
# Funçoes de perda e otimização
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# Treina o modelo
if(not train_on_gpu):
    raise print('Sem gpu') 
epochs = 1 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net.cuda()

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial "min" to infinity    
    
net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())
                valid_loss = np.mean(val_losses)
            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(net.state_dict(), 'model.pt')
                valid_loss_min = valid_loss
#carrega o modelo treinado
net.load_state_dict(torch.load('model.pt',map_location='cpu') )
train_on_gpu = False
# Testa o modelo nos dados de teste

test_losses = [] # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    
    # get predicted outputs
    output, h = net(inputs.long(), h)
    
    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer
    
    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))
# Exemplo de texto que será testado: negative test review
test_review_neg = 'O produto não chegou'
from string import punctuation

def tokenize_review(test_review):
    test_review = test_review.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int.get(word) for word in test_words if vocab_to_int.get(word) != None ])
    

    return test_ints
def predict(net, test_review, sequence_length=200):
    
    net.eval()
    
    # tokenize review
    test_ints = tokenize_review(test_review)
    
    # pad tokenized sequence
    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)
    
    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)
    
    batch_size = feature_tensor.size(0)
    
    # initialize hidden state
    h = net.init_hidden(batch_size)
    
    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()
    
    # get the output from the model
    output, h = net(feature_tensor.long(), h)
    
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze()) 
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    
    # print custom response
    if(pred.item()==1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")
    return feature_tensor
# positive test review
test_review_pos = 'bom '
# negative test review
test_review_neg = 'ruim '
# test otimo, comparar com bom
test_review_poscomp = 'otimo'
# pessimo
# call function
seq_length=20 # good to use the length that was trained on
# positive test review
bom = predict(net, test_review_pos, seq_length)
# negative test review
ruim = predict(net, test_review_neg, seq_length)
# otimo
otimo = predict(net, test_review_poscomp, seq_length)
# perfeito
perfeito = predict(net, 'perfeito', seq_length)
pessimo = predict(net, 'pessimo', seq_length)
return_bom = net.embedding(bom.long())
return_ruim = net.embedding(ruim.long())
return_otimo = net.embedding(otimo.long())
return_perfeito = net.embedding(perfeito.long())
return_pessimo = net.embedding(pessimo.long())
return_bom[0][19]
return_ruim[0][19]
return_otimo[0][19]
# return_otimo.detach().numpy()[0][19] ## Converte em Numpy, tem que usar detach primeiro e depois pegar o que voce^ quer
tensor_dot(return_otimo, return_bom)
def tensor_dot(x,y):
    return np.dot(x.detach().numpy()[0][19], y.detach().numpy()[0][19])

# abs(return_otimo[0][19][0].item() - return_ruim[0][19][0].item())
def diff_tensors(x,y):
    sun = 0
    for i in range(200):
        sun += abs(x[0][19][i].item() - y[0][19][i].item())
    return sun

def tensor_value(x):
    sun = 0
    for i in range(200):
        sun += x[0][19][i].item()
    return sun
dict_tensors = [('otimo', return_otimo),
               ('bom', return_bom),
               ('ruim', return_ruim),
               ('perfeito', return_perfeito),
               ('pessimo', return_pessimo),]
for i in dict_tensors:
    print(i[0])
    print(tensor_value(i[1]), '\n')
tensor_dot(return_otimo, return_bom)
tensor_dot(return_otimo, return_ruim)
tensor_dot(return_otimo, return_perfeito)
# Exemplo de chamada de predição
seq_length=20 # good to use the length that was trained on
print(test_review_neg)
predict(net, test_review_neg, seq_length)
from sklearn.metrics import confusion_matrix,classification_report