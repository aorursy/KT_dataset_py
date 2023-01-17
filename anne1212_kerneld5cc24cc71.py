import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.contrib.layers import fully_connected, dropout
from nltk.tokenize import TweetTokenizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

submission = pd.read_csv('../input/submission/submission.csv')
submission.to_csv('submission.csv',index = False)
train_data = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep='\t')
test_data = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep='\t')
train_data.head()
train_data["SentenceId"].unique().size
corpus = train_data['Phrase'].append(test_data['Phrase']).values
assert corpus.shape[0] == train_data['Phrase'].shape[0] + test_data['Phrase'].shape[0]
def tokenize(x):
    return [x] if len(x) == 1 else TweetTokenizer().tokenize(x)
     
tfVectorizer = TfidfVectorizer(tokenizer=tokenize)
tf_idf_total = tfVectorizer.fit_transform(corpus)
tf_idf_total = tf_idf_total.todense()
tf_idf_dense = tf_idf_total[:train_data['Phrase'].shape[0]]
tf_idf_dense.shape
tfVectorizer.get_params()
train_vocab = tfVectorizer.vocabulary_
train_vocab
tfVectorizer.get_feature_names()
def top_tfidf_words(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_words_in_doc(Xtr, features, row_id, top_n=20):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_words(row, features, top_n)

def top_mean_words(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_words(tfidf_means, features, top_n)

def top_words_by_class(Xtr, y, features, min_tfidf=0.1, top_n=20):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_words(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classWords_h(dfs, num_class=9):
    fig = plt.figure(figsize=(12, 100), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        #z = int(str(int(i/3)+1) + str((i%3)+1))
        ax = fig.add_subplot(num_class, 1, i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=16)
        ax.set_ylabel("Word", labelpad=16, fontsize=16)
        ax.set_title("Class = " + str(df.label), fontsize=25)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
def get_batch(index, batch_size = 2000):
    
    index = shuffle(index)
    
    batch_index = []
    for sample in index:
        batch_index.append(sample)
        
        if len(batch_index) == batch_size:
            yield batch_index
            batch_index= []
        
    if len(batch_index) > 0:
        yield batch_index
test_data.head()
tf_idf_dense2 = tf_idf_total[train_data['Phrase'].shape[0]:]
tf_idf_dense2.shape
assert len(tf_idf_dense) + len(tf_idf_dense2) == len(tf_idf_total)
tfVectorizer2.get_params()
test_vocab = tfVectorizer2.vocabulary_
tfVectorizer2.get_feature_names()
n_sentance, n_words = tf_idf_dense.shape

input_sentence = tf.placeholder(dtype=tf.float32, shape = [None, n_words])
labels = tf.placeholder(dtype=tf.int32, shape = [None])
hidden_layer1 = fully_connected(input_sentence, num_outputs =  50)
hidden_layer1 = dropout(hidden_layer1, keep_prob=0.7)
hidden_layer2 = fully_connected(hidden_layer1, num_outputs =  30)
hidden_layer2 = dropout(hidden_layer2, keep_prob=0.8)
output = fully_connected(hidden_layer2, activation_fn=None, num_outputs =  5)

label_onehot = tf.one_hot(labels, depth = 5)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = label_onehot, logits = output)
loss = tf.reduce_mean(loss)
y_ = tf.nn.softmax(output)

optimize = tf.train.AdamOptimizer().minimize(loss)

predictions = tf.cast(tf.argmax(y_,1), tf.int32)
acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions)
batch_acc = 1 - tf.reduce_mean(tf.cast(tf.cast(labels - predictions, tf.bool), tf.float32))
x_train, x_vad, y_train, y_vad = train_test_split(tf_idf_dense, train_data['Sentiment'], test_size=0.2)
print(x_train.shape)
print(x_vad.shape)
i, j, max_epoch, patience, loss_max, = 0, 0, 50, 5, np.inf 
n = 1
optimal_i = i

optimal_i = 38
gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list='0')
with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # retrain model by using all data with optimal_i
    sess.run(tf.global_variables_initializer())
    index = np.arange(len(tf_idf_dense))
    for i in range(optimal_i):
        sess.run(tf.local_variables_initializer())
        for batch_index in list(get_batch(index)):
            batch_x, batch_y = tf_idf_dense[batch_index], train_data['Sentiment'].values[batch_index]
            _, loss_value, accuracy = sess.run([optimize, loss, acc_op], feed_dict= {input_sentence: batch_x,  labels: batch_y})
        print('accuracy for epoch {} is {}, loss is {}'.format(i, accuracy, loss_value)) 
        
    # --test--
    sess.run(tf.local_variables_initializer())
    pred_value = sess.run(predictions, feed_dict = {input_sentence: tf_idf_dense2}) 
    print(pred_value.shape)
    sentiment = test_data[['PhraseId']]
    sentiment['Sentiment'] = pred_value
    sentiment.to_csv('submission.csv', index = False)
    
tf.reset_default_graph()