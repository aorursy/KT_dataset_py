import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from sklearn import metrics
from matplotlib import cm
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

sns.set(color_codes=True)
sns.set(style="whitegrid")
raw_data = pd.read_json('../input/train.json')
raw_ingredients = [item for sublist in list(raw_data['ingredients']) for item in sublist]
raw_data['seq_length'] = [len(item) for item in raw_data['ingredients']]
raw_data.head()
sns.distplot(raw_data['seq_length'],axlabel='Number of ingredients per dish', color="m")
plt.title('Distribution of number of ingredients per dish')
plt.ylabel('(%)')
plt.show()
def distribution_fit(data, distribution):
    y, x = np.histogram(data, bins=200, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    params = distribution.fit(data)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
    sse = np.sum(np.power(y - pdf, 2.0))
    return pdf,sse
pdf, sse = distribution_fit(raw_data['seq_length'], st.norm)
print('Sum of squared errors: {}' .format(sse))
max_length = 35
new_data = pd.DataFrame()
new_data = raw_data.loc[raw_data['seq_length'] < max_length].sample(frac=1).reset_index(drop=True).copy()


sns.distplot(new_data['seq_length'],axlabel='Number of ingredients per dish', color="m")
plt.title('Distribution of number of ingredients per dish')
plt.ylabel('(%)')
plt.show()

pdf, sse = distribution_fit(new_data['seq_length'], st.norm)
print('Sum of squared errors: {}' .format(sse))
def get_label_representation(dataframe, labels_name):
    num_labels = sorted(set((dataframe[labels_name].values)))
    label_count = {}
    for i,x in enumerate(num_labels):
        label_count[x] = len(dataframe.loc[dataframe[labels_name] == x])
    return label_count
dist = get_label_representation(new_data, 'cuisine')
plt.figure(figsize=(20,3))
sns.barplot(list(dist.keys()), list(dist.values()), color='m')
plt.title('Distribution of dishes')
plt.ylabel('Number')
plt.show()
def balance_dataframe(df,labels_n, hard_balance=False):
    representation = get_label_representation(df, labels_n)
    label_keys = list(representation.keys())
    label_values = list(representation.values())
    soft_value = 3000
    min_value = min(label_values)
    cols = list(df.columns)
    balanced_df = pd.DataFrame()
    if hard_balance:
        thresh = min_value
    else:
        thresh = soft_value
    for i,x in enumerate(label_keys):
        label_slice = df.loc[df[labels_n] == x].sample(min(thresh, label_values[i]))
        balanced_df = balanced_df.append(label_slice)
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    return balanced_df
balanced_df = balance_dataframe(new_data, 'cuisine', hard_balance=True)
almost_balanced_df = balance_dataframe(new_data, 'cuisine', hard_balance=False)
unbalanced_df = new_data

for dataframe in [unbalanced_df, almost_balanced_df, balanced_df]:
    print('\n\n')
    print('Number of dishes: {}' .format(len(dataframe)))
    dist = get_label_representation(dataframe, 'cuisine')
    plt.figure(figsize=(20,3))
    sns.barplot(list(dist.keys()), list(dist.values()), color='m')
    plt.title('Distribution of dishes')
    plt.ylabel('Number')
    plt.show()
def tokenize_df(data_, cutoff = 1):
    data = data_.copy()
    raw_ingredients = [item for sublist in list(data['ingredients']) for item in sublist]
    ingredient_frequencies = collections.Counter(raw_ingredients)
    num_ingredients = len(set(raw_ingredients))
    print('Number of total ingredients: {}' .format(num_ingredients))
    sorted_ingredients_by_prc = ingredient_frequencies.most_common()
    vocabulary_items = [item[0] for item in sorted_ingredients_by_prc if item[1] > cutoff]
    print('Number of common ingredients (appearing more than once): {}' .format(len(vocabulary_items)))
    
    # Create the vocabulary that maps ingredients to the numerical values. 
    # 0 value is reserved for padding while the last value of the vocabulary 
    # is reserved for the UNK token.
    vocabulary = sorted(vocabulary_items)
    unk_index = len(vocabulary) + 1
    voc2idx = {}
    voc2idx.update({'PAD' : 0})
    ind = 1
    for item in sorted(set(raw_ingredients)):
        if item in vocabulary:
            voc2idx.update({item : ind})
            ind += 1
        else:
            voc2idx[item] = unk_index
    idx2voc = dict(zip(voc2idx.values(), voc2idx.keys()))
    idx2voc[unk_index] = 'UNK'
    print('Vocabulary length : {}' .format(len(idx2voc)))
    
    
    # Update the dataframe with tokenized data.
    data['idx_ingredients'] = [[voc2idx[item] for item in sublist]for sublist in data['ingredients']]
    cuisines = sorted(set(list(data['cuisine'])))
    idx2cuisine = dict(enumerate(cuisines))
    cuisine2idx = dict(zip(idx2cuisine.values(), idx2cuisine.keys()))
    data['idx_cuisine'] = [cuisine2idx[item] for item in data['cuisine']]
    data = data.sample(frac=1).reset_index(drop=True)
    data.head()
    return data ,voc2idx, idx2voc, idx2cuisine, cuisine2idx
tokenized_df = tokenize_df(balanced_df)
df = tokenized_df[0]
voc_len = len(tokenized_df[2])
df.head()
train_len, valid_len = np.floor(len(df)*0.8), np.floor(len(df)*0.2)
train_df = df.loc[:train_len-1].sample(frac=1).reset_index(drop=True).copy()
valid_df = df.loc[train_len:(train_len + valid_len)].sample(frac=1).reset_index(drop=True).copy()
print('Training set length: {}\nValidation set length: {}' .format(train_len,valid_len))
class dish_iterator():
    
    def __init__(self, df):
        self.size = len(df)
        self.dfs = df
        self.cursor = 0
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.dfs = self.dfs.sample(frac=1).reset_index(drop=True)
        self.cursor = 0
    
    def next_batch(self, n):
        res = self.dfs.loc[self.cursor:self.cursor+n-1]
        self.cursor += n
        maxlen = max(res['seq_length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['seq_length'].values[i]] = res['idx_ingredients'].values[i]
        
        if self.cursor+n+1 > self.size:
            self.epochs += 1
            self.shuffle()
        return x, res['idx_cuisine']
def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_graph(
    vocab_size,
    output_dims = 50,
    batch_size = 32,
    num_classes = 20,
    learning_rate = 1e-3):

    reset_graph()

    # Placeholders
    x = tf.placeholder(tf.int32, [batch_size, None], name='input_tensor')
    y = tf.placeholder(tf.int32, [batch_size], name='labels_tensor')
    keep_prob = tf.placeholder_with_default(1.0,[])

    # Embedding layer
    embeddings = tf.get_variable('embeddings', [vocab_size, output_dims])
    model_inputs = tf.nn.embedding_lookup(embeddings, x)
    
    # Global Average Pooling to reduce sequences
    pooling_output = tf.reduce_mean(model_inputs, axis = 1)
    pooling_output_d = tf.nn.dropout(pooling_output, keep_prob)

    # Softmax layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [output_dims, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(pooling_output_d, W) + b
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    print('\nGraph summary:')
    print('Embedding layer size: {}' .format(model_inputs.get_shape()))
    print('Global Average Pooling size: {}' .format(pooling_output.get_shape()))
    print('Dropout layer size: {}' .format(pooling_output_d.get_shape()))
    print('Softmax layer size: {}' .format(logits.get_shape()))
    trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Trainable parameters: {}' .format(trainable_params))
    
    return {
        'x': x,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy, 
    }
def train_graph(graph,
                train_dataset,
                validation_dataset,
                batch_size = 32, 
                num_epochs = 50, 
                iterator = dish_iterator, 
                savepath = False):
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tr = iterator(train_dataset)
        vd = iterator(validation_dataset)

        step, accuracy = 0, 0
        tr_acc, vd_acc = [], []
        current_epoch = 0
        early_stopping= 0
        tolerance_flag= False
        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {graph['x']: batch[0], graph['y']: batch[1], graph['dropout']: 0.5}
            accuracy_, _ = sess.run([graph['accuracy'], graph['ts']], feed_dict=feed)
            accuracy += accuracy_
            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_acc.append(accuracy / step)
                step, accuracy = 0, 0
                vd_epoch = vd.epochs
                while vd.epochs == vd_epoch:
                    step += 1
                    batch = vd.next_batch(batch_size)
                    feed = {graph['x']: batch[0], graph['y']: batch[1]}
                    accuracy_ = sess.run([graph['accuracy']], feed_dict=feed)[0]
                    accuracy += accuracy_
                
                vd_acc.append(accuracy / step)
                step, accuracy = 0,0
                
                if (vd_acc[-1] - early_stopping < 3e-3):
                    if tolerance_flag:
                        print('Early stopping at epoch {}: '
                              'Train accuracy {} - Validation accuracy {}' .format(
                            current_epoch, tr_acc[-1], vd_acc[-1]))
                        break
                    tolerance_flag = True
#                     early_stopping = vd_acc[-1]
                else:
                    early_stopping = vd_acc[-1]
                    tolerance_flag = False
        if savepath:
            save_path = saver.save(sess, savepath)
            print("Model saved in path: %s" % save_path)

    return tr_acc, vd_acc
class inference_iterator():
    def __init__(self, df):
        self.dfs = df.reset_index(drop=True)
        self.size = len(self.dfs)
        self.epochs = 0
        self.cursor = 0

    def next_batch(self): 
        res = self.dfs.loc[self.cursor]
        x = np.array(res['idx_ingredients'])
        self.cursor += 1
        if self.cursor  + 1 > self.size:
            self.epochs += 1
        return [x]
def predict_dish(graph, checkpoint, inference_data, iterator= inference_iterator):
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint) 
        
        predictions = []
        tinf = iterator(inference_data)
        while tinf.epochs == 0:
            item = tinf.next_batch()
            feed = {graph['x']: item}
            predictions_ = sess.run([graph['preds']], feed_dict=feed)[0]
            predictions.append(predictions_)
        
    return predictions
def create_confusion_matrix(labels, predictions):
    cm = metrics.confusion_matrix(labels, predictions)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
def plot_distribution(dataframe, label_title):
    dist = get_label_representation(dataframe, label_title)
    sns.barplot(list(dist.keys()), list(dist.values()), color='m')
    plt.title('Distribution of dishes')
    plt.ylabel('Number')
    plt.show()
def session_wrapper(dataframe, cutoff_number, save):
    tokenized_df = tokenize_df(dataframe, cutoff = cutoff_number)
    df = tokenized_df[0]
    voc_len = len(tokenized_df[2])
    train_len, valid_len = np.floor(len(df)*0.8), np.floor(len(df)*0.2)
    train_df, valid_df = df.loc[:train_len-1], df.loc[train_len:(train_len + valid_len)]
    print('Training set length: {}\nValidation set length: {}' .format(train_len,valid_len))

    g = build_graph(vocab_size=voc_len)
    tr_acc, vd_acc = train_graph(g, train_df, valid_df, savepath= save)

    plt.plot(tr_acc,label='Training accuracy')
    plt.plot(vd_acc, label='Validation accuracy')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.show()

    inf_data = valid_df
    g = build_graph(vocab_size=voc_len, batch_size=1)
    preds = predict_dish(g, save, inf_data)
    final_preds = [np.argmax(item) for item in preds]
    create_confusion_matrix(valid_df['idx_cuisine'].values, final_preds)
    plot_distribution(train_df, 'idx_cuisine')
    return tokenized_df
_ = session_wrapper(balanced_df, 3, 'saves/balanced')
_ = session_wrapper(unbalanced_df, 3, 'saves/unbalanced')
_ = session_wrapper(almost_balanced_df, 3, 'saves/semibalanced')
def alt_session_wrapper(dataframe_, cutoff_number, save):
    pre_split_df = dataframe_.copy()
    train_len, valid_len = np.floor(len(pre_split_df)*0.8), np.floor(len(pre_split_df)*0.2)
    train_df = pre_split_df.loc[:train_len-1].sample(frac=1).reset_index(drop=True).copy()
    valid_df = pre_split_df.loc[train_len:(train_len + valid_len)].sample(frac=1).reset_index(drop=True).copy()
    print('Training set length: {}\nValidation set length: {}' .format(train_len,valid_len))
    tokenized_df = tokenize_df(train_df, cutoff = cutoff_number)
    df = tokenized_df[0]
    voc2idx = tokenized_df[1]
    idx2voc = tokenized_df[2]
    idx2cuisine = tokenized_df[3]
    cuisine2idx = tokenized_df[4]
    unk = len(tokenized_df[2])-1
    valid_df['idx_ingredients'] = [[voc2idx[item] if item in list(voc2idx.keys()) else unk for item in sublist]for sublist in valid_df['ingredients']]
    valid_df['idx_cuisine'] = [cuisine2idx[item] for item in valid_df['cuisine']]
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
    voc_len = len(tokenized_df[2])
    
    g = build_graph(vocab_size=voc_len)
    tr_acc, vd_acc = train_graph(g, df, valid_df, savepath= save)

    plt.plot(tr_acc,label='Training accuracy')
    plt.plot(vd_acc, label='Validation accuracy')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.show()

    inf_data = valid_df
    g = build_graph(vocab_size=voc_len, batch_size=1)
    preds = predict_dish(g, save, inf_data)
    final_preds = [np.argmax(item) for item in preds]
    create_confusion_matrix(valid_df['idx_cuisine'].values, final_preds)
    plot_distribution(df, 'idx_cuisine')
    return tokenized_df
_ = alt_session_wrapper(balanced_df, 3, 'saves/alt_balanced')
_ = alt_session_wrapper(unbalanced_df, 3, 'saves/alt_unbalanced')
tokenized_data = alt_session_wrapper(almost_balanced_df, 3, 'saves/alt_semi_balanced')
def alt_balance_dataframe(df,labels_n):
    representation = get_label_representation(df, labels_n)
    label_keys = list(representation.keys())
    label_values = list(representation.values())
    soft_value = 3000
    min_value = min(label_values)
    cols = list(df.columns)
    balanced_df = pd.DataFrame()
    for i,x in enumerate(label_keys):
        label_slice = df.loc[df[labels_n] == x].sample(soft_value, replace=True)
        balanced_df = balanced_df.append(label_slice)
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    return balanced_df
def alt_session_wrapper(dataframe_, cutoff_number, save):
    pre_split_df = dataframe_.copy()
    train_len, valid_len = np.floor(len(pre_split_df)*0.8), np.floor(len(pre_split_df)*0.2)
    train_df_ = pre_split_df.loc[:train_len-1].sample(frac=1).reset_index(drop=True).copy()
    valid_df = pre_split_df.loc[train_len:(train_len + valid_len)].sample(frac=1).reset_index(drop=True).copy()
    print('Training set length: {}\nValidation set length: {}' .format(train_len,valid_len))
    train_df = alt_balance_dataframe(train_df_, 'cuisine')
    tokenized_df = tokenize_df(train_df, cutoff = cutoff_number)
    df = tokenized_df[0]
    voc2idx = tokenized_df[1]
    idx2voc = tokenized_df[2]
    idx2cuisine = tokenized_df[3]
    cuisine2idx = tokenized_df[4]
    unk = len(tokenized_df[2])-1
    valid_df['idx_ingredients'] = [[voc2idx[item] if item in list(voc2idx.keys()) else unk for item in sublist]for sublist in valid_df['ingredients']]
    valid_df['idx_cuisine'] = [cuisine2idx[item] for item in valid_df['cuisine']]
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)
    voc_len = len(tokenized_df[2])
    
    g = build_graph(vocab_size=voc_len)
    tr_acc, vd_acc = train_graph(g, df, valid_df, savepath= save)

    plt.plot(tr_acc,label='Training accuracy')
    plt.plot(vd_acc, label='Validation accuracy')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.show()

    inf_data = valid_df
    g = build_graph(vocab_size=voc_len, batch_size=1)
    preds = predict_dish(g, save, inf_data)
    final_preds = [np.argmax(item) for item in preds]
    create_confusion_matrix(valid_df['idx_cuisine'].values, final_preds)
    plot_distribution(df, 'idx_cuisine')
    return tokenized_df
_ = alt_session_wrapper(unbalanced_df, 3, 'saves/alt_balanced_embeddings_trim')
test_data = pd.read_json('../input/test.json')
test_data.head()
voc2idx = tokenized_data[1]
idx2voc = tokenized_data[2]
idx2cuisine = tokenized_data[3]
unk = len(tokenized_data[2])-1
unk
test_data['idx_ingredients'] = [[voc2idx[item] if item in list(voc2idx.keys()) else unk for item in sublist]for sublist in test_data['ingredients']]
test_data.head()
voc_len = len(tokenized_data[2])
g = build_graph(vocab_size=voc_len, batch_size= 1)
preds = predict_dish(g, 'saves/alt_semi_balanced', test_data)
final_preds = [np.argmax(item) for item in preds]
predictions = [idx2cuisine[item] for item in final_preds]

out = pd.DataFrame()
out['id']= test_data['id'].values
out['predictions'] = predictions
out.to_csv('submission.csv', index=False)
