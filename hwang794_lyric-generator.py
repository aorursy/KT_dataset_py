# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
f = open("/kaggle/input/train-lyric/result.txt", "r")
text = f.read()
print(text[3742:4320])
#导入结巴分词库并定义结巴分词函数
import jieba.posseg as pseg
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return ' '.join([word for word,flag in words])
text = jieba_tokenizer(text)
text=text.split(" ")
print("length of text:", len(text))
print(text[2344:2366])
print("".join(text[2400:2500]))
print(text[2400:2500])
n = len(text)
w = len(set((text)))
print(f"总共有{n}个中文词汇")
print(f"有{w}个不同的中文词汇")
import tensorflow as tf

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=w,
    char_level=False,
    filters=' '
    )
#让tokenizer遍历所有歌词信息
#并建立字典
tokenizer.fit_on_texts(text)
text_as_int = tokenizer.texts_to_sequences([text])[0]
s_idx = 9527
e_idx = 9676
partial_indices = text_as_int[s_idx:e_idx]
partial_texts = [
    tokenizer.index_word[idx] \
    for idx in partial_indices
]
print("原本中文句子：")
print()
print("".join(partial_texts))
print("-"*20)
print("原本的中文字序列：")
print()
print(partial_texts)
print()
print("-" * 20)
print()
print("转换后的索引序列：")
print()
print(partial_indices)
_type = type(text_as_int)
n = len(text_as_int)
print(f"text_as_int 是一个 {_type}\n")
print(f"歌词总长度： {n}\n")
print("前 5 索引：", text_as_int[:5])
SEQ_LENGTH = 10
BATCH_SIZE = 64
#将上面得到的文字序列变成tensor
characters = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = characters.batch(SEQ_LENGTH+1,drop_remainder=True)
#总序列长度
steps_per_epoch=len(text_as_int)

def build_seq_pairs(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

ds = sequences.map(build_seq_pairs)\
    .shuffle(steps_per_epoch)\
    .batch(BATCH_SIZE,drop_remainder=True)
for b_inp, b_tar in ds.take(1):
    print("起始句子的 batch：")
    print(b_inp, "\n")
    print("目標句子的 batch：")
    print(b_tar, "\n")
    print("-" * 20, "\n")
    
    print("第一個起始句子的索引序列：")
    first_i = b_inp.numpy()[0]
    print(first_i, "\n")
    print("第一個目標句子的索引序列：")
    first_t = b_tar.numpy()[0]
    print(first_t, "\n")
    print("-" * 20, "\n")
    
    d = tokenizer.index_word
    print("第一個起始句子的文本序列：")
    print([d[i] for i in first_i])
    print()
    print("第一個目標句子的文本序列：")
    print([d[i] for i in first_t])
EMBEDDING_DIM = 512
RNN_UNITS = 1024
#keras LSTM 模型
model = tf.keras.Sequential()

#词嵌入层
model.add(
    tf.keras.layers.Embedding(
    input_dim=w,
    output_dim=EMBEDDING_DIM,
    batch_input_shape=[
        BATCH_SIZE,None]
    ))

#LSTM层
model.add(
    tf.keras.layers.LSTM(
    units=RNN_UNITS,
    return_sequences=True,
    stateful=True,
    recurrent_initializer='glorot_uniform'))
#全连接层
model.add(
    tf.keras.layers.Dense(
    w))

model.summary()
LEARNING_RATE = 0.001
def loss(y_true,y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(
    y_true,y_pred,from_logits=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE),
        loss = loss)

EPOCHS=10
history = model.fit(ds,
                   epochs=EPOCHS)

model.save_weights('/kaggle/checkpoint/update3_checkpoint')
EMBEDDING_DIM = 512
RNN_UNITS = 1024
BATCH_SIZE = 1
infer_model = tf.keras.Sequential()
infer_model.add(
    tf.keras.layers.Embedding(
        input_dim=w, 
        output_dim=EMBEDDING_DIM,
        batch_input_shape=[
            BATCH_SIZE, None]
))
infer_model.add(
    tf.keras.layers.LSTM(
    units=RNN_UNITS, 
    return_sequences=True, 
    stateful=True
))
infer_model.add(
    tf.keras.layers.Dense(
        w))
infer_model.load_weights('/kaggle/checkpoint/update3_checkpoint')
infer_model.build(
    tf.TensorShape([1, None]))
seed_indices = [3421]
input = tf.expand_dims(seed_indices, axis=0)
predictions = infer_model(input)
predictions = tf.squeeze(predictions, 0)
sampled_indices = tf.random.categorical(predictions, 
                num_samples=1)[-1,0].numpy()
print(sampled_indices)
d = tokenizer.index_word
print(d[sampled_indices])
input_eval = jieba_tokenizer("一样的月光").split(" ")
input_eval = tokenizer.texts_to_sequences([input_eval])[0]
input_eval = tf.expand_dims(input_eval,0)
text_generated = []
#温度越高，文本随机性越高
temperature = 1
model.reset_states()
for i in range(100):
    predictions = infer_model(input_eval)
    predictions = tf.squeeze(predictions,0)
    predictions = predictions/temperature
    predicted_id = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()
    input_eval = tf.expand_dims([predicted_id],0)
    if predicted_id == 1:
        print("next line detected")
    text_generated.append(d[predicted_id])
    
("一样的月光"+''.join(text_generated))
# def generate_lyric(model,start_string,length=100,temperature=0.9):
#     num_generate = length
#     input_eval = tokenizer.texts_to_sequences([start_string])[0]
#     input_eval = tf.expand_dims(input_eval,0)
#     text_generated = []
#     #温度越高，文本随机性越高
#     #temperature = 0.4
    
#     model.reset_states()
#     for i in range(num_generate):
#         predictions = model(input_eval)
#         predictions = tf.squeeze(predictions,0)
#         predictions = predictions/temperature
#         predicted_id = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()
#         input_eval = tf.expand_dims([predicted_id],0)
#         text_generated.append(d[predicted_id])
#     return (start_string+''.join(text_generated))
#print(generate_lyric(infer_model, start_string=u"我人都傻了",temperature=0.9))
