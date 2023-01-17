PLACEHOLDER_CORPUS_FILE_PATH = "../input/santi.txt"

PLACEHOLDER_MODEL_FILE_PATH = "model.pkl"

PLACEHOLDER_SPLIT_MODE = "jieba"

PLACEHOLDER_NGRAM_LEN = 4
## Init

import random

import pickle

import jieba



corpus_file = open(PLACEHOLDER_CORPUS_FILE_PATH, "r", encoding="utf-8")

raw_text = corpus_file.readlines()

text = ""

for line in raw_text:

    text += line.strip()

corpus_file.close()
## Model Config



# 选择分词或者是分字

split_mode = PLACEHOLDER_SPLIT_MODE

if split_mode == "char":

    token_list = [char for char in text]

# 利用jieba库分词

elif split_mode == "jieba":

    token_list = [word for word in jieba.cut(text)]

# 确定ngram的历史检索长度，即n

ngram_len = PLACEHOLDER_NGRAM_LEN
## Run



# 初始化ngram词典

ngram_dict = {}

for i in range(1, ngram_len): # i = 1 2 3

    for j in range(len(token_list) - i - 1):

        # 以前n-1个词为键，第n个词为值，统计映射次数

        key = "".join(token_list[j: j + i + 1])

        value = "".join(token_list[j + i + 1])

        # 为第一次出现的键建立字典

        if key not in ngram_dict:

            ngram_dict[key] = {}

        # 初始化字典内每个键值对映射的计数器

        if value not in ngram_dict[key]:

            ngram_dict[key][value] = 0

        ngram_dict[key][value] += 1



with open(PLACEHOLDER_MODEL_FILE_PATH, 'wb') as f:

    pickle.dump({

        "ngram_dict": ngram_dict,

        "ngram_len": ngram_len

    }, f)
!ls /
PLACEHOLDER_MODEL_FILE_PATH = "model.pkl"
## Init

import random

import pickle

import jieba

with open(PLACEHOLDER_MODEL_FILE_PATH, 'rb') as f:

    tmp = pickle.load(f)

    ngram_dict = tmp["ngram_dict"]

    ngram_len = tmp["ngram_len"]
PLACEHOLDER_START_TEXT = "今天我看到了一个很奇怪的亮光"

PLACEHOLDER_GEN_LEN = 200

PLACEHOLDER_TOPN = 5
## Run

# 对输入进行分字或分词

start_text = PLACEHOLDER_START_TEXT

gen_len = PLACEHOLDER_GEN_LEN

topn = PLACEHOLDER_TOPN



if split_mode == "char":

    word_list = [char for char in start_text]

elif split_mode == "jieba":

    word_list = [word for word in jieba.cut(start_text)]



# gen_len是我们期望的生成字数或词数

for i in range(gen_len):

    temp_list = []

    # 统计给定前小于等于n-1个词的情况下，下一个词的词频分布

    for j in range(1, ngram_len):

        if j >= len(word_list):

            continue

        prefix = "".join(word_list[-(j + 1):])

        if prefix in ngram_dict:

            temp_list.extend(ngram_dict[prefix].items())

    # 按词频对词排序

    temp_list = sorted(temp_list, key=lambda d: d[1], reverse=True)

    next_word = ''

    if len(temp_list) == 0:

        next_word = token_list[random.randint(0, len(token_list))]

    elif temp_list[0] == '，' or temp_list[0] == '。' or temp_list[0] == '\n':

        # 如果最高频词是标点，则选择最高频词

        next_word = temp_list[0]

    else:

    # 否则从前topn中随机选一个

        next_word = random.choice(sorted(temp_list, key=lambda d: d[1], reverse=True)[:topn])[0]

    word_list.append(next_word)

   

print("".join(word_list))