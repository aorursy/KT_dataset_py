import random

import jieba
corpus_file = open("../input/santi.txt", "r", encoding="utf-8")

raw_text = corpus_file.readlines()

text = ""

for line in raw_text:

    text += line.strip()

corpus_file.close()

# 选择分词或者是分字

split_mode = "jieba"

if split_mode == "char":

    token_list = [char for char in text]

# 利用jieba库分词

elif split_mode == "jieba":

    token_list = [word for word in jieba.cut(text)]

# 确定ngram的历史检索长度，即n

ngram_len = 4
# 初始化ngram词典

ngram_dict = {}

for i in range(1, ngram_len): # i = 1 2 3

    for j in range(len(token_list) - i - 1):

        # 统计前缀是[j, j+i]个词的时候第j+i+1个词出现的次数

        key = "".join(token_list[j: j + i + 1])

        value = "".join(token_list[j + i + 1])

        # 为第一次出现的键建立字典

        if key not in ngram_dict:

            ngram_dict[key] = {}

        # 初始化字典内每个键值对映射的计数器

        if value not in ngram_dict[key]:

            ngram_dict[key][value] = 0

        ngram_dict[key][value] += 1
# 对输入进行分字或分词

start_text = "程心觉得"

gen_len = 200

topn = 3



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

    next_word = random.choice(temp_list[:topn])[0]

    word_list.append(next_word)

   

print("".join(word_list))