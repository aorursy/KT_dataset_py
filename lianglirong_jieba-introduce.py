
import numpy as np
import jieba
import glob
import random
import os
print(os.listdir("../input"))

sent = '中文分词是文本处理不可或缺的一步!'
#精准模式
seg_list = jieba.cut(sentence=sent)
print('/'.join(seg_list))
#全模式
seg_list = jieba.cut(sentence=sent,cut_all=True)
print('/'.join(seg_list))
#搜索引擎模式
seg_list = jieba.cut_for_search(sentence=sent)
print('/'.join(seg_list))
#高频词提取
def get_TF(words,topK=10):
    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w,0)+1

    return sorted(tf_dic.items(),key=lambda x:x[1],reverse=True)[:topK]
#获取文件内容
def get_content(path):
    with open(path,'r',encoding='gbk',errors='ignore') as f:
        content = ''
        for line in f:
            content += line.strip()
    return content
def stop_words(path):
    with open(path) as f:
        return [l.strip() for l in f]
files = glob.glob("../input/nlp-data/news/C000013/*.txt")
#print(files)
cropus = [get_content(f) for f in files]
print("文件总数： ",len(cropus))
sample_index = random.randint(0,len(cropus))
print("样本下标：",sample_index)
sample_seg_list = [x for x in list(jieba.cut(cropus[sample_index])) if x not in stop_words("../input/nlp-data/stop_words.utf8")] 
print("样本文本：",cropus[sample_index])
print("样本分词后：",'/'.join(sample_seg_list))
top10 = get_TF(sample_seg_list)
print(top10)
sent = "今天我使用jieba分词，感觉挺好的，非常6"
seg_list = jieba.cut(sent)
print("加载自定义字典前:","/".join(seg_list))
#加载自定义字典
jieba.load_userdict("../input/nlp-data/user_dict.utf8")
sent = "今天我使用jieba分词，感觉挺好的，非常6"
seg_list = jieba.cut(sent)
print("加载自定义字典后:","/".join(seg_list))
