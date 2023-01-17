import numpy as np
import math
import jieba
import jieba.posseg as psg
from jieba import analyse
from gensim import corpora,models
import functools
import os
print(os.listdir("../input"))
#读取停用词
def get_stopword_list():
    with open("../input/keywordextract/stopword.txt") as f:
        return [w.replace('\n','') for w in f.readlines()]
#分词是否包含词性标注
def seg_to_list(sentence,pos=False):
    if not pos:
        return jieba.cut(sentence=sentence)
    else:
        return psg.cut(sentence=sentence)
seg_list = seg_to_list("今天天气很不错喔",True)
for seg in seg_list:
    print("word:%s,flag:%s" %(seg.word,seg.flag))
stopword_list = get_stopword_list()
seg_list = seg_to_list("今天天气很不错喔",True)
for seg in seg_list:
    if (not seg.word in stopword_list) and len(seg.word)>1:
        print(seg.word)
#去掉干扰词.返回结果是有用的词
def word_filter(seg_list,pos=False):
    filter_list = []
    stopword_list = get_stopword_list()
    for seg in seg_list:
        if not pos:
            #不需要过滤掉非n的词性
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        if (not word in stopword_list) and len(word)>1:
            filter_list.append(word)
    return filter_list
    
#加载语料库
def load_data(corpus_path,pos=False):
    doc_list = []
    with open(corpus_path,'r') as f:
        for line in f.readlines():
            content = line.strip()
            seg_list = seg_to_list(content,pos)
            filter_list = word_filter(seg_list,pos)
            doc_list.append(filter_list)
    return doc_list
# idf值统计方法
def train_idf(doc_list):
    idf_dic = {}
    # 总文档数
    tt_count = len(doc_list)
    
    # 每个词出现的文档数
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0
    
    # 按公式转换为idf值，分母加1进行平滑处理
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count / (1.0 + v))
    
    # 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值

    default_idf = math.log(tt_count / (1.0))
    return idf_dic, default_idf
#  排序函数，用于topK关键词的按值排序
def cmp(e1, e2):
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1
# TF-IDF类
class TfIdf(object):
    # 四个参数分别是：训练好的idf字典，默认idf值，处理后的待提取文本，关键词数量
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic, self.default_idf = idf_dic, default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    # 统计tf值
    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0.0) + 1.0

        tt_count = len(self.word_list)
        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / tt_count

        return tf_dic

    # 按公式计算tf-idf
    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0)

            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        tfidf_dic.items()
        # 根据tf-idf排序，去排名前keyword_num的词作为关键词
        for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')
def tfidf_extract(word_list, pos=False, keyword_num=10):
    doc_list = load_data("../input/keywordextract/corpus.txt",pos)
    idf_dic, default_idf = train_idf(doc_list)
    tfidf_model = TfIdf(idf_dic, default_idf, word_list, keyword_num)
    tfidf_model.get_tfidf()
text = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
           '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
           '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
           '重庆市民政局巡视员谭明政。晋江市人大常委会主任陈健倩,以及10余个省、市、自治区民政局' + \
           '领导及四十多家媒体参加了发布会。中华社会救助基金会秘书长时正新介绍本年度“中国爱心城' + \
           '市”公益活动将以“爱心城市宣传、孤老关爱救助项目及第二届中国爱心城市大会”为主要内容,重庆市' + \
           '、呼和浩特市、长沙市、太原市、蚌埠市、南昌市、汕头市、沧州市、晋江市及遵化市将会积极参加' + \
           '这一公益活动。中国雅虎副总编张银生和凤凰网城市频道总监赵耀分别以各自媒体优势介绍了活动' + \
           '的宣传方案。会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理' + \
           '事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。晋江市人大' + \
           '常委会主任陈健倩介绍了大会的筹备情况。'

pos = False
seg_list = seg_to_list(text, pos)
filter_list = word_filter(seg_list, pos)
#print(len(filter_list))
print('不过滤非名词性的TF-IDF模型结果：')
tfidf_extract(filter_list)
pos = True
seg_list = seg_to_list(text, pos)
filter_list = word_filter(seg_list, pos)
#print(len(filter_list))
print('过滤非名词性的TF-IDF模型结果：')
tfidf_extract(filter_list)

