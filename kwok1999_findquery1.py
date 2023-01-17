from gensim.summarization import bm25

from gensim import corpora

import heapq

import pickle

import math

from multiprocessing import Pool

from tqdm import tqdm

import pickle



"""

读文件

"""





def readFile(filename):

    with open(filename, "r", encoding="utf-8") as f:

        temp = f.readlines()

    print(len(temp))

    return temp





'''

获取列表中最大的前n个数值的位置索引,返回list

'''





def getListMaxNumIndex(num_list, topk):

    max_num_index = map(num_list.index, heapq.nlargest(topk, num_list))

#     min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))

    return list(max_num_index)





"""

list去重，保持原来的顺序

"""





def listDeduplicatInOrder(tempList):

    addr_to = list(set(tempList))

    addr_to.sort(key=tempList.index)

    return addr_to





"""

给每个query找到前50个相关文档

要注意，query的ansewer是否被点击过

query的候选集合是数据库ClickQuery中的所有query

输入：query list

输出：query index :对应前50个相关url

"""





def findRelated(queryList):

    queryUrlDict = {}

    for query in tqdm(queryList):

        scores = bm25Model.get_scores(query)

        # 返回前50个相关的title的索引

        res = getListMaxNumIndex(scores, 10000)

        # title_index 排序

        res = listDeduplicatInOrder(res)

        # 构建bm25模型用的是全量的title，所以要把全量title的index转化为非全量的index

        res_new = [text_title_index_list[line] for line in res]

        res_new = listDeduplicatInOrder(res_new)

        # print(res)

        # 找到前50个相关url

        related_url_list = []

#         print([reverse_title_dict[line] for line in res])

        # title_index 转为 url_index

        for i in res_new:

            related_url_list.extend(title_url_dict[i])  # title对应的url列表

        related_url_list = listDeduplicatInOrder(related_url_list)[:1000]

#         print([reverse_url_dict[line] for line in related_url_list])

        query_index = query_dict[" ".join(query)]

        queryUrlDict.update({query_index: related_url_list})

    return queryUrlDict





def run(data, index, size):  # data 传入数据，index 数据分片索引，size进程数

    size = math.ceil(len(data) / size)

    start = size * index

    end = (index + 1) * size if (index + 1) * size < len(data) else len(data)

    temp_data = data[start:end]

    print(start, end)

    '''

    给query找相关文档

    '''

    temp_data_new1 = findRelated(temp_data)

    return temp_data_new1  # 可以返回数据，在后面收集起来





def multi(num, data):

    processor = num

    res = []

    p = Pool(processor)

    for i in range(processor):

        res.append(p.apply_async(run, args=(data, i, processor,)))

        print(str(i) + ' processor started !')

    p.close()

    p.join()

    return res





def loadPickle(path):

    with open(path, "rb") as f:

        return pickle.load(f)





def savePickle(path, obj):

    with open(path, 'wb') as f:

        pickle.dump(obj, f)

# 将id列表转为可以阅读的内容





def saveToRead(res, save_path_name):

    read_res = {}

    for k, v in res.items():

        read_res.update(

            {reverse_query_dict[k]: [reverse_url_dict[value] for value in v]})

    saveFile(save_path_name, read_res)





def saveFile(save_path_name, obj):

    with open(save_path_name, "w") as f:

        f.write(str(obj))





"""

读取两个文件：

1. title

2. query

title用来构造bm25模型

query用来查询

已知的是：每个用户而言，query对应的url和title

必须知道的是：每个用户点击是否在查询到的title里，且是第几位

构造：

    query:index

    url:title:index

"""

import time

start = time.time()

query_list = loadPickle(

    "/kaggle/input/finduse/text_query_list.pickle")

title_list = loadPickle(

    "/kaggle/input/finduse/text_title_list.pickle")

title_url_dict = loadPickle(

    "/kaggle/input/finduse/title_url_dict.pickle")

query_dict = loadPickle(

    "/kaggle/input/finduse/query_dict.pickle")

reverse_url_dict = loadPickle(

    "/kaggle/input/finduse/reverse_url_dict.pickle")

reverse_query_dict = loadPickle(

    "/kaggle/input/finduse/reverse_query_dict.pickle")

query_url_dict = loadPickle(

    "/kaggle/input/finduse/query_url_dict.pickle")

text_title_index_list = loadPickle(

    "/kaggle/input/finduse/text_title_index_list.pickle")



# 根据点击文档的title，建立bm25模型

# titleList是所有titl构造的list,是一个嵌套list,每个list是一个title

bm25Model = bm25.BM25(title_list)

max_length = len(title_list)

# 读取query数据，目标是给每个query拿到对应的url列表

# enData = multi(2, query_list[:1000])

enData = findRelated(query_list[110000:112000])

res = dict()

for i in enData:

    res.update(i.get())



# 如果在1000个以内找不到对应的url，则此条query及对应url作废

# 查看所有找到的url是否是用户点击的

# wrong_query_url = dict()

# temp_res = []

# count = 0

# for k, v in res.items():

#     # print(k,v)

#     # k是query的index

#     # v是找到的url

#     # 用户点击列表

#     user_click_list = query_url_dict[k]

#     # print(user_click_list)

#     # 看user_click_list中的哪一个不在候选列表中

#     temp_list = []

#     temp_list = [line for line in user_click_list if line not in v]

#     if len(user_click_list)-len(temp_list)>0:

#         count = count+1

#     if len(temp_list) > 0:

#         temp_res.append([k, len(user_click_list), len(temp_list)])

#         wrong_query_url.update({k: temp_list})

# print(count)

# saveFile("/kaggle/working1111/temp_res.txt", temp_res)



savePickle(

    "/kaggle/working1111/queryUrl1.pickle", res)

# savePickle("/kaggle/working1111/wrong_query_url.pickle", wrong_query_url)

# saveToRead(

#     res, "/kaggle/working1111/queryUrl_read.txt")

# saveToRead(wrong_query_url,

#            "/kaggle/working1111/wrong_query_url_read.txt")

print("time:{}".format(time.time()-start))