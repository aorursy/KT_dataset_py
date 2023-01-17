import time

from tqdm import tqdm

from scipy.sparse import coo_matrix

import numpy as np

import pickle

import psutil

import os



def notFind(candiList):

    count = 0

    for index, l in enumerate(candiList):

        flag = False

        for small in l:

            if len(small) != 0:

                flag = True

        if not flag:

            count = count+1

    print("not found:", count/index)





def getPartSparse(sparseMatrix, startValue, endValue, startIndex):

    """

    切分文件，拿到切分好的文件和下一次需要传的位置

    """

    row_list = sparseMatrix.row[startIndex:]

    row_list_temp = []

    for line in tqdm(row_list):

        if line >= startValue:

            if line < endValue:

                row_list_temp.append(line-startValue)

            else:

                break

    endIndex = startIndex+len(row_list_temp)

    col_list = sparseMatrix.col[startIndex:endIndex]

    data_list = sparseMatrix.data[startIndex:endIndex]

    shape1 = endValue-startValue

    shape2 = sparseMatrix.shape[1]

    return endIndex, coo_matrix((data_list, (row_list_temp, col_list)), shape=(shape1, shape2), dtype=np.float16)





def getCandidate(postemp, temp):

    # pos为当前位置

    # 以当前位置为中心，总共取50条数据

    # 分别计算向前和向后取多少条数据

    # 分为三种情况：情况1：前面没有25条，情况2:后面没有25条，情况三：前面后面都有25条

    pos = postemp[0][0]

    start, end = 0, 1000

    if pos < 25:

        end = 50

    elif pos > 975:

        start = 950

    else:

        start = pos-25

        end = pos+25

    return temp[start:end].tolist()





def readPickle(path):

    with open(path, "rb") as f:

        return pickle.load(f)





def savePickle(path, obj):

    with open(path, 'wb') as f:

        pickle.dump(obj, f)





def matrixMulti(start, end, startIndex):

    print("start to get part of")

    startIndex, tempMatrix = getPartSparse(

        query_word_matrix_bm, start, end, startIndex)

    temp_score = tempMatrix.tocsr().dot(word_title_matrix_bm)

    temp_score_array = temp_score.astype(

        np.float32).toarray().astype(np.float16)

    print(temp_score_array.shape)

    # 对结果排序,原来的顺序是从小到大，现在需要从大到小的顺序,逆序排序，取前1000个

    # 返回前1000个结果，内容为title的index，同时也是url的index

    print("排序")

    res1000 = np.argsort(-temp_score_array, axis=1).astype(np.int32)[:, :1000]

    print(res1000.shape)

    # 得到4万*1000的矩阵

    # 遍历queryIndex_urlIndex_dict_bm

    print("遍历")

    for index, row in tqdm(enumerate(res1000)):

        index = index+start

        click_url_list = queryIndex_urlIndex_dict_bm[index]

        resList = []

        for pos, url in enumerate(click_url_list):

            if url in row:

                # 找到用户点击的url在1000个候选列表中是第几位

                tempPos = np.argwhere(row == url)  # 此处产生的是二位数组[[value]]

                resList.append(getCandidate(tempPos, row))

            else:

                resList.append([])

        candiList.append(resList)

    return startIndex





# 读文件

print("reading")

query_word_matrix_bm = readPickle(

    "../input/bm25new/query_word_matrix_bm.pickle")

word_title_matrix_bm = readPickle(

    "../input/bm25new/word_title_matrix_bm.pickle")

queryIndex_urlIndex_dict_bm = readPickle(

    "../input/bm25new/queryIndex_urlIndex_dict_bm.pickle")



startValueList = [0, 182000, 364000, 546000, 728000, 910000, 1092000, 1274000, 1456000, 1638000, 1820000, 2002000, 2184000, 2366000, 2548000, 2730000, 2912000, 3094000, 3276000, 3458000]



titleList = [0, 666964, 1349333, 2039914, 2741068, 3442908, 4158214, 4869683, 5588427, 6305231, 7032007, 7753479, 8479275, 9205783, 9930559, 10666745, 11399260, 12139511, 12869350, 13611974]

fileNumber = 17

startTitle = titleList[fileNumber]

startIndex = startTitle

step = 700

print("start")

candiList = []  # 最终的候选列表，长度为4万，每个元素为当前query点击数*50为的二维矩阵

for i in tqdm(range(260)):

    process = psutil.Process(os.getpid())

    print('Used Memory:', process.memory_info().rss / 1024 / 1024, 'MB')

    startTime = time.time()

    start = i*step+startValueList[fileNumber]

    end = (i+1)*step+startValueList[fileNumber]

    startIndex = matrixMulti(start, end, startIndex)

    print("cost time:", time.time()-startTime)

notFind(candiList)

savePickle("/kaggle/working/candiList" +

           str(startTitle)+"-"+str(end)+".pickle", candiList)