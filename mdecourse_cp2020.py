import random

import requests

  

# getNumList 主要在每組最低人數下, 將不足 10 人的分組

# 平均分配至各組

def getNumList(total, eachGrp=10):

    # total is the number of students

    # each group at least 10 students

    #eachGrp = 10;

    # may divide into "grpNum" number of group

    grpNum = total // eachGrp;

    # check grpNum

    #print(grpNum)

    # vacan list

    splits = []

    # find remainder when total number divid into "grpNum" number of group

    remainder = total % grpNum

    # number of people in one group by calculation

    calGrp = total // grpNum

      

    for i in range(grpNum):

        splits.append(calGrp)

          

    # check first splits

    #print(splits)

    

    for i in range(remainder):

        splits[i] += 1

      

    # check final splits

    #print(splits);

    return splits;

  

# 儲存學生名單資料的 url

target_url = "http://mde.tw/cp2020/downloads/get_student_list/2020spring_cd_2a_list.txt"

# 利用 requests 讀取 url 中的資料

f = requests.get(target_url)

# get student list from target_url

# 利用 splitlines() 將資料放入數列 studList

studList = f.text.splitlines()

# 每組預計分組的最低人數

num_in_one_group = 10

# 每組學員暫存數列

gpList = []

# 全班分組數列

group = []

# 各組人數數列

numList = []

# get numList

numList = getNumList(len(studList), num_in_one_group)

# check numList

# 列出已經補齊的各組人數數列

print("預計各組人數數列:" + str(numList))

  

output = ""

gth = 1

inc = 0

  

# 弄亂 studList

random.shuffle(studList)

output += "以下為尚未排序前的各組學員學號: \n"

for i in numList:

    # 列印區隔符號

    output += '=' * 20 + "\n";

    output += "group " + str(gth) + " 有 " + str(i) + " 人: \n"

    # 每組學員暫存數列, 在此 reset

    gpList = []

    for j in range(i):

        output += studList[j+inc] + "\n"

        # 在各分組數列中加入將對應的學員學號

        gpList.append(studList[j+inc])

        

    gth = gth + 1

    inc = inc + j

    # 將各組學員數列依照學號排序

    gpList.sort()

    group.append(gpList)

  

# 列出尚未排序前的分組結果

print(output)

# 列出已經完成排序的分組數列

print("已經排序後的分組數列:" + str(group))

  

output = ""

# 列出已經排序後的分組名單

output += '=' * 25 + "\n"

output += '以下為排序後的各組成員名單:\n'

gth = 1

  

# 先列出純文字以 \n 跳行組員資料

for i in range(len(group)):

    # 列印區隔符號

    output += '=' * 20 + "\n"

    output += "group" + str(gth) + "\n"

    gpList = []

      

    for j in range(len(group[i])):

        output += str(group[i][j]) + "\n"

        

    gth = gth + 1

      

print(output)