# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import re

import os

import glob

a = os.listdir(path='../input/cityofla/CityofLA/Job Bulletins')

#1:FILE NAMEの取得

FileName = []

for i in range(len(a)):

    FileName.append(a[i])



#print(len(FileName)) #683

#print(FileName) #これがFILE NAMEのリスト

#2:JOB_CLASS_TITLEの取得

jobclasstitle=[]

for i in range(len(FileName)):

    contents=[]

    with open('../input/cityofla/CityofLA/Job Bulletins/'+FileName[i], 'r', encoding="latin-1") as f:

        for line in f:

            contents2=[]

            contents2.append(re.sub('^\n*|$\n*','',line))

            contents.extend(list(filter(lambda a: a != "", contents2)))#0行目に空白の行が来ないためのコード

        jobclasstitle.append(re.sub('\t| {2,}', '', contents[0]))

#print(jobclasstitle) #これがJOB_CLASS_TITLEのリスト

#print(len(jobclasstitle)) #683
#3:JOB_CLASS_NOの取得

classcode=[]

classcodeNA=[]

for i in range(len(FileName)):

    contents=[]

    with open('../input/cityofla/CityofLA/Job Bulletins/'+FileName[i], 'r', encoding="latin-1") as f:

        for line in f:

            contents2=[]

            contents2.append(re.sub('^\n*|$\n*','',line))

            contents.extend(list(filter(lambda a: a != "", contents2)))#ここまでは全関数共通コード

        code = [s for s in contents if 'Class Code:' in s]

        #print(code)

        #print(len(code))

        if(len(code)>=1):

            a=re.split(':',code[0])[1]

            a1=re.sub('\t| {1,}|[^0-9]+', '', a)#タブと空白と数字を含まない文字列は消去

            #print(a1)

            classcode.append(a1)

        else:

            classcode.append("NA")#ClassCodeが含まれないテキストの場合はNAを入れる



#classcode

#len(classcode)  #683
req_row_number=[]#各ファイルでREQUIREMENTという単語が初登場した行数のリスト

pro_row_number=[]#NOTEという単語がREQUIREMENTの初登場後に初登場した行数のリスト（＝REQUIREMENTが終わる行）







reqinfo=[]#REQUIREMENTの内容



req_setID=[]#setIDの数

req_subsetID=[]#subsetIDの数



for i in range(len(FileName)):

    contents=[]

    timesOfreq=0

    timesOfpro=0

    with open('../input/cityofla/CityofLA/Job Bulletins/'+FileName[i], 'r', encoding="latin-1") as f:

        for line in f:

            contents2=[]

            contents2.append(re.sub('^\n*|$\n*','',line))

            contents.extend(list(filter(lambda a: a != "", contents2)))#ここまでは全関数共通コード

            

    ###ここからREQUIREMENTの中身を抽出

            

        for j,name in enumerate(contents):          

            if 'REQUIREMENT' in name: 

                timesOfreq+=1

                if timesOfreq==1:#初登場のみ行数を控える

                    req_row_number.append(j)

                else:

                    continue

        for k,name in enumerate(contents):

            if k>req_row_number[i]:

                #print(k)

                if 'NOTE' in name: 

                    #print(k)

                    timesOfpro+=1

                    if timesOfpro==1:#初登場のみ行数を控える

                        pro_row_number.append(k)

                    else:

                        continue

        #if (timesOfreq>=2):

        #print(i,timesOfreq)##ちなみにREQUIREMENTが二回出てくるのは127

        #print(timesOfpro)

        reqinfo.append(contents[req_row_number[i]:pro_row_number[i]])#REQUIREMENTの抜粋コード

    

    ###ここまででREQUIREMENTの中身を抽出

    

    ###ここからsetIDをカウント

    

    setID_number=0 #setIDの数

    setAndsubsetID_row_number=[] #各setIDの行数(1-8(subsetIDの最大値))と各sebsetIDの行数(10-)

    

    for k,name in enumerate(reqinfo[i]):

        a=re.search('\d\.',name)

        if a:

            #print(a)

            setID_number+=1#setIDの回数カウント

            setAndsubsetID_row_number.append(k)#各setIDが登場した行数をsetID_row_numberに格納

    #print(setID_row_number)        

    req_setID.append(setID_number)#req_setIDに格納

    

    

    ###ここからsubsetIDをカウント

    

    subsetID_number=0 #subsetIDの数

    

    for k,name in enumerate(reqinfo[i]):

        a=re.search('^[a-z]\.',name)

        if a:

            subsetID_number+=1#subsetIDの回数カウント

            setAndsubsetID_row_number.append(k*100)#subsetID

    req_subsetID.append(subsetID_number)#req_subsetIDに格納

    

    #print(subsetID_row_number)

    #print(setAndsubsetID_row_number) #これがからの場合はsetIDが1のsubsetIDがAてこと？

    

    ###ここからsetIDとsubsetIDを分ける

    setID=[]

    subsetID=[]

    """

    if len(setAndsubsetID_row_number)>=1:

        setID2

        subsetID2=[]

        for g in range(len(setAndsubsetID_row_number)):

            if setAndsubsetID_row_number[g]<100:#setIDの方

            setID2.append(setAndsubsetID_row_number[g])

            #if setAndsubsetID_row_number[g]>100 and setAndsubsetID_row_number[g]%10==0:#subsetIDの方

            #subsetID2.append(setAndsubsetID_row_number[g]//100)

            

    else:

        setID.append(1)

        subsetID.append("A")

    """

        

    #if (len(setID)==len(subsetID):

    #continue

    #最後に同じになることを確認する

    





#len(pro_row_number) #683

#len(req_row_number)#683

#len(reqinfo)#683

#len(req_setID)#683

#req_subsetID

#max(req_setID) #最大の必要条件数は8(531番目のテキストファイル)

#max(req_subsetID)

req_row_number=[]#各ファイルでREQUIREMENTという単語が初登場した行数のリスト

pro_row_number=[]#NOTEという単語がREQUIREMENTの初登場後に初登場した行数のリスト（＝REQUIREMENTが終わる行）

reqinfo=[]#REQUIREMENTの内容



numlist={"one":1,"two":2,"three":3,"four":4}





for i in range(len(FileName)):

    contents=[]

    timesOfreq=0

    timesOfpro=0

    with open('../input/cityofla/CityofLA/Job Bulletins/'+FileName[i], 'r', encoding="latin-1") as f:

        for line in f:

            contents2=[]

            contents2.append(re.sub('^\n*|$\n*','',line))

            contents.extend(list(filter(lambda a: a != "", contents2)))#ここまでは全関数共通コード

            

    ###ここからREQUIREMENTの中身を抽出

            

        for j,name in enumerate(contents):          

            if 'REQUIREMENT' in name: 

                timesOfreq+=1

                if timesOfreq==1:#初登場のみ行数を控える

                    req_row_number.append(j)

                else:

                    continue

        for k,name in enumerate(contents):

            if k>req_row_number[i]:

                #print(k)

                if 'NOTE' in name: 

                    #print(k)

                    timesOfpro+=1

                    if timesOfpro==1:#初登場のみ行数を控える

                        pro_row_number.append(k)

                    else:

                        continue

        #if (timesOfreq>=2):

        #print(i,timesOfreq)##ちなみにREQUIREMENTが二回出てくるのは127

        #print(timesOfpro)

        reqinfo.append(contents[req_row_number[i]:pro_row_number[i]])#REQUIREMENTの抜粋コード

    #print(i)    

    #print(reqinfo[i])

    for j,name in enumerate(reqinfo[i]):

        if re.search('graduation',name):#graduationでひっぱってくる作戦→失敗？ってくらい少ない

            #education yearをrequirementの中で指定してくる募集は少ないのかな？

            #print(i)

            #print(name)

            if re.search('year|college|university',name):

                #print(name)

                #continue

                print(re.split('year',name)[0])

                #00in numlist.keys())

                #print(name)

            else:

                continue

        else:

            continue

    

    ###ここまででREQUIREMENTの中身を抽出

    ###ここからEDUCATION YEARSを抽出

    

    

#print(reqinfo[0])

#"orange" in numlist.keys()
numberlist={"one":1,"two":2,"three":3,"four":4}

numberlist["one"]
for j,name in enumerate(reqinfo[0]):          

    if 'graduation' in name:

        if re.search('year|college|university',name):

            print(re.split('year',name)[1])

        else:

            continue

    else:

        continue

        
#re.search('year|college|university',name)だったらuniversity or college

#それ以外は空欄でいいのでは

req_row_number=[]#各ファイルでREQUIREMENTという単語が初登場した行数のリスト

pro_row_number=[]#NOTEという単語がREQUIREMENTの初登場後に初登場した行数のリスト（＝REQUIREMENTが終わる行）

reqinfo=[]#REQUIREMENTの内容



major=[]#majorの中身(1ファイル一回)

majorfile=[]#majorの含まれるファイル番号

major_rownumber=[]#REQUIREMENTの中でmajorが初登場する番号





for i in range(len(FileName)):

    

    contents=[]

    timesOfreq=0

    timesOfpro=0

    with open('../input/cityofla/CityofLA/Job Bulletins/'+FileName[i], 'r', encoding="latin-1") as f:

        for line in f:

            contents2=[]

            contents2.append(re.sub('^\n*|$\n*','',line))

            contents.extend(list(filter(lambda a: a != "", contents2)))#ここまでは全関数共通コード

            

    ###ここからREQUIREMENTの中身を抽出

            

        for j,name in enumerate(contents):          

            if 'REQUIREMENT' in name: 

                timesOfreq+=1

                if timesOfreq==1:#初登場のみ行数を控える

                    req_row_number.append(j)

                else:

                    continue

        for k,name in enumerate(contents):

            if k>req_row_number[i]:

                #print(k)

                if 'NOTE' in name: 

                    #print(k)

                    timesOfpro+=1

                    if timesOfpro==1:#初登場のみ行数を控える

                        pro_row_number.append(k)

                    else:

                        continue

        #if (timesOfreq>=2):

        #print(i,timesOfreq)##ちなみにREQUIREMENTが二回出てくるのは127

        #print(timesOfpro)

        reqinfo.append(contents[req_row_number[i]:pro_row_number[i]])#REQUIREMENTの抜粋コード

    #print(i)    

    #print(reqinfo[i])

    majornumber=0

    for j,name in enumerate(reqinfo[i]):

        if re.search('major',name):

            majornumber+=1

            #print(name)

            #print(i)

            #print(re.split('major',name)[1])

            if majornumber==1:

                majorfile.append(i)

                major.append(re.split('major',name)[1])

                major_rownumber.append(j)

                    



#len(major)#40

#len(majorfile)#40

#majorfile

#major_rownumber

#len(major_rownumber)#40

#major#majorの内容

#majorfile#そのmajorが含まれるファイル番号

#major_rownumber#REQUIREMENTの何行目にmajorの詳細が書かれているか#setIDかsubsetIDかが重要
with open('../input/cityofla/CityofLA/Job Bulletins/'+FileName[5], 'r', encoding='latin-1') as f:

    s=f.read()

    print(s)

exam_type = []

# for i in range(len(FileName)):

#     with open('../input/cityofla/CityofLA/Job Bulletins/'+FileName[i], 'r', encoding='latin-1') as f:

#         s=f.read()

#         #print(s)

#         if 'INTERDEPARTMENTAL PROMOTIONAL AND OPEN COMPETITIVE BASIS' in s:

#             exam_type.append('OPEN_INT_PROM')

#         elif 'INTERDEPARTMENTAL PROMOTIONAL BASIS' in s:

#             exam_type.append('INT_PROM')

#         else:

#             exam_type.append(' ')

        

            

# print(exam_type) #ここに入ってる！ 
entrysalary_gen = []

for i in range(len(FileName)):

    with open('../input/cityofla/CityofLA/Job Bulletins/'+FileName[i], 'r', encoding='latin-1') as f:

        s=f.read()

        #print(s)

        if 'ANNUAL SALARY' in s:

            annualsalary = s.split('ANNUAL SALARY')[1]

            annualsalary2 = annualsalary.split('\n')[2]

            entrysalary_gen.append(annualsalary2)

        else:

            entrysalary_gen.append(' ')

            

#print(entrysalary_gen) #ここに入ってる！          

    
#24:entry_salary_dwpの取得

# import re

"""

for i in range(len(FileName)):

    contents=[]

    with open('../input/cityofla/CityofLA/Job Bulletins/'+FileName[i], 'r', encoding='latin-1') as f:

        s=f.read()

        salary = re.findall('\$[1-9,]*',s)

        salary_list.append(salary)

#print(salary_list) #値段と思われるものを全て取ってきた



for j in range(len(salary_list)):

    if salary_list[j]==[]:

        print(j) #値段がかいてないファイルのインデックス #45と534



for k in range(len(salary_list)):

    print(len(salary_list[k]))

"""

dwplist=[]

for i in range(len(FileName)):

    with open('../input/cityofla/CityofLA/Job Bulletins/'+FileName[i], 'r', encoding='latin-1') as f:

        s=f.read()

        #print(s)

        if 'the Department of Water and Power is' in s:

            dwp = s.split( 'the Department of Water and Power is')[1]

            dwp2 = dwp.split('\n')[0]

            dwplist.append(dwp2)

        else:

            dwplist.append(' ')

#print(dwplist) #ここに入ってる！
#filenameをランしてから

#25:open dateの取得

import re

opendate=[]

for i in range(len(FileName)):

    contents=[]

    

    with open('../input/cityofla/CityofLA/Job Bulletins/'+FileName[i], 'r', encoding='latin-1') as f:

        s = f.read()

        date = re.findall('\d{2}-\d{2}-\d{2}|[a-zA-Z]{3,10} \d{2}, \d{4}',s)[0]

#        date = re.findall('\d{2}-\d{2}-\d{2}',s)



        opendate.append(date)



#print(opendate) #これがopendateのリスト

#print(len(opendate)) #683




