%%time

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import glob

import os

stu_det = pd.read_csv('../input/student-data/studentlist.csv')

print(stu_det.shape)

stu_det.head(3)

# Reading batch wise list



path = r'../input/batchwiselist'

all_files = glob.glob(path +"/*.csv")

lst = []

for i in all_files:

    df = pd.read_csv(i)

    lst.append(df)

    

tot_lst = pd.concat(lst)

print(tot_lst.shape)
# Reading batch wise list

path = r'../input/quiz-result/'

all_files = glob.glob(path +"/*.csv")



quiz = []



for i in all_files:

    df = pd.read_csv(i)

    quiz.append(df)

#define each csv

quiz1 = quiz[0]

quiz2 = quiz[1]

#2 file merged

bth_quiz = pd.concat(quiz)

bth_quiz.head(3)

quiz1 = quiz1.drop(columns=['ID number','Institution',"Department",'Email address'])

quiz2 = quiz2.drop(columns=['ID number','Institution',"Department",'Email address'])
print(quiz1.shape)

quiz1.head(2)

stat_df= pd.DataFrame (columns = ["no of present", "lessthan50", "between50and60", 

                                  "between60and70", "between70and80", "greaterthan80"],index=["Quiz_1","Quiz_2"])



stat_df
quiz1["Firstname"] = quiz1["Firstname"].dropna()

quiz2["Firstname"] = quiz2["Firstname"].dropna()
#quiz1["Grade/10.00"].unique()

#quiz2["Grade/10.00"].unique()

quiz1["Grade/10.00"] = quiz1["Grade/10.00"].str.replace("-","0")

quiz2["Grade/10.00"] = quiz2["Grade/10.00"].str.replace("-","0")

quiz1["Grade/10.00"] = quiz1["Grade/10.00"].apply(pd.to_numeric)

quiz2["Grade/10.00"] = quiz2["Grade/10.00"].apply(pd.to_numeric)
#quiz11 = quiz1[quiz1["Grade/10.00"]<5].reset_index()

quiz1[quiz1["Grade/10.00"]<5]["Firstname"].count()

quiz2[quiz2["Grade/10.00"]<5]["Firstname"].count()
quiz1.count()
stat_df["no of present"]= [quiz1["Firstname"].count(),quiz2["Firstname"].count()]

stat_df["lessthan50"] = [quiz1[quiz1["Grade/10.00"].between(0,4.9,inclusive=True)]["Firstname"].count(),

                         quiz2[quiz2["Grade/10.00"].between(0,4.9)]["Firstname"].count()]

stat_df["between50and60"] = [quiz1[quiz1["Grade/10.00"].between(5,5.9,inclusive=True)]["Firstname"].count(),

                             quiz2[quiz2["Grade/10.00"].between(5,5.9,inclusive=True)]["Firstname"].count()]

stat_df["between60and70"] = [quiz1[quiz1["Grade/10.00"].between(6,6.9,inclusive=True)]["Firstname"].count(),

                             quiz2[quiz2["Grade/10.00"].between(6,6.9,inclusive=True)]["Firstname"].count()]

stat_df["between70and80"] = [quiz1[quiz1["Grade/10.00"].between(7,7.9,inclusive=True)]["Firstname"].count(),

                             quiz2[quiz2["Grade/10.00"].between(7,7.9,inclusive=True)]["Firstname"].count()]

stat_df["greaterthan80"] = [quiz1[quiz1["Grade/10.00"].between(8,10,inclusive=True)]["Firstname"].count(),

                            quiz2[quiz2["Grade/10.00"].between(8,10,inclusive=True)]["Firstname"].count()]





stat_df
path = r'../input/batchwiselist'

all_files = glob.glob(path +"/*.csv")

#all_files = [f for f in glob.glob(path +"\\batchwiselist" +"/*.csv")]

all_files[0]

#for i,v in enumerate(all_files):

#   print(i,v)

file = []



for i in all_files:

    df = pd.read_csv(i)

    file.append(df)

    

#file
#bt0 = file[0]
#student

quiz_1 = pd.merge(quiz1,stu_det,left_on='Firstname',right_on='studentname',how='right')

quiz_2 = pd.merge(quiz2,stu_det,left_on='Firstname',right_on='studentname',how='right')

print(quiz_1.shape)

quiz_1["Grade/10.00"] = quiz_1["Grade/10.00"].fillna(0)

quiz_2["Grade/10.00"] = quiz_2["Grade/10.00"].fillna(0)
#quiz_1

#stu1 = pd.merge(quiz2,stu_det,left_on='Firstname',right_on='studentname',how='left')

#stu1

#x = pd.merge(quiz_1,)



#quiz_1['studentname']==file[0]['studentName']

#quiz_1['studentname'].count()

#file[0]['studentName'].count()

#quiz_1['studentname'].str.contains(file[0]['studentName'])
#qz = pd.merge(file[0],quiz_1,left_on='studentName',right_on='studentname',how='left')

#qz.shape



#quiz-1

n = 18

b_q1 = []

for i in range(n+1):

    qz = pd.merge(file[i],quiz_1,left_on='studentName',right_on='studentname',how='left')

    b_q1.append(qz)



print(b_q1[0].shape)

b_q1[0].head(3)



#quiz-2

b_q2 = []

for i in range(n+1):

    qz = pd.merge(file[i],quiz_2,left_on='studentName',right_on='studentname',how='left')

    b_q2.append(qz)



print(b_q2[0].shape)

b_q1[0].head(3)
result1= pd.DataFrame (columns = ["no of present", "lessthan50", "between50and60", 

                                  "between60and70", "between70and80", "greaterthan80"],index=["Quiz_1","Quiz_2"])

result1
#batch[0].info()



result1["no of present"]= [b_q1[0]["studentName"].count(),b_q2[0]["studentName"].count()]

result1["lessthan50"] = [b_q1[0][b_q1[0]["Grade/10.00"].between(0,4.9,inclusive=True)]["studentName"].count(),

                         b_q2[0][b_q2[0]["Grade/10.00"].between(0,4.9)]["studentName"].count()]

result1["between50and60"] = [b_q1[0][b_q1[0]["Grade/10.00"].between(5,5.9,inclusive=True)]["studentName"].count(),

                             b_q2[0][b_q2[0]["Grade/10.00"].between(5,5.9,inclusive=True)]["studentName"].count()]

result1["between60and70"] = [b_q1[0][b_q1[0]["Grade/10.00"].between(6,6.9,inclusive=True)]["studentName"].count(),

                             b_q2[0][b_q2[0]["Grade/10.00"].between(6,6.9,inclusive=True)]["studentName"].count()]

result1["between70and80"] = [b_q1[0][b_q1[0]["Grade/10.00"].between(7,7.9,inclusive=True)]["studentName"].count(),

                             b_q2[0][b_q2[0]["Grade/10.00"].between(7,7.9,inclusive=True)]["studentName"].count()]

result1["greaterthan80"] = [b_q1[0][b_q1[0]["Grade/10.00"].between(8,10,inclusive=True)]["studentName"].count(),

                            b_q2[0][b_q2[0]["Grade/10.00"].between(8,10,inclusive=True)]["studentName"].count()]



result1

#b_q1[0][b_q1[0]["Grade/10.00"].between(0,5,inclusive=True)]["studentname"].count()

#b_q1[0]["Grade/10.00"].unique()
rst = []



for i in range(n+1):

    result= pd.DataFrame (columns = ["no of present", "lessthan50", "between50and60", 

                                  "between60and70", "between70and80", "greaterthan80"],index=["Quiz_1","Quiz_2"])

    result["no of present"]= [b_q1[i]["studentname"].count(),b_q2[i]["studentName"].count()]

    result["lessthan50"] = [b_q1[i][b_q1[i]["Grade/10.00"].between(0,4.9,inclusive=True)]["studentName"].count(),b_q2[i][b_q2[i]["Grade/10.00"].between(0,4.9)]["studentName"].count()]

    result["between50and60"] = [b_q1[i][b_q1[i]["Grade/10.00"].between(5,5.9,inclusive=True)]["studentName"].count(),b_q2[i][b_q2[i]["Grade/10.00"].between(5,5.9,inclusive=True)]["studentName"].count()]

    result["between60and70"] = [b_q1[i][b_q1[i]["Grade/10.00"].between(6,6.9,inclusive=True)]["studentName"].count(),b_q2[i][b_q2[i]["Grade/10.00"].between(6,6.9,inclusive=True)]["studentName"].count()]

    result["between70and80"] = [b_q1[i][b_q1[i]["Grade/10.00"].between(7,7.9,inclusive=True)]["studentName"].count(),b_q2[i][b_q2[i]["Grade/10.00"].between(7,7.9,inclusive=True)]["studentName"].count()]

    result["greaterthan80"] = [b_q1[i][b_q1[i]["Grade/10.00"].between(8,10,inclusive=True)]["studentName"].count(),b_q2[i][b_q2[i]["Grade/10.00"].between(8,10,inclusive=True)]["studentName"].count()]

    rst.append(result)

    

rst[0]

#b_q1[0]["studentName"].count()
txt = pd.read_csv("..//input/input-for-report/testcaseStudent.txt",header=None,names=["inputs"])

txt

#use range

#rst[0]["greaterthan80"]
tst_cs = int(txt.iloc[0,0])

ind = []

for i in range(tst_cs*2+1):

    ind.append(i)

    

ind
file_name = ind[1::2]

column = ind[2::2]

print(file_name)

print(column)
v = []

for i in file_name:

    x = txt.iloc[i,0]

    y = int(x[0])

    v.append(y)

#sub -1 to match index

a = [i - 1 for i in v]



c = []

for i in column:

    x = txt.iloc[i,0]

    c.append(x)



print(v)

print(c)

v
for (x,y) in zip(a,c):

    print(rst[x][y])

    print([x])

for (x,y,z) in zip(a,c,v):

    out = "Output{}.txt".format(z)

    f = open(out, "a")

    print(rst[x][y], file=f)