# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
submit_list = os.listdir("../input/comp/submission/")
print(submit_list)
predict_list = ['cat']* 50
print(predict_list)
a=[1,2,3]
b=['a','b','c']
for first, second in zip(a,b):
    print(first,second)
f = open('temp.csv','w')     #temp.csv 파일을 write모드로 open
f.write("filename,class\n")  #csv파일의 첫번째 row에는 column명이 들어갑니다.  f.write는 글자수를 return하지만 별로 중요한 내용은 아닙니다. 
for fn, class_name in zip(submit_list,predict_list):
    f.write(fn+","+class_name+'\n')     #python에서 +는 문자열을 합쳐주는데도 이용됩니다. "filename,class_name\n"의 형태로 파일에 출력됩니다. 
f.close()    
import pandas as pd
pd.read_csv('temp.csv')
