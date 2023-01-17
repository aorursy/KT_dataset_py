# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#ASSINGMENT1 DATED 18-8-2020
# ANI SIR SESSION
def lst_sqrt(num_list):  #input list
    import math
    sqrt_list = []       #output list
    
    for num in range(len(num_list)):
        root = math.sqrt(num)
        sqrt_list.append(root)
        
    return sqrt_list
        

    
#__MAIN__
l = [10, 20, 30, 40, 50]

result = lst_sqrt(l)

print("SQUARE ROOT LIST IS: ", result)

#ASSINGMENT2 DATED 18-8-2020
# ANI SESSION
def join_str(str_list):    #input list
    
    result_lst = []        #output list
    
    if len(str_list)%2 == 0: #EVEN NUMBER OF ELEMENTS
        
        for i in range(len(str_list)-1):
                result_str = str_list[i] + str_list[i+1]
                result_lst.append(result_str)
                
    else:  #ODD NUMBER OF LIST
        for i in range(len(str_list)-2):
                result_str = str_list[i] + str_list[i+1]
                result_lst.append(result_str)
                
        result_lst.append(str_list[len(str_list)-1])
                
    return result_lst

    
#__main__
print("ÏF NUMBER OF ELEMENTS IN INPUT LIST IS ODD, RESULT IS LIKE\n")
l1 = ['A','B','C','D','E','F','G']
print("ORIGINAL LIST",l1,"\n","RESULT LIST",join_str(l))
print("\n"*5)

print("ÏF NUMBER OF ELEMENTS IN INPUT LIST IS ODD, RESULT IS LIKE\n")
l2 = ['A','B','C','D','E','F']

print("ORIGINAL LIST: ",l2,"\n","RESULT LIST: ",join_str(l2))
            
        
        