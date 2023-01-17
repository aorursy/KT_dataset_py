# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

ipl_data= pd.read_csv('../input/matches.csv')

def top_n_umpires (ipl_data,top=3):
    '''This function returns the top umpires based on number of matches, default top is 3'''
    
    umpire_dict={}
    for i,entry in ipl_data.iterrows():
        if entry['umpire1']  in umpire_dict.keys():
            umpire_dict[entry['umpire1']] +=1
        else:
            umpire_dict[entry['umpire1']] =1
            
        if entry['umpire2']  in umpire_dict.keys():
            umpire_dict[entry['umpire2']] +=1
        else : 
            umpire_dict[entry['umpire2']] =1
                
    sort_result=sorted(umpire_dict.items(),key=lambda value:value[1], reverse=True)
    return dict(sort_result[0:top])


result= top_n_umpires (ipl_data,5)
print(result)


