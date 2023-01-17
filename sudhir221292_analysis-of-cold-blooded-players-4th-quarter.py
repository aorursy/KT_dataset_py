# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/shot_logs.csv")



clutch_data=pd.concat([data.SHOT_CLOCK,data.SHOT_RESULT,data.PERIOD,data.GAME_CLOCK,data.player_name,data.SHOT_DIST],axis=1)

clutch_data["IS_SHOT_MADE"]=(clutch_data.SHOT_RESULT=='made')





    

temp=clutch_data[(clutch_data.SHOT_CLOCK<2) & (clutch_data.PERIOD==4) &(clutch_data.SHOT_DIST>10)].groupby("player_name")

true_list=[]

false_list=[]

names=[]

for i in temp:

    x=pd.DataFrame(i[1])

    

    temp=x.groupby("IS_SHOT_MADE").SHOT_RESULT.count()

    

    if(len(temp.keys())==1):

        if(temp.keys()[0]=="False"):

            false_list.append(temp.get_values()[0])

            true_list.append(0)

        else:

            true_list.append(temp.get_values()[0])

            false_list.append(0)

    else:

        true_list.append(temp[1])

        false_list.append(temp[0])

        

    names.append(i[0])

   

    

final=pd.DataFrame({"player_name":names,"True":true_list,"False":false_list})

final['FG%'] = np.round(final['True'] / (final['True'] + final['False']) * 100, 2)

print (final)
