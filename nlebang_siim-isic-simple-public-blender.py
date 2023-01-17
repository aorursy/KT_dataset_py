# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



""" 

disable the below code,since there are a lot of files in the folde

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

"""



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pathlib import Path



sub_path = Path("../input/melanoma-public")

sub_866_path = sub_path/'submission_866.csv'

sub_877_path = sub_path/'submission_877.csv'

sub_879_path = sub_path/'submission_879.csv'

sub_884_path = sub_path/'submission_884.csv'

sub_892_path = sub_path/'submission_892.csv'

sub_897_path = sub_path/'submission_897.csv'

sub_910_path = sub_path/'submission_910.csv'

sub_914_path = sub_path/'submission_914.csv'

sub_927_path = sub_path/'submission_927.csv'



sub_866 = pd.read_csv(sub_866_path)

sub_877 = pd.read_csv(sub_877_path)

sub_879 = pd.read_csv(sub_879_path)

sub_884 = pd.read_csv(sub_884_path)

sub_892 = pd.read_csv(sub_892_path)

sub_897 = pd.read_csv(sub_897_path)

sub_910 = pd.read_csv(sub_910_path)

sub_914 = pd.read_csv(sub_914_path)

sub_927 = pd.read_csv(sub_927_path)



sub_866 = sub_866.sort_values(by="image_name")

sub_877 = sub_877.sort_values(by="image_name")

sub_879 = sub_879.sort_values(by="image_name")

sub_884 = sub_884.sort_values(by="image_name")

sub_892 = sub_892.sort_values(by="image_name")

sub_897 = sub_897.sort_values(by="image_name")

sub_910 = sub_910.sort_values(by="image_name")

sub_914 = sub_914.sort_values(by="image_name")

sub_927 = sub_927.sort_values(by="image_name")



out1 = sub_866["target"].astype(float).values

out2 = sub_877["target"].astype(float).values

out3 = sub_879["target"].astype(float).values

out4 = sub_884["target"].astype(float).values

out5 = sub_892["target"].astype(float).values

out6 = sub_897["target"].astype(float).values

out7 = sub_910["target"].astype(float).values

out8 = sub_914["target"].astype(float).values

out9 = sub_927["target"].astype(float).values
merge_output = []

n=9



# Dummy weights, find your strategy!

w1 = 0.02

w2 = 0.03

w3 = 0.03

w4 = 0.04

w5 = 0.06

w6 = 0.14

w7 = 0.19

w8 = 0.22

w9 = 0.27



print('Sum weights:',w1+w2+w3+w4+w5+w6+w7+w8+w9)





for o1, o2, o3, o4, o5, o6, o7, o8, o9 in zip(out1, out2, out3, out4, out5, out6, out7, out8, out9):

    #print(o1,type(o1))

    o = float((o1*w1 + o2*w2 + o3*w3 + o4*w4 + o5*w5 + o6*w6 + o7*w7 + o8*w8 + o9*w9)/n)

    merge_output.append(o)

    

sub_866["target"] = merge_output

sub_866["target"] = sub_866["target"].astype(float)

#sub_866 = sub_866.drop(['index'], axis=1)

sub_866.to_csv("submission_simple_bleding1.csv", index=False)



sub_866.head(3)
# Dummy weights, find your strategy!

merge_output2 = []

w1 = 0.03

w2 = 0.03

w3 = 0.03

w4 = 0.05

w5 = 0.05

w6 = 0.15

w7 = 0.20

w8 = 0.21

w9 = 0.25



print('Sum weights:',w1+w2+w3+w4+w5+w6+w7+w8+w9)





for o1, o2, o3, o4, o5, o6, o7, o8, o9 in zip(out1, out2, out3, out4, out5, out6, out7, out8, out9):

    #print(o1,type(o1))

    o = float((o1*w1 + o2*w2 + o3*w3 + o4*w4 + o5*w5 + o6*w6 + o7*w7 + o8*w8 + o9*w9)/n)

    merge_output2.append(o)

    

sub_877["target"] = merge_output

sub_877["target"] = sub_877["target"].astype(float)

sub_877.to_csv("submission_simple_bleding2.csv", index=False)



sub_877.head(3)
# Dummy weights, find your strategy!

merge_output3 = []

w1 = 0.1

w2 = 0.1

w3 = 0.1

w4 = 0.1

w5 = 0.1

w6 = 0.1

w7 = 0.1

w8 = 0.15

w9 = 0.15



print('Sum weights:',w1+w2+w3+w4+w5+w6+w7+w8+w9)





for o1, o2, o3, o4, o5, o6, o7, o8, o9 in zip(out1, out2, out3, out4, out5, out6, out7, out8, out9):

    #print(o1,type(o1))

    o = float((o1*w1 + o2*w2 + o3*w3 + o4*w4 + o5*w5 + o6*w6 + o7*w7 + o8*w8 + o9*w9)/n)

    merge_output3.append(o)

    

sub_866["target"] = merge_output

sub_866["target"] = sub_877["target"].astype(float)

sub_866.to_csv("submission_simple_bleding3.csv", index=False)



sub_866.head(3)