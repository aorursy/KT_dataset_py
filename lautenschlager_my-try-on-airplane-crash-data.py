# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
crashes=pd.read_csv("../input/3-Airplane_Crashes_Since_1908.txt")
crashes.dtypes
crashes.head()
crashes["Date"][1].split("/")
def parse_year(string):

    return int(string.split("/")[2])

crashes["Year"]=crashes["Date"].apply(lambda x: parse_year(x))
def parse_month(string):

    return int(string.split("/")[0])

crashes["Month"]=crashes["Date"].apply(lambda x: parse_month(x))
set(crashes["Operator"].tolist())
import re

def military_private(string):

    string=str(string)

    if re.search("[Mm]ilitary",string)!=None:

        return "military"

    if re.search("[Pp]rivate",string)!=None:

        return "private"

    return "airline"



crashes["category"]=crashes["Operator"].apply(lambda x: military_private(x))
print(crashes.describe())