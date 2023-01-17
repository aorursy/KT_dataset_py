# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import json



with open(r'/kaggle/input/foodyo_output.json') as f:

  data = json.load(f)



def children_func(children, i):

    for child in children:

        if child["selected"] == 1:

            arrow = (i * "-") + "> "

            print(str(arrow) + str(child["name"]))

            children_func(child["children"], i+5)



for restaurant in data["body"]["Recommendations"]:

    print(restaurant["RestaurantName"])

    for menu in restaurant["menu"]:

        if menu["type"] == "sectionheader":

            for children in menu["children"]:

                if children["type"] == "item" and children["selected"] == 1:

                    print("--> " + str(children["name"]))

                    children_func(children["children"], 5)                            