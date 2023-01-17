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
def find_second_tolast(list):

    if list.head is none or list.head.net is none:

        current_el = list.head

        next_el=list.head.next

        while next_el.next is not none:

            current_el = next_el

            next_el = next_el.net

        return current_el
def count_nodes(list):

    if list.current is none:

        return 0

    nodes = 1

    origin_node = list.current

    current_node = origin_node

    while current.next != origin_node:

        count += 1

        return count
def same_list(list1,list2):

    current = list1

    while current.next is x:

        return false

    current = current.next

    return true 
from positional_list import PositionalList

def list_to_positional_list(list_):

    pos_list = PositionalList()

    for element in list_:

        pos_list.add_last(element)

    return pos_list
def max( self ):

    return max(element for element in self )