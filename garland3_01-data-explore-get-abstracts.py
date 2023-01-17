from pathlib import Path

from dataclasses import dataclass

import json

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
dir1 = "/kaggle/input/CORD-19-research-challenge"

dir1 = Path(dir1)
files = list(dir1.rglob("*.readme"))

def show_whole_file(p):

    with open(p) as f:

        for line in f: print(line)

            

show_whole_file(files[0])
## Look at the what else is in the data set
files = list(dir1.rglob("*.*"))

files[0:10]
show_whole_file(files[0])
dir2 = '/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/'

dir2 = Path(dir2)
files = list(dir2.rglob("*.json*"))
# open a single file so we can get confortable with the data and see how to process it

one_json = files[0]

with open(one_json, 'r') as myfile:

    data=myfile.read()

    

j_obj = json.loads(data)
def get_keys(obj):

    for k,v in obj.items():

        if isinstance(v, dict): get_keys(v)

        else: print(k)      



        

def get_values_of_key(obj, mykey = 'country'):

    data = []   

    if isinstance(obj, dict):       

        for k,v in obj.items():

            if isinstance(v, dict) or isinstance(v,list): data.extend(get_values_of_key(v, mykey))

            elif k == mykey: data.append(v)

    if isinstance(obj, list):  

        for v in obj:

            if isinstance(v, dict) or isinstance(v,list): data.extend(get_values_of_key(v, mykey))

    return data



def get_children_of_key(obj, mykey = 'abstract'):

    for k,v in obj.items():

        if isinstance(v, dict):

            r = get_children_of_key(v, mykey)

            if r is not None: return r

        else:

            if k==mykey: return v
# get_keys(j_obj)
# test on somthing simple, "country"

r = get_values_of_key(j_obj)

r
def flatten(r):

    s =""

    for v in r:

        if type(v)==str: s+=v

    return s
r = get_children_of_key(j_obj, 'abstract')

r = get_values_of_key(r,'text')

print(len(r))

r = flatten(r)

print(len(r))

r

def get_key_from_all_files(files, key= 'abstract', key2 = 'text'):

    data = []

    for f in files:

        with open(f, 'r') as myfile:                  

            data.append( flatten(get_values_of_key(get_children_of_key(json.loads(myfile.read()) , key),key2)))

    return data
abstracts = get_key_from_all_files(files, key = 'abstract')
abstracts
df = pd.DataFrame({'abstracts':abstracts})
df.to_csv('abstracts.csv')