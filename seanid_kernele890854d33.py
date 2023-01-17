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

import json

FILENAMES = [
    'yelp_academic_dataset_business.json',
    'yelp_academic_dataset_user.json',
    'yelp_academic_dataset_tip.json',
    'yelp_academic_dataset_checkin.json',
    'yelp_academic_dataset_review.json'
]


def format_json(filename: str):
    """
    Converts weirdly-formatted JSON files to a list of dicts.
    """
    cwd = os.getcwd()
    with open(f'../input/{filename}') as f:
        contents = f.read()
    split = ', '.join(contents.split('\n')[:-1])
    return f'[{split}]'


def to_list(filename: str):
    return json.loads(format_json(filename))


def to_dataframe(filename: str):
    # return pd.DataFrame(format_json†)
    return pd.read_json(format_json(filename))


def first_entry(filename: str):
    with open(f'../input/{filename}') as f:
        line = f.readline()
    return json.loads(line)


def get_headers(filename: str):
    return list(first_entry(filename).keys())


def summarize_file(filename: str):
    print(f'Summarizing file: "{filename}"')
    # print(first_entry(filename))
    # print()
    print(get_headers(filename))
    print()


# for fn in FILENAMES:
#     summarize_file(fn)


for filename in FILENAMES:
    # print(filename)
    # to_dataframe(filename).head(n=10)
    summarize_file(filename)

