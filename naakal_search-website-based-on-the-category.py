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
# Dataframe
df = pd.read_csv("/kaggle/input/sites-information-data-from-alexacom-dataset/alexa.com_site_info.csv", usecols=['site link', 'category','all_topics_top_keywords_name_parameter_1', 'all_topics_top_keywords_name_parameter_2', 'all_topics_top_keywords_name_parameter_3', 'all_topics_top_keywords_name_parameter_4'], delimiter=",")
mapping = {df.columns[0]:'website',df.columns[1]:'category',df.columns[2]:'topkeywords1',df.columns[3]:'topkeywords2',df.columns[4]:'topkeywords3',df.columns[5]:'topkeywords4'}
df = df.rename(columns=mapping)
df.dropna(inplace=True)
df.sample(5)
#query loop
choice = input("Please wish a category = ")

while choice != "exit":
    choice = choice.lower() 
    for index, row in df.iterrows():
        if choice in row['category'].lower():
            print(row['website'])
            choice = input("Please wish a category = ")
        elif choice == "exit":
             choice = "exit"
            
    if choice not in row['category'].lower():
        print("The category {} could not be found".format(choice))
        choice = input("Please wish a category = ")