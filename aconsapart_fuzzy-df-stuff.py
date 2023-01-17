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
df = pd.read_csv("../input/movie_metadata.csv")
df
import difflib



difflib.get_close_matches(df['color'][0].to_string(), df['color'].values.to_string().tolist())

df['color'].unique()
def fuzzy_group_list_elements(ThisList,Tolerance):

    from difflib import SequenceMatcher

    Groups = {}

    TempList = ThisList.copy()

    for Elmt in TempList:

        if Elmt not in Groups.keys():

            Groups[Elmt] = []

        for OtherElmt in TempList:

            if SequenceMatcher(None,Elmt,OtherElmt).quick_ratio() > Tolerance:

                Groups[Elmt] = Groups[Elmt] + [OtherElmt]

                TempList.remove(OtherElmt)

    Groups[Elmt] = list(set(Groups[Elmt]))

    return dict((v,k) for k in Groups for v in Groups[k])
ThisList = df["director_name"].unique().tolist()
Mapping = fuzzy_group_list_elements(ThisList,0.85)

df['director_matched'] = df['director_name'].replace(Mapping)
difflib.get_close_matches('James Cameron', df["director_name"].apply(str))
df['director_name'].unique().tolist()