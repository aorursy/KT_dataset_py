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

categories=pd.read_csv("../input/category_names.csv")
print(categories.info())

print(categories.head(10))
cat_lv1=categories["category_level1"].unique()

print(len(cat_lv1))
print("Total unique Categories Names (Level 3)",len(categories.category_level3.unique()))

print("Total unique Categories Ids",len(categories.category_id.unique()))

print("Number of total observation/rows",len(categories))
categories[categories.category_level3.duplicated()]

cat_bin=categories.iloc[:,[0]]

cat_bin["cl1"]=pd.Series()

cat_bin["cl2"]=pd.Series()

cat_bin["cl3"]=pd.Series()
for i in range(len(cat_lv1)):

    df=categories[categories["category_level1"]==cat_lv1[i]]

    cat_bin.cl1[categories.category_level1==cat_lv1[i]]=i

    cls2=df.category_level2.unique()

    no_cl2=len(cls2)

    for j in range(no_cl2):

        df2=df[df["category_level2"]==cls2[j]]

        cat_bin.cl2[categories.category_level2==cls2[j]]=j

        no_cl3=len(df2)

        cls3=df2.category_level3.unique()

        for k in range(no_cl3):

            cat_bin.cl3[categories.category_level3==cls3[k]]=k

print(cat_bin.head(10))
cat_bin["cl3"][3]=2

cat_bin["cl3"][1149]=23

cat_bin["cl3"][758]=96

cat_bin["cl3"][3704]=1

cat_bin["cl3"][4018]=21

cat_bin["cl3"][893]=11

cat_bin["cl3"][105]=9
cat_bin.to_csv('category_binary',index=False)