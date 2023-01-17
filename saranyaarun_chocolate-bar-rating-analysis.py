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
#Read the File

chocolate = pd.read_csv("/kaggle/input/chocolate-bar-ratings/flavors_of_cacao.csv")
chocolate.head()
#Finding the null values 

chocolate.isnull().sum()
#Changing the column names

chocolate.columns=['Company','Bean_Orgin','REF','Review_Date','Cocoa_Percent','Company_Location','Rating','Bean_Type','Bean_Origin']
chocolate.info()
#summery of the data

chocolate.describe(include='all')
#Question 1: Where are the best cacao beans grown?

chocolate['Bean_Origin'].value_counts().head(10)
#Question:2 Which countries produces the high_rated bars?

chocolate1 = chocolate.groupby(['Bean_Origin'])['Rating'].max()

chocolate1.sort_values(ascending = False).head(20)
#Question:3 What is the realationship between cacao solids percentage and rating?

chocolate.groupby(['Cocoa_Percent'])['Rating'].max().sort_values(ascending = False).head(20)
#Question 4: Countries with highest Vendors

chocolate['Company_Location'].value_counts().head(20)