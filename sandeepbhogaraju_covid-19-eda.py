# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# #This creates a df with all research papers, then load this df as input to the input folder

# import os



# #Load all JSONs to a single df

# df = pd.DataFrame()

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for i,filename in enumerate(filenames):

# #         print(os.path.join(dirname, filename))

#         if('.json' in str(os.path.join(dirname, filename))):

#             df_iter = pd.read_json(str(os.path.join(dirname, filename)),orient='index').transpose()

#             df = df.append(df_iter,ignore_index=True)

#             if(i%5000==0):

#                 print(df.shape)

# df.to_csv('covid_research.csv')



# # Any results you write to the current directory are saved as output.