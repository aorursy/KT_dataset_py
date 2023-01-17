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
df = pd.read_csv("../input/survey_results_public.csv")

#limit to columns of interest

keep = ['StackOverflowFoundAnswer', 'StackOverflowCopiedCode', 'StackOverflowMetaChat',

        'StackOverflowAnswer', 'StackOverflowModeration', 'StackOverflowCommunity', 

        'StackOverflowHelpful', 'StackOverflowBetter']



dfN = pd.DataFrame(df[keep])

#convert frequency answers for first four columns to numeric values

dfN.replace({'At least once each day': 4, 'At least once each week': 3, 

             'Several times': 2, 'Once or twice': 1, 'Haven\'t done at all': 0}, 

                                        inplace = True)

dfN.pivot_table(values = 'StackOverflowCopiedCode', 

                index = 'StackOverflowCommunity', columns= 'StackOverflowModeration', 

                aggfunc = np.mean)
dfN.pivot_table(values = 'StackOverflowFoundAnswer', 

                index = 'StackOverflowCommunity', columns= 'StackOverflowModeration', 

                aggfunc = np.mean)
dfN.pivot_table(values = 'StackOverflowMetaChat', 

                index = 'StackOverflowCommunity', columns= 'StackOverflowModeration', 

                aggfunc = np.mean)
dfN.pivot_table(values = 'StackOverflowAnswer', 

                index = 'StackOverflowCommunity', columns= 'StackOverflowModeration', 

                aggfunc = np.mean)
dfN.pivot_table(values = 'StackOverflowFoundAnswer', 

                index = 'StackOverflowCommunity', columns= 'StackOverflowHelpful', 

                aggfunc = np.mean)
dfN.pivot_table(values = 'StackOverflowAnswer', 

                index = 'StackOverflowCommunity', columns= 'StackOverflowBetter', 

                aggfunc = np.mean)