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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
mcr = pd.read_csv('../input/multipleChoiceResponses.csv',encoding='ISO-8859-1')
mcr.head()
part=mcr['Country'].value_counts()[:15].to_frame()
sns.barplot(part['Country'],part.index,palette='spring')
plt.title('Top 15 Countries by number of respondents')
plt.xlabel('Number of people participated')
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.show()
sns.countplot(y='FormalEducation', data=mcr)
sns.countplot(y='MajorSelect', data=mcr)
sns.countplot(y='LanguageRecommendationSelect',data=mcr)
tools=mcr['MLToolNextYearSelect'].value_counts().head(10)
sns.barplot(y=tools.index,x=tools)
