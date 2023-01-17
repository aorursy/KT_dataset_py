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
train = pd.read_csv("../input/learn-together/train.csv.zip")

test = pd.read_csv("../input/learn-together/test.csv.zip")
print(train.shape)

display(train.head(1))



print(test.shape)

display(test.head(1))
train.head(5)
train.describe()
train.Wilderness_Area1.describe()
numeric_features = train.select_dtypes(include=[np.number])

numeric_features.dtypes
corr = numeric_features.corr()



print (corr['Wilderness_Area1'].sort_values(ascending=False)[1:11], '\n')

print (corr['Wilderness_Area1'].sort_values(ascending=False)[-10:])
train.Soil_Type29.unique()
def visualize_results(self):

        # Visualize logistic curve using seaborn

        sns.set(style="darkgrid")

        sns.regplot(x="pageviews_cumsum",

                    y="is_conversion",

                    data=self.df,

                    logistic=True,

                    n_boot=500,

                    y_jitter=.01,

                    scatter_kws={"s": 60})

        sns.set(font_scale=1.3)

        sns.plt.title('Logistic Regression Curve')

        sns.plt.ylabel('Conversion probability')

        sns.plt.xlabel('Cumulative sum of pageviews')

        sns.plt.subplots_adjust(right=0.93, top=0.90, left=0.10, bottom=0.10)

        sns.plt.show() 
import seaborn as sns
_ = sns.regplot(train['Horizontal_Distance_To_Fire_Points'], train['Wilderness_Area1'])
train=train.drop(train[(train['Horizontal_Distance_To_Fire_Points']>4000) & (train['Wilderness_Area1']<300000)].index)

_ = sns.regplot(train['Horizontal_Distance_To_Fire_Points'], train['Wilderness_Area1'])
_ = sns.regplot(train['Soil_Type4'], train['Wilderness_Area1'])
train = train[train['Soil_Type4'] < 1200]

_ = sns.regplot(train['Soil_Type4'], train['Wilderness_Area1'])
train['log_Wilderness_Area1']=np.log(train['Wilderness_Area1']+1)

Wilderness_Area1s=train[['Wilderness_Area1','log_Wilderness_Area1']]



Wilderness_Area1s.head(5)
_=sns.regplot(train['Soil_Type37'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type1'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type2'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type3'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type4'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type5'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type6'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type7'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type8'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type9'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type10'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type11'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type12'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type13'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type14'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type15'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type16'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type17'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type18'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type19'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type20'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type21'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type22'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type23'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type24'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type25'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type26'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type27'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type28'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type29'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type30'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type31'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type32'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type33'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type34'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type35'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type36'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type38'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type39'],Wilderness_Area1s['Wilderness_Area1'])
_=sns.regplot(train['Soil_Type40'],Wilderness_Area1s['Wilderness_Area1'])
print("Hey Seaborn")

print("Here's looking at You, kid.")