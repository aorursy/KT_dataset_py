# load libraries and check what we have in the "../input" directory
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
test = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test.csv', encoding="utf-8")
train = pd.read_csv('../input/talkingdata-adtracking-subsample-2/train_10mln.csv', encoding="utf-8")
from IPython.display import Markdown, display_html, display, HTML

def print_style(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))
    
## Displays the first 3 rows 
## as well as the type, the number of na values and the number of unique values in the columns.   
def display_df(df, name, nrows=3):
    df_shape = df.shape
    display(HTML('<h4 style=color:{}><h4>{}</h4></h4>'.format('purple',name)))
    display(HTML('<h4 style=color:{}>{}</h4>'.format('blue', df_shape)))
    display(df.head(nrows))
    display(pd.DataFrame([df.dtypes, np.sum(df.isnull()), df.nunique()],
                         columns=df.dtypes.index,
                         index=['type', 'na', 'nunique']))
display_df(test, name='test')
display_df(train, name='train')
import seaborn as sns

# Look at the classes of is_attributed variable in percentages
fraud, normal = train['is_attributed'].value_counts(normalize=True)
print('fraud (class 0): {}, normal (class 1): {}'.format(fraud, normal))
ax = sns.countplot("is_attributed", data = train)
corr_train = train.corr()
print_style('<h4 style="font-family:courier; font-size:200%; color:green ">train</h4>')
sns.heatmap(corr_train, 
            cmap="Greens",
            annot=True) 
corr_test = test.corr()
print_style('<h4 style="font-family:courier; font-size:150%; color:green ">test</h4>')
sns.heatmap(corr_test, 
            cmap="Greens", 
            annot=True)
import matplotlib.pyplot as plt

features = [x for x in ['ip', 'app', 'device', 'os', 'channel']]

# separate fraud from normal
# fraud here are the rows that didn't result in the purchase, thus the value 0
normal = train[train.is_attributed == 1][features]
fraud = train[train.is_attributed == 0][features]

print_style('<h1 style="font-family:courier; font-size:200%; color:green ">ip, app, device, os, channel distributions</h1>')

#### Uncomment the line below and comment out the second line below 
#### to see the histograms with the same x and y axes
#f, axes = plt.subplots(5, 2, figsize = (12,20), sharex='all', sharey = 'all')
f, axes = plt.subplots(5, 2, figsize = (15,25))

## there are 5 rows(i) and 2 columns (1 column for train, the other for test set).
for i in [0,1,2,3,4]:
        #train
        axes[i, 0].hist(fraud[features[i]], label = 'fraud', alpha = 0.5, color = 'red')
        axes[i, 0].hist(normal[features[i]], label = 'normal', alpha = 0.5)
        axes[i, 0].legend(loc='upper right', prop={'size': 15})
        axes[i, 0].set_title('{} (train)\nnunique = {}\nnunique(fraud) = {}\nnunique(normal) = {}'.format(features[i],
                                                                                                          train[features[i]].nunique(),
                                                                                                          fraud[features[i]].nunique(), 
                                                                                                          normal[features[i]].nunique()), 
                             size=17)
        # test
        axes[i, 1].hist(test[features[i]].values, alpha = 0.5)
        axes[i, 1].set_title('{} (test)\nnunique = {}'.format(features[i], 
                                                              test[features[i]].nunique()), 
                             size=17)

f.subplots_adjust(hspace=0.8)
train["click_time"] = train["click_time"].astype("datetime64")
print_style('<h1 style="font-family:courier; font-size:200%; color:green ">click_time distribution (train)</h1>')
train.groupby([train['click_time'].dt.hour]).count().plot(kind="bar", figsize=(20,10))
test["click_time"] = test["click_time"].astype("datetime64")
print_style('<h1 style="font-family:courier; font-size:200%; color:green ">click_time distribution (test)</h1>')
test.groupby([test['click_time'].dt.hour]).count().plot(kind="bar", figsize=(20,10))
