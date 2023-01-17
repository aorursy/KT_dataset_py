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
#load the data

train = pd.HDFStore("../input/train.h5", "r").get("train")

train.shape
# Shape the train set into a panel

df_train = train.set_index(['id','timestamp'])

p_train = df_train.to_panel()

p_train = p_train.transpose(1,2,0)

p_train
y13 = 0     # count of the number of times y_hat is improved by adding technical_13 

yno13 = 0   # count of the number of times y_hat is NOT improved by adding technical_13 

for id in p_train.items:

    dff = p_train[id] # DF of the item or security instrument

    y_feature_13 = dff.technical_20 + dff.technical_13 - dff.technical_30

    y_feature = dff.technical_20 - dff.technical_30

    

    y_hat_13 = (y_feature_13.shift(-1) - (0.92 * y_feature)) / 0.07 

    y_hat = (y_feature.shift(-1) - (0.92 * y_feature)) / 0.07

    

    corr_y_13 = y_hat_13.corr(dff.y, method='spearman')

    corr_y = y_hat.corr(dff.y, method='spearman')

    

    if corr_y < corr_y_13:

        y13 += 1

    else:

        yno13 +=1

print(y13 / (y13 + yno13), y13, yno13)
import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (12.0, 6.0)

plt.style.use('ggplot')
for id in p_train.items[:10]:

    dff = p_train[id] 

    y_feature_13 = dff.technical_20 + dff.technical_13 - dff.technical_30

    y_feature = dff.technical_20 - dff.technical_30



    y_hat_13 = (y_feature_13.shift(-1) - (0.92 * y_feature)) / 0.07 

    y_hat = (y_feature.shift(-1) - (0.92 * y_feature)) / 0.07 



    corr_y_13 = y_hat_13.corr(dff.y, method='spearman')

    corr_y = y_hat.corr(dff.y, method='spearman')



    y_hat_13_s = y_hat_13.dropna()

    y_hat_s = y_hat.dropna()



    ax = y_hat_13_s.cumsum().plot( lw=1,c='b',label='y_hat_13 '+"{0:.4f}".format(corr_y_13),legend=True)

    ax = y_hat_s.cumsum().plot(ax=ax, lw=1,c='g',label='y_hat '+"{0:.4f}".format(corr_y),legend=True)

    ax = dff.y[y_hat_13_s.index[0]:].dropna().cumsum().plot(ax=ax, lw=1, c='r',label='y',legend=True,title='id='+str(id))

    plt.title('id='+str(id))

    plt.show()