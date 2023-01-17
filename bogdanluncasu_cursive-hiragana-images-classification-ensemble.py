# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import pandas as pd

# Any results you write to the current directory are saved as output.
resnet_7_lw = pd.read_csv("../input/ensemble/resnet_c8.csv") # wrong file name

resnet_10 = pd.read_csv("../input/ensemble/resnet_c10.csv") 

resnet_15 = pd.read_csv("../input/ensemble/resnet_c15.csv")

resnet_20 = pd.read_csv("../input/ensemble/resnet_c20.csv") 



vgg_3 = pd.read_csv("../input/ensemble/vgg_c3.csv")

vgg_5 = pd.read_csv("../input/ensemble/vgg_c5.csv")

vgg_6 = pd.read_csv("../input/ensemble/vgg_c6.csv")



# ensemble only on above probabilities results in a 0.99435 private score



# these last two are actually another ensembles - I do not recall on which commits were they done

resnet_19 = pd.read_csv("../input/ensemble/resnet_c19.csv")

resnet_16_lw = pd.read_csv("../input/ensemble/lw_1.csv")



# these last two files improves the score to 0.99464
new_cols = ["p0","p1","p2","p3","p4","p5","p6","p7","p8","p9"]

params=[resnet_7_lw,resnet_10,resnet_15,resnet_20,vgg_3,vgg_5,vgg_6,resnet_16_lw,resnet_19]

def mean(nr=None,*args):

    if not nr:

        nr = len(args)

    s = args[0][new_cols]

    for i in range(1,len(args)):

        s+=args[i][new_cols]

    return s/nr



new_classes = mean(len(params),*params)

arr = new_classes.values



submission = pd.read_csv(os.path.join("../input/cursive-hiragana-classification","sample_submission.csv"))

submission['Class'] = np.argmax(arr, axis=1)

submission.to_csv(os.path.join(".","submission.csv"), index=False)  