import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import os
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
gender=pd.read_csv("../input/gender_submission.csv")


sns.swarmplot(x='Survived',y='Pclass',data=train,hue='Sex')
m = sns.FacetGrid(train, col='Survived')
m.map(plt.hist, 'Age', bins=30,color='r')