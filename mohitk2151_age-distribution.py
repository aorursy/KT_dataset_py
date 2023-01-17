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
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
%matplotlib inline
from plotly import tools
init_notebook_mode(connected=True)
#Read the application train csv
application = pd.read_csv("../input/application_train.csv")
#Q1 Age ranges of the people who has repaid/not repaid the loans?
plt.figure(figsize = (10, 10))

#Loans repaid on time
sns.kdeplot(application.loc[application['TARGET'] == 0, 'DAYS_BIRTH'] / -365, label = 'target == 0')

#Loans which were not repaid on time
sns.kdeplot(application.loc[application['TARGET'] == 1, 'DAYS_BIRTH'] / -365, label = 'target == 1')

#Plot

plt.xlabel('Age'); plt.ylabel('Density'); plt.title('Distribution of Ages');
#Q2 Age count of people who has taken loan?
#PlotStyle
plt.style.use('fivethirtyeight')

#AgeDistribution
plt.hist(application['DAYS_BIRTH'] / -365, edgecolor = 'k', bins = 20)
plt.title('Age'); plt.xlabel('Age'); plt.ylabel('Count');