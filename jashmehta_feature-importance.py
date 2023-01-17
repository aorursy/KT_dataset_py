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
%matplotlib inline 

import matplotlib as mpl

import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsRegressor

from sklearn.cross_validation import cross_val_score, ShuffleSplit, train_test_split

from sklearn.datasets import load_boston

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn import svm

from sklearn import linear_model

import pandas as pd

import numpy as np



%matplotlib inline

#import seaborn as sns

#sns.set_style("white")

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.filterwarnings("ignore")
import pandas as pd

datasource = pd.read_csv('../input/mushrooms.csv', delimiter=',')



#datasource = datasource.dropna(how='any')  #Removing all the rows with Null/NaN values. 

#This step is done because NaN values in features will give an error when we try to fit linear regression model.
datasource.shape[0]
datasource.shape[1] 
datasource.head()
#datasource['class'].head()
datasource['class'] = datasource['class'].replace(['p', 'e'],[0, 1]) 

datasource['cap-shape'] = datasource['cap-shape'].replace(['b', 'c', 'x','f', 'k', 's'],[0, 1,2,3,4,5]) 

datasource['cap-surface'] = datasource['cap-surface'].replace(['f', 'g', 'y','s'],[0, 1,2,3])

datasource['cap-color'] = datasource['cap-color'].replace(['n','b','c','g','r','p','u','e','w','y'],[0, 1,2,3,4,5,6,7,8,9])

datasource['bruises'] = datasource['bruises'].replace(['t','f'],[1, 0])

datasource['odor'] = datasource['odor'].replace(['a','l','c','y','f','m','n','p','s'],[0,1,2,3,4,5,6,7,8])

datasource['gill-attachment'] = datasource['gill-attachment'].replace(['a','d','f','n'],[0,1,2,3])

datasource['gill-spacing'] = datasource['gill-spacing'].replace(['c','w','d'],[0,1,2])

datasource['gill-size'] = datasource['gill-size'].replace(['b','n'],[1,0])

datasource['gill-color'] = datasource['gill-color'].replace(['k','n','b','h','g','r','o','p','u','e','w','y'],[0,1,2,3,4,5,6,7,8,9,10,11])





datasource['stalk-shape'] = datasource['stalk-shape'].replace(['e','t'],[1,0])



datasource['stalk-root'] = datasource['stalk-root'].replace(['b','c','u','e','z','r','?'],[0,1,2,3,4,5,6])



datasource['stalk-surface-above-ring'] = datasource['stalk-surface-above-ring'].replace(['f','y','k','s'],[0,1,2,3])



datasource['stalk-surface-below-ring'] = datasource['stalk-surface-below-ring'].replace(['f','y','k','s'],[0,1,2,3])



datasource['stalk-color-above-ring'] = datasource['stalk-color-above-ring'].replace(['n','b','c','g','o','p','e','w','y'],[0,1,2,3,4,5,6,7,8])

datasource['stalk-color-below-ring'] = datasource['stalk-color-below-ring'].replace(['n','b','c','g','o','p','e','w','y'],[0,1,2,3,4,5,6,7,8])



datasource['veil-type'] = datasource['veil-type'].replace(['p','u'],[0,1])

datasource['veil-color'] = datasource['veil-color'].replace(['n','o','w','y'],[0,1,2,3])



datasource['ring-number'] = datasource['ring-number'].replace(['n','o','t'],[0,1,2])

datasource['ring-type'] = datasource['ring-type'].replace(['c','e','f','l','n','p','s','z'],[0,1,2,3,4,5,6,7])



datasource['spore-print-color'] = datasource['spore-print-color'].replace(['k','n','b','h','r','o','u','w','y'],[0,1,2,3,4,5,6,7,8])

datasource['population'] = datasource['population'].replace(['a','c','n','s','v','y'],[0,1,2,3,4,5])

datasource['habitat'] = datasource['habitat'].replace(['g','l','m','p','u','w','d'],[0,1,2,3,4,5,6])


datasource.head()
#datasource[[0]]
Features = datasource[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]] 
Features.head()
type(Features['cap-color'])

#Features.to_csv('file://localhost/C:/Users/jashm/Desktop/718P/out.csv')
Label = datasource[[0]] 
#Label.head()
LabelMatrix = Label.as_matrix()

FeaturesMatrix = Features.as_matrix()


names1 = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape',

         'stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring',

         'veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']





# univariate feature selection using Random Forest Regressor

# this is a different approach, but confirms our best two features

rf = RandomForestRegressor(n_estimators=20, max_depth=4)

scores1 = []

for i in range(FeaturesMatrix.shape[1]):

     score = cross_val_score(rf, FeaturesMatrix[:, i:i+1], LabelMatrix, scoring="r2",  #'mean_squared_error' sklearn impl is negative.

                              cv=ShuffleSplit(n=len(FeaturesMatrix), n_iter=10, test_size=.1))

     scores1.append((round(np.mean(score), 10), names1[i]))

scores1_df = pd.DataFrame(scores1, columns = ['score', 'feature'])

scores1_df.sort_values(['score'], ascending=False)
type(scores1)
D = pd.DataFrame(scores1, columns=["Score", "Feature Names"])
D.plot(x = 'Feature Names', y='Score', kind='bar');
Mushroom_df = Features

Mushroom_df = pd.concat((Features, Label), axis=1)
Mushroom_df.head()