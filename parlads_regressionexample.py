# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#sklearn imports
from sklearn import (datasets, metrics, model_selection as skms, naive_bayes, neighbors, linear_model)

#warning off
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

pd.options.display.float_format = '{:20,.4f}'.format
diabetis = datasets.load_diabetes()
tts = skms.train_test_split(diabetis.data, diabetis.target, test_size = .30)
(diabetis_train_ftrs, diabetis_test_ftrs,
diabetis_train_tgt, diabetis_test_tgt) = tts
diabetis_df = pd.DataFrame(diabetis.data, columns=diabetis.feature_names)
diabetis_df['target'] = diabetis.target
diabetis_df.head()
sns.pairplot(diabetis_df[['age', 'sex', 'bmi', 'bp', 's1']], 
             height=1.5,
             hue='sex',
             plot_kws={'alpha':.2})
rolls = np.array([1,6,6,6,2,3,4,3,1])
np.mean(rolls) # middle part

np.median(rolls)
distance = np.array([4.0,2.0, 2.0])
closeness = 1.0/distance
closeness
weights = closeness/ closeness.sum()
weights
neigh_values = np.array([79.1, 88.3, 101.2])
np.dot(neigh_values, weights)
knn = neighbors.KNeighborsRegressor(n_neighbors=3)
fit = knn.fit(diabetis_train_ftrs, diabetis_train_tgt)
preds = knn.predict(diabetis_test_ftrs)


metrics.mean_squared_error(diabetis_test_tgt,preds)
#range

diabetis_df['target'].max() - diabetis_df['target'].min()
np.sqrt(3764.2305764411026)
def axis_helper(ax, lims):
    'clean up axes'
    ax.set_xlim(lims); ax.set_xticks([])
    ax.set_ylim(lims); ax.set_yticks([])
    ax.set_aspect('equal')
def process(D,model, ax):
    x ,y = D[:,0], D[:,1]
    m,b = model # y = mx + b
    
    axis_helper(ax, (0,8))
    
    #draws the data
    ax.plot(x,y,'ro')
    
    #drawing the line
    helper_xs = np.array([0,8])
    helper_line  = m * helper_xs + b
    ax.plot(helper_xs, helper_line, color = 'y')
    
    #calculate the errors 
    predictions = m*x + b
    errors = y - predictions
    
    #plot
    ax.values(x, predictions, y)
    
    #return the results
    sse = np.dot(errors, errors) # will give sum of sqared errors
    return (errors, errors.sum(), sse,np.sqrt(sse))
D = np.array([[3,5],
              [4,2]])

#create very simple horizantal lines
# y = mx + b if m =0 , it's horizantal
bs = np.array([1,2,3,3.5,4,5])
horizontal_lines  = np.c_[np.zeros_like(bs), bs]
horizontal_lines
import matplotlib
col_labels = ['Raw Errors', 'Sum', 'SSE', 'TotalDist']
fig = matplotlib.pyplot.figure()
fig, axes = fig.add_subplot(1,6)

records = [process(D, mod, ax) for mod, ax in zip(horizontal_lines, 
                                                 axes.flat)]
df = pd.DataFrame.from_records(records, columns=col_labels)
df
line_mb = np.array([[1,1],
                    [1,1],
                    [1,2],
                    [-1,8],
                    [-3, 14]])

col_labels = ['Raw Errors', 'Sum', 'SSE', 'TotalDist']
fig, axes = plt.subplots(1,5,figsize=(10,5))
records = [process(D, mod, ax) for mod, ax in zip(line_mb, 
                                                 axes.flat)]
df = pd.DataFrame.from_records(records, columns=col_labels)
lr = linear_model.LinearRegression()
fit = lr.fit(diabetis_train_ftrs, diabetis_train_tgt)
preds = fit.predict(diabetis_test_ftrs)

result = metrics.mean_squared_error(diabetis_test_tgt, preds)
result, np.sqrt(result)
