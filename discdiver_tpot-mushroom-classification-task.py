# import the usual packages
import time
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import category_encoders

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

pd.options.display.max_columns = 200
pd.options.display.width = 200

%matplotlib inline
sns.set(font_scale=1.5, palette="colorblind")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/agaricus-lepiota.csv')

X = df.reindex(columns=[x for x in df.columns.values if x != 'class'])        # separate out X
X = X.apply(LabelEncoder().fit_transform)  # encode the x columns string values as integers

y = df.reindex(columns=['class'])   # separate out y
print(y['class'].value_counts())
y = np.ravel(y)                     # flatten the y array
y = LabelEncoder().fit_transform(y) # encode y column strings as integer

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=10) 
print(X_train.describe())
print(X_train.info())
tpot = TPOTClassifier(verbosity=3, 
                      scoring="accuracy", 
                      random_state=10, 
                      periodic_checkpoint_folder="tpot_mushroom_results", 
                      n_jobs=-1, 
                      generations=2, 
                      population_size=10)
times = []
scores = []
winning_pipes = []

# run several fits 
for x in range(10):
    start_time = timeit.default_timer()
    tpot.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    times.append(elapsed)
    winning_pipes.append(tpot.fitted_pipeline_)
    scores.append(tpot.score(X_test, y_test))
    tpot.export('tpot_mushroom.py')

# output results
times = [time/60 for time in times]
print('Times:', times)
print('Scores:', scores)   
print('Winning pipelines:', winning_pipes)
# timeo = [1.6234928817333032, 1.162914126116084, 0.6119730584498029, 0.9018127734161681, 
#          2.0324099983001362, 0.45596561313335165, 0.4123572280164808, 1.9914514322998003, 
#          0.31134609155027043, 2.268216603050435]  # previous times
timeo = np.array(times)
df = pd.DataFrame(np.reshape(timeo, (len(timeo))))
df= df.rename(columns={0: "Times"})
df = df.reset_index()
df = df.rename(columns = {"index": "Runs"})
print(df)
ax = sns.barplot(x= np.arange(1, 11), y = "Times", data = df)
ax.set(xlabel='Run # for Set of 30 Pipelines', ylabel='Time in Minutes')
plt.title("TPOT Run Times for Mushroom Dataset")
plt.show()