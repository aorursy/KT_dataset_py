# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sklearn.metrics
import os

# Any results you write to the current directory are saved as output.
import timeit 

pd.options.display.max_columns = 500
pd.options.display.width = 500
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25, random_state=50)
tpot = TPOTClassifier(verbosity=3, 
                      scoring="accuracy", 
                      random_state=50,  
                      n_jobs=-1, 
                      generations=20, 
                      periodic_checkpoint_folder="intermediate_algos",
                      population_size=60,
                      early_stop=10)
times = []
scores = []
winning_pipes = []

for x in range(1):
    start_time = timeit.default_timer()
    tpot.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    times.append(elapsed)
    winning_pipes.append(tpot.fitted_pipeline_)
    scores.append(tpot.score(X_test, y_test))
    tpot.export('tpot_mnist_pipeline1.py')
times = [time/60 for time in times]
print('Times:', times)
print('Scores:', scores)   
print('Winning pipelines:', winning_pipes)
plt.bar(range(len(scores)), scores)
plt.title('Accuracy on 25% Test Set')
plt.ylabel('Accuracy Score')
plt.xlabel('Winning Model')
plt.ylim((.95, 1))
plt.show()
plt.bar(range(len(times)), times)
plt.title('Time to Search and Train Models')
plt.ylabel('Time in Minutes')
plt.xlabel('Winning Model')
plt.show()