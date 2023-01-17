# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from sklearn import (datasets, neighbors,
                     naive_bayes,
                     model_selection as skms,
                     linear_model, dummy,
                     metrics)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data_train_df = pd.read_csv("/kaggle/input/aiml-wine-quality-dataset/train.csv")
data_test_df = pd.read_csv("/kaggle/input/aiml-wine-quality-dataset/test.csv")
#display(data_train_df)
data_train_ft = data_train_df.drop('quality', axis=1)
data_train_tgt = data_train_df["quality"]
#display(data_train_ft)
dataset = datasets.load_wine()

(train_plus_validation_ftrs, 
 test_ftrs,
 train_plus_validation_tgt, 
 test_tgt) = skms.train_test_split(dataset.data,
                                   dataset.target,
                                   test_size=.25)

# separate training/validation sets
(train_ftrs,
 validation_ftrs,
 train_tgt,
 validation_tgt) = skms.train_test_split(train_plus_validation_ftrs,
                                         train_plus_validation_tgt,
                                         test_size=.33)



models_to_try = {'lr': linear_model.LinearRegression(), 
                 '1-NN': neighbors.KNeighborsRegressor(n_neighbors=1),
                 '3-NN': neighbors.KNeighborsRegressor(n_neighbors=3),
                 '5-NN': neighbors.KNeighborsRegressor(n_neighbors=5),
                 '7-NN': neighbors.KNeighborsRegressor(n_neighbors=7),
                 '9-NN': neighbors.KNeighborsRegressor(n_neighbors=9),
                 '11-NN': neighbors.KNeighborsRegressor(n_neighbors=11),
                 '13-NN': neighbors.KNeighborsRegressor(n_neighbors=13),
                 '15-NN': neighbors.KNeighborsRegressor(n_neighbors=15)}

errors = {}
for model_name in models_to_try:
    fit = models_to_try[model_name].fit(train_ftrs, train_tgt)
    predictions = fit.predict(validation_ftrs)
    rmse = np.sqrt(metrics.mean_squared_error(validation_tgt,
                                              predictions))
    print(f'{model_name} RMSE: {rmse:.2f}')
    errors[model_name] = rmse  
    
# get model with minimum rmse error
best_model_name = min(errors, key=errors.get)
print('Best model: ', best_model_name)
best_model = models_to_try[best_model_name]


fit = best_model.fit(train_ftrs, train_tgt)
predictions = fit.predict(test_ftrs)
rmse = np.sqrt(metrics.mean_squared_error(test_tgt,
                                          predictions))


#predictions should be the result returned by modelName.fit(test_features)
def writeSubmission(predictions):
    i=1
    submissionList = []
    for prediction in predictions:
        submissionList.append([str(i), str(prediction)])
        i+=1

    with open('submission.csv', 'w', newline='') as submission:
        writer = csv.writer(submission)

        writer.writerow(['Id', 'Predicted'])
        for row in submissionList:
            writer.writerow(row)
            
writeSubmission(predictions)
