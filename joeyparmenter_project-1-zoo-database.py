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


data_train_df = pd.read_csv("/kaggle/input/aiml-zoo-dataset/train.csv")
data_test_df = pd.read_csv("/kaggle/input/aiml-zoo-dataset/test.csv")

train_plus_validation_tgt = data_train_df["class"]
train_plus_validation_ftrs = data_train_df.drop('class', axis=1)
train_plus_validation_ftrs = train_plus_validation_ftrs.drop('name', axis=1)



models = {}
models['Naive Bayes'] = naive_bayes.GaussianNB()
for k in range(1,24,2):
    models[f'{k}-NN'] = neighbors.KNeighborsClassifier(n_neighbors=k)

accuracy_scores = {}
for model_name in models:
    scores = skms.cross_val_score(models[model_name],
                                      train_plus_validation_ftrs,
                                      train_plus_validation_tgt,
                                      cv=8,
                                      scoring='accuracy')
    mean_accuracy = scores.mean()
    accuracy_scores[model_name] = mean_accuracy
    print(f'{model_name}: {mean_accuracy:.3f}')

best_model_name = max(accuracy_scores,key=accuracy_scores.get)
best_model = models[best_model_name]

test_features = data_test_df.drop('name', axis=1)

fit = best_model.fit(train_plus_validation_ftrs,train_plus_validation_tgt)
predictions = fit.predict(test_features)

def writeSubmission(predictions):
    i=1
    submissionList = []
    for prediction in predictions:
        submissionList.append([str(i), str(prediction)])
        i+=1

    with open('submission.csv', 'w', newline='') as submission:
        writer = csv.writer(submission)

        writer.writerow(['Id', 'Category'])
        for row in submissionList:
            writer.writerow(row)
            
writeSubmission(predictions)


