import pandas as pd

import numpy  as np
LogisticRegression        = pd.read_csv("../input/logistic-regression-classifier-minimalist-script/submission.csv")

RandomForestClassifier    = pd.read_csv("../input/random-forest-classifier-minimalist-script/submission.csv")

neural_network            = pd.read_csv("../input/very-simple-neural-network-for-classification/submission.csv")

GaussianProcessClassifier = pd.read_csv("../input/gaussian-process-classification-sample-script/submission.csv")

SupportVectorClassifier   = pd.read_csv("../input/support-vector-classifier-minimalist-script/submission.csv")      
all_data = [ LogisticRegression['Survived'] , 

             RandomForestClassifier['Survived'], 

             neural_network['Survived'], 

             GaussianProcessClassifier['Survived'], 

             SupportVectorClassifier['Survived'] ]



votes       = pd.concat(all_data, axis='columns')



predictions = votes.mode(axis='columns').to_numpy()
output = pd.DataFrame({'PassengerId': neural_network.PassengerId, 

                       'Survived'   : predictions.flatten()})

output.to_csv('submission.csv', index=False)