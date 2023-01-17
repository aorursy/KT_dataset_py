import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Let's define a function to calculate the metric



def eval_metric(FVC,FVC_Pred,sigma):

    n = len(sigma)

    a=np.empty(n)

    a.fill(70)

    sigma_clipped = np.maximum(sigma,a) 

    delta = np.minimum(np.abs(FVC,FVC_Pred),1000)

    eval_metric = -np.sqrt(2)*delta/sigma_clipped - np.log(np.sqrt(2)*sigma_clipped)

    return eval_metric
rand_forest_df = pd.read_csv("/kaggle/input/random-forest-submission/submission.csv")
rand_forest_df.head()
nn_df = pd.read_csv("/kaggle/input/neural-network-model-submission-score/submission.csv")
nn_df.head()
FVC = rand_forest_df['FVC'].to_numpy()

FVC_Pred = nn_df['FVC'].to_numpy()

Confidence = nn_df['Confidence'].to_numpy()
eval_metric(FVC,FVC_Pred,Confidence)
np.mean(eval_metric(FVC,FVC_Pred,Confidence))