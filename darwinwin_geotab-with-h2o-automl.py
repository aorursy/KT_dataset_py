# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



pred1 = np.load('../input/geotab-with-h2o-automl-for-totaltimestopped-p20/preds_TotalTimeStopped_p20.npy')

pred2 = np.load('../input/geotab-with-h2o-automl-for-totaltimestopped-p50/preds_TotalTimeStopped_p50.npy')

pred3 = np.load('../input/geotab-with-h2o-automl-for-totaltimestopped-p80/preds_TotalTimeStopped_p80.npy')

pred4 = np.load('../input/geotab-with-h2o-automl-for-distancetofirststop-p20/preds_DistanceToFirstStop_p20.npy')

pred5 = np.load('../input/geotab-with-h2o-automl-for-distancetofirststop-p50/preds_DistanceToFirstStop_p50.npy')

pred6 = np.load('../input/geotab-with-h2o-automl-for-distancetofirststop-p80/preds_DistanceToFirstStop_p80.npy')
# Appending all predictions

all_preds = []

for i in range(len(pred1)):

    for j in [pred1,pred2,pred3,pred4,pred5,pred6]:

        all_preds.append(j[i])   
len(all_preds)

sub  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

sub["Target"] = all_preds

sub.to_csv("benchmark_h2o_automl_1.csv",index = False)
