import pandas as pd 

import numpy as np

import sys

sys.path.insert(0,"../input/")

from cleanml.predict import predict

sub = predict(test_data_path = "../input/categorical-encoding/test_cat.csv",  

              model_type = "randomforest", 

              model_path ="../input/catfeats-model" )
sub.loc[:, "id"] = sub.loc[:, "id"].astype(int)

sub.to_csv("rf_submission.csv", index=False)