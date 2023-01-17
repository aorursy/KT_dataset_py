!pip install -U scikit-learn==0.22 > /dev/null
import sys

sys.path.insert(0, "../input/")

from mlframework.predict import predict
sub = predict(test_data_path="../input/cat-in-the-dat-ii/test.csv",

              model_type="randomforest",

              model_path="../input/catfeats-model/")
sub.loc[:, "id"] = sub.loc[:, "id"].astype(int)

sub.to_csv("submission.csv", index=False)