!pip install https://github.com/dimension23/ludwig/archive/master.zip
import numpy as np

import pandas as pd

import logging

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, roc_curve, precision_recall_curve

from ludwig.api import LudwigModel





import os

print(os.listdir("../input"))
model_definition = {

    "input_features": [

        {

            "bidirectional": True,

            "cell_type": "lstm_cudnn",

            "dropout": True,

            "embedding_size": 300,

            "embeddings_trainable": True,

            "encoder": "rnn",

            "level": "word",

            "name": "question_text",

            "pretrained_embeddings": "../input/embeddings/glove.840B.300d/glove.840B.300d.txt",

            "type": "text"

        }

    ],

    "output_features": [

        {

            "name": "target",

            "type": "category"

        }

    ],

    "preprocessing" : {

        "stratify": "target",

        "text": {

            "lowercase": True

        }

    }

}
model = LudwigModel(model_definition)
input_dataframe = pd.read_csv("../input/train.csv")



training_dataframe, validation_dataframe = train_test_split(input_dataframe,

                                                      test_size=0.1, 

                                                      random_state=42, 

                                                      stratify=input_dataframe["target"])



training_dataframe.reset_index(inplace=True)

validation_dataframe.reset_index(inplace=True)
training_stats = model.train(training_dataframe, logging_level=logging.INFO)
training_stats
predictions_dataframe = model.predict(validation_dataframe, logging_level=logging.INFO)
results_dataframe = validation_dataframe.merge(predictions_dataframe, left_index=True, right_index=True)

results_dataframe["target_predictions"] = pd.to_numeric(results_dataframe["target_predictions"])
f1_score(results_dataframe["target"], results_dataframe["target_predictions"])
test_dataframe = pd.read_csv("../input/test.csv")

test_predictions = model.predict(test_dataframe, logging_level=logging.INFO)
model.close()
submission_dataframe = test_dataframe.merge(test_predictions, left_index=True, right_index=True)[["qid", "target_predictions"]]

submission_dataframe.columns = ["qid", "prediction"]

submission_dataframe.to_csv("submission.csv", index=False)