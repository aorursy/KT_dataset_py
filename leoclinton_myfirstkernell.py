import pandas as pd

sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")

specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")

test = pd.read_csv("../input/data-science-bowl-2019/test.csv")

train = pd.read_csv("../input/data-science-bowl-2019/train.csv")

train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")