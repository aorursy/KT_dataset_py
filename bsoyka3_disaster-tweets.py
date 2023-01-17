import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")



submission = pd.DataFrame(columns=["id", "target"])
for index, row in test.iterrows():

    if any([phrase in row.text.lower() for phrase in ["terrible", "wildfire", "emergency", "alert", "earthquake", "forest fire", "kills", "is ablaze", "accident on"]]):

        submission = submission.append(pd.DataFrame([[row.id, 1]], columns=["id", "target"]), ignore_index=True)

    else:

        submission = submission.append(pd.DataFrame([[row.id, 0]], columns=["id", "target"]), ignore_index=True)

submission.to_csv("submission.csv")