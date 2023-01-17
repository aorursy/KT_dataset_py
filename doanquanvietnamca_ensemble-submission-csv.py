import pandas as pd

import numpy as np

df_submit = pd.read_csv("../input/mberttrain10epochs/submission (1).csv")

# df_2 = pd.read_csv('../input/mberttrain10epochs/tf_roberta-large_submission-prob.csv')

# df_2.head()
df_submit.head()
# final = 0.9*df_2[['prob0','prob1', 'prob2']].values + 0.1*df_submit[['prob0','prob1', 'prob2']].values
# df_submit.prediction = np.argmax(final, axis=1)
# df_submit = df_submit[['id', 'prediction']]

df_submit.to_csv("submission.csv", index=False)

df_submit.head()
df_submit.hist()