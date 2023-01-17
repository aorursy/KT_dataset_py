import sys

import pandas as pd 
sample_sub = pd.read_csv('../input/tensorflow2-question-answering/sample_submission.csv')
sample_sub.head()
test_df = pd.read_json('../input/tensorflow2-question-answering/simplified-nq-test.jsonl',

                      lines=True, orient='records')
test_df.head()
if len(test_df) >= 1000:

    raise ValueError("We'll never see this message again")

else:

    sample_sub.to_csv('submission.csv', index=False)