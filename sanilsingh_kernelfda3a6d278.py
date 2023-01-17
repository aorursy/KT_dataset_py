import pandas as pd



submit = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
submit['prediction'] = 0.5

submit.to_csv('submission.csv', index=False)