import pandas as pd
res = pd.read_csv('/kaggle/input/fakeface-detection/submit.csv')

res['label'] = res['label'].apply(lambda x: 0.001 if x == 0 else 0.999)
res.info()
res.to_csv('/kaggle/working/submission.csv', index=False)