import pandas as pd
submission = pd.read_csv('../input/7-lines-of-code-to-reach-6th-place/submission.csv')
q1 = submission['SalePrice'].quantile(0.0025)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.9)

submission.to_csv('submission.csv', index=False)