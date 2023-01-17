import numpy as np 
import pandas as pd
np.random.seed(42)

submission = pd.read_csv('../input/sampleSubmission.csv')

submission.expected = np.random.laplace(0.0, 0.01, 7200)
submission.to_csv('submission.csv', index=False)