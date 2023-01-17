import numpy as np



ids  = np.arange(90000).reshape((-1,1))

Yhat = np.ones_like(ids)

sample = np.hstack([ids, Yhat])

np.savetxt(fname='sample_submission.csv', X=sample, delimiter=',', header='Id,Predicted',comments='', fmt=['%d', '%0.8f'])