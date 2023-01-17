import os

import numpy as np 

import pandas as pd 

from sklearn.metrics import log_loss 

%pylab inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Read different submissions.

out1 = pd.read_csv("../input/statoil-iceberg-submissions/sub_200_ens_densenet.csv", index_col=0)

out2 = pd.read_csv("../input/statoil-iceberg-submissions/sub_TF_keras.csv", index_col=0)

out3 = pd.read_csv("../input/submission38-lb01448/submission38.csv", index_col=0)

out4 = pd.read_csv("../input/submission38-lb01448/submission43.csv", index_col=0)

out5 = pd.read_csv('../input/submarineering-even-better-public-score-until-now/submission54.csv',index_col=0)
#getting lables from our best scored file. 

labels = (out5>0.5).astype(int)
# out5 score a log_loss of 0.1427 and could be considered also like an error Lerr= 0.1427

# Error produce by itself.

out5err = log_loss(labels, out5)

Lerr =  0.1427

print('out5 Error:', Lerr+out5err)
files = ['out1', 'out2', 'out3', 'out4', 'out5']

ranking = []

for file in files:

    ranking.append(log_loss(labels, eval(file)))

results = pd.DataFrame(files, columns=['Files'])

results['Error'] = ranking

results['Lerr'] = Lerr

results['Total_Error'] = results['Error']+ results['Lerr']

results
results['Total_Error'].plot(kind='bar')
# As before :

out2err = log_loss(labels, out2) + Lerr

out2err
# Apply some clipping : 

OUT2err = log_loss(labels, np.clip(out2, 0.0001, 0.99)) + Lerr

OUT2err