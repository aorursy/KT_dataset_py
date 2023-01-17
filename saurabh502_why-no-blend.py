import numpy as np

import pandas as pd

import os



from scipy.stats import rankdata



LABELS = ["HasDetections"]



!ls ../input/blending-0-697
predict_list = []





predict_list.append(pd.read_csv("../input/ensmbl2/submission.csv")[LABELS].values)

#predict_list.append(pd.read_csv("../input/upd-lightgbm-baseline-model-using-sparse-matrix/lgb_submission.csv")[LABELS].values)

predict_list.append(pd.read_csv("../input/blending-0-697/ens_sub_v12.csv")[LABELS].values)

#predict_list.append(pd.read_csv("../input/malware-predictions-3500-trees/submission0.72968.csv")[LABELS].values)

#predict_list.append(pd.read_csv("../input/detecting-malwares-with-ftrl-proximal/submission.csv")[LABELS].values)

print("Rank averaging on ", len(predict_list), " files")

predictions = np.zeros_like(predict_list[0])

for predict in predict_list:

    for i in range(1):

        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])  

predictions /= len(predict_list)



submission = pd.read_csv('../input/microsoft-malware-prediction/sample_submission.csv')

submission[LABELS] = predictions

submission.to_csv('super_blend.csv', index=False)
submission.head()