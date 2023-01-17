import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn import preprocessing, model_selection, metrics
from imblearn import over_sampling, under_sampling
from collections import Counter
X = pd.read_csv('../input/creditcard.csv')
y = X.pop('Class')
rus = under_sampling.RandomUnderSampler({0:20000,1:len(y[y==1])})
X_res, y_res = rus.fit_sample(X, y)

sm = over_sampling.SMOTE(ratio={0: 20000, 1:1000}, random_state=42)
X_res, y_res = sm.fit_sample(pd.DataFrame(X_res), pd.Series(y_res))
print('Resampled dataset shape {}'.format(Counter(y_res)))
#train X_res y_res, test X, y (WHERE NOT IN X_res, y_res)
gbm = xgb.XGBClassifier().fit(X_res, y_res)
pred_prob = gbm.predict_proba(X.as_matrix())
predictions = gbm.predict(X.as_matrix())
roc_auc = metrics.roc_auc_score(y, predictions)
[x[1] for x in pred_prob]
print(roc_auc)
print(metrics.recall_score(y, predictions))
print(metrics.average_precision_score(y, predictions))
print(metrics.confusion_matrix(y, predictions))
fpr, tpr, _ = metrics.precision_recall_curve(y, [x[1] for x in pred_prob])
#fpr, tpr, _ = metrics.roc_curve(y, [x[1] for x in pred_prob])

# 40000, 2000
# 0.9346814887701135
# 0.8699186991869918
# 0.6355919546223909

# 40000, 1000
# 0.9397311345526096
# 0.8800813008130082
# 0.6259464078330257

# 20000, 1000
# 0.9346902818341445
# 0.8699186991869918
# 0.6410598310513447

#sampling variation has strong affect on performance
plt.figure(figsize=(10,8))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw) #, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Precision recall curve')
#plt.legend(loc="lower right")
plt.show()
