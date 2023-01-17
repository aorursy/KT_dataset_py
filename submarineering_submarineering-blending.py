# Now this gives 0.9824 ROC score. ALL CREDITS GOES TO ORIGINAL KERNEL AUTHORS. I, as a newbie, just created a blend.
import pandas as pd
glove = pd.read_csv("../input/nb-svm-strong-linear-baseline/submission.csv")

subb = pd.read_csv('../input/fasttext-like-baseline-with-keras-lb-0-053/submission_bn_fasttext.csv')

ave = pd.read_csv('../input/toxic-avenger/submission.csv')

lstm = pd.read_csv('../input/toxicfiles/baselinelstm0069.csv')

svm = pd.read_csv("../input/toxicfiles/lstmglove0072ge.csv")
ble = svm.copy()

col = svm.columns
col = col.tolist()

col.remove('id')
for i in col:

    ble[i] = (2*subb[i] + 3*lstm[i] + 4*glove[i] + 5*svm[i] + ave[i]) / 15
ble.to_csv('submission20.csv', index = False)