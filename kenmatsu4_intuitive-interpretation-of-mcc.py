import pandas as pd
from IPython.display import display
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import warnings
warnings.simplefilter(action='ignore')

pd.set_option("display.max_rows", 200)
pd.options.display.float_format = '{:,.2f}'.format
meta_train = pd.read_csv("../input/metadata_train.csv")

n_data = meta_train.shape[0]
n_pos = meta_train.target.sum()
print(f"n_data: {n_data}, n_pos: {n_pos}")

gt = [0]*(n_data-n_pos) + [1]*n_pos
pred = [0] * len(gt)  # initial prediction is all 0

result = []
TN, FP, FN, TP = confusion_matrix(gt, pred).flatten().tolist()
result.append([TN, FP, FN, TP , matthews_corrcoef(gt, pred)])
#result.append([0, matthews_corrcoef(gt, pred)])
for i in range(1, n_pos+1):
    pred[-i] = 1
    TN, FP, FN, TP = confusion_matrix(gt, pred).flatten().tolist()
    result.append([TN, FP, FN, TP , matthews_corrcoef(gt, pred)])
    
print("incleasing True Positive")
df_score = pd.DataFrame(result, columns=["TrueNegative", "FalsePositive", "FalseNegative", "TruePositive", "MCC"])
display(df_score.iloc[::5,:])


result = []
TN, FP, FN, TP = confusion_matrix(gt, pred).flatten().tolist()
result.append([TN, FP, FN, TP , matthews_corrcoef(gt, pred)])
#result.append([0, matthews_corrcoef(gt, pred)])

for i in range(1, n_pos*3+1):
    pred[i] = 1
    TN, FP, FN, TP = confusion_matrix(gt, pred).flatten().tolist()
    result.append([TN, FP, FN, TP , matthews_corrcoef(gt, pred)])
    
print()
print("incleasing False Positive")
df_score = pd.DataFrame(result, columns=["TrueNegative", "FalsePositive", "FalseNegative", "TruePositive", "MCC"])
display(df_score.iloc[::10,:])

