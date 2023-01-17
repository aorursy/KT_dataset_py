import numpy as np
from sklearn.metrics import f1_score
%%time
N = 100000
y_pred = np.random.uniform(size=N)
y_true = ((y_pred + np.random.uniform(high=0.3, size=N)) > 0.5).astype('int32')      # add small noise

thresholds = []
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    res = f1_score(y_true, (y_pred > thresh).astype(int))
    thresholds.append([thresh, res])
    print("F1 score at threshold {0} is {1}".format(thresh, res))
    
thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh, "Best F1: ", thresholds[0][1])
%%time
def f1_smart(y_true, y_pred):
    args = np.argsort(y_pred)
    tp = y_true.sum()
    fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)
    res_idx = np.argmax(fs)
    return 2 * fs[res_idx], (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2

print('Optimal F1: {:.4f} at threshold: {:.4f}'.format(*f1_smart(y_true, y_pred)))
for i in range(10000):
    y_true = (np.random.uniform(size=100) > 0.5).astype('int32')
    y_pred = np.random.uniform(size=100)
    f1_first, threshold = f1_smart(y_true, y_pred)
    f1_second = f1_score(y_true, (y_pred > threshold).astype('int32'))
    assert(np.allclose(f1_first, f1_second))
print('All tests passed!')