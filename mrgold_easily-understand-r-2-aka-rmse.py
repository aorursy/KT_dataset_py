%matplotlib inline
import numpy as np
import math
import matplotlib.pyplot as plt
actual = np.array([1,4,5])
predicted = np.array([3,1,10])
diff = actual - predicted; diff
diff_2 = diff * diff
math.sqrt(np.sum(diff_2)/len(actual))
# Calculate same RMSE in function
def rmse(a,b): return math.sqrt(((a-b)**2).mean())

rmse(actual, predicted)
avg_prediction = np.mean(actual); avg_prediction
rmse(actual, avg_prediction)
actual_as_prediction = actual
rmse(actual, actual_as_prediction)
plt.plot([1,2,3],[rmse(x, actual) for x in [predicted, avg_prediction, actual_as_prediction]], '--bo')
plt.title('RMSE 3 predictions')
plt.ylabel('RMSE')
plt.axis([0, 5, 0, 5])
plt.show()
