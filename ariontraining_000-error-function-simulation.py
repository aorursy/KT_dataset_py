true = [1.5, 2.1, 3.3, 4.7, 2.3, 0.75]  



prediction = [0.5, 1.5, 2.1, 2.2, 0.1, 0.5]
import numpy as np



rmsle = (np.sum((np.log1p(prediction)-np.log1p(true))**2)/len(true))**0.5



print ("Hand calculated RMSLE: {}".format(rmsle))
from sklearn.metrics import mean_squared_error



sklearn_rmsle = np.sqrt(mean_squared_error(np.log1p(true), np.log1p(prediction)))



print ("Sklearn calculated RMSLE: {}".format(sklearn_rmsle))
prediction2 = [1.4, 2.3, 3., 4.8, 2.1, 0.7]  



sklearn_rmsle2 = np.sqrt(mean_squared_error(np.log1p(true), np.log1p(prediction2)))



print ("Sklearn calculated RMSLE: {}".format(sklearn_rmsle2))