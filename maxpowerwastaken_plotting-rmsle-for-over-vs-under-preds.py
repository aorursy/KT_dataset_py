import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#rmsle calc from https://www.kaggle.com/jpopham91/caterpillar-tube-pricing/rmlse-vectorized/code
def rmsle(pred, act):
    return np.sqrt(np.mean(np.power(np.log1p(pred)-np.log1p(act), 2)))


#Get Curves For RMSLE as a Function of Over- and Under-Predicted Amounts
def calcOverPredUnderPredCurves(errorRange):   

    #Initialize Lists to hold Results
    overPredCurve = []
    underPredCurve= []
    
    #Set 'Actual' Data as 100 so [e] of e.g. 5 is 5% error
    actual = 100
    
    for e in range(errorRange):
        
        #Calculate rmsle for overprediction underprediction by [e]
        overPrediction = actual + e
        underPrediction = actual - e
        
        overPredRMSLE = rmsle(overPrediction, actual)
        underPredRMSLE= rmsle(underPrediction, actual)

        #Append to running list of results
        overPredCurve.append(overPredRMSLE)
        underPredCurve.append(underPredRMSLE)
        
    return overPredCurve, underPredCurve
   
over, under = calcOverPredUnderPredCurves(100) 


#Plot Curves
sns.set_style("darkgrid")
plt.plot(under, color="darkblue", label="under-prediction RMSLE")
plt.plot(over, color="green", label="over-prediction RMSLE")
plt.ylabel("RMSLE")
plt.xlabel("% error in Prediction")
plt.legend(loc="best")
plt.show()