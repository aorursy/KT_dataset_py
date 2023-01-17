import pandas as pd

import numpy  as np

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import explained_variance_score
s1 = pd.read_csv("../input/very-simple-neural-network-regression/submission.csv")

s2 = pd.read_csv("../input/gaussian-process-regression-sample-script/submission.csv")

s3 = pd.read_csv("../input/random-forest-regression-minimalist-script/submission.csv")

s4 = pd.read_csv("../input/very-simple-xgboost-regression/submission.csv")

s5 = pd.read_csv("../input/catboost-regression-minimalist-script/submission.csv")





n_submission_files = 5

# also create a placeholder dataFrame

s_final = pd.read_csv("../input/very-simple-xgboost-regression/submission.csv")
solution   = pd.read_csv('../input/house-prices-advanced-regression-solution-file/submission.csv')

y_true     = solution["SalePrice"]
from scipy.optimize import minimize



tmp_scores = []

tmp_weights = []

predictions = []

predictions.append( s1["SalePrice"] )

predictions.append( s2["SalePrice"] )

predictions.append( s3["SalePrice"] )

predictions.append( s4["SalePrice"] )

predictions.append( s5["SalePrice"] )



def scoring_function(weights):

    final_prediction = 0

    for weight, prediction in zip(weights, predictions):

            final_prediction += weight*prediction

    return np.sqrt(mean_squared_log_error(y_true, final_prediction))



for i in range(150):

    starting_values = np.random.uniform(size=n_submission_files)

    bounds = [(0,1)]*len(predictions)

    result = minimize(scoring_function, 

                      starting_values, 

                      method='L-BFGS-B', 

                      bounds=bounds, 

                      options={'disp': False, 'maxiter': 10000})

    tmp_scores.append(result['fun'])

    tmp_weights.append(result['x'])



bestWeight = tmp_weights[np.argmin(tmp_scores)]

print('Best weights', bestWeight)
s_final["SalePrice"] = s1["SalePrice"]*bestWeight[0] + s2["SalePrice"]*bestWeight[1] +  s3["SalePrice"]*bestWeight[2] +  s4["SalePrice"]*bestWeight[3] +  s5["SalePrice"]*bestWeight[4]



print("The new score is %.5f" % np.sqrt( mean_squared_log_error(y_true, s_final["SalePrice"]) ) )

print("The new explained variance is %.5f" % explained_variance_score(y_true, s_final["SalePrice"]) )
s_final.to_csv('submission.csv', index=False)