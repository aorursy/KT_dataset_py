import pandas as pd
import numpy as np
import ast # for dict parsing down the track
test = pd.read_csv('../input/test_v2.csv',dtype={'fullVisitorId': 'str'})
submission_sample = pd.read_csv('../input/sample_submission_v2.csv',dtype={'fullVisitorId': 'str'})
submission_sample.shape
submission_sample.head()
test.shape
test_prep = test.copy()
test_prep['PredictedLogRevenue'] = 0
submission_no_return = test_prep.groupby('fullVisitorId',as_index=False).PredictedLogRevenue.sum()
submission_no_return.head()
submission_no_return.shape
submission_no_return.to_csv('submission_no_return.csv',index=False)
test_prep_2 = test.copy()
test_prep_2['Revenue'] = test_prep_2['totals'].apply(lambda x: ast.literal_eval(x).get('transactionRevenue',np.nan))
test_prep_2['Revenue'] =pd.to_numeric(test_prep_2['Revenue'],errors='coerce')
submission_right_answer = test_prep_2.groupby('fullVisitorId',as_index=False).Revenue.sum()
submission_right_answer['PredictedLogRevenue'] = submission_right_answer.Revenue.apply(lambda x: np.log(x+1))
submission_right_answer.drop(columns='Revenue',inplace=True)
submission_right_answer.head()
submission_right_answer.shape
submission_right_answer.to_csv('submission_right_answer.csv',index=False)