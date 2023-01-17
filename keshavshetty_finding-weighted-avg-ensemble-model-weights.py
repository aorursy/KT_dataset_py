# Lets start

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Install the Library (Refer: https://pypi.org/project/kesh-utils/ )

!pip install kesh-utils
!pip install statsmodels==0.10.0rc2 --pre  # Statsmodel has sme problem with factorial in latest lib
# Ignore the warnings if any

import warnings  

warnings.filterwarnings('ignore')
# Load the dataset 

consolidated_proba_df = pd.read_csv('../input/consolidated_proba.csv')
# Quick check the consolidate dataset

consolidated_proba_df.head(10)
# Load the util

from KUtils.common import blend_util
# Let run and find best weighted avg coefficients for columns 'gnb_proba', 'dt_le_proba', 'rf_proba', 'xgb_proba'



best_blend_df = blend_util.find_best_stacking_blend(

    consolidated_proba_df, 

    actual_target_column_name='Actual', 

    columns_to_blend=['gnb_proba', 'dt_le_proba', 'rf_proba', 'xgb_proba'],

    starting_weight=1,

    max_weight=10,

    step_weight=1,

    minimize_loss='rmse', # other option mae

    verbose=False

)
# The best blend df gets appended as and when new best options are found. So all best blend options are at the end of the dataframe (Sort it if you want)

best_blend_df.tail()
# Create new probablity based new model blend coefficients. Total weight 18

consolidated_proba_df['final_blended_proba'] = (consolidated_proba_df['gnb_proba']*1 + 

                                                consolidated_proba_df['dt_le_proba']*1 + consolidated_proba_df['rf_proba']*10 + consolidated_proba_df['xgb_proba']*6)/18
consolidated_proba_df.head(10)
import matplotlib.pyplot as plt

from KUtils.logistic_regression import auto_logistic_regression as autoglm



from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, recall_score, precision_score
# First(0) column contains prob for 0-class and second(1) contains prob for 1-class

gnb_pred_df = pd.DataFrame({'Actual':consolidated_proba_df['Actual'], 'Probability':consolidated_proba_df['gnb_proba']}) 

return_dictionary = autoglm.calculateGLMKpis(gnb_pred_df, cutoff_by='Sensitivity-Specificity', include_cutoff_df_in_return=True)

cutoff_df = return_dictionary['cutoff_df']
cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])

plt.show()
cutoff_df.plot.line(x='Probability', y=['Precision','Recall'])

plt.show()
prob_column='gnb_proba'

prob_cutoff = 0.04

consolidated_proba_df['predicted'] = consolidated_proba_df[prob_column].map(lambda x: 1 if x > prob_cutoff else 0)



local_confusion_matrix = metrics.confusion_matrix(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'] )

        

accuracy = metrics.accuracy_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

precision = metrics.precision_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

recall = metrics.recall_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

f1_score = metrics.f1_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

roc_auc = metrics.roc_auc_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])



print(" Accuracy {0:.3f}, \n Precision {1:.3f}, \n Recall {2:.3f}, \n f1_score {3:.3f}, \n roc_auc {4:.3f}".format(

    accuracy, precision,recall,f1_score,roc_auc))
# First(0) column contains prob for 0-class and second(1) contains prob for 1-class

rf_pred_df = pd.DataFrame({'Actual':consolidated_proba_df['Actual'], 'Probability':consolidated_proba_df['rf_proba']}) 

return_dictionary = autoglm.calculateGLMKpis(rf_pred_df, cutoff_by='Sensitivity-Specificity', include_cutoff_df_in_return=True)

cutoff_df = return_dictionary['cutoff_df']
cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])

plt.show()
cutoff_df.plot.line(x='Probability', y=['Precision','Recall'])

plt.show()
prob_column='rf_proba'

prob_cutoff = 0.3

consolidated_proba_df['predicted'] = consolidated_proba_df[prob_column].map(lambda x: 1 if x > prob_cutoff else 0)



local_confusion_matrix = metrics.confusion_matrix(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'] )

        

accuracy = metrics.accuracy_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

precision = metrics.precision_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

recall = metrics.recall_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

f1_score = metrics.f1_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

roc_auc = metrics.roc_auc_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])



print(" Accuracy {0:.3f}, \n Precision {1:.3f}, \n Recall {2:.3f}, \n f1_score {3:.3f}, \n roc_auc {4:.3f}".format(

    accuracy, precision,recall,f1_score,roc_auc))
# First(0) column contains prob for 0-class and second(1) contains prob for 1-class

xgb_pred_df = pd.DataFrame({'Actual':consolidated_proba_df['Actual'], 'Probability':consolidated_proba_df['xgb_proba']}) 

return_dictionary = autoglm.calculateGLMKpis(xgb_pred_df, cutoff_by='Sensitivity-Specificity', include_cutoff_df_in_return=True)

cutoff_df = return_dictionary['cutoff_df']
cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])

plt.show()
cutoff_df.plot.line(x='Probability', y=['Precision','Recall'])

plt.show()
prob_column='xgb_proba'

prob_cutoff = 0.2

consolidated_proba_df['predicted'] = consolidated_proba_df[prob_column].map(lambda x: 1 if x > prob_cutoff else 0)



local_confusion_matrix = metrics.confusion_matrix(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'] )

        

accuracy = metrics.accuracy_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

precision = metrics.precision_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

recall = metrics.recall_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

f1_score = metrics.f1_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

roc_auc = metrics.roc_auc_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])



print(" Accuracy {0:.3f}, \n Precision {1:.3f}, \n Recall {2:.3f}, \n f1_score {3:.3f}, \n roc_auc {4:.3f}".format(

    accuracy, precision,recall,f1_score,roc_auc))
# First(0) column contains prob for 0-class and second(1) contains prob for 1-class

final_blend_pred_df = pd.DataFrame({'Actual':consolidated_proba_df['Actual'], 'Probability':consolidated_proba_df['final_blended_proba']}) 

return_dictionary = autoglm.calculateGLMKpis(final_blend_pred_df, cutoff_by='Sensitivity-Specificity', include_cutoff_df_in_return=True)

cutoff_df = return_dictionary['cutoff_df']
cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])

plt.show()
cutoff_df.plot.line(x='Probability', y=['Precision','Recall'])

plt.show()
prob_column='final_blended_proba'

prob_cutoff = 0.28

consolidated_proba_df['predicted'] = consolidated_proba_df[prob_column].map(lambda x: 1 if x > prob_cutoff else 0)



local_confusion_matrix = metrics.confusion_matrix(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'] )

        

accuracy = metrics.accuracy_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

precision = metrics.precision_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

recall = metrics.recall_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

f1_score = metrics.f1_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])

roc_auc = metrics.roc_auc_score(consolidated_proba_df['Actual'], consolidated_proba_df['predicted'])



print(" Accuracy {0:.3f}, \n Precision {1:.3f}, \n Recall {2:.3f}, \n f1_score {3:.3f}, \n roc_auc {4:.3f}".format(

    accuracy, precision,recall,f1_score,roc_auc))