import collections

import math

import os

import pathlib

import platform

import sys

import warnings



import numpy as np

import numpy as np

import pandas as pd

import pandas as pd

import sklearn.metrics

import sympy



from joblib import Parallel, delayed

from tqdm import tqdm





# Filter warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')



# Get working directory

try:

    path_working_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

except:

    path_working_dir = os.path.abspath(str(pathlib.Path().resolve()))



# Set input directory, and make output directory

list_nodename = ['.local', 'CM-1080Ti']

if max([platform.node().find(nodename) for nodename in list_nodename]) > -1:

    path_input  = os.path.abspath(os.path.join(path_working_dir, '../../../input/rsna-intracranial-hemorrhage-detection'))

    path_output = path_working_dir.replace('/src/', '/output/')

    os.makedirs(path_output, exist_ok=True)

else: # maybe kaggle kernel

    path_input  = '../input/rsna-intracranial-hemorrhage-detection'

    path_output = './'

assert os.path.exists(path_input) == True



# Load sample_submission.csv, set labels are 0.5 in case overwrited

df_submission          = pd.read_csv(os.path.join(path_input, 'stage_1_sample_submission.csv'))

df_submission          = pd.read_csv(os.path.join(path_input, 'stage_1_sample_submission.csv'))

df_submission['Label'] = 0.5
# Reduced the number of digits and made it the same as LB for easy calculation

# Define logloss, using LB digits(XX.XXX)

logloss      = {}

logloss[0.5] = -math.log(0.5)           * 1000 * 1000 // 1000 / 1000 #  0.693

logloss[0]   = -math.log(1 - 10**(-15)) * 1000 * 1000 // 1000 / 1000 #  0.000

logloss[1]   = -math.log(    10**(-15)) * 1000 * 1000 // 1000 / 1000 # 34.538
# Define Variables, weights are 0-1(relative value), positives are 0-78545(N of stage 1 test images)

weights_epidural           = sympy.Symbol('weights_epidural')

weights_intraparenchymal   = sympy.Symbol('weights_intraparenchymal')

weights_intraventricular   = sympy.Symbol('weights_intraventricular')

weights_subarachnoid       = sympy.Symbol('weights_subarachnoid')

weights_subdural           = sympy.Symbol('weights_subdural')

weights_any                = sympy.Symbol('weights_any')

positives_epidural         = sympy.Symbol('positives_epidural')

positives_intraparenchymal = sympy.Symbol('positives_intraparenchymal')

positives_intraventricular = sympy.Symbol('positives_intraventricular')

positives_subarachnoid     = sympy.Symbol('positives_subarachnoid')

positives_subdural         = sympy.Symbol('positives_subdural')

positives_any              = sympy.Symbol('positives_any')
# Create submission_file, df_any_{0, 1} means *_any's label is {0, 1} else labels are 0.5

df_epidural_0               = df_submission.copy()

df_epidural_0.iloc[0::6, 1] = 0

df_epidural_0.to_csv(os.path.join(path_output, 'submission_epidural_0.csv'), index=False)



df_epidural_1               = df_submission.copy()

df_epidural_1.iloc[0::6, 1] = 1

df_epidural_1.to_csv(os.path.join(path_output, 'submission_epidural_1.csv'), index=False)



df_intraparenchymal_0               = df_submission.copy()

df_intraparenchymal_0.iloc[1::6, 1] = 0

df_intraparenchymal_0.to_csv(os.path.join(path_output, 'submission_intraparenchymal_0.csv'), index=False)



df_intraparenchymal_1               = df_submission.copy()

df_intraparenchymal_1.iloc[1::6, 1] = 1

df_intraparenchymal_1.to_csv(os.path.join(path_output, 'submission_intraparenchymal_1.csv'), index=False)



df_intraventricular_0               = df_submission.copy()

df_intraventricular_0.iloc[2::6, 1] = 0

df_intraventricular_0.to_csv(os.path.join(path_output, 'submission_intraventricular_0.csv'), index=False)



df_intraventricular_1               = df_submission.copy()

df_intraventricular_1.iloc[2::6, 1] = 1

df_intraventricular_1.to_csv(os.path.join(path_output, 'submission_intraventricular_1.csv'), index=False)



df_subarachnoid_0               = df_submission.copy()

df_subarachnoid_0.iloc[3::6, 1] = 0

df_subarachnoid_0.to_csv(os.path.join(path_output, 'submission_subarachnoid_0.csv'), index=False)



df_subarachnoid_1               = df_submission.copy()

df_subarachnoid_1.iloc[3::6, 1] = 1

df_subarachnoid_1.to_csv(os.path.join(path_output, 'submission_subarachnoid_1.csv'), index=False)



df_subdural_0               = df_submission.copy()

df_subdural_0.iloc[4::6, 1] = 0

df_subdural_0.to_csv(os.path.join(path_output, 'submission_subdural_0.csv'), index=False)



df_subdural_1               = df_submission.copy()

df_subdural_1.iloc[4::6, 1] = 1

df_subdural_1.to_csv(os.path.join(path_output, 'submission_subdural_1.csv'), index=False)



df_any_0               = df_submission.copy()

df_any_0.iloc[5::6, 1] = 0

df_any_0.to_csv(os.path.join(path_output, 'submission_any_0.csv'), index=False)



df_any_1               = df_submission.copy()

df_any_1.iloc[5::6, 1] = 1

df_any_1.to_csv(os.path.join(path_output, 'submission_any_1.csv'), index=False)
# Input LB socre: submission_{epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any}_{0, 1}.csv, 'lb_score_* = - XX.XXX + ...' is LB SCORE

lb_score_epidural_0 = - 0.618 + ((1 - weights_epidural) * logloss[0.5]) + (weights_epidural * (sympy.Symbol('positives_epidural') * logloss[1] + (78545 - sympy.Symbol('positives_epidural')) * logloss[0]) / 78545)

lb_score_epidural_1 = - 5.504 + ((1 - weights_epidural) * logloss[0.5]) + (weights_epidural * (sympy.Symbol('positives_epidural') * logloss[0] + (78545 - sympy.Symbol('positives_epidural')) * logloss[1]) / 78545)



lb_score_intraparenchymal_0 = - 0.817 + ((1 - weights_intraparenchymal) * logloss[0.5]) + (weights_intraparenchymal * (sympy.Symbol('positives_intraparenchymal') * logloss[1] + (78545 - sympy.Symbol('positives_intraparenchymal')) * logloss[0]) / 78545)

lb_score_intraparenchymal_1 = - 5.305 + ((1 - weights_intraparenchymal) * logloss[0.5]) + (weights_intraparenchymal * (sympy.Symbol('positives_intraparenchymal') * logloss[0] + (78545 - sympy.Symbol('positives_intraparenchymal')) * logloss[1]) / 78545)



lb_score_intraventricular_0 = - 0.747 + ((1 - weights_intraventricular) * logloss[0.5]) + (weights_intraventricular * (sympy.Symbol('positives_intraventricular') * logloss[1] + (78545 - sympy.Symbol('positives_intraventricular')) * logloss[0]) / 78545)

lb_score_intraventricular_1 = - 5.375 + ((1 - weights_intraventricular) * logloss[0.5]) + (weights_intraventricular * (sympy.Symbol('positives_intraventricular') * logloss[0] + (78545 - sympy.Symbol('positives_intraventricular')) * logloss[1]) / 78545)



lb_score_subarachnoid_0 = - 0.817 + ((1 - weights_subarachnoid) * logloss[0.5]) + (weights_subarachnoid * (sympy.Symbol('positives_subarachnoid') * logloss[1] + (78545 - sympy.Symbol('positives_subarachnoid')) * logloss[0]) / 78545)

lb_score_subarachnoid_1 = - 5.305 + ((1 - weights_subarachnoid) * logloss[0.5]) + (weights_subarachnoid * (sympy.Symbol('positives_subarachnoid') * logloss[0] + (78545 - sympy.Symbol('positives_subarachnoid')) * logloss[1]) / 78545)



lb_score_subdural_0 = - 0.887 + ((1 - weights_subdural) * logloss[0.5]) + (weights_subdural * (sympy.Symbol('positives_subdural') * logloss[1] + (78545 - sympy.Symbol('positives_subdural')) * logloss[0]) / 78545)

lb_score_subdural_1 = - 5.234 + ((1 - weights_subdural) * logloss[0.5]) + (weights_subdural * (sympy.Symbol('positives_subdural') * logloss[0] + (78545 - sympy.Symbol('positives_subdural')) * logloss[1]) / 78545)



lb_score_any_0 = - 1.855 + ((1 - weights_any) * logloss[0.5]) + (weights_any * (sympy.Symbol('positives_any') * logloss[1] + (78545 - sympy.Symbol('positives_any')) * logloss[0]) / 78545)

lb_score_any_1 = - 9.002 + ((1 - weights_any) * logloss[0.5]) + (weights_any * (sympy.Symbol('positives_any') * logloss[0] + (78545 - sympy.Symbol('positives_any')) * logloss[1]) / 78545)
# Solve {weights, positives}_epidural

solution = []

solution.append(sympy.solve([lb_score_epidural_0,         lb_score_epidural_1        ])[0].values())

solution.append(sympy.solve([lb_score_intraparenchymal_0, lb_score_intraparenchymal_1])[0].values())

solution.append(sympy.solve([lb_score_intraventricular_0, lb_score_intraventricular_1])[0].values())

solution.append(sympy.solve([lb_score_subarachnoid_0,     lb_score_subarachnoid_1    ])[0].values())

solution.append(sympy.solve([lb_score_subdural_0,         lb_score_subdural_1        ])[0].values())

solution.append(sympy.solve([lb_score_any_0,              lb_score_any_1             ])[0].values())
# Output results, 0.1428 ≒ 1 / 7, 7 = 5 + 1 * 2, so I think *_any weight is x2 of others

df_output = pd.DataFrame(solution, columns=['N_positives', 'weight'], index=['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'])

df_output
# Fianly my scoring function



# LB ceiling digits

def ceil(x, digits=0):

    return float(str(x)[:str(x).find('.') + 1 + digits])



# scikit-learn scoring

# sklearn.metrics.log_loss(y_true, y_pred, sample_weights=[1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, ...])

# CAUTION: the function below is for LB probe

def get_score_v5(n_positives, pred_all, weight):

    if weight == 1:

        return ceil(sklearn.metrics.log_loss(y_true=([0] * (78545 * 6 - n_positives) + [1] * n_positives), y_pred=([0.5] * 78545 * 5 + [pred_all] * 78545), labels=[0, 1], sample_weight=[2] * 78545 + [1] * 78545 * 5), 3)

    elif weight == 2:

        return ceil(sklearn.metrics.log_loss(y_true=([0] * (78545 * 6 - n_positives) + [1] * n_positives), y_pred=([0.5] * 78545 * 5 + [pred_all] * 78545), labels=[0, 1], sample_weight=[1] * 78545 * 5 + [2] * 78545), 3)
list_n_positives_epidural = list(range(int(df_output['N_positives']['epidural']) - 25, int(df_output['N_positives']['epidural']) + 25))



list_n_positives_epidural = [n_positives for n_positives in list_n_positives_epidural if 0.618 == get_score_v5(n_positives, pred_all=0,      weight=1)]

list_n_positives_epidural = [n_positives for n_positives in list_n_positives_epidural if 5.504 == get_score_v5(n_positives, pred_all=1,      weight=1)]
list_n_positives_epidural
comment_out = '''

for i in tqdm(range(1, 10000)):

    pred_all      = i / 10000

    list_lb_score = [get_score_v5(n_positives, pred_all=pred_all, weight=1) for n_positives in list_n_positives_epidural]

    counter       = collections.Counter(list_lb_score)

    counter_left  = []

    if len(set(list_lb_score)) != 1:

        print(pred_all, list_lb_score, counter)

        counter_left.append(min([items[1] for items in counter.items()]))

        if len(set(counter_left)) == len(list_n_positives_epidural):

            break

'''

sample_output_subarachnoid = '''

  2%|▏         | 232/9999 [07:44<5:15:49,  1.94s/it]

0.0028 [0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.632, 0.633] Counter({0.632: 8, 0.633: 1})

0.0046 [0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.63] Counter({0.629: 8, 0.63: 1})

0.0054 [0.628, 0.628, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629, 0.629] Counter({0.629: 7, 0.628: 2})

0.0077 [0.626, 0.626, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627, 0.627] Counter({0.627: 7, 0.626: 2})

0.0093 [0.625, 0.625, 0.625, 0.625, 0.626, 0.626, 0.626, 0.626, 0.626] Counter({0.626: 5, 0.625: 4})

0.0113 [0.624, 0.624, 0.624, 0.624, 0.625, 0.625, 0.625, 0.625, 0.625] Counter({0.625: 5, 0.624: 4})

0.0139 [0.623, 0.623, 0.623, 0.624, 0.624, 0.624, 0.624, 0.624, 0.624] Counter({0.624: 6, 0.623: 3})

0.014 [0.623, 0.623, 0.623, 0.623, 0.623, 0.623, 0.623, 0.623, 0.624] Counter({0.623: 8, 0.624: 1})

0.0174 [0.622, 0.623, 0.623, 0.623, 0.623, 0.623, 0.623, 0.623, 0.623] Counter({0.623: 8, 0.622: 1})

0.0175 [0.622, 0.622, 0.622, 0.622, 0.622, 0.623, 0.623, 0.623, 0.623] Counter({0.622: 5, 0.623: 4})

0.0176 [0.622, 0.622, 0.622, 0.622, 0.622, 0.622, 0.622, 0.622, 0.623] Counter({0.622: 8, 0.623: 1})

0.0227 [0.621, 0.622, 0.622, 0.622, 0.622, 0.622, 0.622, 0.622, 0.622] Counter({0.622: 8, 0.621: 1})

0.0228 [0.621, 0.621, 0.621, 0.621, 0.622, 0.622, 0.622, 0.622, 0.622] Counter({0.622: 5, 0.621: 4})

0.0229 [0.621, 0.621, 0.621, 0.621, 0.621, 0.621, 0.622, 0.622, 0.622] Counter({0.621: 6, 0.622: 3})

0.023 [0.621, 0.621, 0.621, 0.621, 0.621, 0.621, 0.621, 0.621, 0.622] Counter({0.621: 8, 0.622: 1})

'''
list_n_positives_epidural = [n_positives for n_positives in list_n_positives_epidural if 0.853 == get_score_v5(n_positives, pred_all=0.838,  weight=1)]

list_n_positives_epidural = [n_positives for n_positives in list_n_positives_epidural if 1.181 == get_score_v5(n_positives, pred_all=0.984,  weight=1)]

list_n_positives_epidural = [n_positives for n_positives in list_n_positives_epidural if 0.599 == get_score_v5(n_positives, pred_all=0.0128, weight=1)]



list_n_positives_epidural
list_n_positives_intraparenchymal = list(range(int(df_output['N_positives']['intraparenchymal']) - 25, int(df_output['N_positives']['intraparenchymal']) + 25))



list_n_positives_intraparenchymal = [n_positives for n_positives in list_n_positives_intraparenchymal if 0.817 == get_score_v5(n_positives, pred_all=0,      weight=1)]

list_n_positives_intraparenchymal = [n_positives for n_positives in list_n_positives_intraparenchymal if 5.305 == get_score_v5(n_positives, pred_all=1,      weight=1)]

list_n_positives_intraparenchymal = [n_positives for n_positives in list_n_positives_intraparenchymal if 0.625 == get_score_v5(n_positives, pred_all=0.0105, weight=1)]

list_n_positives_intraparenchymal = [n_positives for n_positives in list_n_positives_intraparenchymal if 0.627 == get_score_v5(n_positives, pred_all=0.0072, weight=1)]

list_n_positives_intraparenchymal = [n_positives for n_positives in list_n_positives_intraparenchymal if 0.622 == get_score_v5(n_positives, pred_all=0.0955, weight=1)]



list_n_positives_intraparenchymal
list_n_positives_intraventricular = list(range(int(df_output['N_positives']['intraventricular']) - 25, int(df_output['N_positives']['intraventricular']) + 25))



list_n_positives_intraventricular = [n_positives for n_positives in list_n_positives_intraventricular if 0.747 == get_score_v5(n_positives, pred_all=0,      weight=1)]

list_n_positives_intraventricular = [n_positives for n_positives in list_n_positives_intraventricular if 5.375 == get_score_v5(n_positives, pred_all=1,      weight=1)]

list_n_positives_intraventricular = [n_positives for n_positives in list_n_positives_intraventricular if 0.614 == get_score_v5(n_positives, pred_all=0.0245, weight=1)]

list_n_positives_intraventricular = [n_positives for n_positives in list_n_positives_intraventricular if 0.613 == get_score_v5(n_positives, pred_all=0.0249, weight=1)]

list_n_positives_intraventricular = [n_positives for n_positives in list_n_positives_intraventricular if 0.616 == get_score_v5(n_positives, pred_all=0.0098, weight=1)]



list_n_positives_intraventricular
list_n_positives_subarachnoid = list(range(int(df_output['N_positives']['subarachnoid']) - 25, int(df_output['N_positives']['subarachnoid']) + 25))



list_n_positives_subarachnoid = [n_positives for n_positives in list_n_positives_subarachnoid if 0.817 == get_score_v5(n_positives, pred_all=0,      weight=1)]

list_n_positives_subarachnoid = [n_positives for n_positives in list_n_positives_subarachnoid if 5.305 == get_score_v5(n_positives, pred_all=1,      weight=1)]

list_n_positives_subarachnoid = [n_positives for n_positives in list_n_positives_subarachnoid if 0.625 == get_score_v5(n_positives, pred_all=0.0105, weight=1)]

list_n_positives_subarachnoid = [n_positives for n_positives in list_n_positives_subarachnoid if 0.626 == get_score_v5(n_positives, pred_all=0.0072, weight=1)]

list_n_positives_subarachnoid = [n_positives for n_positives in list_n_positives_subarachnoid if 0.628 == get_score_v5(n_positives, pred_all=0.0060, weight=1)]



list_n_positives_subarachnoid
list_n_positives_subdural = list(range(int(df_output['N_positives']['subdural']) - 25, int(df_output['N_positives']['subdural']) + 25))



list_n_positives_subdural = [n_positives for n_positives in list_n_positives_subdural if 0.887 == get_score_v5(n_positives, pred_all=0,      weight=1)]

list_n_positives_subdural = [n_positives for n_positives in list_n_positives_subdural if 5.234 == get_score_v5(n_positives, pred_all=1,      weight=1)]

list_n_positives_subdural = [n_positives for n_positives in list_n_positives_subdural if 0.634 == get_score_v5(n_positives, pred_all=0.0095, weight=1)]

list_n_positives_subdural = [n_positives for n_positives in list_n_positives_subdural if 0.641 == get_score_v5(n_positives, pred_all=0.0038, weight=1)]

list_n_positives_subdural = [n_positives for n_positives in list_n_positives_subdural if 0.640 == get_score_v5(n_positives, pred_all=0.0043, weight=1)]

list_n_positives_subdural = [n_positives for n_positives in list_n_positives_subdural if 0.626 == get_score_v5(n_positives, pred_all=0.0394, weight=1)]



list_n_positives_subdural
# CAUTION: weight = 2

list_n_positives_any = list(range(int(df_output['N_positives']['any']) - 25, int(df_output['N_positives']['any']) + 25))



list_n_positives_any = [n_positives for n_positives in list_n_positives_any if 1.855 == get_score_v5(n_positives, pred_all=0,     weight=2)]

list_n_positives_any = [n_positives for n_positives in list_n_positives_any if 9.002 == get_score_v5(n_positives, pred_all=1,     weight=2)]

list_n_positives_any = [n_positives for n_positives in list_n_positives_any if 0.628 == get_score_v5(n_positives, pred_all=0.046, weight=2)]

list_n_positives_any = [n_positives for n_positives in list_n_positives_any if 0.621 == get_score_v5(n_positives, pred_all=0.058, weight=2)]



list_n_positives_any
# Load train labels

df_input_train_stage_1 = pd.read_csv(os.path.join(path_input, 'stage_1_train.csv'))





# Output summary

df_input_train_stage_1 = pd.read_csv(os.path.join(path_input, 'stage_1_train.csv'))

list_n_positives_train = [df_input_train_stage_1['Label'][i::6].sum() for i in range(6)]

list_n_positives_test  = [list_n_positives_epidural[0], list_n_positives_intraparenchymal[0], list_n_positives_intraventricular[0], list_n_positives_subarachnoid[0], list_n_positives_subdural[0], list_n_positives_any[0]]

df_summary                       = pd.DataFrame([list_n_positives_train, list_n_positives_test], columns=['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'], index=['N_positives_train', 'N_positives_test']).T

df_summary['ratio_N_positives_train'] = df_summary['N_positives_train'] / (len(df_input_train_stage_1) / 6)

df_summary['ratio_N_positives_test']  = df_summary['N_positives_test']  / (len(df_submission) / 6)

df_summary['weight']             = [1, 1, 1, 1, 1, 2]

df_summary