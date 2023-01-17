!cp -r /kaggle/input/kaggle-google-qa-labeling/* ./

!cp -r /kaggle/input/transformersdependencies/dependencies/* ./
from pathlib import Path

import pandas as pd

import numpy as np



from kaggle_google_qa_labeling.ner_detector import NERDetector

from kaggle_google_qa_labeling.evaluator.bi_encoder_evaluator import BiEncoderEvaluator

from kaggle_google_qa_labeling.blend_utils import blend_ranks, blend_mean





def bin_y_pred(y_pred, coeffs):

    y_pred_new = y_pred.copy()

    for j in range(y_pred.shape[1]):

        b = y_pred[:,j]

        backsort_inds = np.argsort(np.argsort(b))

        e = coeffs[j]

        b_ = []



        prev_val = 0

        for val in np.sort(b):

            if val - prev_val < e:

                b_.append(prev_val)

            else:

                b_.append(val)

                prev_val = val



        b_ = np.array(b_)[backsort_inds]



        if len(set(b_)) != 1:

            y_pred_new[:,j] = b_

            

    return y_pred_new
test_df = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

subm_df = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
# experiments_root = Path('/kaggle/input/experiments/')



# # PLACE HERE YOUR EXPERIMENT(or experiments) NAME

# experiment_names = [

#     EXPERIMENT_NAME

# ]



# coeffs = {

#     0: 0.00891250938133745,

#     1: 0.03162277660168379,

#     2: 0.08912509381337454,

#     3: 0.056234132519034905,

#     4: 0.22387211385683392,

#     5: 0.25118864315095796,

#     6: 0.02238721138568339,

#     7: 0.01412537544622754,

#     8: 0.15848931924611132,

#     9: 0.00891250938133745,

#     10: 0.01778279410038923,

#     11: 0.07079457843841377,

#     12: 0.15848931924611132,

#     13: 0.03162277660168379,

#     14: 0.11220184543019633,

#     15: 0.1412537544622754,

#     16: 0.11220184543019633,

#     17: 0.007943282347242814,

#     18: 0.03162277660168379,

#     19: 0.01,

#     20: 0.01258925411794167,

#     21: 0.01258925411794167,

#     22: 0.015848931924611134,

#     23: 0.005623413251903491,

#     24: 0.00501187233627272,

#     25: 0.02238721138568339,

#     26: 0.0630957344480193,

#     27: 0.007079457843841381,

#     28: 0.00891250938133745,

#     29: 0.007943282347242814

#  }



# experiment_dirs = [experiments_root / name for name in experiment_names]

# ner_model_dir = Path('/kaggle/input/google-qa-ner/ner/code/bert_base_cased/')

# process_math = True



# ner_model = NERDetector.from_model_dir(

#     ner_model_dir, 

#     device='cuda', 

#     bs=64, 

#     threshold=0.8, 

#     min_span_len=10

# )





# y_preds = []



# for experiment_dir in experiment_dirs:

    

#     evaluator = BiEncoderEvaluator(

#         experiment_dir=experiment_dir, 

#         device='cuda', 

#         bs=16, 

#         blend_strategy='mean', 

#         models_dir_name='fold_models', 

#         ner_model=ner_model, 

#         process_math=True,

#         ignore_dir_names=list()

#     )

    

#     y_pred = evaluator.run(test_df)

#     y_preds.append(y_pred)





# final_pred = blend_mean(y_preds)

# final_pred = bin_y_pred(final_pred, coeffs)





# out_df = pd.DataFrame(columns=subm_df.columns)

# out_df['qa_id'] = test_df['qa_id']

# out_df.iloc[:,1:] = final_pred

# out_df.to_csv('submission.csv', index=False)