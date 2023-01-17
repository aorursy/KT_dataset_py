# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import torch
CFG = {

    'image_target_cols': [

        'pe_present_on_image', # only image level

    ],

    

    'exam_target_cols': [

        'negative_exam_for_pe', # exam level

        #'qa_motion',

        #'qa_contrast',

        #'flow_artifact',

        'rv_lv_ratio_gte_1', # exam level

        'rv_lv_ratio_lt_1', # exam level

        'leftsided_pe', # exam level

        'chronic_pe', # exam level

        #'true_filling_defect_not_pe',

        'rightsided_pe', # exam level

        'acute_and_chronic_pe', # exam level

        'central_pe', # exam level

        'indeterminate' # exam level

    ], 

    

    'image_weight': 0.07361963,

    'exam_weights': [0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 0.1042944785, 0.06257668712, 0.1042944785, 0.1877300613, 0.09202453988],

}
train = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")

train.head(3)
img_label_mean = train[CFG['image_target_cols']].mean(axis=0)

exam_label_mean = train[CFG['exam_target_cols']].mean(axis=0)

print('===img label mean===\n{} \n\n\n===exam label mean===\n{}\n'.format(img_label_mean, exam_label_mean))



temp_df = train.copy()*0

temp_df[CFG['image_target_cols']] += img_label_mean.values

temp_df[CFG['exam_target_cols']] += exam_label_mean.values



print(temp_df.head(3))
def rsna_torch_wloss(CFG, y_true_img, y_true_exam, y_pred_img, y_pred_exam, chunk_sizes):



    # transform into torch tensors

    y_true_img, y_true_exam, y_pred_img, y_pred_exam = torch.tensor(y_true_img, dtype=torch.float32), torch.tensor(y_true_exam, dtype=torch.float32), torch.tensor(y_pred_img, dtype=torch.float32), torch.tensor(y_pred_exam, dtype=torch.float32)

    

    # split into chunks (each chunks is for a single exam)

    y_true_img_chunks, y_true_exam_chunks, y_pred_img_chunks, y_pred_exam_chunks = torch.split(y_true_img, chunk_sizes, dim=0), torch.split(y_true_exam, chunk_sizes, dim=0), torch.split(y_pred_img, chunk_sizes, dim=0), torch.split(y_pred_exam, chunk_sizes, dim=0)

    

    label_w = torch.tensor(CFG['exam_weights']).view(1, -1)

    img_w = CFG['image_weight']

    bce_func = torch.nn.BCELoss(reduction='none')

    

    total_loss = torch.tensor(0, dtype=torch.float32)

    total_weights = torch.tensor(0, dtype=torch.float32)

    

    for i, (y_true_img_, y_true_exam_, y_pred_img_, y_pred_exam_) in enumerate(zip(y_true_img_chunks, y_true_exam_chunks, y_pred_img_chunks, y_pred_exam_chunks)):

        exam_loss = bce_func(y_pred_exam_[0, :], y_true_exam_[0, :])

        exam_loss = torch.sum(exam_loss*label_w, 1)[0] # Kaggle uses a binary log loss equation for each label and then takes the mean of the log loss over all labels.

        

        image_loss = bce_func(y_pred_img_, y_true_img_)

        img_num = chunk_sizes[i]

        qi = torch.sum(y_true_img_)/img_num

        image_loss = torch.sum(img_w*qi*image_loss)

        

        total_loss += exam_loss+image_loss

        total_weights += label_w.sum() + img_w*qi*img_num

        #print(exam_loss, image_loss, img_num);assert False

        

    final_loss = total_loss/total_weights

    return final_loss



with torch.no_grad():

    loss = rsna_torch_wloss(CFG, train[CFG['image_target_cols']].values, train[CFG['exam_target_cols']].values, 

                      temp_df[CFG['image_target_cols']].values, temp_df[CFG['exam_target_cols']].values, 

                      list(train.groupby('StudyInstanceUID', sort=False)['SOPInstanceUID'].count()))



    print(loss)
from sklearn.metrics import log_loss



def cross_entropy(predictions, targets, epsilon=1e-12, reduction='none'):

    """

    Computes cross entropy between targets (encoded as one-hot vectors)

    and predictions. 

    Input: predictions (N, k1, k2, ...) ndarray

           targets (N, k1, k2, ...) ndarray

           reduction: 'none' | 'mean' | 'sum'

    Returns: scalar

    """

    predictions = np.clip(predictions, epsilon, 1. - epsilon)

    

    ce = -(targets*np.log(predictions) + (1.-targets)*np.log(1.-predictions))

    

    if reduction == 'none':

        return ce

    

    ce = np.sum(ce)

    if reduction == 'sum':

        return ce

    

    if reduction == 'mean':

        ce /= predictions.shape[0]

        return ce



    assert False, "reduction should be 'none' | 'mean' | 'sum'".format(reduction)

    

def rsna_np_wloss(CFG, y_true_img, y_true_exam, y_pred_img, y_pred_exam, split_indices):



    # split into chunks (each chunks is for a single exam)

    y_true_img_chunks, y_true_exam_chunks, y_pred_img_chunks, y_pred_exam_chunks = np.split(y_true_img, split_indices[1:-1], axis=0), np.split(y_true_exam, split_indices[1:-1], axis=0), np.split(y_pred_img, split_indices[1:-1], axis=0), np.split(y_pred_exam, split_indices[1:-1], axis=0)

    

    label_w = np.array(CFG['exam_weights']).reshape((1, -1))

    img_w = CFG['image_weight']

    bce_func = cross_entropy

    

    total_loss = 0.

    total_weights = 0.

    #print(len(y_true_img_chunks))

    

    for i, (y_true_img_, y_true_exam_, y_pred_img_, y_pred_exam_) in enumerate(zip(y_true_img_chunks, y_true_exam_chunks, y_pred_img_chunks, y_pred_exam_chunks)):

        exam_loss = bce_func(y_pred_exam_[0, :], y_true_exam_[0, :])

        exam_loss = np.sum(exam_loss*label_w, 1)[0] # Kaggle uses a binary log loss equation for each label and then takes the mean of the log loss over all labels.

        

        image_loss = bce_func(y_pred_img_, y_true_img_)

        img_num = split_indices[i+1]-split_indices[i]

        qi = np.sum(y_true_img_)/img_num

        image_loss = np.sum(img_w*qi*image_loss)

        

        total_loss += exam_loss+image_loss

        total_weights += label_w.sum() + img_w*qi*img_num

        #print(exam_loss, image_loss, img_num);assert False

        

        

    final_loss = total_loss/total_weights

    return final_loss



img_counts = train.groupby('StudyInstanceUID', sort=False)['SOPInstanceUID'].count()

split_indices = np.concatenate([[0], np.cumsum(img_counts)])

loss = rsna_np_wloss(CFG, train[CFG['image_target_cols']].values, train[CFG['exam_target_cols']].values, 

                     temp_df[CFG['image_target_cols']].values, temp_df[CFG['exam_target_cols']].values, 

                     list(split_indices))



print(loss)
train.StudyInstanceUID.unique()
img_counts