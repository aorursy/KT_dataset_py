import pandas as pd

import numpy as np
sub1 = pd.read_csv('../input/melanoma-dif-sub/pl_0.936.csv')

sub2 = pd.read_csv('../input/melanoma-dif-sub/pl_0.940.csv')

sub3 = pd.read_csv('../input/melanoma-dif-sub/sub_EfficientNetB2_384.csv')

sub4 = pd.read_csv('../input/melanoma-dif-sub/sub_EfficientNetB3_384.csv')

sub5 = pd.read_csv('../input/melanoma-dif-sub/sub_EfficientNetB3_384_v2.csv')



# lets rank each prediction and then divide it by its max value to we have our predictions between 0 and 1

def rank_data(sub):

    sub['target'] = sub['target'].rank() / sub['target'].rank().max()

    return sub



sub1 = rank_data(sub1)

sub2 = rank_data(sub2)

sub3 = rank_data(sub3)

sub4 = rank_data(sub4)

sub5 = rank_data(sub5)

sub1.columns = ['image_name', 'target1']

sub2.columns = ['image_name', 'target2']

sub3.columns = ['image_name', 'target3']

sub4.columns = ['image_name', 'target4']

sub5.columns = ['image_name', 'target5']



f_sub = sub1.merge(sub2, on = 'image_name').merge(sub3, on = 'image_name').merge(sub4, on = 'image_name').merge(sub5, on = 'image_name')

f_sub['target'] = f_sub['target1'] * 0.3 + f_sub['target2'] * 0.3 + f_sub['target3'] * 0.05 + f_sub['target4'] * 0.3 + f_sub['target5'] * 0.05

f_sub = f_sub[['image_name', 'target']]

f_sub.to_csv('blend_sub.csv', index = False)