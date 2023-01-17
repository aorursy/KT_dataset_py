import pandas as pd
sub_1 = pd.read_csv('../input/third-ensemble/effB1_512_5x15epochs_3Stratif.csv')

sub_2 = pd.read_csv('../input/third-ensemble/submission_blending.csv')

sub_3 = pd.read_csv('../input/third-ensemble/submission_blending_2.csv')
def rank_data(sub):

    sub['target'] = sub['target'].rank() / sub['target'].rank().max()

    return sub
sub_1 = rank_data(sub_1)

sub_2 = rank_data(sub_2)

sub_3 = rank_data(sub_3)
sub_1.columns = ['image_name', 'target1']

sub_2.columns = ['image_name', 'target2']

sub_3.columns = ['image_name', 'target3']
f_sub = sub_1.merge(sub_2, on = 'image_name').merge(sub_3, on ='image_name')

f_sub['target'] = f_sub['target1'] * 0.6 + f_sub['target2'] * 0.2 + f_sub['target3'] * 0.2 
f_sub = f_sub[['image_name', 'target']]

f_sub.to_csv('Finalblend_sub_3.csv', index = False)