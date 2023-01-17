import numpy as np
import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
res = pd.read_csv('../input/resources.csv')
res.drop('description',inplace=True,axis=1)
res['total_price'] = res.quantity*res.price
quantityres = res.groupby('id').quantity.sum().reset_index()
totalprice = res.groupby('id').total_price.sum().reset_index()
train = train.merge(quantityres,on='id')
train = train.merge(totalprice,on='id')
test = test.merge(quantityres,on='id')
test = test.merge(totalprice,on='id')
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
loo_features = ['teacher_id', 'teacher_prefix', 'school_state',
                'project_grade_category',
                'project_subject_categories', 'project_subject_subcategories',
                'project_resource_summary']
test['project_is_approved'] = -1
for col in loo_features:
    print(col)
    train[col], test[col] = target_encode(train[col], 
                                          test[col], 
                                          target=train.project_is_approved, 
                                          min_samples_leaf=10,
                                          smoothing=10,
                                          noise_level=0.01)
trainapproved = train.project_is_approved.values
colstodrop = ['project_submitted_datetime',
              'project_title',
              'project_essay_1',
              'project_essay_2',
              'project_essay_3',
              'project_essay_4',
              'project_is_approved'
             ]
train.drop(colstodrop,inplace=True,axis=1)
test.drop(colstodrop,inplace=True,axis=1)
train['project_is_approved'] = trainapproved
train.quantity = np.log(train.quantity)
train.total_price = np.log(train.total_price)
train.teacher_number_of_previously_posted_projects = np.log1p(train.teacher_number_of_previously_posted_projects)

test.quantity = np.log(test.quantity)
test.total_price = np.log(test.total_price)
test.teacher_number_of_previously_posted_projects = np.log1p(test.teacher_number_of_previously_posted_projects)
def Output(p):
    return 1.0/(1.0+np.exp(-p))

def GP(data):
    p = (1.0*np.tanh((((((8.0)) * (((-1.0) + (((((data["project_resource_summary"]) * 2.0)) * (data["project_resource_summary"]))))))) * 2.0)) +
        1.0*np.tanh(((((((data["total_price"]) * ((-1.0*(((((0.79683798551559448)) - (data["project_resource_summary"])))))))) * 2.0)) * ((6.0)))) +
        1.0*np.tanh((((((((((4.46330404281616211)) * (data["project_resource_summary"]))) - (3.0))) * ((6.38417387008666992)))) - (2.0))) +
        1.0*np.tanh(((((((((((((data["project_resource_summary"]) * (data["project_resource_summary"]))) * 2.0)) + (-1.0))) * 2.0)) * 2.0)) * 2.0)) +
        1.0*np.tanh(((3.0) * (((((-3.0) - (((-2.0) * (((data["project_resource_summary"]) * 2.0)))))) * 2.0)))) +
        1.0*np.tanh(((data["project_resource_summary"]) * ((((((13.21048259735107422)) * (((data["project_resource_summary"]) - (np.tanh((1.0))))))) * 2.0)))) +
        1.0*np.tanh(((((((np.tanh((-1.0))) + (data["teacher_id"]))) * (((data["project_resource_summary"]) * 2.0)))) * ((9.97741889953613281)))) +
        1.0*np.tanh(((((data["teacher_id"]) + (np.where(((data["project_resource_summary"]) - ((0.84962385892868042)))>0, data["teacher_id"], -1.0 )))) * 2.0)) +
        1.0*np.tanh(((((3.0) * (((data["project_resource_summary"]) - (np.tanh((np.tanh((2.0))))))))) * 2.0)) +
        1.0*np.tanh(((((((((data["teacher_id"]) - (data["project_subject_categories"]))) * 2.0)) * (((data["project_resource_summary"]) * 2.0)))) * 2.0)))
    return Output(p)
from sklearn.metrics import roc_auc_score
roc_auc_score(train.project_is_approved,GP(train))