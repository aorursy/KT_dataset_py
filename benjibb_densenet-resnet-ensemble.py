from fastai.tabular import *

from fastai.callbacks import ReduceLROnPlateauCallback, EarlyStoppingCallback, SaveModelCallback

from sklearn.metrics import roc_auc_score

import gc
dense161 = pd.read_csv("../input/cancer-densenet161-v2-for-ensemble/validation_0.976066529750824.csv")

dense161_test = pd.read_csv("../input/cancer-densenet161-v2-for-ensemble/submission_0.976066529750824.csv")
dense201 = pd.read_csv("../input/cancer-densenet201-v2-for-ensemble/validation_0.9749373197555542.csv")

dense201_test = pd.read_csv("../input/cancer-densenet201-v2-for-ensemble/submission_0.9749373197555542.csv")
res50 = pd.read_csv("../input/cancer-resnet50-v2-for-ensemble/validation_0.9727705717086792.csv")

res50_test = pd.read_csv("../input/cancer-resnet50-v2-for-ensemble/submission_0.9727705717086792.csv")
#trydf = pd.DataFrame({'dense161':dense161.ground_truth_label, 

#                      'dense201':dense201.ground_truth_label, 

#                      'res50':res50.ground_truth_label})

#trydf['1and2'] = trydf.dense161==trydf.dense201

#trydf['2and3'] = trydf.res50==trydf.dense201

#trydf['1and2'].nunique() == 1

#trydf['2and3'].nunique() == 1
def softmax_df(df, model_name, test=False):

    if test:

            df[model_name+'_0'] = np.exp(df['pred_0'])

            df[model_name+'_1'] = np.exp(df['pred_1'])

    else:

        df[model_name+'_0'] = np.exp(df['val_0'])

        df[model_name+'_1'] = np.exp(df['val_1'])

    df[model_name+'sum'] = df[model_name+'_0'] + df[model_name+'_1']

    df[model_name+'softmax'] = df[model_name+'_1'] / df[model_name+'sum']

    return df[model_name+'softmax']
#dense161_sm = softmax_df(dense161, 'dense161')

#dense201_sm = softmax_df(dense201, 'dense201')

#res50_sm = softmax_df(res50, 'res50')

#dense161_sm_test = softmax_df(dense161_test, 'dense161_test', True)

#dense201_sm_test = softmax_df(dense201_test, 'dense201_test', True)

#res50_sm_test = softmax_df(res50_test, 'res50_test', True)

#train = pd.DataFrame({'dense161_sm':dense161_sm, "dense201_sm":dense201_sm, "res50_sm":res50_sm, "y":dense161.ground_truth_label})

#test = pd.DataFrame({'dense161_sm':dense161_sm_test, "dense201_sm":dense201_sm_test, "res50_sm":res50_sm_test})

#test.y=0
dense161.head()
train = pd.DataFrame({'dense161_0':dense161.val_0, 'dense161_1':dense161.val_1, 

                      'dense201_0':dense201.val_0, 'dense201_1':dense201.val_1,

                      'res50_0':res50.val_0, 'res50_1':res50.val_1,

                      "y":dense161.ground_truth_label})

test = pd.DataFrame({'dense161_0':dense161_test.pred_0, 'dense161_1':dense161_test.pred_1, 

                      'dense201_0':dense201_test.pred_0, 'dense201_1':dense201_test.pred_1,

                      'res50_0':res50_test.pred_0, 'res50_1':res50_test.pred_1})

test.y=0
dep_var = 'y'

#cont_names = ['dense161_sm','dense201_sm', 'res50_sm']

cont_names = ['dense161_0', 'dense161_1', 'dense201_0', 'dense201_1', 'res50_0','res50_1']



data = (TabularList.from_df(train, cont_names=cont_names)

            .split_by_rand_pct(seed=47)

            .label_from_df(cols=dep_var)

            .add_test(TabularList.from_df(test, cont_names=cont_names))

            .databunch())
def roc_score(inp, target):

    _, indices = inp.max(1)

    return torch.Tensor([roc_auc_score(target, indices)])[0]
learn = tabular_learner(data, layers=[10, 10, 10], metrics=[accuracy, roc_score],  ps=0.5, wd=1e-1, model_dir='./').to_fp16()
#learn.lr_find()
#learn.recorder.plot()
from fastai.callbacks import ReduceLROnPlateauCallback, EarlyStoppingCallback, SaveModelCallback

ES = EarlyStoppingCallback(learn, monitor='roc_score',patience = 5)

RLR = ReduceLROnPlateauCallback(learn, monitor='roc_score',patience = 2)

SAVEML = SaveModelCallback(learn, every='improvement', monitor='roc_score', name='best')

learn.fit_one_cycle(20, 1e-3, callbacks = [ES, RLR, SAVEML])
learn.load('best')
preds, _ = learn.get_preds(DatasetType.Test)

preds = torch.softmax(preds, dim=1)[:, 1].numpy()
auc_val = learn.validate()[2].item()

auc_val
sub = pd.read_csv("../input/histopathologic-cancer-detection/sample_submission.csv")

sub.head()
sub['label'] = preds
sub.to_csv(f'submission_{auc_val}.csv', header=True, index=False)
sub.head()