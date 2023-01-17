# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# In[0]

!pip install pretrainedmodels

import os



import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import pretrainedmodels as ptms

import torchvision.models as tmodels

from fastai.vision import (

    DatasetType, ImageList, Path, accuracy, cnn_learner, error_rate,

    get_transforms, imagenet_stats, models, Learner)

from fastai import metrics





# In[1]

print(os.listdir("../input/"))

print(dir(tmodels))

print(dir(models))



train_dir = "../input/train/train"

test_dir = "../input/test/train"

train = pd.read_csv('../input/train.csv')

test = pd.read_csv("../input/sample_submission.csv")

data_folder = Path("../input")

# print(train.head(5), train.describe(), sep='\n')





# In[2]

test_img = ImageList.from_df(test, path=data_folder/'test', folder='test')

# # Applying Data augmentation

# trfm = get_transforms(do_flip=True, flip_vert=True)

# #trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1)

# train_img = (

#     ImageList.from_df(train, path=data_folder, folder='train')

#     .split_by_rand_pct(0.01)

#     .label_from_df()

#     .add_test(test_img)

#     .transform(trfm, size=128)

#     .databunch(path='.', bs=64, device=torch.device('cuda:0'))

#     .normalize(imagenet_stats)

# )





# In[3]

def predict(learn: Learner, name: str):

    # submission.csv

    preds, _ = learn.get_preds(ds_type=DatasetType.Test)

    test['has_cactus'] = preds.numpy()[:, 0]

    test.to_csv('submission_{}.csv'.format(name), index=False)

    print('Finish creating submission_{}.csv'.format(name))

    # loss.csv

    id_ = range(len(learn.recorder.losses))

    loss_df = pd.DataFrame(

        {'id': id_, 'loss': np.array(learn.recorder.losses)})

    loss_df.to_csv('loss_{}.csv'.format(name), index=False)

    print('Finish creating loss_{}.csv'.format(name))

    # TODO 计算训练集上的一些指标

    preds, targets = learn.get_preds(ds_type=DatasetType.Train)

    preds_label = np.argmax(preds.numpy(), axis=1)

    id_ = range(len(preds))

    train_pred_df = pd.DataFrame({'id': id_, 'preds': preds.numpy(

    )[:, 0], 'preds_label': preds_label, 'targets': targets.numpy()})

    train_pred_df.to_csv('./train_pred_{}.csv'.format(name))

    print('Finish creating train_pred_{}.csv'.format(name))

    correct_count = np.equal(preds_label, targets.numpy()).sum()

    len_preds = len(preds)

    incorrect_count = len_preds - correct_count

    fpr, tpr = metrics.roc_curve(preds[:, 0], targets)

    fpr, tpr = fpr.numpy(), tpr.numpy()

    FP = np.floor(fpr * len_preds)

    FN = incorrect_count - FP

    TP = np.floor(tpr * len_preds)

    TN = correct_count - TP

    id_ = range(len(fpr))

    train_index_df = pd.DataFrame(

        {'id': id_, 'fpr': fpr, 'tpr': tpr, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN})

    train_index_df.to_csv('./train_index_{}.csv'.format(name))

    print('Finish creating train_index_{}.csv'.format(name))

    # 销毁learn和保存模型

    learn.export('./model_{}.pth'.format(name), destroy=True)





def get_model(model_name: str, pretrained: bool, seq: bool = False, pname: str = 'imagenet', **kwargs):

    pretrained = pname if pretrained else None

    model = getattr(ptms, model_name)(

        pretrained=pretrained, **kwargs)

    return nn.Sequential(*model.children()) if seq else model





def dpn68b(pretrained: bool = False): return get_model(

    'dpn68b', pretrained, pname='imagenet+5k', seq=True)





def dpn92(pretrained: bool = False): return get_model(

    'dpn92', pretrained, pname='imagenet+5k', seq=True)





def dpn107(pretrained: bool = False): return get_model(

    'dpn107', pretrained, pname='imagenet+5k', seq=True)





def se_resnet50(pretrained: bool = False): return get_model(

    'se_resnet50', pretrained)





def se_resnet101(pretrained: bool = False): return get_model(

    'se_resnet101', pretrained)





def se_resnet152(pretrained: bool = False): return get_model(

    'se_resnet152', pretrained)





# In[4]

model_normal_dict = {

    'densenet': tmodels.densenet161,

    'vgg': tmodels.vgg16_bn,

    'resnet': tmodels.resnet101,

    # 'xresnet': models.xresnet2.xresnet101,

    # 'present': models.presnet.presnet101,

    #'dpn': dpn92,

    'resnext': tmodels.resnext101_32x8d,

    'se_resnet': se_resnet101,

}

# 模型层数对比

model_layer_dict = {

    'densenet': {

        '121': tmodels.densenet121,

        # '161': tmodels.densenet161,

        '161': None,

        '169': tmodels.densenet169

    },

    'vgg': {

        '13_bn': tmodels.vgg13_bn,

        # '16_bn': tmodels.vgg16_bn,

        '16_bn': None,

        '19_bn': tmodels.vgg19_bn,

    },

    'resnet': {

        '50': tmodels.resnet50,

        # '101': tmodels.resnet101,

        '101': None,

        '152': tmodels.resnet152,

    },

    # 'xresnet': {

    #     '50_2': models.xresnet2.xresnet50_2,

    #     # '101': models.xresnet2.xresnet101,

    #     '152': models.xresnet2.xresnet152,

    # },

    # 'presnet': {

    #     '50': models.presnet.presnet50,

    #     # '101': models.presnet.presnet101,

    #     '152': models.presnet.presnet152,

    # },

    #'dpn': {

    #    '68b': dpn68b,

    #    # '92': dpn92,

    #    '92': None,

    #    '107': dpn107,

    #},

    'resnext': {

        '101': tmodels.resnext101_32x8d,

        '50': tmodels.resnext50_32x4d,

        '169': None,

    },

    'se_resnet': {

        '50': se_resnet50,

        # '101': se_resnet101,

        '101': None,

        '152': se_resnet152,

    }

}

# 模型学习率对比

lr_dict = {

    '1e-1': 1e-1,

    # '1e-2': 1e-2,

    '1e-3': 1e-3,

}

# 优化器对比

optim_dict = {

    'adam': torch.optim.Adam,

    # 'sgd': torch.optim.SGD,

    'rprop': torch.optim.Rprop,

}

# 1 cycle 策略

one_cycle_dict = {

    # 'use-one-cycle': True,

    'not-use-one-cycle': False,

}

# 不同的数据增强方法对比

transform_dict = {

    # 'dofilp-flipvert': [True, True],

    'dofilp': [True, True],

    'nothing': [True, True],

}







# In[5]

target_id = 3

index = 0

for name, model in model_normal_dict.items():

    if index != target_id:

        index = index + 1

        continue

    # 默认设置

    trfm = get_transforms(do_flip=True, flip_vert=True)

    train_img = (

        ImageList.from_df(train, path=data_folder/'train', folder='train')

        .split_by_rand_pct(0.01)

        .label_from_df()

        .add_test(test_img)

        .transform(trfm, size=128)

        .databunch(path='.', bs=64, device=torch.device('cuda:0'))

        .normalize(imagenet_stats)

    )

    learn = cnn_learner(train_img, model, metrics=[

        error_rate, accuracy], opt_func=torch.optim.SGD)

    lr = 1e-2

    learn.fit_one_cycle(2, slice(lr))

    predict(learn, '{}_{}_{}_{}_{}_{}'.format(name, list(

        model_layer_dict[name].keys())[1], '1e-2', 'sgd', 'use-one-cycle', 'dofilp-flipvert'))

    # 层数

    for layer_name, true_model in model_layer_dict[name].items():

        if true_model is None:

            continue

        learn = cnn_learner(train_img, true_model, metrics=[

            error_rate, accuracy], opt_func=torch.optim.SGD)

        lr = 1e-2

        learn.fit_one_cycle(2, slice(lr))

        predict(learn, '{}_{}'.format(name, layer_name))

    # 学习率

    for lr_name, lr in lr_dict.items():

        learn = cnn_learner(train_img, model, metrics=[

            error_rate, accuracy], opt_func=torch.optim.SGD)

        learn.fit_one_cycle(2, slice(lr))

        predict(learn, '{}_{}'.format(name, lr_name))

    # 优化器

    for optim_name, optim in optim_dict.items():

        learn = cnn_learner(train_img, model, metrics=[

            error_rate, accuracy], opt_func=optim)

        lr = 1e-2

        learn.fit_one_cycle(2, slice(lr))

        predict(learn, '{}_{}'.format(name, optim_name))

    # one cycle

    for cycle_name, cycle_strategy in one_cycle_dict.items():

        learn = cnn_learner(train_img, model, metrics=[

            error_rate, accuracy], opt_func=torch.optim.SGD)

        lr = 1e-2

        if cycle_strategy:

            learn.fit_one_cycle(2, slice(lr))

        else:

            learn.fit(2, slice(lr))

        predict(learn, '{}_{}'.format(name, cycle_name))

    # 数据增强策略

    for transform_name, transform_strategy in transform_dict.items():

        trfm = get_transforms(

            do_flip=transform_strategy[0], flip_vert=transform_strategy[1])

        train_img = (

            ImageList.from_df(train, path=data_folder/'train', folder='train')

            .split_by_rand_pct(0.01)

            .label_from_df()

            .add_test(test_img)

            .transform(trfm, size=128)

            .databunch(path='.', bs=64, device=torch.device('cuda:0'))

            .normalize(imagenet_stats)

        )

        learn = cnn_learner(train_img, model, metrics=[

            error_rate, accuracy], opt_func=torch.optim.SGD)

        lr = 1e-2

        learn.fit_one_cycle(2, slice(lr))

        predict(learn, '{}_{}'.format(name, transform_name))

    index = index + 1