from fastai.vision import *
path = Path('../input')

path.ls()
path_trainval = path/'voctrainval_06-nov-2007/VOCdevkit/VOC2007'

path_test = path/'voctest_06-nov-2007/VOCdevkit/VOC2007'

path_label = path/'pascal_voc/PASCAL_VOC'
path_trainval.ls()
path_test.ls()
path_label.ls()
trn_j = json.load((path_label/'pascal_train2007.json').open())

trn_j.keys()