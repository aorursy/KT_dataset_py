# 导入必要的库
import os, sys, glob, shutil, json
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2

from PIL import Image
import numpy as np

from tqdm import tqdm, tqdm_notebook

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
# 步骤1：定义读取数据集
class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl)  + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)
# 步骤2：定义读取数据dataloader
train_path = glob.glob('../input/vision/mchar_train/mchar_train/*.png')
train_path.sort()
train_json = json.load(open('../input/vision/mchar_train.json'))

train_label = [train_json[x]['label'] for x in train_json]
print(len(train_path), len(train_label))

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((80, 160)),
                    transforms.RandomCrop((64, 128)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),     # 0.6046
                    transforms.RandomRotation(10),            # 0.6057
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    batch_size=40,
    shuffle=True,
    num_workers=10,
)

val_path = glob.glob('../input/vision/mchar_val/mchar_val/*.png')
val_path.sort()
val_json = json.load(open('../input/vision/mchar_val.json'))

val_label = [val_json[x]['label'] for x in val_json]
print(len(val_path), len(val_label))

val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path, val_label,
                transforms.Compose([
                    transforms.Resize((80, 160)),
                    transforms.CenterCrop((64, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    batch_size=40,
    shuffle=False,
    num_workers=10,
)
# 步骤3：定义好字符分类模型，使用resnet18的模型作为特征提取模块
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
                
        # resnet18
        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])  # 去除最后一个fc layer
        self.cnn = model_conv

        self.hd_fc1 = nn.Linear(512, 128)
        self.hd_fc2 = nn.Linear(512, 128)
        self.hd_fc3 = nn.Linear(512, 128)
        self.hd_fc4 = nn.Linear(512, 128)
        self.hd_fc5 = nn.Linear(512, 128)
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.25)
        self.dropout_3 = nn.Dropout(0.25)
        self.dropout_4 = nn.Dropout(0.25)
        self.dropout_5 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128, 11)
        self.fc2 = nn.Linear(128, 11)
        self.fc3 = nn.Linear(128, 11)
        self.fc4 = nn.Linear(128, 11)
        self.fc5 = nn.Linear(128, 11)



    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)

        feat1 = self.hd_fc1(feat)
        feat2 = self.hd_fc2(feat)
        feat3 = self.hd_fc3(feat)
        feat4 = self.hd_fc4(feat)
        feat5 = self.hd_fc5(feat)
        feat1 = self.dropout_1(feat1)
        feat2 = self.dropout_2(feat2)
        feat3 = self.dropout_3(feat3)
        feat4 = self.dropout_4(feat4)
        feat5 = self.dropout_5(feat5)

        c1 = self.fc1(feat1)
        c2 = self.fc2(feat2)
        c3 = self.fc3(feat3)
        c4 = self.fc4(feat4)
        c5 = self.fc5(feat5)

        return c1, c2, c3, c4, c5
# 步骤4：定义好训练、验证和预测模块
def train(train_loader, model, criterion, optimizer):
    # 切换模型为训练模式
    model.train()
    train_loss = []
    train_loss_c1 = []
    train_loss_c2 = []
    train_loss_c3 = []
    train_loss_c4 = []
    train_loss_c5 = []
    
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
            
        c1, c2, c3, c4, c5 = model(input)

        loss_c1 =criterion(c1, target[:, 0])
        loss_c2 =criterion(c2, target[:, 1])
        loss_c3 =criterion(c3, target[:, 2])
        loss_c4 =criterion(c4, target[:, 3])
        loss_c5 =criterion(c5, target[:, 4])
        loss = loss_c1 + loss_c2 + loss_c3 + loss_c4 + loss_c5
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_loss_c1.append(loss_c1.item())
        train_loss_c2.append(loss_c2.item())
        train_loss_c3.append(loss_c3.item())
        train_loss_c4.append(loss_c4.item())
        train_loss_c5.append(loss_c5.item())

    loss_detail = [0] * 5
    loss_detail[0] = np.mean(train_loss_c1)
    loss_detail[1] = np.mean(train_loss_c2)
    loss_detail[2] = np.mean(train_loss_c3)
    loss_detail[3] = np.mean(train_loss_c4)
    loss_detail[4] = np.mean(train_loss_c5)
    return np.mean(train_loss), loss_detail


def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []
    val_loss_c1 = []
    val_loss_c2 = []
    val_loss_c3 = []
    val_loss_c4 = []
    val_loss_c5 = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()
            
            c1, c2, c3, c4, c5 = model(input)

            loss_c1 =criterion(c1, target[:, 0])
            loss_c2 =criterion(c2, target[:, 1])
            loss_c3 =criterion(c3, target[:, 2])
            loss_c4 =criterion(c4, target[:, 3])
            loss_c5 =criterion(c5, target[:, 4])
            loss = loss_c1 + loss_c2 + loss_c3 + loss_c4 + loss_c5

            val_loss.append(loss.item())
            val_loss_c1.append(loss_c1.item())
            val_loss_c2.append(loss_c2.item())
            val_loss_c3.append(loss_c3.item())
            val_loss_c4.append(loss_c4.item())
            val_loss_c5.append(loss_c5.item())

    loss_detail = [0] * 5
    loss_detail[0] = np.mean(val_loss_c1)
    loss_detail[1] = np.mean(val_loss_c2)
    loss_detail[2] = np.mean(val_loss_c3)
    loss_detail[3] = np.mean(val_loss_c4)
    loss_detail[4] = np.mean(val_loss_c5)
    return np.mean(val_loss), loss_detail


def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None
    
    # TTA 次数
    for _ in range(tta):
        test_pred = []
    
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()
                
                c0, c1, c2, c3, c4 = model(input)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(), 
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(), 
                        c3.data.cpu().numpy(),
                        c4.data.cpu().numpy()], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(), 
                        c1.data.numpy(),
                        c2.data.numpy(), 
                        c3.data.numpy(),
                        c4.data.numpy()], axis=1)
                
                test_pred.append(output)
        
        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    
    return test_pred_tta
# 步骤5：迭代训练和验证模型
model = SVHN_Model1()

# TODO: set train parameters
max_epochs = 27
lr = 0.0005
submit_base_name = 'ex10-2'
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

lr_decay_node = (13, 22)
lr_decay_ratio = 0.1

best_loss_model = 'model_best_loss.pt'
best_acc_model = 'model_best_acc.pt'
if os.path.exists(best_loss_model):
    os.remove(best_loss_model)
if os.path.exists(best_acc_model):
    os.remove(best_acc_model)

use_cuda = True
if use_cuda:
    model = model.cuda()


best_val_loss = 1000.0
best_val_acc = 0
for epoch in range(max_epochs):
    print("---------------------------------------")
    if epoch in lr_decay_node:
        lr = lr * lr_decay_ratio
        optimizer = torch.optim.Adam(model.parameters(), lr)
        #optimizer = torch.optim.SGD(model.parameters(), lr)

    train_loss, train_loss_detail = train(train_loader, model, criterion, optimizer)
    val_loss, val_loss_detail = validate(val_loader, model, criterion)

    val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
    val_predict_label = predict(val_loader, model, 1)
    val_predict_label = np.vstack([
        val_predict_label[:, :11].argmax(1),
        val_predict_label[:, 11:22].argmax(1),
        val_predict_label[:, 22:33].argmax(1),
        val_predict_label[:, 33:44].argmax(1),
        val_predict_label[:, 44:55].argmax(1),
    ]).T
    val_label_pred = []
    for x in val_predict_label:
        val_label_pred.append(''.join(map(str, x[x!=10])))

    val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))

    print('Epoch: %d, Train loss: %.5f \t Val loss: %.5f \t Val Acc: %.5f' % (epoch, train_loss, val_loss, val_char_acc))

    tr_c1, tr_c2, tr_c3, tr_c4, tr_c5 = train_loss_detail
    vl_c1, vl_c2, vl_c3, vl_c4, vl_c5 = val_loss_detail
    print("train loss detail: %.5f   %.5f   %.5f   %.5f   %.5f" % (tr_c1, tr_c2, tr_c3, tr_c4, tr_c5))
    print("val loss detail:   %.5f   %.5f   %.5f   %.5f   %.5f" % (vl_c1, vl_c2, vl_c3, vl_c4, vl_c5))

    # 如果是最优结果，保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print('Find better model in Epoch {0}, saving model.'.format(epoch))
        torch.save(model.state_dict(), best_loss_model)
    if val_char_acc > best_val_acc:
        best_val_acc = val_char_acc
        print('Find better model in Epoch {0}, saving model.'.format(epoch))
        torch.save(model.state_dict(), best_acc_model)
print("best val acc:  {}".format(best_val_acc))
print("best val loss: {}".format(best_val_loss))
# 步骤6：对测试集样本进行预测
test_path = glob.glob('../input/vision/mchar_test_a/mchar_test_a/*.png')
test_path.sort()
test_label = [[1]] * len(test_path)
print(len(test_path), len(test_label))

test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path, test_label,
                transforms.Compose([
                    transforms.Resize((68, 136)),
                    transforms.RandomCrop((64, 128)),
                    # transforms.RandomCrop((60, 120)),
                    # transforms.ColorJitter(0.3, 0.3, 0.2),
                    # transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    batch_size=40,
    shuffle=False,
    num_workers=10,
)
# -----------------------------------
# 测试最优loss模型
model.load_state_dict(torch.load(best_loss_model))

test_predict_label = predict(test_loader, model, 10)
print(test_predict_label.shape)

test_label = [''.join(map(str, x)) for x in test_loader.dataset.img_label]
test_predict_label = np.vstack([
    test_predict_label[:, :11].argmax(1),
    test_predict_label[:, 11:22].argmax(1),
    test_predict_label[:, 22:33].argmax(1),
    test_predict_label[:, 33:44].argmax(1),
    test_predict_label[:, 44:55].argmax(1),
]).T

test_label_pred = []
for x in test_predict_label:
    test_label_pred.append(''.join(map(str, x[x!=10])))

import pandas as pd
df_submit = pd.read_csv('../input/vision/sample_submit_A.csv')
df_submit['file_code'] = test_label_pred
submit_file = submit_base_name + '_best_loss.csv'
df_submit.to_csv(submit_file, index=None)
# -----------------------------------
# 测试最优acc模型
model.load_state_dict(torch.load(best_acc_model))

test_predict_label = predict(test_loader, model, 10)
print(test_predict_label.shape)

test_label = [''.join(map(str, x)) for x in test_loader.dataset.img_label]
test_predict_label = np.vstack([
    test_predict_label[:, :11].argmax(1),
    test_predict_label[:, 11:22].argmax(1),
    test_predict_label[:, 22:33].argmax(1),
    test_predict_label[:, 33:44].argmax(1),
    test_predict_label[:, 44:55].argmax(1),
]).T

test_label_pred = []
for x in test_predict_label:
    test_label_pred.append(''.join(map(str, x[x!=10])))

import pandas as pd
df_submit = pd.read_csv('../input/vision/sample_submit_A.csv')
df_submit['file_code'] = test_label_pred
submit_file = submit_base_name + '_best_acc.csv'
df_submit.to_csv(submit_file, index=None)