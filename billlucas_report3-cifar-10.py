from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss,nn
import mxnet as mx
import os
import pandas as pd
import shutil          
import time
#先使用小容量的样本进行训练
data_source = '../input/cifar10zip-for-train-and-test'
train_source = 'train/train'
test_source = 'test/test'

demo = False
if demo:
    train_dir, test_dir, batch_size = 'train_tiny', 'test_tiny', 1
    data_dir = '/kaggle/working/cifartiny'
else:
    train_dir, test_dir, batch_size = 'train', 'test', 128
    data_dir, label_file = '/kaggle/working', 'trainLabels.csv'
input_dir, valid_ratio = 'train_valid_test', 0.1

#下面定义一个辅助函数，从而仅在路径不存在的情况下创建路径。
def mkdir_if_not_exist(path):  # 函数创建目录
    if not os.path.exists(os.path.join(*path)):   #此处考虑path可能为列表的形式
        print(os.path.join(*path))
        os.makedirs(os.path.join(*path))
#在‘/kaggle/working/'里新建cifartiny，用于后面的读取
#mkdir_if_not_exist([data_dir, train_dir])
shutil.copytree(os.path.join(data_source, train_source), os.path.join(data_dir, train_dir))
#读取测试集的标签文件，valid_ratio是验证集占原始训练集的比例。
#读取trainLabels.csv里的序号和label,将它们转为字典类型
def read_label_file(data_dir, label_file, train_dir, valid_ratio):
    with open(os.path.join(data_dir, label_file), 'r') as f:
        lines = f.readlines()[1:]  # 跳过文件头行（栏名称，id和label）
        #此时lines为列表类型，其中元素是由序号+ 逗号 +label+\n组成的字符串
        tokens = [l.rstrip().split(',') for l in lines]  #rstrip()默认在字符串尾删除\n、\t、空格等字符串
        #tokens亦为列表类型，其中元素由split隔开，为包含两个子元素的子列表类型
        idx_label = dict(((int(idx), label) for idx, label in tokens))
        #先加括号构造含有两个字符串的元组，再构造generator、构造字典类型
    labels = set(idx_label.values())       #构造元组，删除重复元素
    n_train_valid = len(os.listdir(os.path.join(data_dir, train_dir)))     #统计文件夹train_dir中子文件（图片）的个数
    n_train = int(n_train_valid * (1 - valid_ratio))      #计算训练数目
    assert 0 < n_train < n_train_valid        #判断valid_ratio是否有效
    return n_train // len(labels), idx_label        # //代表整除，返回每个label所需的训练数，以及字典类型的label文件

n_train_per_label, idx_label = read_label_file(data_dir, label_file, train_dir, valid_ratio)
#下面定义一个辅助函数，从而仅在路径不存在的情况下创建路径。
def mkdir_if_not_exist(path):  
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))
    
#切割训练集，以valid_ratio为标准
#在input_dir下新增文件夹train_valid, train, valid, 其中分别包含原始训练集、训练数据集以及测试数据集
#三个子文件夹里均是各个类型所包含的图片
def reorg_train_valid(data_dir, train_dir, input_dir, n_train_per_label,
                      idx_label):
    #input_dir为新建的文件夹,为train_valid_test
    label_count = {}  #构造空字典
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        #train_file的一个例子：23.png。
        idx = int(train_file.split('.')[0])   
        label = idx_label[idx]  #str类型,与字典序号相对应
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        #构造新文件夹train_valid,里面存在包含各个类型图片的小文件夹
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        #将各个png图片放到他们label对应的文件夹里
        if label not in label_count or label_count[label] < n_train_per_label:
            #当字典的key进行查找或者每个label的图片数小于预定值
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            #构造新文件夹train,里面也是包含各个类型图片的小文件夹
            
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))
            label_count[label] = label_count.get(label, 0) + 1
            #字典可以通过指定key的value来增加dict中的元素，get方法返回指定键的值，如不存在则返回0
        else:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            #构建新文件夹valid,存放train_valid中包含但是train中不包含的值，即用于验证的值
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))
    
    #print('train_valid')
    #for dirpath, dirnames, filenames in os.walk(os.path.join(data_dir, input_dir, 'train_valid')):
    #    file_count = 0
    #    for file in filenames:
    #        file_count = file_count +1
    #    print(dirpath, file_count)
    #print(label_count)
reorg_train_valid(data_dir, train_dir, input_dir, n_train_per_label, idx_label)
#下面的reorg_test函数用来整理测试集，从而方便预测时的读取。
def reorg_test(data_dir, test_dir, input_dir):
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))        
reorg_test(data_dir, test_dir, input_dir) 
#进行简易的图像增广
transform_train = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(40),
    gdata.vision.transforms.RandomResizedCrop(32, scale = (0.64, 1.0), ratio = (1.0, 1.0)),
    #先裁剪出原图像0.64-1.0倍的图像，在变换为大小为32的图像
    gdata.vision.transforms.RandomFlipLeftRight(),    #图像左右翻转
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010]) 
])
transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])
])
#加载文件内容构造数据集，flag为1表明输入图像含有3个通道，即彩色
train_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, input_dir, 'train'), flag = 1)
valid_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, input_dir, 'valid'), flag = 1)
train_valid_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, input_dir, 'train_valid'), flag = 1)
test_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, input_dir, 'test'), flag = 1)

#从数据集加载数据并返回小批量数据。
#train_ds返回带有每个样本的第一个元素作为输入、由函数fn转换的数据集。在保持label不变的同时转换数据
train_iter = gdata.DataLoader(train_ds.transform_first(transform_train), batch_size, shuffle = True, last_batch = 'keep')
#gdata.vision.ImageFolderDataset.transform_first(fn, lazy)，其中lazy默认为False，转换个例，否则转换所有。当fn为随机时，必须为False
valid_iter = gdata.DataLoader(valid_ds.transform_first(transform_test), batch_size, shuffle = True, last_batch = 'keep')
train_valid_iter = gdata.DataLoader(train_valid_ds.transform_first(transform_train), batch_size, shuffle = True, last_batch = 'keep')
test_iter = gdata.DataLoader(test_ds.transform_first(transform_test), batch_size, shuffle = True, last_batch = 'keep')

#此处的ResNet与之前的不同：在Residual上使用HybirdBlock而不是Block,定义的顺序网络改为HybirdSequential()
#首个卷积层使用通道为64，步幅为1的3*3卷积层，其后接BatchNorm和relu函数，而没有最大池化层

#传统的ResNet在residual类中为2个具有相同通道数的3*3卷积层，后接批量归一化层和激活函数。取定strides和use_1conv后，第一个卷积层和self.conv3
#(即use_1conv)可以对输出形状进行减半。在残差块中对非首块的首层网络取定strides和use_1conv使得对非首块的输出减半，在Resnet中的首个卷积层
#是通道数为64，步幅大小为2的7*7卷积层，后接批量归一、激活函数，以及步幅为2的3*3最大池化层
class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1conv = False, strides = 1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size = 3, strides = strides, padding = 1)
        self.conv2 = nn.Conv2D(num_channels, kernel_size = 3, padding = 1)
        if use_1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size = 1, strides = strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
    
    def hybrid_forward(self, F, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(X + Y)
            
def resnet_18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size = 3, padding = 1), nn.BatchNorm(), nn.Activation('relu'))
    def resnet_block(num_channels, num_residuals, first_block = False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1conv = True, strides = 2))
            else:
                blk.add(Residual(num_channels))
        return blk
    net.add(
        resnet_block(64, 2, first_block = True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2)
    )
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
    
#构造网络并进行初始化
def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

def get_net(ctx):
    num_classes = 10
    net = resnet_18(num_classes)
    net.initialize(ctx = ctx, init = init.Xavier())
    return net

def evaluate_accuracy(data_iter, net, ctx=mx.cpu()):
    """Evaluate accuracy of a model on the given data set."""
    
    acc_sum, n = nd.array([0], ctx = ctx), 0
    for X, y in data_iter:
        X = X.as_in_context(ctx)
        y = y.astype('float32').as_in_context(ctx)
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size

    return acc_sum.asscalar() / n

loss = gloss.SoftmaxCrossEntropyLoss()
#momentum对初始化的权值进行优化，wd对权重进行弱化，权重越低，复杂度越低，拟合的效果越好
#gluon.Trainer.step()函数在内调用allreduce_grads()和update()对参数进行更新
#梯度由1/batch_size进行标准化
def train(net, train_iter, valid_iter, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr, 'momentum':0.9, 'wd':wd})
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(learning_rate * lr_decay)
        for X,y in train_iter:
            y = y.astype('float32').as_in_context(ctx)
            with autograd.record():
                y_hat = net(X.as_in_context(ctx))
                l = loss(y_hat, y).sum()
            l.backward()
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis = 1) == y).sum().asscalar()
            n += y.size
        time_s = 'time %.2f sec'% (time.time() - start)
        if valid_iter is not None:
            valid_acc = evaluate_accuracy(valid_iter, net, ctx)
            epoch_s = 'epoch %d, loss %f, train_acc %f, valid_acc %f'%(epoch + 1, train_l_sum/n, train_acc_sum/n, valid_acc)
        else:
            epoch_s = 'epoch %d, loss %f, train_acc %f' %(epoch +1, train_l_sum/n, train_acc_sum/n)    
    print(epoch_s + time_s + ' ,lr ' + str(trainer.learning_rate))
ctx, num_epochs, lr, wd = try_gpu(), 1, 0.1, 5e-4
lr_period, lr_decay, net = 80, 0.1, get_net(ctx)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, ctx, lr_period, lr_decay)
net, preds = get_net(ctx), []
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, ctx, lr_period, lr_decay)
for X, _ in test_iter:
    y_hat = net(X.as_in_context(ctx))
    preds.extend(y_hat.argmax(axis =1).astype('int32').asnumpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label':preds})
print(df)
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index = False)
import os
os.listdir('/kaggle/working')
print(sorted_ids, preds)
