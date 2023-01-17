import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as p
import random
import scipy as sp, scipy.ndimage
import os
SCALED = 64
LARGES = 64
class mainrun:
    def __init__(self, DIR=""):
        self.DIR = DIR
    def validate_images_and_reshape(self, images, check_number=True):
        shape = images.shape
        total_images = shape[0]
        assert shape[1] == 2
        for i in range(total_images):
            if check_number:
                assert images[i, 0] == i
            assert images[i, 1].shape == (10000,)
            images[i, 1].shape = (100, 100)
    def prefixfilename(self, n):
        return os.path.join(self.DIR, n)
    def preprocessing(self):
        train_images = np.load(self.prefixfilename("train_images.npy"), encoding='bytes')
        train_labels = np.loadtxt(self.prefixfilename("train_labels.csv"),
                                  skiprows=1,
                                  delimiter=',',
                                  dtype={'names': ('id', 'category'),
                                         'formats': ('i4', 'U32')})
        self.validate_images_and_reshape(train_images)
        print("Loaded %d images" % train_images.shape[0])
        
        labels = [None] * len(train_labels)
        for i in range(len(train_labels)):
            assert train_labels[i][0] == i
            labels[i] = train_labels[i][1]
        ncat = len({label: 1 for label in labels})
        print("Loaded %d labels, with a total of %d unique categories" %
              (len(labels), ncat))
        self.labels = labels
        
        images = train_images.shape[0]
        self.im = np.zeros((images, LARGES, LARGES), dtype=np.int16)
        self.copy_and_denoise(train_images, self.im)
        print("Denoised and scaled %d training images" % train_images.shape[0])
        
        del train_images
        del train_labels
        
        def buildlabel():
            from sklearn.preprocessing import LabelEncoder
            labelencoder = LabelEncoder()
            labelencoder.fit(labels)
            self.labelencoder = labelencoder
            self.imlabel = labelencoder.transform(labels)
            print("Encoded labels:", self.imlabel)
        
        buildlabel()
        
        def test():
            np.random.seed(12)
            perm = np.random.permutation(len(labels))
            test = int(len(labels) * 0.9)
            trainx, validx = perm[:test], perm[test:]
            im, holdout = self.im[trainx, :], self.im[validx, :]
            imlabel, holdlabel = self.imlabel[trainx], self.imlabel[validx]
            self.__dict__.update({
                'im': im,
                'holdout': holdout,
                'imlabel': imlabel,
                'holdlabel': holdlabel,
                'perm': perm,
                'smallim': 0,
                'smallhold': 0
            })
        
        test()
        self.smallhold = self.generate_holdouts()
    def getimage(self, i):
        return self.train_images[self.perm[i], 1], self.train_labels[self.perm[i]][1]
    def getlabel(self, i):
        return self.labels[self.perm[i]]
    def getholdo(self, i):
        return self.labelencoder.inverse_transform([self.holdlabel[i]])[0]
    def getnameo(self, i):
        return self.labelencoder.inverse_transform([i])[0]
    def testimages(self):
        print("Loading test images")
        test_images = np.load(self.prefixfilename("test_images.npy"), encoding='bytes')
        self.validate_images_and_reshape(test_images)
        images = test_images.shape[0]
        print("Denoising %d images" % images)
        self.testim = np.zeros((images, LARGES, LARGES), dtype=np.int16)
        self.copy_and_denoise(test_images, self.testim)
    def autocrop(self, x):
        def border(xx, axis):
            aa = np.max(xx, axis=axis)
            if np.all(aa == 0):
                return 0, 0
            return np.argmax(aa), aa.shape[0] - 1 - np.argmax(aa[::-1])
        xx = x > 0
        ymin, ymax = border(xx, 1)
        xmin, xmax = border(xx, 0)
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        picture = x[ymin:ymax+1, xmin:xmax+1]
        return picture
    def zoom_to_size(self, x, size):
        height, width = x.shape
        x = sp.ndimage.zoom(x, size / max(width, height), prefilter=False)
        return x
    def randomly_repad(self, x, size):
        def random_padding(size):
            a = random.randint(0, size)
            b = size - a
            return (a, b)
        height, width = x.shape
        return np.pad(x, (random_padding(size - height), random_padding(size - width)), 'constant',
                      constant_values = (0, 0))
    def delete_all_but_largest(self, xx):
        aa = sp.ndimage.label(xx)
        l = np.bincount(aa[0].flat)[1:]
        m = np.argmax(l) + 1
        if l[m-1] <= 60:
            xx[:] = 0
        else:
            cluster = sp.ndimage.morphology.binary_fill_holes(aa[0] == m)
            xx[np.logical_not(cluster)] = 0
    def copy_and_denoise(self, x, im):
        for i in range(x.shape[0]):
            self.delete_all_but_largest(x[i, 1])
            im[i, :, :] = self.randomly_repad(self.autocrop(x[i, 1]), LARGES)
    def autozooms(self, x, test=False):
        x = self.autocrop(x)
        if test:
            z = SCALED
        else:
            if random.randint(0, 1):
                x = x[:, ::-1]
            z = SCALED * (0.75 + random.random() * 0.25)
        x = self.zoom_to_size(x, z)
        return self.randomly_repad(x, SCALED)
    def generate(self, m):
        smallim = np.zeros((m, SCALED, SCALED), dtype=np.int16)
        label = [None] * m
        for i in range(m):
            j = random.randint(0, self.im.shape[0] - 1)
            smallim[i, :, :] = self.autozooms(self.im[j, :, :])
            label[i] = self.imlabel[j]
        return smallim, label
    def generate_holdouts(self):
        s = random.getstate()
        random.seed(12)
        nh = self.holdout.shape[0]
        smallhold = np.zeros((nh, SCALED, SCALED), dtype=np.int16)
        for i in range(nh):
            smallhold[i, :, :] = self.autozooms(self.holdout[i, :, :])
        random.setstate(s)
        return smallhold
    def generate_scaled_test_set(self):
        s = random.getstate()
        random.seed(12)
        ntest = self.testim.shape[0]
        smallt = np.zeros((ntest, SCALED, SCALED), dtype=np.int16)
        for i in range(ntest):
            smallt[i, :, :] = self.autozooms(self.testim[i, :, :])
        random.setstate(s)
        return smallt
class SqueezeModule(nn.Module):
    def __init__(self, features):
        super(SqueezeModule, self).__init__()
        self.features = features
        self.broadcast_squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(features, features//16),
            nn.ReLU(),
            nn.Linear(features//16, features),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.broadcast_squeeze[0](x)
        out = out.view(-1, self.features)
        out = self.broadcast_squeeze[1](out)
        out = self.broadcast_squeeze[2](out)
        out = self.broadcast_squeeze[3](out)
        out = self.broadcast_squeeze[4](out)
        out = out.view(-1, self.features, 1, 1)
        return out * x
            

class SEResnetBlock(nn.Module):
    # from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    # (modified)
    def __init__(self, input_features, features, stride=1):
        super(SEResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_features, features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(features)
        
        self.squeeze = SqueezeModule(features)

        self.shortcut = nn.Sequential()
        if input_features != features or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_features, features, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(features)
            )
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(self.shortcut(x) + self.squeeze(out))

class ResnetCNN(nn.Module):
    def __init__(self):
        super(ResnetCNN, self).__init__()
        self.conv0 = nn.Conv2d(1, 31, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(31)
        self.conv0a = nn.Conv2d(32, 32, 3, padding=1)
        self.bn0a = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False) # 7
        self.bn1 = nn.BatchNorm2d(64)
        self.squeeze1 = SqueezeModule(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.short12 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.short23 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer1 = self.makelayer(64, 64, 3)
        self.layer2 = self.makelayer(128, 128, 4, stride=2)
        self.layer3 = self.makelayer(256, 256, 6, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 31))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def makelayer(self, input_features, features, depth, stride=1):
        layers = [SEResnetBlock(input_features, features, stride)] + \
            [SEResnetBlock(features, features, 1) for i in range(depth)]
        return nn.Sequential(*layers)
    def forward(self, x):
        # vectors: (batch, channels, height, width)
        # dense preprocessing layer
        # x is (batch, 1, 64, 64)
        
        dense = torch.relu(self.bn0(self.conv0(x)))
        x = torch.cat([x, dense], dim=1)
        # x is (batch, 32, 64, 64)
        
        dense = torch.relu(self.bn0a(self.conv0a(x)))
        x = torch.cat([x, dense], dim=1)
        # x is (batch, 64, 64, 64)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        # x is (batch, 64, 64, 64)
        x = self.squeeze1(self.maxpool(x))
        # x is (batch, 64, 32, 32)
        x = torch.cat([x, self.layer1(x)], dim=1)
        # x is (batch, 128, 32, 32)
        x = torch.cat([self.short12(x), self.layer2(x)], dim=1)
        # x is (batch, 256, 16, 16)
        x = torch.cat([self.short23(x), self.layer3(x)], dim=1)
        # x is (batch, 512, 8, 8)
        x = self.avgpool(x)
        # x is (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)
        # x is (batch, 512)
        x = self.dense(x)
        # x is (batch, 31)
        return x

class WhateverNet(nn.Module):
    def __init__(self):
        super(WhateverNet, self).__init__()
        self.net = ResnetCNN()
        pass
    def forward(self, x):
        return self.net(x)
device = torch.device('cuda')

def process_images(x):
    return torch.tensor(x / 128.0 - 1, dtype=torch.float, device=device).view(-1, 1, SCALED, SCALED)
def process_labels(x):
    return torch.tensor(x, dtype=torch.long, device=device)

class runloop:
    def __init__(self):
        self.loop = 1
        self.logs = {}
        self.smallt = runner.generate_scaled_test_set()
        self.output = []
    def train(self):
        import torch.optim as optim
        net = WhateverNet()
        net = net.to(device)
        
        from functools import reduce
        print("net built. parameters %d" % sum(reduce(lambda a, b: a*b, x.size()) for x in net.parameters()))

        trainloss_log = []
        validloss_log = []
        def train():
            criterion = nn.CrossEntropyLoss()
            self.parm = {
                'lr': 0.025 + 0.075 * random.random(),
                'momentum': 0.81 + 0.09 * random.random(),
                'weight_decay': .0001 + .0002 * random.random()}
            print("lr=%.4f, momentum=%.4f, weight_decay=%.4e" %
                 (self.parm['lr'], self.parm['momentum'], self.parm['weight_decay']))
            optimizer = optim.SGD(net.parameters(), **self.parm)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6, verbose=True)
            # On my desktop, this was 32, but these gpus have more memory, so:
            MINIBATCH = 128
            IMAGES_PER_EPOCH = 256*19 #4864
            assert IMAGES_PER_EPOCH % MINIBATCH == 0
            for epoch in range(70):
                trainloss = 0.0
                smallim, labels = runner.generate(IMAGES_PER_EPOCH)
                for start in range(0, IMAGES_PER_EPOCH, MINIBATCH):
                    end = min(start + MINIBATCH, IMAGES_PER_EPOCH)
                    scaled_images = process_images(smallim[start:end])
                    label = process_labels(labels[start:end])
                    optimizer.zero_grad()
                    out = net(scaled_images)
                    loss = criterion(out, label)
                    loss.backward()
                    optimizer.step()
                    trainloss += loss.item() * (end - start)
                del smallim
                del labels
                del loss
                totalloss = 0.0
                for start in range(0, runner.holdout.shape[0], MINIBATCH):
                    with torch.no_grad():
                        end = min(start + MINIBATCH, runner.holdout.shape[0])
                        current_slice = slice(start, start + MINIBATCH)
                        scaled_images = process_images(runner.smallhold[start:end])
                        labels = process_labels(runner.holdlabel[start:end])
                        out = net(scaled_images)
                        loss = criterion(out, labels)
                        totalloss += loss.item() * (end - start)
                del scaled_images
                del labels
                del loss

                trainloss /= IMAGES_PER_EPOCH
                totalloss /= runner.holdout.shape[0]
                trainloss_log.append(trainloss)
                validloss_log.append(totalloss)

                scheduler.step(totalloss)
                if epoch % 10 == 0:
                    print("Epoch %d/%d, best %.4f" % (epoch, 70, min(validloss_log)))
                if totalloss < 2 and len(validloss_log) > 1 and totalloss < np.min(validloss_log[:-1]):
                    f = "savenet-%d" % self.loop
                    torch.save(net.state_dict(), f)
            return (trainloss_log, validloss_log)
        self.logs[self.loop] = train()
        print("Loop %d: best %.4f" % (self.loop, min(self.logs[self.loop][1])))
    def eval(self):
        device = torch.device('cuda')
        net = WhateverNet() #ResnetCNN()
        net.load_state_dict(torch.load("savenet-%d" % self.loop))
        net.to(device)
        net.eval()

        import warnings
        warnings.filterwarnings("ignore")

        import random
        correct = 0
        total = 0
        correct = np.zeros(1000, dtype=np.int32)
        for i in range(1000):
            label = runner.getholdo(i)
            t = net(process_images(runner.smallhold[i, :, :]))
            guess = runner.labelencoder.inverse_transform([t.argmax().item()])
            correct[i] = label == guess
        print("Loop %d: correct %f%%" % (self.loop, np.mean(correct)*100))

        print("Classifying test... ", end='')
        f = open("answers-%d" % self.loop, "w")
        l = open("logprob-%d" % self.loop, "w")
        f.write("Id,Category\n")
        smallt = self.smallt
        for i in range(smallt.shape[0]):
            t = net(process_images(smallt[i, :, :]))
            guess = runner.labelencoder.inverse_transform([t.argmax().item()])
            l.write(" ".join(["%.2f" % a for a in t[0].tolist()]))
            l.write("\n")
            f.write("%d,%s\n" % (i, guess[0]))
        f.close()
        l.close()
        self.output += ["logprob-%d" % self.loop]
        print("done")
        
        self.loop += 1
runner = mainrun("../input")
runner.preprocessing()
runner.testimages()

loop = runloop()
loop.loop = 1
print("goin for it")
# The actual submission was an ensemble of 19,
# but this kernel has to complete in less than 6 hours,
# so let's set it to 7.
TOTAL_NUMBER_OF_ENSEMBLES = 7
for i in range(TOTAL_NUMBER_OF_ENSEMBLES):
    loop.train()
    loop.eval()
def logprobabilities(fn):
    return np.loadtxt(fn)
logprob = [logprobabilities(x) for x in loop.output]
guesses = [np.argmax(x, axis=1) for x in logprob]
def show_agreement(x):
    m = len(x)
    agreement = np.zeros((m, m))
    for i in range(m):
        agreement[i, i] = 1
        for j in range(i, m):
            tot = np.mean(x[i] == x[j])
            agreement[i, j] = tot
            agreement[j, i] = tot
    print("Agreement")
    print(agreement)
    p.matshow(agreement, vmin=0, vmax=1)
    p.show()
show_agreement(guesses)
average = sum(logprob)/len(logprob)
classes = ['apple', 'empty', 'moustache', 'mouth', 'mug', 'nail', 'nose', 'octagon',
           'paintbrush', 'panda', 'parrot', 'peanut', 'pear', 'pencil', 'penguin', 'pillow',
           'pineapple', 'pool', 'rabbit', 'rhinoceros', 'rifle', 'rollerskates', 'sailboat',
           'scorpion', 'screwdriver', 'shovel', 'sink', 'skateboard', 'skull', 'spoon', 'squiggle']
ensembled_guesses = [classes[j] for j in np.argmax(average, axis=1).tolist()]
f = open("comboanswers", "w")
f.write("Id,Category\n")
for i in range(len(ensembled_guesses)):
    f.write("%d,%s\n" % (i, ensembled_guesses[i]))
f.close()
print(len(ensembled_guesses), "written")
