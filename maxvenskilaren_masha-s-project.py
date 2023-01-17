import os

import numpy as np

import subprocess

from scipy.io import loadmat, savemat

from skimage.transform import resize

from torchvision import datasets, transforms

from torch.utils.data import DataLoader



batch_size = 64

test_batch_size = 64





# Transformations

mnist_data_transformations = transforms.Compose([

                                transforms.Resize((32, 32)),

                                transforms.Grayscale(num_output_channels=3),

                                transforms.ToTensor()

                       ])



svhn_data_transformations = transforms.Compose([

                                transforms.Resize((32, 32)),

                                transforms.ToTensor()

])



# Data Source

svhn_train = datasets.SVHN('../data', split='train',

                           download=True,

                           transform=svhn_data_transformations)



svhn_test = datasets.SVHN('../data', split='test',

                          download=True,

                          transform=svhn_data_transformations)



mnist_test = datasets.MNIST('../data', train=False,

                            download=True,

                            transform=mnist_data_transformations)



mnist_train = datasets.MNIST('../data', train=True,

                            download=True,

                            transform=mnist_data_transformations)



# Data loaders

svhn_train_loader = DataLoader(svhn_train,

                               batch_size=batch_size, shuffle=True)

mnist_train_loader = DataLoader(mnist_train,

                                batch_size=batch_size, shuffle=True)

mnist_test_loader = DataLoader(mnist_test,

                               batch_size=test_batch_size, shuffle=True)

svhn_test_loader = DataLoader(svhn_test,

                              batch_size=test_batch_size, shuffle=True)
import torch

from torch import nn





class GaussianNoise(nn.Module):

    def __init__(self, sigma=1.0):

        super().__init__()

        self.sigma = sigma

        self.noise = torch.tensor(0.0).cuda()



    def forward(self, x):

        if self.training:

            sampled_noise = self.noise.repeat(*x.size()).normal_(mean=0, std=self.sigma)

            x = x + sampled_noise

        return x





class Classifier(nn.Module):

    def __init__(self, large=False):

        super(Classifier, self).__init__()



        n_features = 192 if large else 64



        self.feature_extractor = nn.Sequential(

            nn.InstanceNorm2d(3, momentum=1, eps=1e-3),  # L-17

            nn.Conv2d(3, n_features, 3, 1, 1),  # L-16

            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-16

            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-16

            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-15

            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-15

            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-15

            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-14

            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-14

            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-14

            nn.MaxPool2d(2),  # L-13

            nn.Dropout(0.5),  # L-12

            GaussianNoise(1.0),  # L-11

            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-10

            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-10

            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-10

            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-9

            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-9

            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-9

            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-8

            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-8

            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-8

            nn.MaxPool2d(2),  # L-7

            nn.Dropout(0.5),  # L-6

            GaussianNoise(1.0),  # L-5

        )



        self.classifier = nn.Sequential(

            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-4

            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-4

            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-4

            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-3

            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-3

            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-3

            nn.Conv2d(n_features, n_features, 3, 1, 1),  # L-2

            nn.BatchNorm2d(n_features, momentum=0.99, eps=1e-3),  # L-2

            nn.LeakyReLU(negative_slope=0.1, inplace=True),  # L-2

            nn.AdaptiveAvgPool2d(1),  # L-1

            nn.Conv2d(n_features, 10, 1)

        )



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

                m.track_running_stats = False



    def track_bn_stats(self, track):

        for m in self.modules():

            if isinstance(m, nn.BatchNorm2d):

                m.track_running_stats = track



    def forward(self, x, track_bn=False):



        if track_bn:

            self.track_bn_stats(True)



        features = self.feature_extractor(x)

        logits = self.classifier(features)



        if track_bn:

            self.track_bn_stats(False)



        return features, logits.view(x.size(0), 10)





class Discriminator(nn.Module):

    def __init__(self, large=False):

        super(Discriminator, self).__init__()



        n_features = 192 if large else 64



        self.disc = nn.Sequential(

            nn.Linear(n_features * 1 * 8 * 8, 100),

            nn.ReLU(True),

            nn.Linear(100, 1)

        )



    def forward(self, x):

        x = x.view(x.size(0), -1)

        return self.disc(x).view(x.size(0), -1)





class EMA:

    def __init__(self, decay):

        self.decay = decay

        self.shadow = {}



    def register(self, model):

        for name, param in model.named_parameters():

            if param.requires_grad:

                self.shadow[name] = param.data.clone()

        self.params = self.shadow.keys()



    def __call__(self, model):

        if self.decay > 0:

            for name, param in model.named_parameters():

                if name in self.params and param.requires_grad:

                    self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)

                    param.data = self.shadow[name]

import torch

import torch.nn as nn

import torch.nn.functional as F





class ConditionalEntropyLoss(torch.nn.Module):

    def __init__(self):

        super(ConditionalEntropyLoss, self).__init__()



    def forward(self, x):

        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

        b = b.sum(dim=1)

        return -1.0 * b.mean(dim=0)





class VAT(nn.Module):

    def __init__(self, model):

        super(VAT, self).__init__()

        self.n_power = 1

        self.XI = 1e-6

        self.model = model

        self.epsilon = 3.5



    def forward(self, X, logit):

        vat_loss = self.virtual_adversarial_loss(X, logit)

        return vat_loss



    def generate_virtual_adversarial_perturbation(self, x, logit):

        d = torch.randn_like(x, device='cuda')



        for _ in range(self.n_power):

            d = self.XI * self.get_normalized_vector(d).requires_grad_()

            _, logit_m = self.model(x + d)

            dist = self.kl_divergence_with_logit(logit, logit_m)

            grad = torch.autograd.grad(dist, [d])[0]

            d = grad.detach()



        return self.epsilon * self.get_normalized_vector(d)



    def kl_divergence_with_logit(self, q_logit, p_logit):

        q = F.softmax(q_logit, dim=1)

        qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))

        qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))

        return qlogq - qlogp



    def get_normalized_vector(self, d):

        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())



    def virtual_adversarial_loss(self, x, logit):

        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)

        logit_p = logit.detach()

        _, logit_m = self.model(x + r_vadv)

        loss = self.kl_divergence_with_logit(logit_p, logit_m)

        return loss
import torch

from torch import nn

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot





# discriminator network

feature_discriminator = Discriminator(large=False).cuda()



# classifier network.

classifier = Classifier(large=False).cuda()



# loss functions

cent = ConditionalEntropyLoss().cuda()

xent = nn.CrossEntropyLoss(reduction='mean').cuda()

sigmoid_xent = nn.BCEWithLogitsLoss(reduction='mean').cuda()

vat_loss = VAT(classifier).cuda()



# optimizer.

optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=1e-3, betas=(0.5, 0.999))

optimizer_disc = torch.optim.Adam(feature_discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))



# loss params.

dw = 1e-2

cw = 1

sw = 0

tw = 1e-2

bw = 1e-2





''' Exponential moving average (simulating teacher model) '''

ema = EMA(0.998)

ema.register(classifier)



array_for_graph = []

svhn_array_for_graph = []

svhn_test_array_for_graph = []

epochs = []

# training..

for epoch in range(1, 51):



    loss_main_sum, n_total = 0, 0

    loss_domain_sum, loss_src_class_sum, \

    loss_src_vat_sum, loss_trg_cent_sum, loss_trg_vat_sum = 0, 0, 0, 0, 0

    loss_disc_sum = 0

    

    

    for (images_source, labels_source), (images_target, labels_target) in zip(svhn_train_loader, mnist_train_loader):

        images_source, labels_source, images_target, labels_target = images_source.cuda(), labels_source.cuda(), images_target.cuda(), labels_target.cuda()



        # pass images through the classifier network.

        feats_source, pred_source = classifier(images_source)

        feats_target, pred_target = classifier(images_target, track_bn=True)



        ' Discriminator losses setup. '

        # discriminator loss.

        real_logit_disc = feature_discriminator(feats_source.detach())

        fake_logit_disc = feature_discriminator(feats_target.detach())



        loss_disc = 0.5 * (

                sigmoid_xent(real_logit_disc, torch.ones_like(real_logit_disc, device='cuda')) +

                sigmoid_xent(fake_logit_disc, torch.zeros_like(fake_logit_disc, device='cuda'))

        )



        ' Classifier losses setup. '

        # supervised/source classification.

        loss_src_class = xent(pred_source, labels_source)



        # conditional entropy loss.

        loss_trg_cent = cent(pred_target)



        # virtual adversarial loss.

        loss_src_vat = vat_loss(images_source, pred_source)

        loss_trg_vat = vat_loss(images_target, pred_target)



        # domain loss.

        real_logit = feature_discriminator(feats_source)

        fake_logit = feature_discriminator(feats_target)



        loss_domain = 0.5 * (

                sigmoid_xent(real_logit, torch.zeros_like(real_logit, device='cuda')) +

                sigmoid_xent(fake_logit, torch.ones_like(fake_logit, device='cuda'))

        )



        # combined loss.

        loss_main = (

                0.01 * loss_domain +

                1 * loss_src_class +

                0 * loss_src_vat +

                0.01 * loss_trg_cent +

                0.01 * loss_trg_vat

        )



        ' Update network(s) '



        # Update discriminator.

        optimizer_disc.zero_grad()

        loss_disc.backward()

        optimizer_disc.step()



        # Update classifier.

        optimizer_cls.zero_grad()

        loss_main.backward()

        optimizer_cls.step()



        # Polyak averaging.

        ema(classifier)  # TODO: move ema into the optimizer step fn.



        loss_domain_sum += loss_domain.item()

        loss_src_class_sum += loss_src_class.item()

        loss_src_vat_sum += loss_src_vat.item()

        loss_trg_cent_sum += loss_trg_cent.item()

        loss_trg_vat_sum += loss_trg_vat.item()

        loss_main_sum += loss_main.item()

        loss_disc_sum += loss_disc.item()

        n_total += 1



    # validate.

    if epoch % 1 == 0:

        classifier.eval()

        feature_discriminator.eval()



        with torch.no_grad():

            preds_mnist_val, gts_mnist_val = [], []

            preds_svhn_train_val, gts_svhn_train_val = [], []

            preds_svhn_test_val, gts_svhn_test_val = [], []

            val_loss = 0

            for images_target, labels_target in mnist_test_loader:

                images_target, labels_target = images_target.cuda(), labels_target.cuda()



                # cross entropy based classification

                _, pred_val = classifier(images_target)



                pred_val = np.argmax(pred_val.cpu().data.numpy(), 1)



                preds_mnist_val.extend(pred_val)

                gts_mnist_val.extend(labels_target)

                

            for images_source, labels_source in svhn_train_loader:

                images_source, labels_source, images_target, labels_target = images_source.cuda(), labels_source.cuda(), images_target.cuda(), labels_target.cuda()

                # cross entropy based classification

                _, pred_val = classifier(images_source)



                pred_val = np.argmax(pred_val.cpu().data.numpy(), 1)



                preds_svhn_train_val.extend(pred_val)

                gts_svhn_train_val.extend(labels_source)

                

            for images_target, labels_target in svhn_test_loader:

                images_target, labels_target = images_target.cuda(), labels_target.cuda()



                # cross entropy based classification

                _, pred_val = classifier(images_target)



                pred_val = np.argmax(pred_val.cpu().data.numpy(), 1)



                preds_svhn_test_val.extend(pred_val)

                gts_svhn_test_val.extend(labels_target)



            

            preds_mnist_val = np.asarray(preds_mnist_val)

            gts_mnist_val = np.asarray(gts_mnist_val)

            preds_svhn_train_val = np.asarray(preds_svhn_train_val)

            gts_svhn_train_val = np.asarray(gts_svhn_train_val)

            preds_svhn_test_val = np.asarray(preds_svhn_test_val)

            gts_svhn_test_val = np.asarray(gts_svhn_test_val)



            score_cls_val = ((np.mean(preds_mnist_val == gts_mnist_val))).astype(np.float)

            score_svhn_train_val = ((np.mean(preds_svhn_train_val == gts_svhn_train_val))).astype(np.float)

            score_svhn_test_val = ((np.mean(preds_svhn_test_val == gts_svhn_test_val))).astype(np.float)

            print('\n{} MNSIT {:.3f} SVHN:train {:.3f} SVHN:test {:.3f}\n'.format(epoch, score_cls_val, score_svhn_train_val, score_svhn_test_val))

            array_for_graph.append(score_cls_val)

            svhn_array_for_graph.append(score_svhn_train_val)

            svhn_test_array_for_graph.append(score_svhn_test_val)

            epochs.append(epoch)

            

            matplotlib.pyplot.plot(epochs, array_for_graph, color='green', marker='o', linestyle='solid',linewidth=2, markersize=2)

            matplotlib.pyplot.plot(epochs, svhn_array_for_graph, color='red', marker='o', linestyle='solid',linewidth=2, markersize=2)

            matplotlib.pyplot.plot(epochs, svhn_test_array_for_graph, color='blue', marker='o', linestyle='solid',linewidth=2, markersize=2)

            matplotlib.pyplot.show()



        feature_discriminator.train()

        classifier.train()