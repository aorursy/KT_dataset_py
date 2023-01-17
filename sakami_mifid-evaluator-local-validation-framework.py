import os

from pathlib import Path

import shutil



import numpy as np

import torch

from torch import nn, optim

import torch.nn.functional as F

from torchvision import datasets, transforms

from torchvision.utils import save_image
batch_size = 32

epochs = 2



MODEL_PATH = '../input/dog-face-generation-competition-kid-metric-input/classify_image_graph_def.pb'

TRAIN_DIR = Path('../input/generative-dog-images/all-dogs/')

OUT_DIR = Path('../output_images/')

OUT_DIR.mkdir(exist_ok=True)



device = torch.device('cuda')



lr = 0.0003

beta1 = 0.5



nz = 100

fixed_noise = torch.randn(25, nz, 1, 1, device=device)



real_label = 0.9

fake_label = 0
class Generator(nn.Module):



    def __init__(self, nz, nfeats, nchannels):

        super(Generator, self).__init__()



        # input is Z, going into a convolution

        self.conv1 = nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False)

        self.bn1 = nn.BatchNorm2d(nfeats * 8)

        # state size. (nfeats * 8) x 4 x 4

        

        self.conv2 = nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(nfeats * 8)

        # state size. (nfeats * 8) x 8 x 8

        

        self.conv3 = nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False)

        self.bn3 = nn.BatchNorm2d(nfeats * 4)

        # state size. (nfeats*4) x 16 x 16

        

        self.conv4 = nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False)

        self.bn4 = nn.BatchNorm2d(nfeats * 2)

        # state size. (nfeats * 2) x 32 x 32

        

        self.conv5 = nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False)

        self.bn5 = nn.BatchNorm2d(nfeats)

        # state size. (nfeats) x 64 x 64

        

        self.conv6 = nn.ConvTranspose2d(nfeats, nchannels, 3, 1, 1, bias=False)

        # state size. (nchannels) x 64 x 64

        

    def forward(self, x):

        x = F.leaky_relu(self.bn1(self.conv1(x)))

        x = F.leaky_relu(self.bn2(self.conv2(x)))

        x = F.leaky_relu(self.bn3(self.conv3(x)))

        x = F.leaky_relu(self.bn4(self.conv4(x)))

        x = F.leaky_relu(self.bn5(self.conv5(x)))

        x = torch.tanh(self.conv6(x))

        

        return x





class Discriminator(nn.Module):



    def __init__(self, nchannels, nfeats):

        super(Discriminator, self).__init__()



        # input is (nchannels) x 64 x 64

        self.conv1 = nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)

        # state size. (nfeats) x 32 x 32

        

        self.conv2 = nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(nfeats * 2)

        # state size. (nfeats * 2) x 16 x 16

        

        self.conv3 = nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False)

        self.bn3 = nn.BatchNorm2d(nfeats * 4)

        # state size. (nfeats * 4) x 8 x 8

       

        self.conv4 = nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False)

        self.bn4 = nn.BatchNorm2d(nfeats * 8)

        # state size. (nfeats * 8) x 4 x 4

        

        self.conv5 = nn.Conv2d(nfeats * 8, 1, 4, 1, 0, bias=False)

        # state size. 1 x 1 x 1

        

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x), 0.2)

        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)

        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)

        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)

        x = torch.sigmoid(self.conv5(x))

        

        return x.view(-1, 1)
transform = transforms.Compose([transforms.Resize(64),

                                transforms.CenterCrop(64),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

                                           

imgs, label = next(iter(train_loader))

imgs = imgs.numpy().transpose(0, 2, 3, 1)



net_g = Generator(100, 32, 3).to(device)

net_d = Discriminator(3, 48).to(device)



criterion = nn.BCELoss()



optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))

optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))
for epoch in range(epochs):

    for i, (real_images, train_labels) in enumerate(train_loader):

        # --------------------------------------

        # Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))

        # --------------------------------------

        # train with real

        net_d.zero_grad()

        real_images = real_images.to(device)

        batch_size = real_images.size(0)

        labels = torch.full((batch_size, 1), real_label, device=device)



        output = net_d(real_images)

        err_d_real = criterion(output, labels)

        err_d_real.backward()

        d_x = output.mean().item()



        # train with fake

        noise = torch.randn(batch_size, nz, 1, 1, device=device)

        fake = net_g(noise)

        labels.fill_(fake_label)

        output = net_d(fake.detach())

        err_d_fake = criterion(output, labels)

        err_d_fake.backward()

        d_g_z1 = output.mean().item()

        err_d = err_d_real + err_d_fake

        optimizer_d.step()



        # --------------------------------------

        # Update Generator network: maximize log(D(G(z)))

        # --------------------------------------

        net_g.zero_grad()

        labels.fill_(real_label)  # fake labels are real for generator cost

        output = net_d(fake)

        err_g = criterion(output, labels)

        err_g.backward()

        d_g_z2 = output.mean().item()

        optimizer_g.step()

        

        if (i + 1) % (len(train_loader) // 2) == 0:

            print('[{}/{}][{}/{}] Loss_d: {:.4f} Loss_g: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}'.format(

                epoch + 1, epochs, i + 1, len(train_loader), err_d.item(), err_g.item(), d_x, d_g_z1, d_g_z2))
im_batch_size = 50

n_images = 10000



for i_batch in range(0, n_images, im_batch_size):

    gen_z = torch.randn(im_batch_size, 100, 1, 1, device=device)

    gen_images = net_g(gen_z)

    images = gen_images.to('cpu').clone().detach()

    images = images.numpy().transpose(0, 2, 3, 1)

    for i_image in range(gen_images.size(0)):

        save_image(gen_images[i_image, :, :, :], OUT_DIR / f'image_{i_batch + i_image:05d}.png')
from dataclasses import dataclass

from pathlib import Path

import warnings



import numpy as np

from PIL import Image

from scipy import linalg

import tensorflow as tf





class KernelEvalException(Exception):

    pass





@dataclass

class MiFIDEvaluator(object):

    model_path: str

    train_images_path: str

    feature_path: str = None

    imsize: int = 64

    output_layer: str = 'Pretrained_Net/pool_3:0'

    input_layer: str = 'Pretrained_Net/ExpandDims:0'

    output_shape: int = 2048

    cosine_distance_eps: float = 0.1

    batch_size: int = 50

    fid_epsilon: float = 1e-14

    

    def __post_init__(self):

        tf.reset_default_graph()

        self.create_model_graph()

        with tf.Session() as sess:

            if self.feature_path is None:

                self.mu2, self.sigma2, self.features2 = self._handle_path_memorization(

                    self.train_images_path, sess, is_checksize=False, is_check_png=False)

            else:

                with np.load(self.feature_path) as f:

                    self.mu2, self.sigma2, self.features2 = f['m'], f['s'], f['features']

    

    def create_model_graph(self):

        with tf.gfile.FastGFile(self.model_path, 'rb') as f:

            graph_def = tf.GraphDef()

            graph_def.ParseFromString(f.read())

            _ = tf.import_graph_def(graph_def, name='Pretrained_Net')

            

    def img_read_checks(self, filename, is_checksize=False, is_check_png=False):

        im = Image.open(str(filename))

        if is_checksize and im.size != (self.imsize, self.imsize):

            raise KernelEvalException(f'The images are not of size {check_imsize}')

        if is_check_png and im.format != 'PNG':

            raise KernelEvalException('Only PNG images should be submitted.')



        if self.imsize is None:

            return im

        else:

            return im.resize((self.imsize, self.imsize), Image.ANTIALIAS)

        

    def _get_model_layer(self, sess):

        layer = sess.graph.get_tensor_by_name(self.output_layer)

        ops = layer.graph.get_operations()

        for op_idx, op in enumerate(ops):

            for o in op.outputs:

                shape = o.get_shape()

                if shape._dims != []:

                    shape = [s.value for s in shape]

                    new_shape = []

                    for j, s in enumerate(shape):

                        if s == 1 and j == 0:

                            new_shape.append(None)

                        else:

                            new_shape.append(s)

                    o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

        return layer

        

    def get_activations(self, images, sess):

        inception_layer = self._get_model_layer(sess)

        n_images = images.shape[0]

        if self.batch_size > n_images:

            warnings.warn('batch size is bigger than the data size. setting batch size to data size')

            self.batch_size = n_images

        n_batches = n_images // self.batch_size + 1

        pred_arr = np.empty((n_images, self.output_shape))

        for i in range(n_batches):

            start = i * self.batch_size

            if start + self.batch_size < n_images:

                end = start + self.batch_size

            else:

                end = n_images



            batch = images[start:end]

            pred = sess.run(inception_layer, {self.input_layer: batch})

            pred_arr[start:end] = pred.reshape(-1, self.output_shape)

        return pred_arr

        

    def calculate_activation_statistics(self, images, sess):

        act = self.get_activations(images, sess)

        mu = np.mean(act, axis=0)

        sigma = np.cov(act, rowvar=False)

        return mu, sigma, act

            

    def _handle_path_memorization(self, path, sess, is_checksize, is_check_png):

        path = Path(path)

        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))



        # In production we don't resize input images. This is just for demo purpose. 

        x = np.array([np.array(self.img_read_checks(fn, is_checksize, is_check_png)) for fn in files])

        m, s, features = self.calculate_activation_statistics(x, sess)

        del x

        return m, s, features

    

    def calculate_frechet_distance(self, mu1, sigma1):

        mu1 = np.atleast_1d(mu1)

        mu2 = np.atleast_1d(self.mu2)

        sigma1 = np.atleast_2d(sigma1)

        sigma2 = np.atleast_2d(self.sigma2)



        assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'

        assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'



        # product might be almost singular

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():

            msg = f'fid calculation produces singular product; adding {self.eps} to diagonal of cov estimates'

            warnings.warn(msg)

            offset = np.eye(sigma1.shape[0]) * self.eps

            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

            

        # numerical error might give slight imaginary component

        if np.iscomplexobj(covmean):

            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):

                m = np.max(np.abs(covmean.imag))

                raise ValueError(f'Imaginary component {m}')

            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (mu1 - mu2).dot(mu1 - mu2) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    

    @staticmethod

    def normalize_rows(x):

        return np.nan_to_num(x / np.linalg.norm(x, ord=2, axis=1, keepdims=True))

    

    def cosine_distance(self, features1):

        features1_nozero = features1[np.sum(features1, axis=1) != 0]

        features2_nozero = self.features2[np.sum(self.features2, axis=1) != 0]

        norm_f1 = self.normalize_rows(features1_nozero)

        norm_f2 = self.normalize_rows(features2_nozero)



        d = 1.0 - np.abs(np.matmul(norm_f1, norm_f2.T))

        mean_min_d = np.mean(np.min(d, axis=1))

        return mean_min_d

            

    def calculate_kid_given_paths(self, user_images_unzipped_path):

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            m1, s1, features1 = self._handle_path_memorization(

                user_images_unzipped_path, sess, is_checksize=True, is_check_png=True)



            fid_value = self.calculate_frechet_distance(m1, s1)

            distance = self.cosine_distance(features1)

            return fid_value, distance

        

    def distance_thresholding(self, d):

        if d < self.cosine_distance_eps:

            return d

        else:

            return 1

        

    def evaluate(self, user_images_unzipped_path):

        fid_value, distance = self.calculate_kid_given_paths(user_images_unzipped_path)

        distance = self.distance_thresholding(distance)

        return fid_value, distance, fid_value / (distance + self.fid_epsilon)
evaluator = MiFIDEvaluator(MODEL_PATH, TRAIN_DIR / 'all-dogs/')

fid_value, distance, mi_fid_score = evaluator.evaluate(OUT_DIR)

print(f'FID: {fid_value:.5f}')

print(f'distance: {distance:.5f}')

print(f'MiFID: {mi_fid_score:.5f}')
shutil.make_archive('images', 'zip', OUT_DIR)