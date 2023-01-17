!pip install tensorlayer

!pip install easydict

# !pip install json

import tensorflow as tf

#import tensorlayer as tl

from tensorlayer.layers import (Input, Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Dense)

from tensorlayer.models import Model

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def get_G(input_shape):

    w_init = tf.random_normal_initializer(stddev=0.02)

    g_init = tf.random_normal_initializer(1., 0.02)



    nin = Input(input_shape)

    n = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init)(nin)

    temp = n



    # B residual blocks

    for i in range(12):

        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)

        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)

        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)

        nn = BatchNorm2d(gamma_init=g_init)(nn)

        nn = Elementwise(tf.add)([n, nn])

        n = nn



    n = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)

    n = BatchNorm2d(gamma_init=g_init)(n)

    n = Elementwise(tf.add)([n, temp])

    # B residual blacks end



    n = Conv2d(256, (3, 3), (1, 1), padding='SAME', W_init=w_init)(n)

    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)



    n = Conv2d(256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init)(n)

    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)



    nn = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init)(n)

    G = Model(inputs=nin, outputs=nn,name='gen')

    return G



def get_D(input_shape):

    w_init = tf.random_normal_initializer(stddev=0.02)

    gamma_init = tf.random_normal_initializer(1., 0.02)

    df_dim = 64

    lrelu = lambda x: tl.act.lrelu(x, 0.2)



    nin = Input(input_shape)

    n = Conv2d(df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init)(nin)



    n = Conv2d(df_dim * 2, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)

    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 4, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)

    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 8, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)

    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 16, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)

    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 32, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)

    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 16, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)

    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 8, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)

    nn = BatchNorm2d(gamma_init=gamma_init)(n)



    n = Conv2d(df_dim * 2, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)

    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 2, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)

    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 8, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)

    n = BatchNorm2d(gamma_init=gamma_init)(n)

    n = Elementwise(combine_fn=tf.add, act=lrelu)([n, nn])



    n = Flatten()(n)

    no = Dense(n_units=1, W_init=w_init)(n)

    D = Model(inputs=nin, outputs=no,name='dis')

    return D

from easydict import EasyDict as edict

import json

config = edict()

config.TRAIN = edict()



## Adam

config.TRAIN.batch_size = 8 # [16] use 8 if your GPU memory is small, and use [2, 4] in tl.vis.save_images / use 16 for faster training

config.TRAIN.lr_init = 1e-4

config.TRAIN.beta1 = 0.9



## initialize G

config.TRAIN.n_epoch_init = 100

    # config.TRAIN.lr_decay_init = 0.1

    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)



## adversarial learning (SRGAN)

config.TRAIN.n_epoch = 100

config.TRAIN.lr_decay = 0.1

config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)



## train set location

#config.TRAIN.hr_img_path = 'DIV2K/DIV2K_train_HR/'

#config.TRAIN.hr_img_path = '/kaggle/input/DIV2K_HR/DIV2K_train_HR/'

config.TRAIN.hr_img_path = '/kaggle/input/div2k-hr/DIV2K_train_HR/'

config.TRAIN.lr_img_path = '/kaggle/input/dev2k-hr/DIV2K_train_LR/'



config.VALID = edict()

## test set location

config.VALID.hr_img_path = 'DIV2K/DIV2K_valid_HR/'

config.VALID.lr_img_path = 'DIV2K/DIV2K_valid_LR_bicubic/X4/'



def log_config(filename, cfg):

    with open(filename, 'w') as f:

        f.write("================================================\n")

        f.write(json.dumps(cfg, indent=4))

        f.write("\n================================================\n")





import os

import time

import random

import numpy as np

import scipy, multiprocessing

import tensorflow as tf

import tensorlayer as tl

#from model import get_G, get_D

#from config import config



###====================== HYPER-PARAMETERS ===========================###

## Adam

batch_size = config.TRAIN.batch_size  # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]

lr_init = config.TRAIN.lr_init

beta1 = config.TRAIN.beta1

## initialize G

n_epoch_init = config.TRAIN.n_epoch_init

## adversarial learning (SRGAN)

n_epoch = config.TRAIN.n_epoch

lr_decay = config.TRAIN.lr_decay

decay_every = config.TRAIN.decay_every

shuffle_buffer_size = 128



# ni = int(np.sqrt(batch_size))



# create folders to save result images and trained models

save_dir = "samples"

tl.files.exists_or_mkdir(save_dir)

checkpoint_dir = "models"

tl.files.exists_or_mkdir(checkpoint_dir)





def get_train_data():

    # load dataset

    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))  # [0:20]

    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))

    # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))

    # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))



    ## If your machine have enough memory, please pre-load the entire train set.

    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)



    # for im in train_hr_imgs:

    #     print(im.shape)

    # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)

    # for im in valid_lr_imgs:

    #     print(im.shape)

    # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)

    # for im in valid_hr_imgs:

    #     print(im.shape)



    # dataset API and augmentation

    def generator_train():

        for img in train_hr_imgs:

            yield img



    def _map_fn_train(img):

        hr_patch = tf.image.random_crop(img, [384, 384, 3])

        hr_patch = hr_patch / (255. / 2.)

        hr_patch = hr_patch - 1.

        hr_patch = tf.image.random_flip_left_right(hr_patch)

        lr_patch = tf.image.resize(hr_patch, size=[96, 96])

        return lr_patch, hr_patch



    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))

    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())

    # train_ds = train_ds.repeat(n_epoch_init + n_epoch)

    train_ds = train_ds.shuffle(shuffle_buffer_size)

    train_ds = train_ds.prefetch(buffer_size=2)

    train_ds = train_ds.batch(batch_size)

    # value = train_ds.make_one_shot_iterator().get_next()

    return train_ds





def train():

    with tf.device('/gpu:0'):

        G = get_G((batch_size, 96, 96, 3))

        D = get_D((batch_size, 384, 384, 3))

        savepoint=0

        saveafter=2

        VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')



        lr_v = tf.Variable(lr_init)

        g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)

        g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

        d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)



        G.train()

        D.train()

        VGG.train()



        train_ds = get_train_data()

        print("INITIALIZING TRAINING")

        ## initialize learning (G)

        n_step_epoch = round(n_epoch_init // batch_size)

        for epoch in range(n_epoch_init):

            for step, (lr_patchs, hr_patchs) in enumerate(train_ds):

                if lr_patchs.shape[0] != batch_size:  # if the remaining data in this epoch < batch_size

                    break

                step_time = time.time()

                with tf.GradientTape() as tape:

                    fake_hr_patchs = G(lr_patchs)

                    mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)

                grad = tape.gradient(mse_loss, G.trainable_weights)

                g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))

                print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(

                    epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))

            if (epoch != 0) and (epoch % 10 == 0):

                tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4],

                                   os.path.join(save_dir, 'train_g_init_{}.png'.format(epoch)))

            if (savepoint % saveafter == saveafter - 1):

                G.save("./gmodel.h5")

                print("SAVED MODEL\nSAVED MODEL\n")

            savepoint+=1

        print("OVER THE INITIAL LEARNING\n")

        #tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_{}.png'.format(epoch)))

        G.save_weights(os.path.join(checkpoint_dir, 'g12.h5'))

        D.save_weights(os.path.join(checkpoint_dir, 'd12.h5'))

        ## adversarial learning (G, D)

        n_step_epoch = round(n_epoch // batch_size)

        for epoch in range(n_epoch):

            for step, (lr_patchs, hr_patchs) in enumerate(train_ds):

                if lr_patchs.shape[0] != batch_size:  # if the remaining data in this epoch < batch_size

                    break

                step_time = time.time()

                with tf.GradientTape(persistent=True) as tape:

                    fake_patchs = G(lr_patchs)

                    logits_fake = D(fake_patchs)

                    logits_real = D(hr_patchs)

                    feature_fake = VGG((fake_patchs + 1) / 2.)  # the pre-trained VGG uses the input range of [0, 1]

                    feature_real = VGG((hr_patchs + 1) / 2.)

                    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))

                    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))

                    d_loss = d_loss1 + d_loss2

                    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))

                    mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)

                    vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)

                    g_loss = mse_loss + vgg_loss + g_gan_loss

                grad = tape.gradient(g_loss, G.trainable_weights)

                g_optimizer.apply_gradients(zip(grad, G.trainable_weights))

                grad = tape.gradient(d_loss, D.trainable_weights)

                d_optimizer.apply_gradients(zip(grad, D.trainable_weights))

                print(

                    "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(

                        epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss,

                        d_loss))



            # update the learning rate

            if epoch != 0 and (epoch % decay_every == 0):

                new_lr_decay = lr_decay ** (epoch // decay_every)

                lr_v.assign(lr_init * new_lr_decay)

                log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)

                print(log)



            if  ( (epoch != 0) and (epoch % 10 == 0) )or True:

                tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_{}.png'.format(epoch)))

                G.save_weights(os.path.join(checkpoint_dir, 'g12.h5'))

                D.save_weights(os.path.join(checkpoint_dir, 'd12.h5'))





def evaluate():

    ###====================== PRE-LOAD DATA ===========================###

    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))

    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))

    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))

    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))



    ## if your machine have enough memory, please pre-load the whole train set.

    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)

    # for im in train_hr_imgs:

    #     print(im.shape)

    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)

    # for im in valid_lr_imgs:

    #     print(im.shape)

    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)

    # for im in valid_hr_imgs:

    #     print(im.shape)



    ###========================== DEFINE MODEL ============================###

    imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡

    valid_lr_img = valid_lr_imgs[imid]

    valid_hr_img = valid_hr_imgs[imid]

    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image

    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]

    # print(valid_lr_img.min(), valid_lr_img.max())



    G = get_G([1, None, None, 3])

    G.load_weights(os.path.join(checkpoint_dir, 'g12.h5'))

    G.eval()



    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)

    valid_lr_img = valid_lr_img[np.newaxis, :, :, :]

    size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]



    out = G(valid_lr_img).numpy()



    print("LR size: %s /  generated HR size: %s" % (

    size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)

    print("[*] save images")

    tl.vis.save_image(out[0], os.path.join(save_dir, 'valid_gen.png'))

    tl.vis.save_image(valid_lr_img[0], os.path.join(save_dir, 'valid_lr.png'))

    tl.vis.save_image(valid_hr_img, os.path.join(save_dir, 'valid_hr.png'))



    out_bicu = scipy.misc.imresize(valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)

    tl.vis.save_image(out_bicu, os.path.join(save_dir, 'valid_bicubic.png'))



train()