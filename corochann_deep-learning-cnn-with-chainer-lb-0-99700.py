# Make sure you installed latest version of scikit-learn

#!pip install -U scikit-learn



# You may install `chaineripy` to show training progress bar nicely on jupyter notebook.

# https://github.com/grafi-tt/chaineripy

#!pip install chaineripy
# Make it False when you want to execute full training.

# It takes a long time to train deep CNN with CPU, but much less time with GPU.

# It is nice if you can utilize GPU, when DEBUG = False.

DEBUG = True
import os



import pandas as pd

import numpy as np



# Load data

print('Loading data...')

DATA_DIR = '../input'

#DATA_DIR = './input'

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))



train_x = train.iloc[:, 1:].values.astype('float32')

train_y = train.iloc[:, 0].values.astype('int32')

test_x = test.values.astype('float32')



print('train_x', train_x.shape)

print('train_y', train_y.shape)

print('test_x', test_x.shape)
# reshape and rescale value

train_imgs = train_x.reshape((-1, 1, 28, 28)) / 255.

test_imgs = test_x.reshape((-1, 1, 28, 28)) / 255.

print('train_imgs', train_imgs.shape, 'test_imgs', test_imgs.shape)
%matplotlib inline

#import matplotlib

#matplotlib.use('agg')

import matplotlib.pyplot as plt





def show_image(img):

    plt.figure(figsize=(1.5, 1.5))

    plt.axis('off')

    if img.ndim == 3:

        img = img[0, :, :]

    plt.imshow(img, cmap=plt.cm.binary)           

    plt.show()



print('index0, label {}'.format(train_y[0]))

show_image(train_imgs[0])

print('index1, label {}'.format(train_y[1]))

show_image(train_imgs[1])

#show_image(train_imgs[2])

#show_image(train_imgs[3])
import chainer

import chainer.links as L

import chainer.functions as F

from chainer.dataset.convert import concat_examples





class CNNMedium(chainer.Chain):

    def __init__(self, n_out):

        super(CNNMedium, self).__init__()

        with self.init_scope():

            self.conv1 = L.Convolution2D(None, 16, 3, 1)

            self.conv2 = L.Convolution2D(16, 32, 3, 1)

            self.conv3 = L.Convolution2D(32, 32, 3, 1)

            self.conv4 = L.Convolution2D(32, 32, 3, 2)

            self.conv5 = L.Convolution2D(32, 64, 3, 1)

            self.conv6 = L.Convolution2D(64, 32, 3, 1)

            self.fc7 = L.Linear(None, 30)

            self.fc8 = L.Linear(30, n_out)



    def __call__(self, x):

        h = F.leaky_relu(self.conv1(x), slope=0.05)

        h = F.leaky_relu(self.conv2(h), slope=0.05)

        h = F.leaky_relu(self.conv3(h), slope=0.05)

        h = F.leaky_relu(self.conv4(h), slope=0.05)

        h = F.leaky_relu(self.conv5(h), slope=0.05)

        h = F.leaky_relu(self.conv6(h), slope=0.05)

        h = F.leaky_relu(self.fc7(h), slope=0.05)

        h = self.fc8(h)

        return h



    def _predict_batch(self, x_batch):

        with chainer.no_backprop_mode(), chainer.using_config('train', False):

            h = self.__call__(x_batch)

            return F.softmax(h)



    def predict_proba(self, x, batchsize=32, device=-1):

        if device >= 0:

            chainer.cuda.get_device_from_id(device).use()

            self.to_gpu()  # Copy the model to the GPU



        y_list = []

        for i in range(0, len(x), batchsize):

            x_batch = concat_examples(x[i:i + batchsize], device=device)

            y = self._predict_batch(x_batch)

            y_list.append(chainer.cuda.to_cpu(y.data))

        y_array = np.concatenate(y_list, axis=0)

        return y_array



    def predict(self, x, batchsize=32, device=-1):

        proba = self.predict_proba(x, batchsize=batchsize, device=device)

        return np.argmax(proba, axis=1)
if DEBUG:

    print('DEBUG mode, reduce training data...')

    # Use only first 1000 example to reduce training time

    train_x = train_x[:1000]

    train_imgs = train_imgs[:1000]

    train_y = train_y[:1000]

else:

    print('No DEBUG mode')
from chainer import iterators, training, optimizers, serializers

from chainer.datasets import TupleDataset

from chainer.training import extensions





# -1 indicates to use CPU, 

# positive value indicates GPU device id.

device = -1  # If you use CPU.

#device = 0  # If you use GPU. (You need to install chainer & cupy with CUDA/cudnn installed)

batchsize = 16

class_num = 10

out_dir = '.'

if DEBUG:

    epoch = 5  # This value is small. Change to more than 20 for Actual running.

else:

    epoch = 20





def train_main(train_x, train_y, val_x, val_y, model_path='cnn_model.npz'):

    # 1. Setup model    

    model = CNNMedium(n_out=class_num)

    classifier_model = L.Classifier(model)

    if device >= 0:

        chainer.cuda.get_device(device).use()  # Make a specified GPU current

        classifier_model.to_gpu()  # Copy the model to the GPU



    # 2. Setup an optimizer

    optimizer = optimizers.Adam()

    #optimizer = optimizers.MomentumSGD(lr=0.001)

    optimizer.setup(classifier_model)



    # 3. Load the dataset

    #train_data = MNISTTrainImageDataset()

    train_dataset = TupleDataset(train_x, train_y)

    val_dataset = TupleDataset(val_x, val_y)



    # 4. Setup an Iterator

    train_iter = iterators.SerialIterator(train_dataset, batchsize)

    #train_iter = iterators.MultiprocessIterator(train, args.batchsize, n_prefetch=10)

    val_iter = iterators.SerialIterator(val_dataset, batchsize, repeat=False, shuffle=False)

    

    # 5. Setup an Updater

    updater = training.StandardUpdater(train_iter, optimizer, 

                                       device=device)

    # 6. Setup a trainer (and extensions)

    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out_dir)



    # Evaluate the model with the test dataset for each epoch

    trainer.extend(extensions.Evaluator(val_iter, classifier_model, device=device), trigger=(1, 'epoch'))



    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PlotReport(

        ['main/loss', 'validation/main/loss'],

        x_key='epoch', file_name='loss.png'))

    trainer.extend(extensions.PlotReport(

        ['main/accuracy', 'validation/main/accuracy'],

        x_key='epoch',

        file_name='accuracy.png'))



    try:

        # Use extension library, chaineripy's PrintReport & ProgressBar

        from chaineripy.extensions import PrintReport, ProgressBar

        trainer.extend(ProgressBar(update_interval=5))

        trainer.extend(PrintReport(

            ['epoch', 'main/loss', 'validation/main/loss',

             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))



    except:

        print('chaineripy is not installed, run `pip install chaineripy` to show rich UI progressbar')

        # Use chainer's original ProgressBar & PrintReport

        # trainer.extend(extensions.ProgressBar(update_interval=5))

        trainer.extend(extensions.PrintReport(

            ['epoch', 'main/loss', 'validation/main/loss',

             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))



    # Resume from a snapshot

    # serializers.load_npz(args.resume, trainer)



    # Run the training

    trainer.run()

    # Save the model

    serializers.save_npz('{}/{}'

                         .format(out_dir, model_path), model)

    return model
import numpy as np

from sklearn.model_selection import train_test_split

seed = 777

model_simple = 'cnn_model_simple.npz'



train_idx, val_idx = train_test_split(np.arange(len(train_x)),

                                      test_size=0.20, random_state=seed)

print('train size', len(train_idx), 'val size', len(val_idx))

train_main(train_imgs[train_idx], train_y[train_idx], train_imgs[val_idx], train_y[val_idx], model_path=model_simple)
class_num = 10



def predict_main(model_path='cnn_model.npz'):

    # 1. Setup model

    model = CNNMedium(n_out=class_num)

    classifier_model = L.Classifier(model)

    if device >= 0:

        chainer.cuda.get_device(device).use()  # Make a specified GPU current

        classifier_model.to_gpu()  # Copy the model to the GPU



    # load trained model

    serializers.load_npz(model_path, model)



    # 2. Prepare the dataset --> it's already prepared

    # test_imgs



    # 3. predict the result

    t = model.predict(test_imgs, device=device)

    return t

 



def create_submission(submission_path, t):

    result_dict = {

        'ImageId': np.arange(1, len(t) + 1),

        'Label': t

    }

    df = pd.DataFrame(result_dict)

    df.to_csv(submission_path,

              index_label=False, index=False)

    print('submission file saved to {}'.format(submission_path))
predict_label = predict_main(model_path=model_simple)

print('predict_label = ', predict_label, predict_label.shape)



create_submission('submission_simple.csv', predict_label)
from chainer.datasets import TransformDataset

import skimage

import skimage.transform

from skimage.transform import AffineTransform, warp

import numpy as np



def affine_image(img):

    #ch, h, w = img.shape

    #img = img / 255.

    

    # --- scale ---

    min_scale = 0.8

    max_scale = 1.2

    sx = np.random.uniform(min_scale, max_scale)

    sy = np.random.uniform(min_scale, max_scale)

    

    # --- rotation ---

    max_rot_angle = 7

    rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.

    

    # --- shear ---

    max_shear_angle = 10

    shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.

    

    # --- translation ---

    max_translation = 4

    tx = np.random.randint(-max_translation, max_translation)

    ty = np.random.randint(-max_translation, max_translation)

    

    tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle, 

                            translation=(tx, ty))

    transformed_image = warp(img[0, :, :], tform.inverse, output_shape=(28, 28))

    return transformed_image



transformed_imgs = TransformDataset(train_imgs / 255., affine_image)



print('Affine transformation, image: ', transformed_imgs[0].shape)

#print(train_imgs[0])

print('Original image')

show_image(train_imgs[3])

print('Transformed image')

show_image(transformed_imgs[3])

show_image(transformed_imgs[3])
class MNISTTrainImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, imgs, labels, train=True, augmentation_rate=1.0,

                 min_scale=0.90, max_scale=1.10, max_rot_angle=4,

                 max_shear_angle=2, max_translation=3):

        self.imgs = imgs.reshape((-1, 1, 28, 28))

        self.labels = labels

        self.train = train



        # affine parameters

        self.augmentation_rate = augmentation_rate

        self.min_scale = min_scale  # 0.85

        self.max_scale = max_scale  # 1.15

        self.max_rot_angle = max_rot_angle  # 5

        self.max_shear_angle = max_shear_angle  # 5

        self.max_translation = max_translation



    def __len__(self):

        """return length of this dataset"""

        return len(self.labels)



    def affine_image(self, img):

        # ch, h, w = img.shape



        # --- scale ---

        sx = np.random.uniform(self.min_scale, self.max_scale)

        sy = np.random.uniform(self.min_scale, self.max_scale)



        # --- rotation ---

        rot_angle = np.random.uniform(-self.max_rot_angle,

                                      self.max_rot_angle) * np.pi / 180.



        # --- shear ---

        shear_angle = np.random.uniform(-self.max_shear_angle,

                                        self.max_shear_angle) * np.pi / 180.



        # --- translation ---

        tx = np.random.randint(-self.max_translation, self.max_translation)

        ty = np.random.randint(-self.max_translation, self.max_translation)



        tform = AffineTransform(scale=(sx, sy), rotation=rot_angle,

                                shear=shear_angle,

                                translation=(tx, ty))

        transformed_image = warp(img[0, :, :], tform.inverse,

                                 output_shape=(28, 28))

        return transformed_image.astype('float32').reshape(1, 28, 28)



    def get_example(self, i):

        """Return i-th data"""

        img = self.imgs[i]

        label = self.labels[i]



        # Data augmentation...

        if self.train:

            if np.random.uniform() < self.augmentation_rate:

                img = self.affine_image(img)



        return img, label
# 3. Load the dataset

train_data = MNISTTrainImageDataset(train_imgs, train_y)
# train_data[i] is `i`-th dataset, with format (img, label)



# extract 3rd dataset

index = 3

img, label = train_data[index]



show_image(train_data[index][0])

show_image(train_data[index][0])

show_image(train_data[index][0])
# -1 indicates to use CPU, 

# positive value indicates GPU device id.

device = -1  # If you use CPU.

#device = 0  # If you use GPU. (You need to install chainer & cupy with CUDA/cudnn installed)



batchsize = 16

class_num = 10

out_dir = '.'

if DEBUG:

    epoch = 5  # This value is small. Change to more than 20 for Actual running.

else:

    epoch = 30





def train_main2(train_x, train_y, val_x, val_y, model_path='cnn_model.npz', model_class=CNNMedium):

    # 1. Setup model

    model = model_class(n_out=class_num)

    classifier_model = L.Classifier(model)

    if device >= 0:

        chainer.cuda.get_device(device).use()  # Make a specified GPU current

        classifier_model.to_gpu()  # Copy the model to the GPU



    # 2. Setup an optimizer

    optimizer = optimizers.Adam()

    # optimizer = optimizers.MomentumSGD(lr=0.01)

    optimizer.setup(classifier_model)



    # 3. Load the dataset

    # --- Use custom dataset to train model with data augmentation ---

    train_dataset = MNISTTrainImageDataset(train_x, train_y, augmentation_rate=0.5,

                                           min_scale=0.95, max_scale=1.05, max_rot_angle=7,

                                           max_shear_angle=3, max_translation=2)

    val_dataset = MNISTTrainImageDataset(val_x, val_y, train=False)

    # --- end of modification ---



    # 4. Setup an Iterator

    train_iter = iterators.SerialIterator(train_dataset, batchsize)

    #train_iter = iterators.MultiprocessIterator(train, args.batchsize, n_prefetch=10)

    val_iter = iterators.SerialIterator(val_dataset, batchsize, repeat=False, shuffle=False)

    

    # 5. Setup an Updater

    updater = training.StandardUpdater(train_iter, optimizer, 

                                       device=device)

    # 6. Setup a trainer (and extensions)

    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out_dir)



    # Evaluate the model with the test dataset for each epoch

    trainer.extend(extensions.Evaluator(val_iter, classifier_model, device=device), trigger=(1, 'epoch'))



    # --- Learning rate decay scheduling ---

    def decay_lr(trainer):

        print('decay_lr at epoch {}'.format(trainer.updater.epoch_detail))

        # optimizer.lr *= 0.1  # for MomentumSGD optimizer

        optimizer.alpha *= 0.1

        print('optimizer lr has changed to {}'.format(optimizer.lr))

    trainer.extend(decay_lr, 

                   trigger=chainer.training.triggers.ManualScheduleTrigger([10, 20], 'epoch'))

    # --- end of modification ---



    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PlotReport(

        ['main/loss', 'validation/main/loss'],

        x_key='epoch', file_name='loss.png'))

    trainer.extend(extensions.PlotReport(

        ['main/accuracy', 'validation/main/accuracy'],

        x_key='epoch',

        file_name='accuracy.png'))



    try:

        # Use extension library, chaineripy's PrintReport & ProgressBar

        from chaineripy.extensions import PrintReport, ProgressBar

        trainer.extend(ProgressBar(update_interval=5))

        trainer.extend(PrintReport(

            ['epoch', 'main/loss', 'validation/main/loss',

             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))



    except:

        print('chaineripy is not installed, run `pip install chaineripy` to show rich UI progressbar')

        # Use chainer's original ProgressBar & PrintReport

        # trainer.extend(extensions.ProgressBar(update_interval=5))

        trainer.extend(extensions.PrintReport(

            ['epoch', 'main/loss', 'validation/main/loss',

             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))



    # Resume from a snapshot

    # serializers.load_npz(args.resume, trainer)



    # Run the training

    trainer.run()

    # Save the model

    serializers.save_npz('{}/{}'.format(out_dir, model_path), model)

                         

    return model
from sklearn.model_selection import KFold, StratifiedKFold



if DEBUG:

    N_SPLIT_CV = 2

else:

    N_SPLIT_CV = 5



cv_step = 0

# for train_idx, valid_idx in StratifiedKFold(n_splits=N_SPLIT_CV, shuffle=True, random_state=7).split(train_imgs, train_y):

for train_idx, valid_idx in StratifiedKFold(n_splits=N_SPLIT_CV).split(train_imgs, train_y):

    print('Training cv={} ...'.format(cv_step))

    train_main2(train_imgs[train_idx], train_y[train_idx], 

                train_imgs[val_idx], train_y[val_idx], 

                model_path='cnn_model_cv{}.npz'.format(cv_step))

    cv_step += 1
class_num = 10

# device= -1



def predict_proba_main(model_path='cnn_model.npz'):

    # 1. Setup model

    model = CNNMedium(n_out=class_num)

    classifier_model = L.Classifier(model)

    if device >= 0:

        chainer.cuda.get_device(device).use()  # Make a specified GPU current

        classifier_model.to_gpu()  # Copy the model to the GPU



    # load trained model

    serializers.load_npz(model_path, model)



    # 2. Load the dataset

    # test_imgs is already prepared



    # 3. predict the result

    proba = model.predict_proba(test_imgs, device=device)

    return proba





def create_submission(submission_path, t):

    result_dict = {

        'ImageId': np.arange(1, len(t) + 1),

        'Label': t

    }

    df = pd.DataFrame(result_dict)

    df.to_csv(submission_path,

              index_label=False, index=False)

    print('submission file saved to {}'.format(submission_path))
proba_list = []

for i in range(N_SPLIT_CV):

    print('predicting {}-th model...'.format(i))

    proba = predict_proba_main(model_path='./cnn_model_cv{}.npz'.format(i))

    proba_list.append(proba)

  

proba_array = np.array(proba_list)
proba_ensemble = np.mean(proba_array, axis=0)  # Take each model's mean as ensembled prediction.

predict_ensemble = np.argmax(proba_ensemble, axis=1)



# --- Check shape ---

# 0th axis represents each model, 1st axis represents test data index, 2nd axis represents the probability of each label

print('proba_array', proba_array.shape)

# 0th axis represents test data index, 1st axis represents the probability of each label

print('proba_ensemble', proba_ensemble.shape)

# 0th axis represents final label prediction for test data index

print('predict_ensemble', predict_ensemble.shape, predict_ensemble)
create_submission('submission_ensemble.csv', predict_ensemble)
from chainer import cuda



def calc_loss_and_prob(model_path='cnn_model.npz'):

    # 1. Setup model

    model = CNNMedium(n_out=class_num)

    classifier_model = L.Classifier(model)

    if device >= 0:

        chainer.cuda.get_device(device).use()  # Make a specified GPU current

        classifier_model.to_gpu()  # Copy the model to the GPU



    # load trained model

    serializers.load_npz(model_path, model)



    # 2. Load the dataset

    # test_imgs is already prepared



    # 3. predict the result

    if device >= 0:

        h = model(cuda.to_gpu(train_imgs))

        loss = F.softmax_cross_entropy(h, cuda.to_gpu(train_y), reduce='no')

        return cuda.to_cpu(loss.data), cuda.to_cpu(h.data)

    else:

        h = model(train_imgs)

        loss = F.softmax_cross_entropy(h, train_y, reduce='no')

        return loss.data, h.data
loss_list = []

prob_list = []

for i in range(N_SPLIT_CV):

    print('calc loss for {}-th model...'.format(i))

    loss, prob = calc_loss_and_prob(model_path='./cnn_model_cv{}.npz'.format(i))

    loss_list.append(loss)

    prob_list.append(prob)



loss_array = np.array(loss_list)

prob_array = np.array(prob_list)
loss_ensemble = np.mean(loss_array, axis=0)  # Take each model's mean loss.

predict_ensemble = np.mean(prob_array, axis=0)  # Take each model's mean as ensembled prediction.



# --- Check shape ---

# 0th axis represents each model, 1st axis represents test data index, 2nd axis represents the probability of each label

print('loss_array', loss_array.shape)

# 0th axis represents test data index, 1st axis represents the probability of each label

print('loss_ensemble', loss_ensemble.shape)
loss_index = np.argsort(loss_ensemble)

print('BEST100:  ', loss_index[:30], loss_ensemble[loss_index[:30]])

print('WORST100: ', loss_index[::-1][:100], loss_ensemble[loss_index[::-1][:100]])
def show_images(imgs):

    num_imgs = len(imgs)

    fig, axs = plt.subplots(nrows=1, ncols=num_imgs)

    # plt.figure(figsize=(1.5, 1.5))

    plt.axis('off')

    #if img.ndim == 3:

    #    img = img[0, :, :]

    print('imgs shape', imgs.shape)

    for i in range(num_imgs):

        axs[i].imshow(imgs[i, 0, :, :], cmap=plt.cm.binary)           

        axs[i].axis('off')

    plt.show()
worst_index = loss_index[::-1][:10]

print('label ', train_y[worst_index])

print('predict ', np.argmax(predict_ensemble, axis=1)[worst_index])

show_images(train_imgs[worst_index])



#for i in loss_index[::-1][:10]:

#    print('label ', train_y[i])

#    show_images(train_imgs[])