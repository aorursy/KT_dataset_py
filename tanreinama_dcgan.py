import chainer

import chainer.functions as F

import chainer.links as L

from chainer import training, datasets, iterators, optimizers

from chainer.training import extensions

import numpy as np

import os

import math

from numpy import random

from PIL import Image



batch_size = 10

uses_device = 0

image_size = 128

neuron_size = 64

n_epochs = 10000



stand_left = ['n02087046_2744.jpg','n02087046_924.jpg','n02087394_2319.jpg','n02087394_7254.jpg','n02087394_9695.jpg','n02088364_16881.jpg','n02088364_5147.jpg','n02088466_12388.jpg','n02088632_1463.jpg','n02088632_2145.jpg','n02089078_1174.jpg','n02089078_1735.jpg','n02089078_2542.jpg','n02089078_2574.jpg','n02089078_2921.jpg','n02089078_3923.jpg','n02089078_393.jpg','n02089078_877.jpg','n02089078_901.jpg','n02089867_2688.jpg','n02089867_3044.jpg','n02089867_3524.jpg','n02099267_1862.jpg','n02099267_198.jpg','n02099267_2121.jpg','n02099267_5073.jpg','n02099429_1234.jpg','n02099429_1377.jpg','n02099429_159.jpg','n02099429_1758.jpg','n02099429_2650.jpg','n02099429_2698.jpg','n02099429_2756.jpg','n02099429_355.jpg','n02099429_448.jpg','n02099429_537.jpg','n02099429_618.jpg','n02099601_1010.jpg','n02099601_3414.jpg','n02099849_501.jpg','n02100236_2392.jpg','n02100236_4723.jpg','n02100236_4755.jpg','n02100735_5713.jpg','n02100735_6660.jpg','n02100877_1062.jpg','n02101006_114.jpg','n02101006_751.jpg','n02101388_2142.jpg','n02101388_2324.jpg','n02101388_2522.jpg','n02101388_3003.jpg','n02101388_4229.jpg','n02101388_4632.jpg','n02101388_6081.jpg','n02102177_1520.jpg','n02102973_2449.jpg','n02102973_2805.jpg','n02102973_3584.jpg','n02105162_4569.jpg','n02105251_8643.jpg','n02106166_1460.jpg','n02106166_321.jpg','n02106166_6833.jpg','n02106550_6286.jpg','n02106662_10122.jpg','n02106662_15858.jpg','n02106662_1841.jpg','n02106662_21715.jpg','n02106662_26335.jpg','n02106662_27186.jpg','n02106662_4522.jpg','n02107142_15377.jpg','n02107142_16400.jpg','n02107142_16917.jpg','n02107142_278.jpg','n02107142_3094.jpg','n02107142_3171.jpg','n02107312_1586.jpg','n02107312_3449.jpg','n02107908_1855.jpg','n02107908_2365.jpg','n02108000_2357.jpg','n02108000_2536.jpg','n02108000_3207.jpg','n02108000_3305.jpg','n02108000_3500.jpg','n02108089_122.jpg','n02108089_14112.jpg','n02108089_5977.jpg','n02108422_3576.jpg','n02108422_3709.jpg','n02108422_5609.jpg','n02109047_10414.jpg','n02109047_1533.jpg','n02109047_16735.jpg','n02109047_2527.jpg','n02109047_2553.jpg','n02109047_32010.jpg','n02109047_34162.jpg','n02109047_4267.jpg','n02109525_7497.jpg','n02109961_1076.jpg','n02109961_1235.jpg','n02109961_5035.jpg','n02109961_997.jpg','n02110063_11887.jpg','n02110063_5676.jpg','n02110063_9112.jpg','n02110063_9259.jpg','n02110185_10967.jpg','n02110185_4294.jpg','n02110185_6263.jpg','n02110185_6850.jpg','n02110185_7936.jpg','n02110627_11819.jpg','n02110627_11875.jpg','n02110627_12077.jpg','n02110627_12973.jpg','n02110806_1485.jpg','n02110806_1577.jpg','n02110806_1637.jpg','n02110806_1639.jpg','n02110806_1902.jpg','n02110806_2497.jpg','n02110806_2627.jpg','n02110806_2995.jpg','n02110806_2997.jpg','n02110806_3008.jpg','n02110806_3966.jpg','n02110806_4024.jpg','n02110806_4142.jpg','n02110806_4242.jpg','n02110806_513.jpg','n02110806_581.jpg','n02111129_1014.jpg','n02111129_1181.jpg','n02111129_1429.jpg','n02111129_1684.jpg','n02111129_2047.jpg','n02111129_2359.jpg','n02111129_2400.jpg','n02111129_2594.jpg','n02111129_2620.jpg','n02111129_2700.jpg','n02111129_2750.jpg','n02111129_2881.jpg','n02111129_3087.jpg','n02111129_4305.jpg','n02111129_4698.jpg','n02111129_4725.jpg','n02111129_4958.jpg','n02111129_686.jpg','n02111277_13070.jpg','n02111277_5845.jpg','n02111277_5932.jpg','n02111277_6017.jpg','n02111277_980.jpg','n02112706_2149.jpg','n02112706_442.jpg','n02112706_473.jpg','n02112706_762.jpg','n02113978_131.jpg','n02113978_1823.jpg','n02113978_1868.jpg','n02113978_1939.jpg','n02113978_1970.jpg','n02113978_1983.jpg','n02113978_2143.jpg','n02113978_2441.jpg','n02113978_2707.jpg','n02113978_2798.jpg','n02113978_3419.jpg','n02113978_3722.jpg','n02113978_386.jpg','n02113978_903.jpg','n02113978_937.jpg','n02113978_961.jpg','n02113978_996.jpg','n02116738_10038.jpg','n02116738_10215.jpg','n02116738_10614.jpg','n02116738_1739.jpg','n02116738_8489.jpg','n02116738_8719.jpg']



if uses_device >= 0:

    import cupy as cp

    import chainer.cuda

else:

    cp = np



class DCGAN_Generator_NN(chainer.Chain):



    def __init__(self):

        w = chainer.initializers.Normal(scale=0.02, dtype=None)

        super(DCGAN_Generator_NN, self).__init__()

        with self.init_scope():

            self.l0 = L.Linear(100, neuron_size * image_size * image_size // 8 // 8,

                               initialW=w)

            self.dc1 = L.Deconvolution2D(neuron_size, neuron_size // 2, 4, 2, 1, initialW=w)

            self.dc2 = L.Deconvolution2D(neuron_size // 2, neuron_size // 4, 4, 2, 1, initialW=w)

            self.dc3 = L.Deconvolution2D(neuron_size // 4, neuron_size // 8, 4, 2, 1, initialW=w)

            self.dc4 = L.Deconvolution2D(neuron_size // 8, 3, 3, 1, 1, initialW=w)

            self.bn0 = L.BatchNormalization(neuron_size * image_size * image_size // 8 // 8)

            self.bn1 = L.BatchNormalization(neuron_size // 2)

            self.bn2 = L.BatchNormalization(neuron_size // 4)

            self.bn3 = L.BatchNormalization(neuron_size // 8)



    def __call__(self, z):

        shape = (len(z), neuron_size, image_size // 8, image_size // 8)

        h = F.reshape(F.relu(self.bn0(self.l0(z))), shape)

        h = F.relu(self.bn1(self.dc1(h)))

        h = F.relu(self.bn2(self.dc2(h)))

        h = F.relu(self.bn3(self.dc3(h)))

        x = F.sigmoid(self.dc4(h))

        return x



class DCGAN_Discriminator_NN(chainer.Chain):



    def __init__(self):

        w = chainer.initializers.Normal(scale=0.02, dtype=None)

        super(DCGAN_Discriminator_NN, self).__init__()

        with self.init_scope():

            self.c0_0 = L.Convolution2D(3, neuron_size //  8, 3, 1, 1, initialW=w)

            self.c0_1 = L.Convolution2D(neuron_size //  8, neuron_size // 4, 4, 2, 1, initialW=w)

            self.c1_0 = L.Convolution2D(neuron_size //  4, neuron_size // 4, 3, 1, 1, initialW=w)

            self.c1_1 = L.Convolution2D(neuron_size //  4, neuron_size // 2, 4, 2, 1, initialW=w)

            self.c2_0 = L.Convolution2D(neuron_size //  2, neuron_size // 2, 3, 1, 1, initialW=w)

            self.c2_1 = L.Convolution2D(neuron_size //  2, neuron_size, 4, 2, 1, initialW=w)

            self.c3_0 = L.Convolution2D(neuron_size, neuron_size, 3, 1, 1, initialW=w)

            self.l4 = L.Linear(neuron_size * image_size * image_size // 8 // 8, 1, initialW=w)

            self.bn0_1 = L.BatchNormalization(neuron_size // 4, use_gamma=False)

            self.bn1_0 = L.BatchNormalization(neuron_size // 4, use_gamma=False)

            self.bn1_1 = L.BatchNormalization(neuron_size // 2, use_gamma=False)

            self.bn2_0 = L.BatchNormalization(neuron_size // 2, use_gamma=False)

            self.bn2_1 = L.BatchNormalization(neuron_size, use_gamma=False)

            self.bn3_0 = L.BatchNormalization(neuron_size, use_gamma=False)



    def __call__(self, x):

        h = F.leaky_relu(self.c0_0(x))

        h = F.dropout(F.leaky_relu(self.bn0_1(self.c0_1(h))),ratio=0.2)

        h = F.dropout(F.leaky_relu(self.bn1_0(self.c1_0(h))),ratio=0.2)

        h = F.dropout(F.leaky_relu(self.bn1_1(self.c1_1(h))),ratio=0.2)

        h = F.dropout(F.leaky_relu(self.bn2_0(self.c2_0(h))),ratio=0.2)

        h = F.dropout(F.leaky_relu(self.bn2_1(self.c2_1(h))),ratio=0.2)

        h = F.dropout(F.leaky_relu(self.bn3_0(self.c3_0(h))),ratio=0.2)

        return self.l4(h)



class DCGANUpdater(training.StandardUpdater):



    def __init__(self, train_iter, optimizer, device):

        super(DCGANUpdater, self).__init__(

            train_iter,

            optimizer,

            device=device

        )

    

    def loss_dis(self, dis, y_fake, y_real):

        batchsize = len(y_fake)

        L1 = F.sum(F.softplus(-y_real)) / batchsize

        L2 = F.sum(F.softplus(y_fake)) / batchsize

        loss = L1 + L2

        return loss



    def loss_gen(self, gen, y_fake):

        batchsize = len(y_fake)

        loss = F.sum(F.softplus(-y_fake)) / batchsize

        return loss



    def update_core(self):

        batch = self.get_iterator('main').next()

        src = self.converter(batch, self.device)

        

        optimizer_gen = self.get_optimizer('opt_gen')

        optimizer_dis = self.get_optimizer('opt_dis')

        gen = optimizer_gen.target

        dis = optimizer_dis.target



        rnd = random.uniform(-1, 1, (src.shape[0], 100))

        rnd = cp.array(rnd, dtype=cp.float32)

        

        x_fake = gen(rnd)

        y_fake = dis(x_fake)

        y_real = dis(src)



        optimizer_dis.update(self.loss_dis, dis, y_fake, y_real)

        optimizer_gen.update(self.loss_gen, gen, y_fake)

        

model_gen = DCGAN_Generator_NN()

model_dis = DCGAN_Discriminator_NN()



if uses_device >= 0:

    chainer.cuda.get_device_from_id(0).use()

    chainer.cuda.check_cuda_available()

    model_gen.to_gpu()

    model_dis.to_gpu()



images = []



import cv2



for fn in stand_left:

    if os.path.isfile("../input/all-dogs/all-dogs/"+fn):

        im = cv2.imread("../input/all-dogs/all-dogs/"+fn)

        im = cv2.resize(im, (image_size,image_size), cv2.INTER_CUBIC).astype(np.float32) / 255.0

        images.append(im.transpose((2,0,1)))



train_iter = iterators.SerialIterator(images, batch_size, shuffle=True)



optimizer_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)

optimizer_gen.setup(model_gen)

optimizer_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)

optimizer_dis.setup(model_dis)



updater = DCGANUpdater(train_iter, \

        {'opt_gen':optimizer_gen, 'opt_dis':optimizer_dis}, \

        device=uses_device)

trainer = training.Trainer(updater, (n_epochs, 'epoch'), out="result")

trainer.extend(extensions.ProgressBar(update_interval=7400))



trainer.run()
from matplotlib import pyplot as plt



model = model_gen

num_generate = 200



rnd_to_gen = random.uniform(-1, 1, (num_generate, 100, 1, 1))

rnd = cp.array(rnd_to_gen, dtype=cp.float32)



with chainer.using_config('train', False):

    result_gen = model(rnd)



data = np.zeros((640, 1280, 3), dtype=np.uint8)



for i in range(10):

    for j in range(20):

        dst = result_gen.data[i*10+j] * 255.0

        if uses_device >= 0:

            dst = chainer.cuda.to_cpu(dst)

        tmp = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        tmp[:,:,0] = dst[0]; tmp[:,:,1] = dst[1]; tmp[:,:,2] = dst[2]

        data[i*64:i*64+64,j*64:j*64+64,:] = cv2.resize(tmp, (64,64), cv2.INTER_CUBIC)

plt.figure(figsize=(40, 20), dpi=50)

plt.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

plt.show()
others = []

for fn in os.listdir("../input/all-dogs/all-dogs/"):

    if fn not in stand_left and os.path.isfile("../input/all-dogs/all-dogs/"+fn):

        im = cv2.imread("../input/all-dogs/all-dogs/"+fn)

        im = cv2.resize(im, (128,128), cv2.INTER_CUBIC).astype(np.float32) / 255.0

        others.append(im.transpose((2,0,1)))

        if len(others) >= 1000:

            break



triplet_pos = [0,0]

def get_one(o=False):

    global triplet_pos

    im = images if o else others

    ip = 0 if o else 1

    data = im[triplet_pos[ip]]

    triplet_pos[ip] = triplet_pos[ip]+1

    if triplet_pos[ip] >= len(im):

        triplet_pos[ip] = 0

    return data

def get_one_triple():

    a = (random.random() < 0.5)

    c = get_one(a)

    d = get_one(not a)

    e = get_one(not a)

    return (c,d,e)



class Triplet_NN(chainer.Chain):

 

    def __init__(self):

        super(Triplet_NN, self).__init__()

        with self.init_scope():

            self.layer1 = L.Convolution2D(3, 16, 3, 1, 1)

            self.layer2 = L.Convolution2D(16, 16, 3, 1, 1)

            self.layer3 = L.Convolution2D(16, 32, 3, 1, 1)

            self.layer4 = L.Convolution2D(32, 32, 3, 1, 1)

            self.layer5 = L.Linear(32*32*32, 2)

 

    def __call__(self, x):

        x = F.relu(self.layer1(x))

        x = F.relu(self.layer2(x))

        x = F.max_pooling_2d(x, 2)

        x = F.relu(self.layer3(x))

        x = F.relu(self.layer4(x))

        x = F.max_pooling_2d(x, 2)

        return self.layer5(x)



class TripletUpdater(training.StandardUpdater):

 

    def __init__(self, optimizer, device):

        self.loss_val = []

        super(TripletUpdater, self).__init__(

            None,

            optimizer,

            device=device

        )

 

    @property

    def epoch(self):

        return 0

 

    @property

    def epoch_detail(self):

        return 0.0

 

    @property

    def previous_epoch_detail(self):

        return 0.0

 

    @property

    def is_new_epoch(self):

        return False

        

    def finalize(self):

        pass

    

    def update_core(self):

        batch_size = 1000

        optimizer = self.get_optimizer('main')

        anchor = []

        positive = []

        negative = []

        for i in range(batch_size):

            in_data = get_one_triple()

            anchor.append(in_data[0])

            positive.append(in_data[1])

            negative.append(in_data[2])

        anchor = cp.array(anchor)

        positive = cp.array(positive)

        negative = cp.array(negative)

        model = optimizer.target

        anchor_r = model(anchor)

        positive_r = model(positive)

        negative_r = model(negative)

        optimizer.update(F.triplet, anchor_r, positive_r, negative_r)



model_tri = Triplet_NN().to_gpu()

optimizer = optimizers.Adam()

optimizer.setup(model_tri)

updater = TripletUpdater(optimizer, device=0)

trainer = training.Trainer(updater, (1000, 'iteration'), out="result")

trainer.run()
with chainer.using_config('train', False):

    vectors_gen = model_tri(cp.array(result_gen.data)).data

with chainer.using_config('train', False):

    vectors_train = model_tri(cp.array(images)).data

min_deltas = []

min_deltaindex = []

for vt in vectors_train:

    md = np.inf

    mnd_index = 0

    for i in range(num_generate):

        delta = np.sum(np.absolute(vt - vectors_gen[i]))

        if delta < md:

            md = delta

            mnd_index = i

    min_deltas.append(md)

    min_deltaindex.append(mnd_index)

pi = np.argsort(min_deltas)

picture1idx = min_deltaindex[pi[0]]

gi = 1

for i in range(gi,len(pi)):

    if min_deltaindex[pi[i]] != picture1idx:

        picture2idx = min_deltaindex[pi[i]]

        gi = i+1

        break

for i in range(gi,len(pi)):

    if min_deltaindex[pi[i]] != picture1idx and  min_deltaindex[pi[i]] != picture2idx:

        picture3idx = min_deltaindex[pi[i]]

        gi = i+1

        break

for i in range(gi,len(pi)):

    if min_deltaindex[pi[i]] != picture1idx and  min_deltaindex[pi[i]] != picture2idx and  min_deltaindex[pi[i]] != picture3idx:

        picture4idx = min_deltaindex[pi[i]]

        break
model = model_gen

rnd = np.zeros((num_generate, 100, 1, 1))

for i in range(num_generate):

    rnd[i] = rnd_to_gen[picture1idx] + (rnd_to_gen[picture2idx] - rnd_to_gen[picture1idx]) * i / num_generate

rnd = cp.array(rnd, dtype=cp.float32)



with chainer.using_config('train', False):

    result = model(rnd)



data = np.zeros((640, 640, 3), dtype=np.uint8)



for i in range(10):

    for j in range(10):

        dst = result.data[i*10+j] * 255.0

        if uses_device >= 0:

            dst = chainer.cuda.to_cpu(dst)

        tmp = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        tmp[:,:,0] = dst[0]; tmp[:,:,1] = dst[1]; tmp[:,:,2] = dst[2]

        data[i*64:i*64+64,j*64:j*64+64,:] = cv2.resize(tmp, (64,64), cv2.INTER_CUBIC)

plt.figure(figsize=(20, 20), dpi=50)

plt.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

plt.show()
model = model_gen

rnd = np.zeros((num_generate, 100, 1, 1))

for i in range(num_generate):

    rnd[i] = rnd_to_gen[picture3idx] + (rnd_to_gen[picture4idx] - rnd_to_gen[picture3idx]) * i / num_generate

rnd = cp.array(rnd, dtype=cp.float32)



with chainer.using_config('train', False):

    result = model(rnd)



data = np.zeros((640, 640, 3), dtype=np.uint8)



for i in range(10):

    for j in range(10):

        dst = result.data[i*10+j] * 255.0

        if uses_device >= 0:

            dst = chainer.cuda.to_cpu(dst)

        tmp = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        tmp[:,:,0] = dst[0]; tmp[:,:,1] = dst[1]; tmp[:,:,2] = dst[2]

        data[i*64:i*64+64,j*64:j*64+64,:] = cv2.resize(tmp, (64,64), cv2.INTER_CUBIC)

plt.figure(figsize=(20, 20), dpi=50)

plt.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

plt.show()
import zipfile

z = zipfile.PyZipFile('images.zip', mode='w')

generated_num = 0



sample = np.zeros((640, 640, 3), dtype=np.uint8)

model = model_gen

mn = np.min([rnd_to_gen[a] for a in (picture1idx,picture2idx,picture3idx,picture4idx)],axis=0).reshape((100,))

mx = np.max([rnd_to_gen[a] for a in (picture1idx,picture2idx,picture3idx,picture4idx)],axis=0).reshape((100,))

for i in range(100):

    data = np.zeros((64, 64, 3), dtype=np.uint8)



    rnd = mn + (mx - mn) * np.random.rand(100, 100)

    rnd = rnd.reshape((100, 100, 1, 1))

    rnd = cp.array(rnd, dtype=cp.float32)



    with chainer.using_config('train', False):

        result = model(rnd)



    for j in range(100):

        dst = result.data[j] * 255.0

        if uses_device >= 0:

            dst = chainer.cuda.to_cpu(dst)

        tmp = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        tmp[:,:,0] = dst[0]; tmp[:,:,1] = dst[1]; tmp[:,:,2] = dst[2]

        data = cv2.resize(tmp, (64,64), cv2.INTER_CUBIC)

        f = str(generated_num)+'.png'

        cv2.imwrite(f, data); z.write(f); os.remove(f)

        generated_num += 1

    sample[(i//10)*64:(i//10)*64+64,(i%10)*64:(i%10)*64+64,:] = data

z.close()

plt.figure(figsize=(20, 20), dpi=50)

plt.imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))

plt.show()