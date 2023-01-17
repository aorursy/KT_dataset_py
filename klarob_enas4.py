# scripts from the ENAS project

import child_model as CM

import controller2 as C

import utils as U

import dataa as D



# torch imports

import torch

import torch.optim as optim

import torch.nn.functional as F



# general imports

import numpy as np

import time

import os

import glob

import tarfile



# set RNG seed for reproducibility

seed = 42

torch.manual_seed(seed)

np.random.seed(seed)



# extract CIFAR10 dataset

tar = tarfile.open("../input/cifar10-python/cifar-10-python.tar.gz", "r:gz")

tar.extractall()

tar.close()



# load the dataset

BATCH_SIZE = 100

TEST_BATCH_SIZE = 200

cifar10 = D.CIFAR10(batch_size=BATCH_SIZE, path="./", test_batch_size=TEST_BATCH_SIZE)

test_set = list(iter(cifar10.test))

train_set = list(iter(cifar10.train))
train_set[0][0]
def enas(experiment,

         nodes,

         num_child_samples,

         iterations,

         train_set=train_set,

         test_set=test_set,

         lr=0.01,

         mom=0.8,

         wd=1e-4,

         checkpoint=None,

         max_batches=None,

         checkpoint_interval=5,

         save_path="../saved",

         log_interval=20

        ):

    # ENAS epoch

    enas_epoch = 1

    if not checkpoint is None:

        enas_epoch = checkpoint["epoch"] + 1

    

    # create controller  LSTM

    controller = C.Controller(nodes, num_child_samples, learning_rate=0.007, kl_weight=1, sk_prob_target=0.4)

    if not checkpoint is None:

        controller.load_state_dict(checkpoint["controller"])

    # optimizer for child model training

    optimizer = optim.SGD

    # loss function

    loss = F.cross_entropy

    # initialize shared weights

    if checkpoint is None:

        omega = CM.create_shared_weights(nodes)

    else:

        omega = checkpoint["shared_weights"]

        

    # Controller input (empty embedding)

    emb = torch.zeros(1, controller.num_hidden)

    

    # track best child

    best_child_acc = 0

    best_child = None

    best_accs = []

    mean_accs = []

    

    # run 'iterations' ENAS steps

    for i in range(iterations):

        print("Start of ENAS epoch {}".format(enas_epoch))

        ts = time.time()

        # sample child models from the controller

        _, Psk1, cm_step1 = controller.forward_with_feedback()

        #print("PSK1 ",Psk1)

        #cm_step1 = C.sampler(Pop, Psk, 1).get_childmodel(0)

        #cm_step2 = C.sampler(Pop, Psk, num_child_samples)

        

        print("Step 1")

        print("Model ops: ",cm_step1.ops)

        print("Model skips: ",cm_step1.skips)

        

        # Step 1: shared weights training

        

        tm_step1 = cm_step1.to_torch_model(omega) 



        opt = optimizer(tm_step1.parameters(), lr=lr, momentum=mom, weight_decay=wd, nesterov=True)

        C.train1(tm_step1, train_set, opt, loss, max_batches=max_batches, log_interval=log_interval)

        omega = tm_step1.get_shared_weights(omega) # update shared weights

        

        print("Step 2")

        t2 = time.time()

        # Step 2: controller training

        # sample many child models

        tcms = []

        grads = []

        kl_skips = []

        for csind in range(num_child_samples ):

            Pop, Psk, cm = controller.forward_with_feedback()

            tcm = cm.to_torch_model(omega)

            tcms.append(tcm)

            pgrad, kl_skip = controller.backward(tcm.childmodel, Pop, Psk)

            #print("Skip penalty: ",kl_skip)

            grads.append(pgrad)

            kl_skips.append(kl_skip)

            #print("Skip fraction: ", cm.skips.nonzero().size(0)/cm.skips.size(0))

            

        acc = C.test_one_batch(tcms, test_set) # test child model performance

        best_ind = np.argmax(acc)

        best_acc = np.max(acc)

        best_accs.append(best_acc)

        mean_acc = np.mean(acc)

        mean_accs.append(mean_acc)

        print("Best accuracy = {:.0f}%".format(best_acc*100))

        print("Mean accuracy = {:.0f}%".format(mean_acc*100))



        # update best child

        if best_child_acc <= best_acc:

            best_child = tcms[best_ind]

            best_child_acc = best_acc

            

        #controller.update_step_adam(acc, grads, kl_skips, baseline=False) # update controller weights with ADAM

        controller.update_step_naive(acc, grads, kl_skips, baseline=False) # update controller weights naively

        print("End of ENAS epoch {}".format(enas_epoch))

        print("Took {:.0f} seconds to complete (Step 1 {:.0f}s, Step 2 {:.0f}s)".format((time.time() - ts), t2 - ts, time.time() - t2))

        

        if enas_epoch % checkpoint_interval == 0:

            U.save_checkpoint(experiment, enas_epoch, controller, best_child, best_accs, mean_accs, omega, path=save_path)

        

        enas_epoch += 1

        

    return mean_accs, best_accs
experiment = "enas_naive_with_KL1"

fraction = 1

max_batches = int(len(train_set)*fraction)

#ckpt = load_checkpoint(experiment)

mean_accs, best_accs = enas(experiment, 12, 250, 50, max_batches=max_batches, checkpoint_interval=5, checkpoint=None, log_interval=200) # nodes, num_child_samples, iterations
import matplotlib.pyplot as plt

plt.figure()

plt.plot(mean_accs)

plt.xlabel("epoch")

plt.ylabel("accuracy")

plt.show()
plt.figure()

plt.plot(best_accs)

plt.xlabel("epoch")

plt.ylabel("accuracy")

plt.show()