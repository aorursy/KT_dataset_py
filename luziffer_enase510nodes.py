import torch

from torch.nn import Parameter

import numpy as np

import fccontroller as FCC

import dataa

import torch.optim as optim

import child_model as CM

import torch.nn.functional as F

import tarfile

import time

import utils as U

import matplotlib.pyplot as plt



seed = 42

torch.manual_seed(seed)

np.random.seed(seed)
# extract CIFAR10 dataset

tar = tarfile.open("../input/cifar10-python/cifar-10-python.tar.gz", "r:gz")

tar.extractall()

tar.close()
BATCH_SIZE = 100

TEST_BATCH_SIZE = 200

cifar10 = dataa.CIFAR10(batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, path="./")

test_set = list(cifar10.test)

train_set = list(cifar10.train)
# ENAS run parameters

path = "saved"

nodes, child_samples, epochs, channels = 8, 500, 350, 240

checkpoint_interval = 15

exp_name = "FC_{}_{}_005".format(nodes, child_samples)



# child model training parameters

optimizer=optim.SGD

opt_args={"lr": 0.01, "momentum": 0.8, "weight_decay": 1e-4, "nesterov": True}

train_args={"loss_func": F.cross_entropy, "log_interval": 200, "max_batches": None}



# create controller network

layer_sizes = [100, 200, 300, 200, 100]

fcc = FCC.FCController(nodes, num_child_samples=child_samples, learning_rate=0.005, gamma=0.9,

                       input_amplitude=0.01, allowed_ops=[0, 1, 4, 5], layer_sizes=layer_sizes)



# create shared weights

omega = CM.create_shared_weights(nodes)



# best child and metrics

best_child = None

best_child_acc = 0

best_accs = []

mean_accs = []



plt.figure()

plt.ylabel("accuracy")

plt.xlabel("epochs")



# start ENAS loop

for ep in range(epochs):

    ts = time.time()

    print("Start of ENAS epoch {}".format(ep))

    P_net = fcc.forward()

    cm = FCC.sample_model(P_net, fcc.allowed_ops)

    

    print("Training shared weights ...")

    tcm = cm.to_torch_model(omega, channels)

    opt = optimizer(tcm.parameters(), **opt_args)

    FCC.train1(tcm, train_set, opt, **train_args)

    omega = tcm.get_shared_weights(omega)



    t2 = time.time()

    print("Generating {} child models ...".format(fcc.num_child_samples))



    tcms = []        

    for _ in range(fcc.num_child_samples):

        fcc.zero_grad()

        Pop = fcc.forward()

        cm = FCC.sample_model(Pop, fcc.allowed_ops)

        tcm = cm.to_torch_model(omega, channels)

        tcms.append(tcm)

        fcc.backward(tcm.childmodel, Pop)



    print("Validating child models ...")

    acc = FCC.test_one_batch(tcms, test_set) # test child model performance

    

    # record metrics

    best_ind = np.argmax(acc)

    best_acc = np.max(acc)

    best_accs.append(best_acc)

    mean_acc = np.mean(acc)

    mean_accs.append(mean_acc)

    #print("Best accuracy = {:.2f}%".format(best_acc*100))

    print("Mean accuracy = {:.2f}%".format(mean_acc*100))

    print("Mean of mean accuracies = {:.2f}%".format(np.mean(mean_accs)*100))

    plt.plot(mean_accs)

    plt.show()



    # update best child

    if best_child_acc <= best_acc:

        best_child = tcms[best_ind]

        best_child_acc = best_acc

    

    print("Updating controller weights ...")

    fcc.update_step_naive(acc, baseline=False, log=False) # update controller weights naively

    print("End of ENAS epoch {} took {:.0f} seconds to complete (Step 1 {:.0f}s, Step 2 {:.0f}s)".format(ep, time.time() - ts, t2 - ts, time.time() - t2))



    if ep % checkpoint_interval == 0:

        print("Saving checkpoint at epoch {} ...".format(ep))

        U.save_checkpoint(exp_name, ep, fcc, best_child, omega, best_accs, mean_accs, path=path)
print(mean_accs)
import matplotlib.pyplot as plt

plt.figure()

plt.plot(mean_accs)

plt.xlabel("epoch")

plt.ylabel("accuracy")
chk = U.load_checkpoint(exp_name, path="saved")
chk["best_child_model"].ops
max(chk["best_accs"])
chk