import child_model as CM

import controller as C

import utils as U

import dataa as D



import torch

import torch.optim as optim

import torch.nn.functional as F

import torchvision.datasets as datasets

import torchvision.transforms as transforms



import numpy as np

import time

import os

#os.chdir(r'/kaggle/working')
import tarfile

tar = tarfile.open("../input/cifar10-python/cifar-10-python.tar.gz", "r:gz")

tar.extractall()

tar.close()
cifar10 = D.CIFAR10(batch_size=100, path="./", test_batch_size=200)

test_set = list(cifar10.test)

train_set = list(cifar10.train)
torch.cuda.is_available()
def enas(experiment,

         nodes,

         num_child_samples,

         iterations,

         dataset=cifar10,

         lr=0.01,

         mom=0.8,

         wd=1e-4,

         checkpoint=None,

         max_batches=None,

         checkpoint_interval=5,

         save_path="saved",

         log_interval=20

        ):

    # ENAS epoch

    enas_epoch = 1

    if not checkpoint is None:

        enas_epoch = checkpoint["epoch"] + 1

    

    # create controller  LSTM

    controller = C.Controller(nodes, num_child_samples)

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

    emb = torch.zeros(1, controller.num_hidden, device=U.device)

    

    # track best child

    best_child_acc = 0

    best_child = None

    best_accs = []

    avg_accs = []





    

    # run 'iterations' ENAS steps

    for i in range(iterations):

        print("Start of ENAS epoch {}".format(enas_epoch))

        ts = time.time()

        # sample child models from the controller

        Pop, Psk = controller.forward(emb)

        #print("P_op = ",Pop)

        #print("P_sk = ",Psk)

        cm_step1 = C.sampler(Pop, Psk, 1).get_childmodel(0)

        cm_step2 = C.sampler(Pop, Psk, num_child_samples)

        

        print("Step 1")

        print("Model ops: ",cm_step1.ops)

        print("Model skips: ",cm_step1.skips)

        

        # Step 1: shared weights training

        

        tm_step1 = cm_step1.to_torch_model(omega) # TODO use to_torch_model to combine with set_shared_weights 



        opt = optimizer(tm_step1.parameters(), lr=lr, momentum=mom, weight_decay=wd, nesterov=True)

        C.train1(tm_step1, train_set, opt, loss, max_batches=max_batches, log_interval=log_interval)

        omega = tm_step1.get_shared_weights(omega) # update shared weights

        

        print("Step 2")

        t2 = time.time()

        # Step 2: controller training        

        tm_step2 = cm_step2.to_torch_models(omega) # TODO: readd weights parameter for setting the shared weights

        acc = C.test_one_batch(tm_step2, test_set) # test child model performance

        best_ind = np.argmax(acc)

        best_acc = np.max(acc)

        avg_acc = np.mean(acc)

        print("Best accuracy = {:.0f}%".format(best_acc*100))

        print("Avg accuracy = {:.3f}%".format(avg_acc))

        best_accs.append(best_acc)

        avg_accs.append(avg_acc)

        

        # update best child

        if best_child_acc <= best_acc:

            best_child = tm_step2[best_ind]

            best_child_acc = best_acc

            

        controller.update_step(cm_step2, Pop, Psk, acc) # update controller weights with ADAM

        print("End of ENAS epoch {}".format(enas_epoch))

        print("Took {:.0f} seconds to complete (Step 1 {:.0f}s, Step 2 {:.0f}s)".format((time.time() - ts), t2 - ts, time.time() - t2))

        

        if enas_epoch % checkpoint_interval == 0:

            U.save_checkpoint(experiment, enas_epoch, controller, best_child, omega, best_accs, avg_accs, path=save_path)

        

        enas_epoch += 1
experiment = "long_enas_run1"

fraction = 1.0
#ckpt = load_checkpoint(experiment)int(len(cifar10.train.dataset)/cifar10.train.batch_size*fraction)

enas(experiment, 12, 500, 200, max_batches=None, checkpoint_interval=5, checkpoint=None, log_interval=10, save_path="../working/saved") # nodes, num_child_samples, iterations
#check = U.load_checkpoint("long_enas_run1", path="../working/saved")

#check["epoch"]