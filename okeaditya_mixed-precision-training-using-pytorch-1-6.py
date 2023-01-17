! pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!pip install timm
# PyTorch Image model from Ross Wightman
import timm
# All these models can be used with this code.
# print(timm.list_models())
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import os
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv",dtype = np.float32)
train.head()
targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,targets_numpy,
                                                                              test_size = 0.2,random_state = 2) 
X_train=torch.from_numpy(features_train)
y_train=torch.from_numpy(targets_train).type(torch.LongTensor)

X_test = torch.from_numpy(features_test)
y_test = torch.from_numpy(targets_test).type(torch.LongTensor)

X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
batch_size = 256

train=torch.utils.data.TensorDataset(X_train,y_train)
test=torch.utils.data.TensorDataset(X_test,y_test)


# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

# visualize one of the images in data set
plt.imshow(features_numpy[8].reshape(28,28))
plt.axis("off")
plt.title(str(targets_numpy[8]))
plt.savefig('graph.png')
plt.show()
from torch.cuda import amp
MODEL_NAME = "efficientnet_b3"
NUM_ClASSES = 10
IN_CHANNELS = 1
PRETRAINED = True  # If True -> Fine Tuning else Scratch Training
EPOCHS = 3
EARLY_STOPPING = True  # If you need early stoppoing for validation loss
SAVE_PATH = "{}.pt".format(MODEL_NAME)
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Creating Model")

model = timm.create_model(MODEL_NAME, num_classes=NUM_ClASSES, in_chans=IN_CHANNELS, pretrained=True)
if torch.cuda.is_available():
    print("Model Created. Moving it to CUDA")
else:
    print("Model Created. Training on CPU only")
_ = model.to(device)


# Creates a GradScaler once at the beginning of training.
# This is the different step while defining the model
scaler = amp.GradScaler()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].view(-1).float().sum(0) * 100.0 / batch_size for k in topk]
def train_step(
    model,
    train_loader,
    criterion,
    device,
    optimizer,
    scheduler=None,
    num_batches: int = None,
    log_interval: int = 100,
    grad_penalty: bool = False,
    fp16_scaler=None,
):
    """
    Performs one step of training. Calculates loss, forward pass, computes gradient and returns metrics.
    Args:
        model : A pytorch CNN Model.
        train_loader : Train loader.
        criterion : Loss function to be optimized.
        device : "cuda" or "cpu"
        optimizer : Torch optimizer to train.
        scheduler : Learning rate scheduler.
        num_batches : (optional) Integer To limit training to certain number of batches.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
        grad_penalty : (optional) To penalize with l2 norm for big gradients.
        fp16_scaler: (optional) If True uses PyTorch native mixed precision Training.
    """

    start_train_step = time.time()

    model.train()
    last_idx = len(train_loader) - 1
    batch_time_m = AverageMeter()
    # data_time_m = utils.AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    cnt = 0
    batch_start = time.time()
    # num_updates = epoch * len(loader)

    for batch_idx, (inputs, target) in enumerate(train_loader):
        last_batch = batch_idx == last_idx
        # data_time_m.update(time.time() - batch_start)
        inputs = inputs.to(device)
        target = target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        if fp16_scaler is not None:
            with amp.autocast():
                output = model(inputs)
                loss = criterion(output, target)
                # Scale the loss using Grad Scaler

            if grad_penalty is True:
                # Scales the loss for autograd.grad's backward pass, resulting in scaled grad_params
                scaled_grad_params = torch.autograd.grad(
                    fp16_scaler.scale(loss), model.parameters(), create_graph=True
                )
                # Creates unscaled grad_params before computing the penalty. scaled_grad_params are
                # not owned by any optimizer, so ordinary division is used instead of fp16_scaler.unscale_:
                inv_scale = 1.0 / fp16_scaler.get_scale()
                grad_params = [p * inv_scale for p in scaled_grad_params]
                # Computes the penalty term and adds it to the loss
                with amp.autocast():
                    grad_norm = 0
                    for grad in grad_params:
                        grad_norm += grad.pow(2).sum()

                    grad_norm = grad_norm.sqrt()
                    loss = loss + grad_norm

            fp16_scaler.scale(loss).backward()
            # Step using fp16_scaler.step()
            fp16_scaler.step(optimizer)
            # Update for next iteration
            fp16_scaler.update()

        else:
            output = model(inputs)
            loss = criterion(output, target)

            if grad_penalty is True:
                # Create gradients
                grad_params = torch.autograd.grad(
                    loss, model.parameters(), create_graph=True
                )
                # Compute the L2 Norm as penalty and add that to loss
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                loss = loss + grad_norm

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        cnt += 1
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1_m.update(acc1.item(), output.size(0))
        top5_m.update(acc5.item(), output.size(0))
        losses_m.update(loss.item(), inputs.size(0))

        batch_time_m.update(time.time() - batch_start)
        batch_start = time.time()
        if last_batch or batch_idx % log_interval == 0:  # If we reach the log intervel
            print(
                "Batch Train Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                "Top 1 Accuracy: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                "Top 5 Accuracy: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                    batch_time=batch_time_m, loss=losses_m, top1=top1_m, top5=top5_m
                )
            )

        if num_batches is not None:
            if cnt >= num_batches:
                end_train_step = time.time()
                metrics = OrderedDict(
                    [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
                )
                print("Done till {} train batches".format(num_batches))
                print(
                    "Time taken for train step = {} sec".format(
                        end_train_step - start_train_step
                    )
                )
                return metrics

    metrics = OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
    )
    end_train_step = time.time()
    print(
        "Time taken for train step = {} sec".format(end_train_step - start_train_step)
    )
    return metrics

def val_step(
    model, val_loader, criterion, device, num_batches=None, log_interval: int = 100
):
    """
    Performs one step of validation. Calculates loss, forward pass and returns metrics.
    Args:
        model : A pytorch CNN Model.
        val_loader : Validation loader.
        criterion : Loss function to be optimized.
        device : "cuda" or "cpu"
        num_batches : (optional) Integer To limit validation to certain number of batches.
        log_interval : (optional) Defualt 100. Integer to Log after specified batch ids in every batch.
    """
    start_test_step = time.time()
    last_idx = len(val_loader) - 1
    batch_time_m = AverageMeter()
    # data_time_m = utils.AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    cnt = 0
    model.eval()
    batch_start = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(val_loader):
            last_batch = batch_idx == last_idx
            inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            reduced_loss = loss.data

            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - batch_start)

            batch_start = time.time()

            if (
                last_batch or batch_idx % log_interval == 0
            ):  # If we reach the log intervel
                print(
                    "Batch Inference Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                    "Top 1 Accuracy: {top1.val:>7.4f} ({top1.avg:>7.4f})  "
                    "Top 5 Accuracy: {top5.val:>7.4f} ({top5.avg:>7.4f})".format(
                        batch_time=batch_time_m, loss=losses_m, top1=top1_m, top5=top5_m
                    )
                )

            if num_batches is not None:
                if cnt >= num_batches:
                    end_test_step = time.time()
                    metrics = OrderedDict(
                        [
                            ("loss", losses_m.avg),
                            ("top1", top1_m.avg),
                            ("top5", top5_m.avg),
                        ]
                    )
                    print("Done till {} validation batches".format(num_batches))
                    print(
                        "Time taken for validation step = {} sec".format(
                            end_test_step - start_test_step
                        )
                    )
                    return metrics

        metrics = OrderedDict(
            [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)]
        )
        print("Finished the validation epoch")

    end_test_step = time.time()
    print(
        "Time taken for validation step = {} sec".format(
            end_test_step - start_test_step
        )
    )
    return metrics

for epoch in tqdm(range(EPOCHS)):
    print()
    print("Training Epoch = {}".format(epoch))
    train_metrics = train_step(model, train_loader, criterion, device, optimizer, fp16_scaler=scaler)
    print()
    print("Validating Epoch = {}".format(epoch))
    valid_metrics = val_step(model, test_loader, criterion, device)
    validation_loss = valid_metrics["loss"]

    print("Done Training, Model Saved to Disk")