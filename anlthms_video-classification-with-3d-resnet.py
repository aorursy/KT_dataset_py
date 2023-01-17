import os

import sys

import cv2

import json

import glob

import random

import logging

import numpy as np

from tqdm import tqdm



import glob

import torch

import torch.utils.data as data

from torch import nn

from shutil import copyfile





copyfile(src = "../input/resnetpy/resnet.py", dst = "../working/resnet.py")

from resnet import *

        

metafiles = sorted(glob.glob('/kaggle/input/dfdc-video-faces/part*/*/*.json'))

train_metafiles = metafiles[2:]

eval_metafiles = metafiles[:2]

print(train_metafiles)

print(eval_metafiles)
class VideoDataset(data.Dataset):



    def __init__(self, dataset_type, meta_files, batch_size):

        self.dataset_type = dataset_type

        self.batch_size = batch_size

        self.clip_shape = (16, 112, 112)

        self.load_meta(meta_files)



        print('%s dataset: %d real and %d fake samples' % (

              dataset_type, self.n_reals, self.n_fakes))



    def load_clip(self, filename, target):

        depth, height, width = self.clip_shape

        clip = np.zeros((3, depth, height, width), dtype=np.uint8)

        reader = cv2.VideoCapture(filename)

        if not reader.isOpened():

            logging.warn('could not open %s' % filename)

            return torch.from_numpy(clip).float()



        # If training, use the same cropping parameters for an entire set

        # of video clips

        if target == 0 or self.dataset_type == 'eval':

            nframes = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))

            self.start_frame = random.randint(0, nframes - self.clip_shape[0])

            self.start_row = random.randint(0, frame_height - height)

            self.start_col = random.randint(0, frame_width - width)



        for i in range(self.start_frame):

            reader.grab()



        for i in range(depth):

            reader.grab()

            success, frame = reader.retrieve()

            if not success:

                logging.warn('could not load frame %d in %s' % (

                             start_frame + i, filename))

                break

            frame = frame[self.start_row:self.start_row + height,

                          self.start_col:self.start_col + width]

            clip[:, i] = frame.transpose((2, 0, 1))



        reader.release()

        return torch.from_numpy(clip).float()



    def load_meta(self, meta_files):

        meta = []

        for meta_file in meta_files:

            dirname = os.path.dirname(meta_file)

            with open(meta_file) as meta_fd:

                meta_dict = json.load(meta_fd)

                new_dict = {}

                # Expand filenames to their paths

                for real in meta_dict:

                    fakes = meta_dict[real]

                    fakes = [os.path.join(dirname, fake) for fake in fakes]

                    new_dict[os.path.join(dirname, real)] = fakes

                meta_list = list(new_dict.items())

                meta.extend(meta_list)



        random.shuffle(meta)

        self.clips = []

        self.targets = []

        for item in meta:

            real = item[0]

            fakes = item[1]

            random.shuffle(fakes)

            for fake in fakes:

                # Oversample from non-fake videos

                self.clips.append(real)

                self.targets.append(0)

                self.clips.append(fake)

                self.targets.append(1)

                # Use a small subset for evaluation

                if self.dataset_type == 'eval':

                    break



        if self.dataset_type == 'eval':

            # Make 4 copies to get random crops from

            for _ in range(2):

                self.clips.extend(self.clips)

                self.targets.extend(self.targets)



        self.len = len(self.clips)

        self.n_fakes = np.sum(self.targets)

        self.n_reals = self.len -  self.n_fakes



    def __getitem__(self, index):

        filename = self.clips[index]

        target = self.targets[index]

        clip = self.load_clip(filename, target)



        return clip, target



    def __len__(self):

        return self.len





epochs = 8

batch_size = 16



train_dataset = VideoDataset('train', train_metafiles, batch_size)



train_loader = torch.utils.data.DataLoader(

    train_dataset, batch_size=batch_size, shuffle=False,

    num_workers=1, pin_memory=True, sampler=None)



eval_dataset = VideoDataset('eval', eval_metafiles, batch_size)



eval_loader = torch.utils.data.DataLoader(

    eval_dataset, batch_size=batch_size, shuffle=False,

    num_workers=1, pin_memory=True, sampler=None)
device = 'cuda'

model = resnet3d18(num_classes=2)

model = model.to(device)



crit = nn.CrossEntropyLoss().to(device)

opt = torch.optim.Adam(model.parameters())
def train(loader, model, crit, optimizer, epoch):

    model.train()



    loss_sum = 0

    for clips, targets in tqdm(loader):

        clips = clips.to(device)

        targets = targets.to(device)



        logits = model(clips)

        loss = crit(logits, targets)

        loss_sum += loss.data.cpu().numpy()



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



    return (loss_sum / len(loader))



def bce(probs, labels):

    safelog =  lambda x: np.log(np.maximum(x, np.exp(-50.)))

    return np.mean(-labels * safelog(probs) - (1 - labels) * safelog(1 - probs))



def validate(loader, model, crit):

    model.eval()

    sm = nn.Softmax(dim=1)

    labels = np.zeros((len(loader.dataset)), dtype=np.float32)

    probs = np.zeros((len(loader.dataset), 2), dtype=np.float32)

    with torch.no_grad():

        for i, (clips, targets) in enumerate(tqdm(loader)):

            start = i*batch_size

            end = start + clips.shape[0]

            labels[start:end] = targets

            clips = clips.to(device)



            logits = model(clips)

            probs[start:end] = sm(logits).cpu().numpy()



    probs = probs.reshape(4, -1, 2).mean(axis=0)

    labels = labels.reshape(4, -1).mean(axis=0)



    preds = probs.argmax(axis=1)

    correct = (preds == labels).sum()

    acc = correct*100//preds.shape[0]

    loss = bce(probs[:, 1], labels)

    print('validation accuracy %d%%' % acc)

    return loss



model_file = 'model.pth'

if os.path.exists(model_file):

    checkpoint = torch.load(model_file)

    model.load_state_dict(checkpoint['state_dict'])

    print('loaded %s' % model_file)



try:

    for epoch in range(epochs):

        # Train for one epoch

        train_loss = train(train_loader, model, crit, opt, epoch)



        # Evaluate on validation set

        val_loss = validate(eval_loader, model, crit)

        print('epoch %d training loss %.2f validation loss %.2f\n' % (

              epoch, train_loss, val_loss))

finally:

    torch.save({'state_dict': model.state_dict()}, model_file)

print('done')