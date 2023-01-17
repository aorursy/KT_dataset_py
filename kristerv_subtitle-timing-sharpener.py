!pip install torch-lr-finder
import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
%matplotlib inline

import re
from datetime import datetime
from itertools import zip_longest
from tqdm import tqdm

import torch.optim as optim
from torch_lr_finder import LRFinder
from torch.optim import lr_scheduler
torch.manual_seed(0)
audio_file = '/kaggle/input/TH_S01E01.wav'
os.listdir('/kaggle/input/')
waveform, sample_rate = torchaudio.load(audio_file)
duration_min = waveform.size()[1] / sample_rate / 1000 / 1000 / 60

print("duration_min", duration_min)
print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))
fft_length = 10 # how many ms goes into a single line of pixels
spectrogram = T.Spectrogram(win_length=16*fft_length)(waveform)

plt.figure(figsize=(20, 20))
plt.imshow(spectrogram.log2()[0,:,:2000].numpy())
plt.figure(figsize=(20, 20))
plt.imshow(spectrogram.log2()[0,:,1750:1751].t())
spectrogram.log2().size()
fft_list = spectrogram[0].t()
fft_list.size()
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def pessimistic_subs(times):
    crop_by = 200
    return [[start + crop_by, end - crop_by] for start, end in times if end - start > crop_by*2+100]
# Pessimistic times
sub_file_path = '/kaggle/input/TerraceHouse_OpeningNewDoors_S01E01_ja.vtt'
sub_file = open(sub_file_path, "r")
lines = sub_file.read().split("\n")
times_arr = [ int(datetime.strptime(time, '%H:%M:%S.%f').replace(year=1970).timestamp() * 1000) for line in lines if "-->" in line for time in re.findall('\d\d:\d\d:\d\d\.\d{3}', line) ]
sub_times_speech = pessimistic_subs(grouper(times_arr, 2))
sub_times_silence = pessimistic_subs(grouper([0] + times_arr[:-1], 2))
# Separate speech from silence

train_speech_a = []
train_silence_a = []
train_leftover_a = []

for fft_i in tqdm(range(len(fft_list)), desc="Matching fft to sub times"):
    fft_ms = fft_i * fft_length
    added = 0
    found_start = -1
    fft_item = fft_list[fft_i].numpy()
    
    # FIX: This doesn't train on silence..
    if fft_ms < times_arr[0]:
        train_silence_a.append(fft_item)
        continue
        
    for sub_start, sub_end in sub_times_speech:
        if fft_ms > sub_start and fft_ms < sub_end:
            added += 1
            found_start = sub_start
            train_speech_a.append(fft_item)
            break
            
    for sub_start, sub_end in sub_times_silence:
        if fft_ms > sub_start and fft_ms < sub_end:
            added += 1
            found_start = sub_start
            train_silence_a.append(fft_item)
            break
            
    if added == 0:
        train_leftover_a.append(fft_item)
            
    assert added <= 1, "Should only add once. added: " + str(added) + ", index: " + str(fft_i)

print("Speech FFT count:", len(train_speech_a))
print("Silence FFT count:", len(train_silence_a))
print("Leftover FFT count:", len(train_leftover_a))
class MyDataset(Dataset):
    def __init__(self, speech, silence):
        self.data = list(map(lambda x: (x, torch.tensor(0)), silence)) + list(map(lambda x: (x, torch.tensor(1)), speech))
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
batch_size = 1024

whole_ds = MyDataset(train_speech_a, train_silence_a)

valid_size_perc = 0.2
train_size = int(len(whole_ds) * (1 - valid_size_perc))
valid_size = int(len(whole_ds) - train_size)

train_ds, valid_ds = torch.utils.data.random_split(whole_ds, [train_size, valid_size])
test_ds = torch.tensor(train_leftover_a)

train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True)
train_dl_normal = DataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True)
valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=batch_size, pin_memory=True, drop_last=True)
test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, pin_memory=True, drop_last=True)

print("Train dataset size:", len(train_ds))
print("Valid dataset size:", len(valid_ds))
print("Test dataset size:", len(test_ds))
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
device
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
test_dl = DeviceDataLoader(test_dl, device)
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(201, 400),
            nn.Linear(400, 200),
            nn.Linear(200, 50),
            nn.Linear(50, 2)
        )

    def forward(self, x):
        return self.model(x)
def accuracy(out, targets):
    targets_indexes = targets.cpu().data.numpy()
    out_indexes = torch.max(out, -1, True)[1].cpu().data.numpy()
    assert len(targets_indexes) == len(out_indexes)
    
    hits = []
    for i in range(len(targets_indexes)):
        out_item = out_indexes[i].item()
#         print("out", out_item, "actual", targets_indexes[i])
        hits.append(targets_indexes[i] == out_item)
            
    return (hits, out_indexes)
net = Net()
criterion_lr = nn.CrossEntropyLoss()
optimizer_lr = optim.Adam(net.parameters(), lr=1e-7)
lr_finder = LRFinder(net, optimizer_lr, criterion_lr, device=device)
lr_finder.range_test(train_dl_normal, end_lr=100, num_iter=100)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state
epochs = 100
net = Net()
net = to_device(net, device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-5)
# cycle_lr = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, total_steps=epochs*len(train_dl), cycle_momentum=False)
# cycle_lr = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
# cycle_lr = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
loss_history = []
valid_history = []
valid_acc_history = []
lr_history = []
silence_count_history = []
speech_count_history = []
for epoch in tqdm(range(epochs)):
    train_loss_epoch = []
    valid_loss_epoch = []
    valid_acc_epoch = []
    lr_epoch = []
    silence_count_epoch = []
    speech_count_epoch = []
    net.train(True)
    for (inputs, targets) in train_dl:
        lr_epoch.append(optimizer.param_groups[0]['lr'])
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
#         cycle_lr.step()
        train_loss_epoch.append(loss.item())
    net.train(False)
    for (inputs, targets) in valid_dl:
        out = net(inputs)
        loss = criterion(out, targets)
        valid_loss_epoch.append(loss.item())
        hits, out_indexes = accuracy(out, targets)
        silence_count_epoch.append((np.array(out_indexes) == False).sum())
        speech_count_epoch.append((np.array(out_indexes) == True).sum())
        valid_acc_epoch.append(hits)
    loss_history.append(np.average(train_loss_epoch))
    valid_history.append(np.average(valid_loss_epoch))
    valid_acc_flat = np.array(valid_acc_epoch).flatten().flatten()
    valid_acc_history.append(np.average(valid_acc_flat))
    lr_history.append(np.average(lr_epoch))
    silence_count_history.append(np.sum(silence_count_epoch))
    speech_count_history.append(np.sum(speech_count_epoch))
plt.figure(figsize=(9,6))
plt.plot(lr_history)
plt.figure(figsize=(9,6))
plt.plot(silence_count_history)
# plt.plot(speech_count_history)
plt.figure(figsize=(9,6))
# plt.ylim([0,1])
plt.plot(loss_history, label="train loss")
plt.plot(valid_history, label="validation loss")
plt.plot(valid_acc_history, label="validation accuracy")
plt.legend()
plt.show()
result = net(test_ds.to(device))
print_count = 10
skip = int(len(result)/print_count)
for i in range(print_count):
    print(result[i*skip])
preds = net(fft_list.to(device))
preds[:10]
def gen_timecodes(preds):
    detections = torch.max(preds, -1, True)[1].cpu().data.numpy()
    timecodes = []
    started = None
    for i in range(len(detections)):
        detection = detections[i].item()
        ms = fft_length * i
        if started == None and detection == 1:
            started = ms
        elif started != None and detection == 0:
            timecodes.append([started, ms])
            started = None
    return timecodes

timecodes = gen_timecodes(preds)
timecodes[:10]
def gen_vtt(times, text):
    vtt_str = """WEBVTT

NOTE Netflix
NOTE Profile: webvtt-lssdh-ios8
NOTE Date: 2018/01/29 18:46:46

NOTE SegmentIndex
NOTE Segment=596.554 16670@498 125
NOTE Segment=597.597 20459@17168 159
NOTE Segment=596.471 23265@37627 178
NOTE Segment=596.137 24518@60892 185
NOTE Segment=122.539 4791@85410 37
NOTE /SegmentIndex



"""

    for i, [start_ms, end_ms] in enumerate(times, 1):
        start_time = datetime.fromtimestamp(start_ms/1000.0).strftime('%H:%M:%S.%f')[:-3]
        end_time = datetime.fromtimestamp(end_ms/1000.0).strftime('%H:%M:%S.%f')[:-3]
        vtt_str += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
        
    return vtt_str
print(gen_vtt(timecodes, "AImazing"))
print(gen_vtt(sub_times_speech, "speech"))
print(gen_vtt(sub_times_silence, "silence"))






