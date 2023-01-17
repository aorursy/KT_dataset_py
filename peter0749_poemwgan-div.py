import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import json
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_words = 20 # 五言絕句
n_class = 0
n_noise = 64
def encode_context(context, charset):
    def f(x):
        return charset[x] if x in charset else 0
    return list(map(f, list(context)))
def one_hot(x, n_class):
    ohe = np.zeros((len(x), n_class), dtype=np.uint8)
    ohe[np.arange(len(x)), x] = 1
    return ohe
def str2ohe(x, charset):
    return one_hot(encode_context(x, charset), len(charset))
def ohe2str(x, charset_inv):
    x = np.argmax(x,axis=-1)
    return ''.join(list(map(lambda a: charset_inv[a], list(x))))
strainset = {'平': 0, '仄': 1}
strainset_inv = {0: '平', 1: '仄'}
with open('../input/charset.json', 'r') as fp:
    charset = json.loads(fp.read())
charset_inv = {}
for key, value in charset.items():
    charset_inv[value] = key
n_class = len(charset)
class G(nn.Module):
    def __init__(self, n_words=20, n_class=5000, n_noise=128):
        super(G, self).__init__()
        self.n_words = n_words
        self.n_class = n_class
        self.n_noise = n_noise
        self.fc1 = nn.Linear(self.n_noise, 64*(self.n_words//4), bias=False)
        self.net = nn.Sequential(*[
            nn.ConvTranspose1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # 10
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), # 20
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2, bias=False),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Conv1d(512, self.n_class, kernel_size=1, padding=0, bias=False),
        ])
    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.size(0), 64, self.n_words//4)
        x = self.net(x)
        x = torch.tanh(x)
        return x
G_net = G(n_words, n_class, n_noise)
G_net.load_state_dict(torch.load('../input/iter-13000-G.ckpt', map_location='cpu'))
G_net = G_net.to(device)
_ = G_net.train() # enable dropout
samples_preview = torch.randn(8, n_noise).clamp(-2,2).to(device)
generated = G_net(samples_preview)
generated = generated.detach().cpu().numpy().transpose(0,2,1)
generated = list(map(lambda x: ohe2str(x, charset_inv), generated))
for poet in generated:
    print(poet[:5]+'，'+poet[5:10]+'。\n'+poet[10:15]+'，'+poet[15:20]+'。\n')
