import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import json
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_words = 28 # 七言絕句
n_class = 0
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
    def __init__(self, n_words=20, n_class=5000):
        super(G, self).__init__()
        self.n_words = n_words
        self.n_class = n_class
        
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, stride=1, padding=1, bias=False) # 20
        self.norm1 = nn.InstanceNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False) # 10
        self.norm2 = nn.InstanceNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False) # 10
        self.norm3 = nn.InstanceNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1, bias=False) # 5
        self.norm4 = nn.InstanceNorm1d(512)
        self.upconv5 = nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1, bias=False) # 10
        self.norm5 = nn.InstanceNorm1d(256+256)
        self.upconv6 = nn.ConvTranspose1d(256+256, 300, kernel_size=4, stride=2, padding=1, bias=False) # 20
        self.norm6 = nn.InstanceNorm1d(300+64)
        self.conv7 = nn.Conv1d(300+64, self.n_class, kernel_size=1, stride=1, padding=0, bias=False) # 20
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, 0.5)
        s1 = x
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, 0.5)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.leaky_relu(x, 0.1) # 10
        s2 = x
        
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.leaky_relu(x, 0.1) # 5
        
        x = self.upconv5(x)
        x = torch.cat((x, s2), 1)
        x = self.norm5(x)
        x = F.leaky_relu(x, 0.1) # 10
        
        x = self.upconv6(x)
        x = torch.cat((x, s1), 1)
        x = self.norm6(x)
        x = F.leaky_relu(x, 0.1) # 20
        
        x = self.conv7(x) # 20
        x = torch.tanh(x)
        return x

G_net = G(n_words, n_class)
G_net.load_state_dict(torch.load('../input/iter-36000-G.ckpt', map_location='cpu'))
G_net = G_net.to(device)
_ = G_net.train() # enable dropout
strains = [
    '仄平平仄仄平平仄仄平平仄仄平平仄仄平平仄仄仄平平仄仄平平',
    '仄仄平平仄仄平仄平平仄仄平平平平仄仄平平仄仄仄平平仄仄平',
    '仄平仄仄仄平平仄仄仄平仄仄平仄仄平平平仄仄平平平仄仄平平'
]

strains_ohe = torch.from_numpy(np.asarray(list(map(lambda x: str2ohe(x, strainset), strains)), dtype=np.float32).transpose(0,2,1)).to(device)
generated = G_net(strains_ohe)
generated = generated.detach().cpu().numpy().transpose(0,2,1)
generated = list(map(lambda x: ohe2str(x, charset_inv), generated))
for l, poet in zip(strains,generated):
    print(poet[:7]+'，'+poet[7:14]+'。\n'+poet[14:21]+'，'+poet[21:28]+'。\n')
