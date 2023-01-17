import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
all_data = []
for i, path in enumerate(sorted(glob.glob('/kaggle/input/*.nc'))):
    print('Reading', i, path)
    ds  = xr.open_dataset(path)
    data = torch.stack([torch.tensor(ds['tcc'].values), torch.tensor(ds['msl'].values)], dim=1)
    resized = F.interpolate(data, size=(32, 64), mode='bilinear')
    all_data.append(resized)
    
all_data = torch.cat(all_data, dim=0)
all_data.size() # Shape of this is (num_observations, num_channels, height, width). The channels are tcc an msl.
# Normalize the data to 0..1
m = all_data.min(dim=0, keepdim=True)[0].min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
M = all_data.max(dim=0, keepdim=True)[0].max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
all_data = (all_data - m)/(M - m)
# A function to plot a TSNE with colors representing the seasons
def show_tsne(x, num_years=None):
    x_embedded = TSNE(n_components=2, perplexity=30).fit_transform(x)
    if num_years is None:
        colors = plt.cm.jet(np.linspace(0,1,x.shape[0]))
    else:
        colors = plt.cm.cool(np.linspace(0,1,365))
        colors = np.concatenate([colors,  colors[::-1]])
        colors = np.concatenate([colors for _ in range(num_years)])
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=5, c=colors)
    plt.show()
class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

        self.conv_1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=2, padding=0)
#         self.conv_1_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.conv_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0)
#         self.linear = nn.Linear(in_features=192, out_features=8)

    def forward(self, x):

        x = self.conv_1(x).relu()
        x = self.conv_2(x).relu()
        x = self.conv_3(x).relu()
        x = self.conv_4(x).relu()
        x = x.flatten(start_dim=1)
#         x = self.linear(x)

        return x

model = Model()
def get_triplet(batch_size=32, max_interval=10, device='cpu'):

    i_ref = torch.randint(low=max_interval, high=all_data.size(0)-max_interval, size=(batch_size,), device=device)
    i_neg = torch.randint(low=0, high=all_data.size(0), size=(batch_size,), device=device)

    intervals = torch.randint(low=1, high=max_interval+1, size=(batch_size,), device=device)
    signs = torch.rand(size=(batch_size,), device=device)
    signs[signs>0.5] = 1
    signs[signs<=0.5] = -1
    i_pos = i_ref + signs.type(torch.int32)*intervals
    
    x_ref = all_data[i_ref]
    x_pos = all_data[i_pos]
    x_neg = all_data[i_neg]

    return x_ref, x_pos, x_neg
device = 'cuda'
model = model.to(device)
all_data = all_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
running_loss_alpha = 0.99
running_loss = 0
for i in range(200000):

    optimizer.zero_grad()
    x_ref, x_pos, x_neg = get_triplet(device=device)

    emb_ref = model(x_ref) # shape (batch_size, 192)
    emb_pos = model(x_pos) # shape (batch_size, 192)
    emb_neg = model(x_neg) # shape (batch_size, 192)

    s_pos = torch.sum(emb_ref * emb_pos, dim=-1) # shape (batch_size,)
    s_neg = torch.sum(emb_ref * emb_neg, dim=-1) # shape (batch_size,)

    loss = -s_pos + torch.log(s_pos.exp() + s_neg.exp())
    loss = loss.sum()

    loss.backward()
    optimizer.step()

    running_loss = running_loss * running_loss_alpha + (1-running_loss_alpha) * loss.item()
    
    if (i+1) % 500 == 0:
        print('step {0} - running loss {1}'.format(i + 1, running_loss))
torch.save(model.state_dict(), 'model.pth')
show_tsne(all_data[:2*365*4].reshape(2*365*4, -1).cpu(), num_years=4)
with torch.no_grad():
    show_tsne(model(all_data[:2*365*8]).cpu(), num_years=8)
# take = torch.rand(size=(all_data.size(0),)) < 0.1
with torch.no_grad():
    predictions = all_data.flatten(start_dim=1)[take].cpu().numpy()

colors = plt.cm.cool(np.linspace(0,1,365))
colors = np.concatenate([colors,  colors[::-1]])
colors = np.concatenate([colors for _ in range(41)])
colors = np.concatenate([colors, colors[:20]])
colors = colors[take]
x_embedded = TSNE(n_components=2, perplexity=30).fit_transform(predictions)
plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=5, c=colors)
plt.show()
# take = torch.rand(size=(all_data.size(0),)) < 0.1
with torch.no_grad():
    predictions = model(all_data[take]).cpu().numpy()

colors = plt.cm.cool(np.linspace(0,1,365))
colors = np.concatenate([colors,  colors[::-1]])
colors = np.concatenate([colors for _ in range(41)])
colors = np.concatenate([colors, colors[:20]])
colors = colors[take]
x_embedded = TSNE(n_components=2, perplexity=30).fit_transform(predictions)
plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=5, c=colors)
plt.show()