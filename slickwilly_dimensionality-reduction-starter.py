# notebook settings
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# increase plot resolution
%config InlineBackend.figure_format = 'retina'
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.decomposition import PCA
# Custom functions
def run_pca(data, meta, n_components=50, data_only=False):
    """Applies PCA to `data` and returns a tuple of principle components with metadata and just principle components."""
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    print("Explained variance ratio: {}".format(np.sum(pca.explained_variance_ratio_)))
    # create dataframe
    data_pca = pd.DataFrame((data_pca),
                             index=data.index,
                             columns=["PCA_{}".format(i) for i in range(n_components)])
    # join with metadata
    joined = meta.merge(data_pca, how='inner', left_index=True, right_index=True)
    
    if data_only:
        return joined.drop(columns=meta.columns)
    else:
        return joined
# metadata
meta = pd.read_csv('/kaggle/input/ai4all-project/data/metatable_with_viral_status.csv', delimiter=',', index_col=0)
# get viral calls data
viral = pd.read_csv("/kaggle/input/ai4all-project/results/viral_calls/viruses_with_pval.csv", delimiter=",", index_col=0)
# samples to CZB_IDS
sample_map = pd.read_csv("/kaggle/input/ai4all-project/data/viral_calls/ngs_samples.csv", index_col=0)
# merge tables
viral = viral.join(sample_map, how='inner')
# remove rows with non-significant viral detection
rows, cols = viral.shape
viral = viral[viral['p_val']<=0.05]
print("Removed {} rows".format(rows - viral.shape[0]))
# now lets pivot
viral_pivot = viral.pivot_table(values='nt_rpm', index='CZB_ID', columns='name', fill_value=0.)
viral_pivot.head()
viral['name'].value_counts().head()
sc2_meta = meta[meta['viral_status']=='SC2']
sc2_viral = viral[viral['CZB_ID'].isin(sc2_meta.index)]
coinf = sc2_viral['CZB_ID'].value_counts()[sc2_viral['CZB_ID'].value_counts() > 1].index
coinf_vir = viral[viral['CZB_ID'].isin(coinf)]['name'].value_counts()
coinf_vir.head(10)
coinf_vir = coinf_vir[coinf_vir > 3].index
viral[(viral['CZB_ID'].isin(coinf)) & (viral['name'].isin(coinf_vir))].boxplot('nt_rpm', by='name', rot=45)
all_viral_pca = run_pca(viral_pivot, meta)
all_viral_pca.head()
sns.scatterplot(x="PCA_0", y="PCA_1", hue='viral_status', data=all_viral_pca)
# is this outlier skewing our PCA?
# lets remove and rerun
outlier = all_viral_pca['PCA_1'].nlargest(1).index

all_viral_pca = run_pca(viral_pivot.drop(index=outlier), meta.drop(index=outlier))
sns.scatterplot(x="PCA_0", y="PCA_1", hue='viral_status', data=all_viral_pca)
# ok, thhe scales seem to be extreme so lets rescale each feature
from sklearn.preprocessing import minmax_scale
scaled_viral_pivot = pd.DataFrame(minmax_scale(viral_pivot),
                                  index=viral_pivot.index,
                                  columns=viral_pivot.columns)
all_viral_pca = run_pca(scaled_viral_pivot.drop(index=outlier), meta.drop(index=outlier))
sns.scatterplot(x="PCA_0", y="PCA_1", hue='viral_status', data=all_viral_pca)
# ok lets try this with only SC2+ patients because the variance from some outliers is still skewing the results
sc2_meta = meta[meta['viral_status']=='SC2']
sc2_scaled_viral_pivot = scaled_viral_pivot.reindex(sc2_meta.index).dropna()
sc2_viral_pca = run_pca(sc2_scaled_viral_pivot, sc2_meta)
sns.scatterplot(x="PCA_0", y="PCA_1", hue='SC2_rpm', data=sc2_viral_pca)
# again, is this outlier skewing our PCA?
# lets remove and rerun
outlier = sc2_viral_pca['PCA_0'].nlargest(1).index

sc2_viral_pca = run_pca(sc2_scaled_viral_pivot.drop(index=outlier), sc2_meta.drop(index=outlier))
sns.scatterplot(x="PCA_0", y="PCA_1", hue='SC2_rpm', data=sc2_viral_pca)
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
# utility functions
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
class Autoencoder(nn.Module):
    def __init__(self, n_features):
        self.n_features = n_features
        super(Autoencoder, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=self.n_features, out_features=128)
        self.enc2 = nn.Linear(in_features=128, out_features=64)
        self.enc3 = nn.Linear(in_features=64, out_features=16)
        self.enc4 = nn.Linear(in_features=16, out_features=2)
 
        # decoder
        self.dec1 = nn.Linear(in_features=2, out_features=16)
        self.dec2 = nn.Linear(in_features=16, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=self.n_features)
 
    def encoder(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        
        return x
    
    def get_encodings(self, x):
        self.eval()
        with torch.no_grad():
            x = self.encoder(x)

        return x
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
def train(net, trainloader, testloader, NUM_EPOCHS):
    train_loss = []
    test_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        net.train()
        for data in trainloader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        
        # Test at the end of each epoch
        test_loss_epoch = test(net, testloader)
        test_loss.append(test_loss_epoch)
        print('Epoch {} of {}, Train Loss: {:.3f}, Test Loss: {:.3f}'.format(
            epoch+1, NUM_EPOCHS, loss, test_loss_epoch))

    return train_loss, test_loss

def test(net, testloader):
    running_loss = 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            outputs = net(data)
            loss = criterion(outputs, data)
            running_loss += loss.item()
    return running_loss / len(testloader)
device = get_device()
device

# hyperparameters
NUM_EPOCHS = 100
LEARNING_RATE = 1e-6
BATCH_SIZE = 8
# Convert pd dataframe to pytorch tensor
X = torch.tensor(sc2_scaled_viral_pivot.to_numpy(), device=device).float()

# train/test split
X_train, X_test, train_idx, test_idx = train_test_split(X, sc2_scaled_viral_pivot.index, test_size=0.15, random_state=42)
X_train.shape, X_test.shape
# dataloaders
trainloader = DataLoader(
    X_train, 
    batch_size=BATCH_SIZE,
    shuffle=True
)
testloader = DataLoader(
    X_test, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)
# define network
net = Autoencoder(n_features=X_train.shape[1])
print(net)

# define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# load the neural network onto the device
net.to(device)

# train the network
train_loss, test_loss = train(net, trainloader, testloader, NUM_EPOCHS)
plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.legend()
plt.title('Training and Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
train_enc = net.get_encodings(X_train)
test_enc = net.get_encodings(X_test)
all_enc = pd.DataFrame(np.concatenate((train_enc, test_enc)),
                       index=np.concatenate((train_idx, test_idx)),
                       columns=['AE_{}'.format(i) for i in range(2)])
all_enc['split'] = np.concatenate((np.repeat('train', len(train_idx)),
                                   np.repeat('test', len(test_idx))))
all_enc = all_enc.join(meta)
sns.scatterplot(x='AE_0', y='AE_1', hue='split', data=all_enc)
sns.scatterplot(x='AE_0', y='AE_1', hue='SC2_rpm', data=all_enc)
