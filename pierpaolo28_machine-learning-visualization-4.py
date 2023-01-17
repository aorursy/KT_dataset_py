import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.animation as animation
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv',error_bad_lines=False)
df = df.reset_index(drop=True)
df.authors = [i.split('/')[0] for i in df.authors]
df.head()
from wordcloud import WordCloud

d = {}
for x, a in zip(df.authors.value_counts(), df.authors.value_counts().index):
    d[a] = x

wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud", fontsize=20)
plt.savefig("cloud.png", dpi=200)
plt.show()
df = df.drop_duplicates('authors')
df = df.reset_index(drop=True)
df = df[:30]
size = len(df.authors)
encoder, scaler = LabelEncoder(), MinMaxScaler()
aut = encoder.fit_transform(df.authors) 
rat = scaler.fit_transform(df[['average_rating']])
class Latent_Embed(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Latent_Embed, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(3, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = F.relu(self.linear1(embeds))
        out = F.relu(self.linear2(out))
        return out


aut_t = torch.tensor(aut)
rat_t = torch.tensor(rat)
loss_function = nn.MSELoss()
model = Latent_Embed(size, 3)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(10):
    total_loss = 0
    for context, target in zip(aut_t, rat_t):
        model.zero_grad()
        log_probs = model(context)
        loss = loss_function(log_probs.double(), target.view(1).double())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Loss: ", total_loss/len(aut_t))
embedding_weights = pd.DataFrame(model.embeddings.weight.detach().numpy())
embedding_weights.columns = ['X1','X2','X3']
embedding_weights
fig = plt.figure(num=None, figsize=(14, 12), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection='3d')
for index, (x, y, z) in enumerate(zip(embedding_weights['X1'], 
                                      embedding_weights['X2'], 
                                      embedding_weights['X3'])):
    ax.scatter(x, y, z, color='b', s=12)
    ax.text(x, y, z, str(df.authors[index]), size=12, zorder=2.5, color='k')

ax.set_title("Word Embedding", fontsize=20)
ax.set_xlabel("X1", fontsize=20)
ax.set_ylabel("X2", fontsize=20)
ax.set_zlabel("X3", fontsize=20)
plt.show()
def rotate(angle):
    ax.view_init(azim=angle)

print("Making animation")
res_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 365, 2), interval=100)
res_animation.save('embedding.gif', dpi=100, writer='imagemagick')