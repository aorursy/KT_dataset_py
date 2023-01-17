import pandas as pd

import numpy as np



from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



import torch

from torch.nn import Linear, Tanh, Sigmoid, Module, BCELoss

from torch.optim import Adam



from umap import UMAP



import plotly.express as px



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/lyrics-dataset/lyrics_features.csv', index_col=0)

labels = data.columns[-10:]

X, Y = data.iloc[:, 2:-10].values, data.iloc[:, -10:].values

print(data.shape)

data.head()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, 

                                                    test_size=0.1)

to_tensor = lambda M: torch.from_numpy(M).float().cuda()



X = to_tensor(X)

Y = to_tensor(Y)



X_train = to_tensor(X_train)

X_test = to_tensor(X_test)

Y_train = to_tensor(Y_train)

Y_test = to_tensor(Y_test)
class MultiLabelNet(Module):

    def __init__(self, n_feature, n_hidden, n_classes):

        super().__init__()

        

        self.fc1 = Linear(n_feature, n_hidden)

        self.act1 = Tanh()

        self.fc2 = Linear(n_hidden, n_hidden)

        self.act2 = Tanh()

        self.fc3 = Linear(n_hidden, n_feature)

        self.act3 = Tanh()

        self.fc4 = Linear(n_feature, n_classes)

        self.act4 = Sigmoid()

    

    def forward(self, x):

        x = self.fc1(x)

        x = self.act1(x)

        x = self.fc2(x)

        x = self.act2(x)

        x = self.fc3(x)

        x = self.act3(x)

        x = self.fc4(x)

        x = self.act4(x)

        return x
def get_score_over_labels(Y_pred, Y_true=Y.detach().cpu().numpy(), threshold=0.5):

    scores = []

    print('F1 Score over labels:')

    for label in range(Y_true.shape[1]):

        y_true = Y_true[:, label]

        y_pred = np.where(Y_pred[:, label] > threshold, 1, 0)

        score = f1_score(y_true, y_pred)

        scores.append(score)

        print(f'{labels[label].rjust(20)}: {score:.4f}')

    print(f'Mean F1 Score: {np.mean(scores):.4f}')





def train(X_train, X_test, Y_train, Y_test, 

          n_epochs, model, optimizer, criterion, save_path):

    valid_loss_min = np.Inf

    best_model = None



    for epoch in range(1, n_epochs+1):

        # train the model

        model.train()



        optimizer.zero_grad()

        output = model(X_train)



        train_loss = criterion(output, Y_train)

        train_loss.backward()

        optimizer.step()

        

        # validate the model

        model.eval()



        output = model(X_test)

        valid_loss = criterion(output, Y_test)



        # print train/val statistics

        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')

        

        # update best model if validation loss has decreased

        if valid_loss < valid_loss_min:

            print(f'Validation loss decreased ({valid_loss_min:.4f} --> {valid_loss:.4f}).')

            best_model = model

            valid_loss_min = valid_loss



    print('Saving best model ...')

    torch.save(best_model, save_path)

    print(f'Minimum value of loss: {valid_loss_min:.4f}')



    return best_model
path_to_model = './model.pth'

model = MultiLabelNet(X.shape[1], 100, Y.shape[1]).cuda()

criterion = BCELoss()

optimizer = Adam(model.parameters(), lr=0.001, amsgrad=True)



model = train(X_train, X_test, Y_train, Y_test, 2000, model, optimizer, criterion, path_to_model)
model = torch.load(path_to_model).cuda()

model.eval()



prediction = model(X).detach().cpu().numpy()

get_score_over_labels(prediction, threshold=0.15)
cutted_model = torch.nn.Sequential(*(list(model.children())[:-2])).cuda()

cutted_model.eval()



song_embeddings = cutted_model(X).detach().cpu().numpy()
X = X.detach().cpu().numpy()

X.shape, song_embeddings.shape
# just space cleaning

torch.cuda.empty_cache()
%%time

reducer = UMAP(n_components=3)

X_reduced = reducer.fit_transform(X)
%%time

reducer = UMAP(n_components=3)

song_embeddings_reduced = reducer.fit_transform(song_embeddings)
X_reduced.shape, song_embeddings_reduced.shape
songs = pd.DataFrame(np.concatenate([data.iloc[:, :2].values.reshape(-1, 2), 

                                     X_reduced, 

                                     song_embeddings_reduced], 

                                    axis=1), 

                     columns=['Singer', 'Song', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2'])



for col in songs.columns[2:]: # I dont know why are they casting to

    songs[col] = songs[col].astype(float) # `object` while creating DF

songs.head()
singers = songs.groupby(['Singer'], as_index=False)[songs.columns[2:]].mean()

get_label = np.vectorize(lambda x: labels[x])

singers['Genre'] = get_label(data.groupby(['Singer'])[labels].sum().values.argmax(axis=1))

singers.head()
fig = px.scatter_3d(singers, x='x1', y='y1', z='z1',

                    color='Genre', hover_name='Singer')

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()
fig = px.scatter_3d(singers, x='x2', y='y2', z='z2',

                    color='Genre', hover_name='Singer')

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()
def find_nearest_singers(singer_df, singer, metric, k=10):

    dist_func = euclidean_distances if metric == 'euclidean' else cosine_similarity

    ascending = True  if metric == 'euclidean' else False



    singers_similarity = pd.DataFrame(dist_func(singer_df))

    singer_index = singers[singers['Singer'] == singer].index[0]

    top_similar = singers_similarity[singer_index].sort_values(ascending=ascending)[1:k+1]

    top_similar_names = singers.loc[top_similar.index]['Singer'].values

    top_similar_values = map(lambda x: round(x, 4), top_similar.values)



    print(f'The nearest artists to {singer}:')

    print('\n'.join([f'{name.rjust(25)} - {values}' for name, values in zip(top_similar_names, top_similar_values)]))
find_nearest_singers(singers[['x2', 'y2', 'z2']], 'Eminem', 'euclidean')
find_nearest_singers(singers[['x1', 'y1', 'z1']], 'Eminem', 'euclidean')