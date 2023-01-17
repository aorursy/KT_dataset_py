
# first read the data
import pandas as pd
video=pd.read_json("../input/train.json")
video.head(3)


dicti={}
for i in range(1195):
    liste=[]
    for frame in video.audio_embedding[i]:
        for value in frame:
            liste.append(value)
    dicti[i]=liste
for i in range(1195):
    while len(dicti[i])<1280:
        dicti[i].extend(dicti[i])
    dicti[i]=dicti[i][:1280]  
audio=pd.DataFrame.from_dict(dicti, orient='index')
audio.head(3)

#now we can create the model using Logistic Regression and check . Accuracy of the model is 93.6%
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
is_turkey=video.is_turkey # target variable
from sklearn.model_selection import cross_val_score
cross_val_score(model, audio, is_turkey, cv=5).mean()  
# to visualize the clusters, I project data into 2 dimensions with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(audio)
%matplotlib inline
import matplotlib.pyplot as plt
plt.scatter(proj[:, 0], proj[:, 1], c=is_turkey) 
plt.colorbar() 