import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.plotly as py

import matplotlib.style as style

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from scipy.stats import pearsonr

from IPython.display import HTML



# Init plotly

init_notebook_mode(connected=True)



# Seaborn style

sns.set(style="white")

style.use('seaborn-poster') #sets the size of the charts

style.use('ggplot')



# Filter warnings

import warnings

warnings.filterwarnings("ignore")



pd.set_option('display.max_columns', 7)
from sklearn.utils.multiclass import unique_labels



def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax

df = pd.read_csv("../input/global.csv")

df = df.dropna()



df.shape
df.head(10)
df.info()
df.describe()
audio_feature_headers = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
sns.pairplot(df[audio_feature_headers].sample(1000))

plt.show()
corr = df[audio_feature_headers].corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(11, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.6, vmin=-.4, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5});
g = sns.jointplot(x='loudness', y='energy', data=df.sample(1000), kind='reg');
plt.tight_layout()

plt.subplots_adjust(top=0.4)



fig, ax = plt.subplots(2, 3, figsize=(25, 10), sharex=True)



sample = df.sample(1000)



fig.suptitle("Correlation between some of the audio features", fontsize=20)

sns.regplot(x='valence', y='danceability', data=sample, ax=ax[0, 0]);

sns.regplot(x='valence', y='energy', data=sample, ax=ax[0, 1]);

sns.regplot(x='valence', y='loudness', data=sample, ax=ax[0, 2]);

sns.regplot(x='acousticness', y='energy', data=sample, ax=ax[1, 0]);

sns.regplot(x='acousticness', y='loudness', data=sample, ax=ax[1, 1]);
la_he = df[(df['energy'] > 0.8) & (df['acousticness'] < 0.2)][['artist', 'track_name', 'energy', 'acousticness', 'danceability']] # Low-acousticness - High-energy

la_he.head(5)
from IPython.display import HTML

HTML("""

<iframe width="560" height="315" src="https://www.youtube.com/embed/ynlQ8Oz8Qmk?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/UqyT8IEBkvY?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>

""")
songs_grouped_mean = df.drop(['day', 'month', 'year'], axis=1)

songs_grouped_mean = songs_grouped_mean.groupby('url').mean() # Group unique songs by its 'url' (unique identifier)



songs_grouped_mean.head(10)
songs_grouped_mean_ranges = songs_grouped_mean.groupby(pd.cut(songs_grouped_mean["position"], np.arange(0, 101, 10))).count()



f, ax = plt.subplots(figsize=(15, 5))

sns.barplot(x=songs_grouped_mean_ranges.index.values, y=songs_grouped_mean_ranges["position"]);



ax.set(title="Number of songs that reached a top position", ylabel="Count");
songs_grouped_min = df.drop(['day', 'month', 'year'], axis=1)

songs_grouped_min = songs_grouped_min.groupby('url').min() # Group unique songs by its 'url' (unique identifier)



gsongs = songs_grouped_min # Alias
songs_grouped_min_ranges = gsongs.groupby(pd.cut(gsongs["position"], np.arange(0, 101, 10))).count()



f, ax = plt.subplots(figsize=(15, 5))

sns.barplot(x=songs_grouped_min_ranges.index.values, y=songs_grouped_min_ranges["position"]);



ax.set(title="Number of songs that reached a top position", ylabel="Count");
def tempo_to_rythm(tempo):

    if(tempo < 66):

        return 'lento'

    if(66 <= tempo < 76):

        return 'adagio'

    if(76 <= tempo < 108):

        return 'andante'

    if(108 <= tempo < 168):

        return 'allegro'

    if(168 <= tempo):

        return 'presto'



gsongs['rythm'] = gsongs['tempo'].transform(tempo_to_rythm)
fig, ax = plt.subplots(figsize=(8, 8))

ax.pie(gsongs["rythm"].value_counts(), labels=gsongs['rythm'].value_counts().axes[0], autopct='%1.1f%%', shadow=True, textprops={'fontsize': 16});

ax.set_title("Rythm tag distribution");
ax = sns.barplot(x='position', y='rythm', data=gsongs[gsongs['position'] <= 20])

ax.set(title="Rythm tag for the Top 20 songs", xlim=(1,20));



# Count the number of songs occurences on each group

print(gsongs[gsongs['position'] <= 20].groupby('rythm').count()['position'])
keys = np.array(['Do', 'Do#/Re♭', 'Re', 'Re#/Mi♭', 'Mi', 'Mi#/Fa♭', 'Fa', 'Fa#/Sol♭', 'Sol', 'Sol#/La♭', 'La', 'La#/Si♭', 'Si', 'Si#/Do♭'])

keys_top10_categorical = pd.Series(keys[gsongs[gsongs['position'] < 10]['key']])
keys_count = keys_top10_categorical.value_counts()



f, ax = plt.subplots(figsize=(15, 5))

sns.barplot(x=keys_count.axes[0], y=keys_count.values)

ax.set(title="Most repeated keys on the top 10 songs", ylabel="Number of songs");
from wordcloud import WordCloud, STOPWORDS



t10_titles = gsongs[gsongs["position"] <= 10]["track_name"].values

wc = WordCloud(stopwords=STOPWORDS).generate(" ".join(t10_titles))



plt.figure(figsize=(20, 20))



plt.subplot(1, 2, 2)

plt.imshow(wc, interpolation='bilinear')

plt.title('Most repeated words for Top 10 song titles', fontsize=25)

plt.axis("off");
gsongs["success"] = gsongs["position"] <= 30

gsongs["success"] = gsongs["success"].astype(int)



sns.countplot(x="success", data=gsongs)
from sklearn.model_selection import train_test_split





feature_headers = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

predict_headers = ['success']



X_all = gsongs[feature_headers]

Y_all = gsongs[predict_headers]



X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42);
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

prediction = knn.predict(X_test)
from sklearn.metrics import confusion_matrix



print("La precisión de nuestro modelo es: ", knn.score(X_test, Y_test))



plot_confusion_matrix(Y_test, prediction, np.array([0, 1]));
X_all = gsongs[feature_headers]

X_all = X_all.drop(["duration_ms", "mode", "loudness", "danceability", "acousticness", "liveness", "instrumentalness"], axis=1)



X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

prediction = knn.predict(X_test)



print("La precisión de nuestro modelo es: ", knn.score(X_test, Y_test))



plot_confusion_matrix(Y_test, prediction, np.array([0, 1]));
X_all = gsongs[feature_headers]

X_all = X_all.drop(["duration_ms", "mode", "loudness", "danceability", "acousticness", "liveness", "instrumentalness"], axis=1)



X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)



test_accuracy = []

rg = np.arange(1, 25)



for i, k in enumerate(rg):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, Y_train)

    test_accuracy.append(knn.score(X_test, Y_test))

    

plt.figure(figsize=[13,8])

plt.plot(rg, test_accuracy, label = 'Precisión de test')

plt.legend()

plt.title('K VS Precisión')

plt.xlabel('K')

plt.ylabel('Precisión')

print("La mejor precisión que podemos obtener es {} con K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

    
from sklearn.ensemble import RandomForestClassifier



X_all = gsongs[feature_headers]



X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)



rfc = RandomForestClassifier(max_depth=10, random_state=43)

rfc.fit(X_train, Y_train)



df = pd.DataFrame({'group': X_train.columns, 'values': rfc.feature_importances_})



# Reorder it following the values:

ordered_df = df.sort_values(by='values')



my_range=range(1,len(df.index)+1)



plt.hlines(y=my_range, xmin=0, xmax=ordered_df['values'], color='skyblue')

plt.plot(ordered_df['values'], my_range, "o", color="skyblue")

 

# Add titles and axis names

plt.yticks(my_range, ordered_df['group'])

plt.title("Importance of features for RandomForest", loc='left')

plt.xlabel('Importance')

plt.ylabel('Feature');
useless_headers = df[df["values"] < 0.10]["group"].tolist()



filtered_headers = [header for header in feature_headers if header not in useless_headers]



filtered_headers
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification



X_all = gsongs[filtered_headers]



X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)



test_accuracy = []

rg = np.arange(1, 200)



for i, max_depth in enumerate(rg):

    rfc = RandomForestClassifier(max_depth=max_depth, random_state=47)

    rfc.fit(X_train, Y_train)

    test_accuracy.append(rfc.score(X_test, Y_test))

    

plt.figure(figsize=[13,8])

plt.plot(rg, test_accuracy, label = 'Precisión de test')

plt.legend()

plt.title('max_depth VS Precisión')

plt.xlabel('max_depth')

plt.ylabel('Precisión')

print("La mejor precisión que podemos obtener es {} con max_depth = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))