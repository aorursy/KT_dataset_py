import pandas as pd
df_spotify = pd.read_csv('../input/data.csv', sep=',')
df_spotify.drop('id', axis=1, inplace=True)

df_spotify.drop('song_title',axis=1, inplace=True)
df_spotify.head()
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

inteiros = enc.fit_transform(df_spotify['artist'])

df_spotify['artistas_inteiros'] = inteiros

df_spotify.drop('artist', axis=1, inplace=True)
df_spotify.columns
features = ['acousticness', 'danceability', 'duration_ms', 'energy',

       'instrumentalness', 'key', 'liveness', 'loudness', 'mode',

       'speechiness', 'tempo', 'time_signature', 'valence',

       'artistas_inteiros']

classes = ['target']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df_spotify[features], df_spotify[classes], test_size=0.20)

df_spotify[classes].head()
## from sklearn.model_selection import train_test_split

## X_train, X_test, y_train, y_test = train_test_split(df_spotify, classes)

from sklearn.svm import SVC
classificador_svm = SVC()
classificador_svm.fit(X_train,y_train)
y_pred = (classificador_svm.predict(X_test)).reshape(-1,1)

y_test.shape, y_pred.shape
print(classificador_svm.score(X_test,y_test))
from sklearn.model_selection import cross_val_predict
resultados = cross_val_predict(classificador_svm, df_spotify[features], df_spotify[classes], cv=10)
from sklearn import metrics
print (metrics.accuracy_score(y_test, y_pred))
# Importe as bibliotecas de Pipelines e Pré-processadores



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler
# Criando Pipeline



MD_01 = Pipeline([

    ('scaler', StandardScaler()),

    ('classificador', SVC(kernel="rbf", C=100, gamma=0.01))

])
def Acuracia(clf,X,y):

    y_pred = cross_val_predict(clf, X, y, cv=5)

    return metrics.accuracy_score(y,y_pred)
# Teste o modelo usando o pipeline(criado anteriormente)

Acuracia(MD_01,df_spotify[features],df_spotify[classes])
MD_02 = Pipeline([

    ('scaler', StandardScaler()),

    ('classificador', SVC(kernel="rbf", C=100, gamma=0.01))

])



MD_03 = Pipeline([

    ('scaler', StandardScaler()),

    ('classificador', SVC(kernel="linear", C=100, gamma=0.01))

])



MD_04 = Pipeline([

    ('scaler', StandardScaler()),

    ('classificador', SVC(kernel="poly", C=100, gamma=0.01))

])



MD_05 = Pipeline([

    ('scaler', StandardScaler()),

    ('classificador', SVC(kernel="rbf", C=100, gamma=0.1))

])



MD_06 = Pipeline([

    ('scaler', StandardScaler()),

    ('classificador', SVC(kernel="linear", C=100, gamma=0.1))

])



MD_07 = Pipeline([

    ('scaler', StandardScaler()),

    ('classificador', SVC(kernel="poly", C=100, gamma=0.1))

])
def Acuracia(clf,X,y):

    y_pred = cross_val_predict(clf, X, y, cv=5)

    return print("Acurácia para a Pipeline  é de ", round(metrics.accuracy_score(y,y_pred), 2)*100,"%.")
Acuracia(MD_01,df_spotify[features],df_spotify[classes])

Acuracia(MD_02,df_spotify[features],df_spotify[classes])

Acuracia(MD_03,df_spotify[features],df_spotify[classes])

Acuracia(MD_04,df_spotify[features],df_spotify[classes])

Acuracia(MD_05,df_spotify[features],df_spotify[classes])

Acuracia(MD_06,df_spotify[features],df_spotify[classes])

Acuracia(MD_07,df_spotify[features],df_spotify[classes])
