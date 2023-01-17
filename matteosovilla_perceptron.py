import numpy as np

from tqdm import trange

"""

trange serve a disegnare barre di avanzamento. 

Per esempi di uso si rimanda a https://medium.com/better-programming/python-progress-bars-with-tqdm-by-example-ce98dbbc9697

"""

from numpy import sign



class Perceptron:



  def __init__(self, random_state=None, max_step=100, rate=0.0001):

    """

    random_state permette di rendere ripetibili i risultati ottenuti pur utilizzando una componente casuale.

    come criterio di stop si considera il numero massimo di passi, si potrebbero valutare criteri più fini

    basandosi sugli score del modello (es: negli ultimi N passi il miglioramento è stato minore di una soglia)

    """

    self.rate = rate

    self.rng = np.random.RandomState(seed=random_state)

    self.max_step = max_step

    self.rate = rate

    self.best_score = 0



  def _perceptron_step(self, x, y):

    """

    passo base del perceptron, aggiorna i diversi pesi w in base al rate

    """

    for i, p in enumerate(x):

      # aggiungo 1 per la costante b

      p = np.append([1.0], p)

      t = y[i]

      o = self._output(p)

      if o != t:

        self.w = self.w + self.rate * (t - o) * p



  def _output(self, p):

    return sign(np.dot(p, self.w))



  def score(self, x, y):

    """

    score restituisce l'accuratezza media, ovvero il numero di casi correttamente etichettati diviso per il numero totale di test

    """

    correct = 0

    for i, p in enumerate(x):

      p = np.append([1.0], p)

      if self._output(p) == y[i]:

        correct += 1

    return correct / len(y)

  

  def fit(self, x, y):

    """

    fit genera casualmente un vettore di pesi w e ripete max_pass volte il perceptron.

    mantiene il miglior valore di w

    """

    # + 1 perchè un peso corrisponde alla costante b (traslazione della retta)

    self.w = self.rng.random_sample(x.shape[1] + 1)

    with trange(self.max_step) as t:

      for i in t:

        self._perceptron_step(x, y)

        score = self.score(x, y)

        t.set_description(f"step {i}")

        t.set_postfix(score=score)

        if score > self.best_score:

          self.best_score = score

          self.best_w = self.w

        if score == 100:

          break

    self.w = self.best_w

  

  def predict(self, x):

    y = []

    for p in x:

      y.append(self._output(p))

    return np.array(y)
import pandas as pd





from sklearn.model_selection import train_test_split





df = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')



# devo preprocessare i dati perché le classi siano +1 e -1

df["blueWins"] = df["blueWins"].replace([0],-1)

df = df.drop(columns=['gameId'])



display(df.head())



train, test = train_test_split(df, test_size=0.2, random_state=0)



# separo i dati sulle vittorie dal resto del dataset

y_train = train["blueWins"].values

y_test = test["blueWins"].values

x_train = train.drop(columns=['blueWins']).values

x_test = test.drop(columns=['blueWins']).values



print(f"Dimensione del dataset di training: {train.size}\n")



model = Perceptron(random_state=1, max_step=1000, rate=0.000001)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)



print(f"Score finale: {score}")