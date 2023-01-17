from sklearn.model_selection import train_test_split



import pandas as pd





df = pd.read_csv('../input/videogamesales/vgsales.csv')

train, test = train_test_split(df, test_size=0.2, random_state=0)

print(f"Dimensione del dataset di training: {train.size}")

df.head()
import matplotlib.pyplot as plt



plt.xlabel("NA sales")

plt.ylabel("EU sales")

plt.xscale("symlog")

plt.scatter(train.NA_Sales, train.EU_Sales)
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression



import numpy as np





# Configurazione necessaria per disegnare i grafici

X_seq = np.linspace(train.NA_Sales.values.min(),train.NA_Sales.values.max(),300).reshape(-1,1) # crea dei valori adatti per il plot della curva ideale

fig, axs = plt.subplots(3, 3, figsize=(20, 10)) # divide la figura in 9 quadranti

for ax in axs.flat:

        ax.set(xlabel="NA sales", ylabel="EU sales", xscale="symlog") # associa le etichette agli assi dei grafici e ne imposta la scala dell'asse x

for ax in fig.get_axes():

    ax.label_outer() # "Raggruppa" le etichette degli assi: mostra solo quelle degli assi più esterni





results = []

    

for degree in range(1,10):

    model = make_pipeline(

      PolynomialFeatures(degree=degree),

      LinearRegression()

    )

    model.fit(train.NA_Sales.values.reshape(-1,1), train.EU_Sales)

    

    # Valuta il modello appena ottenuto

    train_score = model.score(train.NA_Sales.values.reshape(-1,1), train.EU_Sales)

    test_score = model.score(test.NA_Sales.values.reshape(-1,1), test.EU_Sales)

    norm = np.linalg.norm(model.named_steps["linearregression"].coef_)

    results.append([train_score, test_score, norm])

    

    plot = axs[(degree-1) // 3, (degree-1) % 3]

    plot.scatter(train.NA_Sales,train.EU_Sales)

    plot.plot(X_seq,model.predict(X_seq),color="black")

    plot.set_title("Polynomial regression with degree "+str(degree))



res_df = pd.DataFrame(data=results, columns=["training score", "validation score", "norm"])

res_df[["training score", "validation score", "norm"]]

import numpy as np



from sklearn.linear_model import Ridge



alphas = np.logspace(-10, 0.1, 200)



# Configurazione necessaria per disegnare i grafici

X_seq = np.linspace(train.NA_Sales.values.min(),train.NA_Sales.values.max(),300).reshape(-1,1) # crea dei valori adatti per il plot della curva ideale

fig, axs = plt.subplots(2, 4, figsize=(30, 10)) # divide la figura in quadranti



for ax in fig.get_axes():

    ax.label_outer() # "Raggruppa" le etichette degli assi: mostra solo quelle degli assi più esterni





for degree in range(1, 8, 2):

    results = []

    

    for alpha in alphas:

        model = make_pipeline(

          PolynomialFeatures(degree=degree),

          Ridge(alpha, normalize=True)

        )

        model.fit(train.NA_Sales.values.reshape(-1,1), train.EU_Sales)



        # Valuta il modello appena ottenuto

        train_score = model.score(train.NA_Sales.values.reshape(-1,1), train.EU_Sales)

        test_score = model.score(test.NA_Sales.values.reshape(-1,1), test.EU_Sales)

        norm = np.linalg.norm(model.named_steps["ridge"].coef_)

        results.append([alpha, train_score, test_score, norm])

    

    res_df = pd.DataFrame(data=results, columns=["alpha", "training score", "validation score", "norm"])

    res_df[["alpha", "training score", "validation score", "norm"]]

    

    # scores / alpha

    plot = axs[0, (degree-1) // 2]

    plot.set(xlabel="Alpha", ylabel="Score", xscale="log")

    plot.plot(res_df["alpha"], res_df["validation score"], label="Validation")

    plot.plot(res_df["alpha"], res_df["training score"], label="Training")

    plot.set_title("Ridge regression with degree "+str(degree))

    plot.legend()

    

    

    # norm / alpha

    plot = axs[1, (degree-1) // 2]

    plot.set(xlabel="Alpha", ylabel="Norm", xscale="log")

    plot.plot(res_df["alpha"], res_df["norm"])

    plot.set_title("Ridge regression with degree "+str(degree))

    

    

#     min_score = res_df["validation score"].min()

#     max_score = res_df["validation score"].max()

#     print("min score: {}, max score: {} [delta {}]\n".format(min_score, max_score, max_score - min_score ))