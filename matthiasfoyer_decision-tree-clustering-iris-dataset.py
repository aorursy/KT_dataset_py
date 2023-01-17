import pandas as pd



iris_dataframe = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')



iris_dataframe.head()
import seaborn as sns



sns.pairplot(iris_dataframe, hue='species')
from math import log



def entropy (dataframe):

    rows = dataframe.shape[0]

    series = dataframe['species'].value_counts()

    occurences = [series.get('Iris-setosa'),series.get('Iris-versicolor'),series.get('Iris-virginica')]

    e = 0

    k = 3 # nombre de classes dans le dataset

    for i in range(0,k):

        if occurences[i] is not None:

            e += occurences[i]/rows*log(occurences[i]/rows,2)

    return -e
entropy(iris_dataframe)
def decision_tree(dataframe):

    dataframeEntropy = entropy(dataframe)

    N = dataframe.shape[0]

    G = 3 # nombre de groupes

    discBest = 0 # meilleur discriminant

    

    for i in dataframe.keys()[:-1]: # on omet le dernier element

        dfCopy = dataframe.sort_values(by=i)

        subG = []

        hGroupes = []

        for j in range(G):

            start = int(j*(N/G))

            end = int((j+1)*(N/G)-1)

            hGroupes.append(dfCopy[start:end])

            subG.append(entropy(dfCopy[start:end]))

        

        # calcul du pouvoir discriminant

        sum = 0

        for k in subG:

            sum += ((int(N/G)-1)/N)*k

        disc = dataframeEntropy-sum

        

        if disc > discBest:

            discBest = disc

            meilleurAttribut = i

            meilleursGroupes = hGroupes

    

    return (meilleurAttribut, meilleursGroupes)



decision_tree(iris_dataframe)