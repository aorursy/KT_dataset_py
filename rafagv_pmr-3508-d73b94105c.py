import pandas as pd
import sklearn

df = pd.read_csv("../input/costa-rica/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
test = pd.read_csv("../input/costa-rica/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
idt = test.Id
df.shape

ndf = df.dropna()
ndf.head(2)
ndf.columns
xndf = ndf[["v2a1", "v18q1", 'rooms', 'SQBescolari', 'SQBedjefe', 'SQBovercrowding']]
yndf = ndf.Target
xtest = test[["v2a1", "v18q1", 'rooms', 'SQBescolari', 'SQBedjefe', 'SQBovercrowding']]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=16, p=1)
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=16, p=1)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, xndf, yndf, cv=5)
scores
knn.fit(xndf,yndf)
testPred = knn.predict(xtest.fillna(0))
output = pd.DataFrame(idt)
output["Target"] = testPred
out = pd.DataFrame(output)
out.to_csv("resposta.csv", index =False)

