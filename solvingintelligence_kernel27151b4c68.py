# Modelo e parâmetros utilizados na melhor performance obtida:

from sklearn.ensemble import ExtraTreesClassifier

modelo=ExtraTreesClassifier(n_estimators=1500, max_depth=19, criterion='entropy', min_samples_split=4, min_samples_leaf=3, bootstrap=True, max_features=None, n_jobs=-1)

#modelo.fit(x_treino, y_treino)