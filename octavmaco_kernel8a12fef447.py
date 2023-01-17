import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression



# Importul și afișarea primelor cinci rânduri din setul de date privind publicitatea

advert = pd.read_csv('../input/Advertising.csv')

advert.head()
# Construirea modelului de regresie liniară multiplă utilizînd publicitatea TV și Radio ca predictori

# Separarea datelor în predictori X și rezultate Y

predictori = ['TV', 'radio']

X = advert[predictori]

y = advert['vanzari']



# Initialise and fit model

lm = LinearRegression()

model = lm.fit(X, y)
print(f'alfa = {model.intercept_}')

print(f'coeficientii_beta = {model.coef_}')
y_pred = model.predict(X)

y_pred
X_nou = [[400, 200]]

print(model.predict(X_nou))
from sklearn.metrics import r2_score

r2_score(y, y_pred) 