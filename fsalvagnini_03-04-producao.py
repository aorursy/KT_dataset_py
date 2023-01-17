from sklearn.decomposition import PCA
from sklearn import datasets

wine = datasets.load_wine()
X = wine.data
y = wine.target
import pandas as pd

df = pd.DataFrame(X)
df.describe()
!pip install joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train_scaled, y_train)

persistence = {}
persistence['scaler'] = scaler
persistence['model']  = model

dump(persistence, 'persist.joblib')
persistence = load('persist.joblib')

scaler = persistence['scaler']
model = persistence['model']

X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)

from sklearn.metrics import accuracy_score

print('Acurácia:', accuracy_score(y_test, y_pred))
