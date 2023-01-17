import pandas as pd
raw_data = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')
raw_data.head()
raw_test = pd.read_csv('../input/contradictory-my-dear-watson/test.csv')
raw_combined = pd.concat([raw_data, raw_test])
!pip install laserembeddings
import os
for l in raw_combined.lang_abv.unique():
    os.system(f'pip install laserembeddings[{l}]')
!python -m laserembeddings download-models
from laserembeddings import Laser
import numpy as np

laser = Laser()

vectors = []

for i, r in raw_combined.iterrows():
    vectors.append(
        np.concatenate(
            (
                laser.embed_sentences(r.hypothesis, lang=r.lang_abv)[0], 
                laser.embed_sentences(r.premise, lang=r.lang_abv)[0],
            )
        )
    )
vectors = np.array(vectors)
data = np.concatenate(
    (
        pd.get_dummies(raw_combined.lang_abv, drop_first=True).to_numpy(),
        vectors
    ),
    axis=1
)
X = data[:len(raw_data)]
y = raw_data.label.to_numpy()
test = data[len(raw_data):]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
### SVC
from sklearn.svm import LinearSVC

model_svc = LinearSVC(random_state=1, dual=False)
model_svc.fit(X_train, y_train)

print(f'Model test accuracy: {model_svc.score(X_test, y_test)*100:.3f}%')
result_svc = model_svc.predict(test)
### Logistic
from sklearn.linear_model import LogisticRegression

model_logit = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000)
model_logit.fit(X_train, y_train)

print(f'Model test accuracy: {model_logit.score(X_test, y_test)*100:.3f}%')
result_logit = model_logit.predict(test)
### Random Forest
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=3500, max_depth=35, random_state=0)
model_rf.fit(X_train, y_train)

print(f'Model test accuracy: {model_rf.score(X_test, y_test)*100:.3f}%')
result_rf = model_rf.predict(test)
from keras.models import Sequential
from keras import layers
input_dim = X_train.shape[1]

model_nn = Sequential()
model_nn.add(layers.Dense(60, input_dim=input_dim, activation='relu'))
model_nn.add(layers.Dense(30, input_dim=input_dim, activation='relu'))
model_nn.add(layers.Dense(1, activation='sigmoid'))
model_nn.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# model_nn.summary()

history = model_nn.fit(X_train, y_train,
                    epochs=50,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model_nn.evaluate(X_train, y_train, verbose=False)
print(f'Loss/Accuracy:', loss, accuracy)

result_nn = [r[0] for r in model_nn.predict_classes(test)]