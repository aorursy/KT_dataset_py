import numpy as np

import pandas as pd

from patsy import dmatrices

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import metrics

import matplotlib.pyplot as plt
data = pd.read_csv("../input/train.csv")

data.head()
columns = data.iloc[:, 1:-1].columns

X = data[columns]

y = data.iloc[:, -1].ravel()

X.head()
y = np.ravel(data['target'])

y = y.ravel()

print(type(y), y.shape)
target_size = data.groupby("target").size()

target_size.plot(kind='bar')

plt.show()
for id in range(1, 10):

    plt.subplot(3, 3, id)

    data[data['target'] == "Class_" + str(id)]["feat_20"].hist()

plt.show()
plt.scatter(data['feat_19'], data['feat_20'])

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(X.corr(), interpolation='nearest')

fig.colorbar(cax)

plt.show()
import seaborn as sns

plt.subplots(figsize=(12, 9))

sns.heatmap(X.iloc[:, :].corr(), square=True, annot=False, fmt='.2f')
num_fea = X.shape[1]

M = num_fea

N = len(data['target'].unique())



neuron_n1 = int((M * N) ** 0.5 + 5)

neuron_n2 = int((M + N) * 2/3)

print((M,N), neuron_n1, neuron_n2)

# (M + N) * 2/3
model_1 = MLPClassifier(hidden_layer_sizes =(30, 10), solver='lbfgs', random_state = 1, alpha =1e-5, verbose = False)
model_1.get_params()["hidden_layer_sizes"]
model_2 = MLPClassifier(hidden_layer_sizes =(neuron_n1, 10), solver='lbfgs', random_state = 1, alpha =1e-5, verbose = False)
model_3 = MLPClassifier(hidden_layer_sizes =(neuron_n2, 10), solver='lbfgs', random_state = 1, alpha =1e-5, verbose = False)
model_4 = MLPClassifier(hidden_layer_sizes =(70, 70), solver='lbfgs', random_state = 1, alpha =1e-5, verbose = False)
import time

models = [model_1, model_2, model_3, model_4]

times = []

for i, model in enumerate(models):

    print("Fitting Model_" + str(i) + ".....", end ="")

    start_time = time.time()

    model.fit(X,y)

    end_time = time.time()

    training_time = end_time - start_time

    times.append(training_time)

    print(training_time, "Secs", ", hidden layer:", model.get_params()["hidden_layer_sizes"])

#model_1.fit(X,y)
#model_2.fit(X,y)
#model_3.fit(X,y)
#model_4.fit(X,y)
model.intercepts_
print(model.coefs_[0].shape)

print(model.coefs_[1].shape)

print(model.coefs_[2].shape)
pred = model_1.predict(X)

pred
model_1.score(X,y)
scores = []

for model in models:

    scores.append(model.score(X, y))



scores_improve = [0]

times_improve = [0]

for i in range(1, len(scores)):

    s_improve = 100 * (scores[i] - scores[i - 1]) / scores[i-1] 

    t_improve = 100 * (-1) *(times[i] - times[i - 1]) / times[i - 1]

    scores_improve.append(s_improve)

    times_improve.append(t_improve)

    

plt.title("Score change in time %")

plt.plot(scores_improve, "ro-")

plt.ylabel("%")

plt.show()



plt.title("Performance change in time %")

plt.ylabel("%")

plt.plot(times_improve, "bo-")

plt.show()
sum(pred == y) / len(y)
test_data = pd.read_csv("../input/test.csv")

Xtest = test_data.iloc[:, 1:]

Xtest.head()
prediction = model_4.predict(Xtest)

prediction_proba = model_4.predict_proba(Xtest)
solution = pd.DataFrame(prediction_proba, columns = ['Class_1', 'Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
solution['id'] = test_data['id']

col = list(solution.columns)

col = col[-1:] + col[1:]

solution = solution[col]
solution.head()
solution.to_csv("./ooto_prediction.csv", index=False)