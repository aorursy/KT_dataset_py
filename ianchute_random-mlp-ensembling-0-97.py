import pandas as pd
from sklearn import preprocessing, manifold, model_selection, decomposition, metrics
from sklearn import naive_bayes, tree, ensemble, neighbors, linear_model, svm, neural_network
import seaborn as sb
import numpy as np
%matplotlib inline

def random_mlp(n):
    model = neural_network.MLPClassifier(
        epsilon=np.random.uniform(1e-6, 1e-8),
        activation=np.random.choice(['logistic', 'tanh', 'relu']),
        alpha=np.random.uniform(0.00001, 0.01),
        batch_size=np.random.randint(100,300),
        hidden_layer_sizes=\
            np.sort([int(np.abs(np.random.normal(loc=0, scale=n)) + n)\
                     for i in range(0, np.random.randint(1,20))])[::-1],
        beta_1=np.random.uniform(low=0.888,high=0.999),
        beta_2=np.random.uniform(low=0.888,high=0.999),
        random_state=np.random.randint(1000000),
        learning_rate_init=np.random.uniform(low=0.00001,high=0.05),
        max_iter=np.random.randint(low=5000,high=20000)
    )
    print(model.get_params()['activation'], model.get_params()['hidden_layer_sizes'])
    return model
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/train.csv')
data = pd.concat([train, test]).reset_index()

len(data), len(train), len(test)
data['label'] = data['label'].fillna(-1)
X = data[data['label'] != -1]
y = X['label']
X = X.drop(['index', 'label'], axis=1)
estimators = [('mlp' + str(i), random_mlp(len(X.columns))) for i in range(20)]
model = ensemble.VotingClassifier(estimators=estimators, voting='soft')
score = model_selection.cross_val_score(estimator=model, X=X, y=y)
score.mean(), score
model.fit(X, y)
test = data[data['label'] == -1]
test = test.drop(['index', 'label'], axis=1)
results = pd.DataFrame()
results['ImageId'] = list(range(1, len(test) + 1))
results['label'] = list(map(int, model.predict(test)))

results.to_csv('results.csv', index=False, header=True)