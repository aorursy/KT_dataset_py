import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Normalizer

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
train = pd.read_csv('../input/adultpmr3508/train_data.csv',
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values='?')

test = pd.read_csv('../input/adultpmr3508/test_data.csv',
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values='?')
train.info()
train.head()
train.describe()
# Note abaixo que somente 3 feature tẽm missing data.
test.isnull().any(axis=0)
for col in train.columns:
    # Comente o if abaixo para plotar dados numéricos e categóricos. A plotagem de dados numéricos é muito lenta.
    #if train[col].dtype.name == 'object':
        train[col].value_counts().plot(title=col, kind='pie' if train[col].dtype.name == 'object' else 'bar', label='')
        plt.show()
        print()
from scipy.stats import pearsonr

# Para calcular correlações, é necessário ter dados numéricos. Bools são considerados numéricos (True=1, False=0)
income_num = train['income'] == '>50K'

print('Correlações e p-values entre algumas features e income:')
for k in ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']:
    print(k, pearsonr(train[k], income_num))
print('Frequência de valores nulos:')
print('capital.gain =', sum(train['capital.gain'] == 0) / train.shape[0] * 100)
print('capital.loss =', sum(train['capital.loss'] == 0) / train.shape[0] * 100)
def label_encode(xtrain, xtest):
    """Converte features categóricas de tipo object (string/nan) para tipo int (exceto income).
       Retorna os índices das features categóricas."""
    cat = []
    for i, col in enumerate(xtrain.columns):
        if col != 'income' and xtrain[col].dtype.name == 'object':
            cat.append(i)
            enc1 = LabelEncoder()
            xtrain[col] = enc1.fit_transform(xtrain[col].astype(str))
            xtest[col] = enc1.transform(xtest[col].astype(str)) # evita erros com nan
    return cat


def setup(features, normalize=False, dropna=False):
    """Retorna xtrain, xtest, pipeline."""
    xtrain = train[features].copy()
    xtest = test[features].copy()
    ytrain = train['income'].copy()
    if dropna:
        ytrain.drop(xtrain[xtrain.isna().any(axis=1)].index, inplace=True)
        xtrain.dropna(inplace=True)
        xtest.dropna(inplace=True)
    cat_features = label_encode(xtrain, xtest)
    
    steps = [('onehot', OneHotEncoder(categorical_features=cat_features, sparse=False))]
    if normalize:
        steps.append(('normalizer', Normalizer()))
    steps.append(('knn', KNeighborsClassifier(n_jobs=-1)))
    return xtrain, ytrain, xtest, Pipeline(steps)


def avg(nums):
    return sum(nums)/len(nums)
# Teste para ver quais parametros existe
_, _, _, pipeline = setup(['Id'], normalize=True, dropna=True)
pipeline.get_params()
# Hiperparâmetros relevantes:
# - knn__n_neighbors
# - knn__p (distância euclideana x Manhattan)
# - normalizer__norm (distância euclideana x Manhattan)

features = ['age', 'education.num', 'workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week']
xtrain, ytrain, xtest, pipeline = setup(features)

# Verificação inicial: valores pequenos de cv (cv=3, nesse caso), garantem validação cruzada mais rápida.
scores = {}
for k in [10, 20, 30, 40]:
    for p in [1, 2]:
        pipeline.set_params(knn__p=p, knn__n_neighbors=k)
        scores[k] = avg(cross_val_score(pipeline, xtrain, ytrain, cv=3))
        print('k={0:2d}  p={1}  score={2}'.format(k, p, scores[k]), flush=True)
pipeline.set_params(knn__p=1)
scores = {}
for k in range(10, 35, 2):
    pipeline.set_params(knn__n_neighbors=k)
    scores[k] = avg(cross_val_score(pipeline, xtrain, ytrain, cv=3))
    print('k={0:2d}  score={1}'.format(k, scores[k]), flush=True)

plt.plot(scores.keys(), [1 - acc for acc in scores.values()])
plt.xlabel('k')
plt.ylabel('erro')

by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print('Melhores k:', *[k for k, score in by_score])
xtrain, ytrain, xtest, pipeline = setup(features, normalize=True)
pipeline.set_params(knn__p=1)

scores_norm1 = {}
scores_norm2 = {}
for k in [14, 16, 18, 20]:
    pipeline.set_params(knn__n_neighbors=k, normalizer__norm='l1')
    scores_norm1[k] = avg(cross_val_score(pipeline, xtrain, ytrain, cv=3))
    pipeline.set_params(knn__n_neighbors=k, normalizer__norm='l2')
    scores_norm2[k] = avg(cross_val_score(pipeline, xtrain, ytrain, cv=3))
    print(k, end=' ')
fig, ax = plt.subplots()

for scores_, label in [(scores, 'no_norm'), (scores_norm1, 'norm1'), (scores_norm2, 'norm2')]:
    ax.plot(scores_.keys(), [1-acc for acc in scores_.values()], label=label)
plt.xlabel('k')
plt.ylabel('erro')
ax.legend()
xtrain, ytrain, xtest, pipeline = setup(features, dropna=True)
pipeline.set_params(knn__p=1)

scores_dropna = {}
for k in [14, 16, 18, 20]:
    pipeline.set_params(knn__n_neighbors=k)
    scores_dropna[k] = avg(cross_val_score(pipeline, xtrain, ytrain, cv=3))
    print(k, end=' ')
fig, ax = plt.subplots()
for scores_, label in [(scores, 'keepna'), (scores_dropna, 'dropna')]:
    ax.plot(scores_.keys(), [1-acc for acc in scores_.values()], label=label)
plt.xlabel('k')
plt.ylabel('erro')
ax.legend()

for k in [14, 16, 18, 20]:
    print('k={0:2}   score_keepna={1:.6f}   score_dropna={2:.6f}'.format(k, scores[k], scores_dropna[k]))
features_no_capital = list(features)
features_no_capital.remove('capital.gain')
features_no_capital.remove('capital.loss')

feature_sets = {
    'no_capital': features_no_capital,
    'country':    features + ['native.country'],
    'fnlwgt':     features + ['fnlwgt'],
    'education':  features + ['education'],
}

fig, ax = plt.subplots()
ax.plot(scores.keys(), [1-acc for acc in scores.values()], label='original')
scores_per_fts = {}
for label, fts in feature_sets.items():
    scores_per_fts[label] = {}
    xtrain, ytrain, xtest, pipeline = setup(fts)
    pipeline.set_params(knn__p=1)
    
    for k in [14, 16, 18, 20]:
        pipeline.set_params(knn__n_neighbors=k)
        scores_per_fts[label][k] = avg(cross_val_score(pipeline, xtrain, ytrain, cv=3))
        print(k, end=' ')
    print()
    ax.plot(scores_per_fts[label].keys(), [1-acc for acc in scores_per_fts[label].values()], label=label)
    
ax.legend()
plt.xlabel('k')
plt.ylabel('erro')
scores_per_fts['both'] = {}
xtrain, ytrain, xtest, pipeline = setup(features + ['native.country', 'education'])
pipeline.set_params(knn__p=1)

for k in [14, 16, 18, 20]:
    pipeline.set_params(knn__n_neighbors=k)
    scores_per_fts['both'][k] = avg(cross_val_score(pipeline, xtrain, ytrain, cv=3))
    print(k, end=' ')
print()

fig, ax = plt.subplots()
ax.plot(scores.keys(), [1-acc for acc in scores.values()], label='original')
for label in ['education', 'country', 'both']:
    ax.plot(scores_per_fts[label].keys(), [1-acc for acc in scores_per_fts[label].values()], label=label)

ax.legend()
plt.xlabel('k')
plt.ylabel('erro')
# Sem native.country

xtrain, ytrain, xtest, pipeline = setup(features)
pipeline.set_params(knn__p=1)

scores = {}
for k in range(10, 41):
    pipeline.set_params(knn__n_neighbors=k)
    scores[k] = avg(cross_val_score(pipeline, xtrain, ytrain, cv=10))
    print('k={0:2d}  score={1}'.format(k, scores[k]), flush=True)

plt.plot(scores.keys(), [1 - acc for acc in scores.values()])
plt.xlabel('k')
plt.ylabel('erro')

by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print('Melhores k:', *[k for k, _ in by_score])
# Com native.country. k está em um intervalo restringido, pois a adição de native.country aumenta o tempo de execução

xtrain, ytrain, xtest, pipeline = setup(features + ['native.country'])
pipeline.set_params(knn__p=1)

scores = {}
for k in range(15, 36):
    pipeline.set_params(knn__n_neighbors=k)
    scores[k] = avg(cross_val_score(pipeline, xtrain, ytrain, cv=10))
    print('k={0:2d}  score={1}'.format(k, scores[k]), flush=True)

plt.plot(scores.keys(), [1 - acc for acc in scores.values()])
plt.xlabel('k')
plt.ylabel('erro')
    
by_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print('Melhores k:', *[k for k, _ in by_score])
features = ['age', 'education.num', 'workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week']

def make_submission(features, k, out):
    """Cria uma submissão sem normalização ou dropna."""
    xtrain, ytrain, xtest, pipeline = setup(features)
    pipeline.set_params(knn__p=1, knn__n_neighbors=k)
    pipeline.fit(xtrain, ytrain)
    ytest = pipeline.predict(xtest)
    df=pd.DataFrame({'Id': test['Id'], 'income': ytest})
    df.to_csv(out, index=False)
    print('{0}: k={1}, features={2}'.format(out, k, features))

make_submission(features, 32, 'sub1.csv')

make_submission(features + ['native.country'], 33, 'sub2.csv')
make_submission(features + ['native.country'], 35, 'sub3.csv')
make_submission(features + ['native.country'], 24, 'sub4.csv')

# Obs: a inclusão de education nas submissões 5 e 6 foi um erro. Esse erro foi retificado nas submissões 7 e 8.
make_submission(features + ['education'], 26, 'sub5.csv')
make_submission(features + ['education'], 34, 'sub6.csv')

make_submission(features, 26, 'sub7.csv')
make_submission(features, 34, 'sub8.csv')