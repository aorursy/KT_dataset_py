import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA, KernelPCA

from sklearn.ensemble import RandomForestRegressor

%pylab inline

from matplotlib import cm

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/solutions.csv')

df.info()

df.Status.unique()

df = df[['UserID', 'QCode', 'Status']].dropna()



df.Status.unique()

mapping = {'accepted': 1, 'wrong answer': 0, 'compilation error': 0, 'time limit exceeded': 0,

           'runtime error(SIGSEGV)': 0, 'runtime error(NZEC)': 0, 'runtime error(SIGABRT)': 0,

           'runtime error(SIGFPE)': 0, 'runtime error(SIGXFSZ)': 0,

           'runtime error(OTHER)': 0, 'internal error': 0, 'running..': 0,

           'compiling..': 0, 'running judge..': 0

          }

df['Status'] = df.Status.map(mapping)

ctab = pd.crosstab(df.UserID, df.QCode, values=df.Status, aggfunc=pd.np.sum).fillna(0)



y = ctab.sum(axis=1)

ctab['target'] = y

ctab
rf = RandomForestRegressor()

rf.fit(ctab.drop('target', axis=1), y)
varexp = list(sorted(zip(rf.feature_importances_, ctab.columns), key=lambda x: x[0], reverse=True))

var = list(zip(*varexp))[0]

plt.plot(np.cumsum(var))

plt.xlabel('Question number')

plt.ylabel('Variance in user score explained')

plt.title('When user score = no of questions they got right, which questions explain maximum variance?')
Q = pd.read_csv('../input/questions.csv')

Q.info()
# Let's get the top 10 questions.

N = 10



qnames = list(zip(*varexp))[1][:N]

data = []

for q in qnames:

    qs = Q.loc[Q.QCode == q][['QCode', 'Title', 'statement']].values

    data.append(qs[0])



fizz = pd.DataFrame(data, columns=['QCode', 'Title', 'statement'])

fizz.style.applymap(lambda x: 'word-wrap: normal')