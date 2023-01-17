from fastai.tabular import *
from sklearn import datasets

diabetes = datasets.load_diabetes()



with np.printoptions(linewidth=130):

    print('Data - first 5\n', diabetes.data[0:5,:])

    print('Target - first 5\n', diabetes.target[0:5])
df = pd.DataFrame(data=diabetes.data, columns=['age', 'sex', 'bmi', 'abp', 's1', 's2', 's3', 's4', 's5', 's6'])

df['target'] = diabetes.target

df.head()
df.shape
df.describe()
data = (TabularList.from_df(df, cont_names=df.columns[:-1])

                           .split_by_rand_pct(valid_pct=.2, seed=42)

                           .label_from_df(cols='target')

                           .databunch())
data.show_batch(rows=5)
learn=None

learn = tabular_learner(data,

                        layers=[500,200,100],

                        metrics=[explained_variance, mean_squared_error, r2_score])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, 1e-1)
learn.unfreeze()

learn.fit_one_cycle(5, 1e-3)
row = df.iloc[0]

row
learn.predict(row)
url="https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt"

df=pd.read_csv(url, sep='\t')

# change column names to lowercase for easier reference

df.columns = [x.lower() for x in df.columns]

df.head()
df.shape
df.describe()
# if target variable is int, library assumes it is a categorical

df.y = df.y.astype(float)
cat_names = ['sex']

cont_names = ['age', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

procs = [Categorify, Normalize]

data = (TabularList.from_df(df, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_rand_pct(valid_pct=.2, seed=42)

                           .label_from_df(cols='y')

                           .databunch())
data.show_batch(rows=5)
learn = tabular_learner(data,

                        layers=[500,200,100],

                        metrics=[explained_variance, mean_squared_error, r2_score])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, 1e-1)
learn.unfreeze()

learn.fit_one_cycle(5, 1e-3)
row = df.iloc[0]

row
learn.predict(row)