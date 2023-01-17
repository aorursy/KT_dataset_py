from fastai.tabular import *
path = untar_data(URLs.ADULT_SAMPLE)

df = pd.read_csv(path/'adult.csv')

df.head()
dep_var = 'salary'

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']

cont_names = ['age', 'fnlwgt', 'education-num']

procs = [FillMissing, Categorify, Normalize]
test = TabularList.from_df(df.iloc[800:1000].copy(), path=path, cat_names=cat_names, cont_names=cont_names)
data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_idx(list(range(800,1000)))

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch())
data.show_batch(rows=10)
learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
learn.fit(1, 1e-2)
row = df.iloc[0]
learn.predict(row)