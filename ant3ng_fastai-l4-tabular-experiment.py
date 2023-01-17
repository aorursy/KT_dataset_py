from fastai.tabular import *
path = untar_data(URLs.ADULT_SAMPLE)

df = pd.read_csv(path/'adult.csv')



dep_var = 'salary'

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

cont_names = list((set(df.columns) - set(cat_names) - set([dep_var])))

# same as cont_names = ['age', 'fnlwgt', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']

procs = [FillMissing, Categorify, Normalize]
# Following are made in fastai notebook, but some of columns are missing (e.g. native-country, sex, hours-per-week etc..)

# Missing columns actually affect result of training in worse way

# cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']

# cont_names = ['age', 'fnlwgt', 'education-num']
test = TabularList.from_df(df.iloc[800:1000].copy(), cat_names, cont_names, path=path)
data = (TabularList.from_df(df, cat_names, cont_names, procs, path=path)

                   .split_by_idx(list(range(800,1000)))

                   .label_from_df(dep_var)

                   .add_test(test)

                   .databunch())
# Experiments

learn1 = tabular_learner(data, [200,100], metrics=accuracy) # fit           without min_grad_lr

learn2 = tabular_learner(data, [200,100], metrics=accuracy) # fit           with    min_grad_lr

learn3 = tabular_learner(data, [200,100], metrics=accuracy) # fit_one_cycle without min_grad_lr

learn4 = tabular_learner(data, [200,100], metrics=accuracy) # fit_one_cycle with    min_grad_lr
learn2.lr_find()

learn2.recorder.plot(suggestion=True)

lr2 = learn2.recorder.min_grad_lr;



learn4.lr_find()

learn4.recorder.plot(suggestion=True)

lr4 = learn4.recorder.min_grad_lr;



lr2, lr4
learn1.fit(5, 1e-2)
learn2.fit(5, lr2)
learn3.fit_one_cycle(5, 1e-2)
learn4.fit_one_cycle(5, lr4)
learn.predict(df.iloc[0])