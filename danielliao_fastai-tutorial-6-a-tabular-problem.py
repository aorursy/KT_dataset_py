import fastai.tabular as fta
adult = fta.untar_data(fta.URLs.ADULT_SAMPLE)

df = fta.pd.read_csv(adult/'adult.csv')

dep_var = 'salary'

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

cont_names = ['education-num', 'hours-per-week', 'age', 'capital-loss', 'fnlwgt', 'capital-gain']

procs = [fta.FillMissing, fta.Categorify, fta.Normalize]
data = (fta.TabularList.from_df(df, path=adult, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_idx(valid_idx=range(800,1000))

                           .label_from_df(cols=dep_var)

                           .databunch())
data.show_batch()
learn = fta.tabular_learner(data, layers=[200,100], metrics=fta.accuracy)

learn.fit(5, 1e-2)

learn.save('mini_train')
learn.show_results()