from fastai.tabular import TabularDataBunch, Normalize, tabular_learner, accuracy

from pandas import read_csv

from sklearn.metrics import classification_report
df = read_csv('../input/diabetes.csv')

df.head()
path = '../'

dep_var = 'Outcome'

cont_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

valid_idx = list(range(len(df) - int(len(df)*0.1), len(df)))

procs = [Normalize]



data = TabularDataBunch.from_df(path=path, df=df, dep_var=dep_var, cont_names=cont_names, valid_idx=valid_idx, procs=procs)

learner = tabular_learner(data=data, metrics=accuracy, layers=[200, 100])

learner.fit_one_cycle(10)
preds = [int(learner.predict(df.iloc[idx])[0]) for idx in valid_idx]

actuals = df.iloc[valid_idx].Outcome.values

print(classification_report(y_pred=preds, y_true=actuals))