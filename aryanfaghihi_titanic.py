!pip install fastai -U 
import fastai
fastai.__version__
from fastai.tabular.all import *
path = '../input/titanic'
train_path = path + '/train.csv'
test_path = path + '/test.csv'
train = pd.read_csv(train_path)
train.head()
train.Parch.describe()
train.columns
splits = RandomSplitter(valid_pct=0.2)(range_of(train))
data = TabularPandas(
    train,
    procs=[Categorify, FillMissing,Normalize],
    cat_names = ['Pclass', 'SibSp', 'Sex', 'Parch','Cabin','Embarked'],
    cont_names = ['Age','Fare'],
    y_names='Survived',
    splits=splits)
dls = data.dataloaders(bs=64)
learn = tabular_learner(dls, metrics=accuracy, cbs=ShowGraphCallback())
learn.fit_one_cycle(150)
test = pd.read_csv(test_path)
test.head()
test = test.fillna(0)
dl = learn.dls.test_dl(test)
result = learn.get_preds(dl=dl)
results = pd.DataFrame(columns=['PassengerId', 'Survived'])
for i, row in test.iterrows():
    pass_id = row['PassengerId']
    pred = learn.predict(row)
    if (pred[1] < 0.5):
        survived = 0
    else:
        survived = 1
    
    results.loc[i] = [pass_id, survived]
        
results
results.to_csv('preds.csv', index=False)
