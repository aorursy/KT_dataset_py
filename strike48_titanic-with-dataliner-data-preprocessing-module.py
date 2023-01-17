! pip install -U dataliner
import pandas as pd 
import dataliner as dl
df = pd.read_csv('/kaggle/input/titanic/train.csv')
target_col = 'Survived'
X = df.drop(target_col, axis=1)
y = df[target_col]
X
trans = dl.DropColumns('PassengerId')
trans.fit_transform(X)
X_test = pd.read_csv('/kaggle/input/titanic/test.csv')
trans.transform(X_test)
X['Test_Feature'] = 1
trans = dl.DropNoVariance()
trans.fit_transform(X)
trans.drop_columns_
trans = dl.DropHighCorrelation(threshold=0.5)
trans.fit_transform(X, y)
trans.drop_columns_
trans = dl.ImputeNaN() 
trans.fit_transform(X)
trans = dl.CountRowNaN() 
trans.fit_transform(X)
trans = dl.ClipData() 
trans.fit_transform(X)
trans = dl.StandardScaling() 
trans.fit_transform(X)
trans = dl.MinMaxScaling() 
trans.fit_transform(X)
trans = dl.CountEncoding()
trans.fit_transform(X)
trans = dl.RankedCountEncoding()
trans.fit_transform(X)
trans = dl.OneHotEncoding()
trans.fit_transform(X)
trans = dl.TargetMeanEncoding()
trans.fit_transform(X, y)
trans = dl.RankedTargetMeanEncoding()
trans.fit_transform(X, y)
from sklearn.pipeline import make_pipeline

process = make_pipeline(
    dl.DropNoVariance(),
    dl.DropHighCardinality(),
    dl.BinarizeNaN(),
    dl.ImputeNaN(),
    dl.TargetMeanEncoding(),
    dl.StandardScaling(),
    dl.DropLowAUC(),
)

process.fit_transform(X, y)
process.transform(X_test)
