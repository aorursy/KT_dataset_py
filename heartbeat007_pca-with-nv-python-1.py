import h2o

h2o.init()
url = "https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv"
df = h2o.import_file(url)
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator
df.pca = H2OPrincipalComponentAnalysisEstimator(k=4)
response = "Name"
fm = df.col_names
fm.remove(response)
df.pca.train(x=fm, training_frame=df)
df
df.pca.varimp(use_pandas=True)
df.pca
df_new = df.pca.predict(df)
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator
nv = H2ONaiveBayesEstimator()
df_new  = df_new.cbind(df[response])
df_new
fm = df_new.col_names

response = "Name"

fm.remove(response)
nv.train(fm,response,df_new)
nv
### tou can split the data in train and test

### but i leave it their