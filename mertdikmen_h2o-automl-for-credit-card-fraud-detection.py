import h2o
h2o.init(ip="localhost", port=54323)
data = h2o.import_file('../input/creditcard/creditcard.csv')
data.head()
x = data.columns
y = "Class"# response
x.remove(y)
x.remove("Id")
x.remove("Time")
train,test,validation = data.split_frame(ratios=[0.7, 0.1])
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()
# Run AutoML for 10 minutes
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_runtime_secs = 2*60)
aml.train(x = x, y = y,
          training_frame = train, leaderboard_frame = validation)
lb = aml.leaderboard
lb
model = aml.leader

preds = aml.leader.predict(test)
preds
perf_test = model.model_performance(test_data=test)
perf_test.plot()
perf_test.confusion_matrix()
merged = test.cbind(preds)
merged_pred = merged[:, ["Id", "Class","predict"]]
submission = merged_pred.as_data_frame()
submission.head()
submission.to_csv('submission.csv', index=False)