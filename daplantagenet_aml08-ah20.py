import h2o

from h2o.automl import H2OAutoML, get_leaderboard



h2o.init()
# Import train/test dataset into H2O

df = h2o.import_file("../input/bank-marketing/bank-additional-full.csv", sep=";")
df.describe()
# convert response column to a factor

df["y"] = df["y"].asfactor()



# set the predictor names and the response column name

predictors = df.columns

response = "y"

predictors.remove(response)



# split into train and validation sets

train, test, valid = df.split_frame(ratios=[0.7, 0.15], seed = 42)
# train model

aml = H2OAutoML(max_models=20, max_runtime_secs=600, seed=42)



aml.train(x = predictors, y = response, training_frame = train, leaderboard_frame = valid)
# AutoML Leaderboard

lb = aml.leaderboard



# Optional: add extra model information to the leaderboard

lb = get_leaderboard(aml, extra_columns='ALL')
lb
# plot performance for the validation data

perf_valid = aml.leader.model_performance(test_data=valid)

perf_valid.plot()
# plot performance for the test data

perf_test = aml.leader.model_performance(test_data=test)

perf_test.plot()
# Error metrics

perf_test
perf_test.confusion_matrix()
aml.leader.varimp_plot()
aml.leader.partial_plot(data = train, cols = ["duration","emp.var.rate"], server=True, plot=True)
lb.head(rows=lb.nrows)
print (aml.leaderboard.as_data_frame()['model_id'])
glm=h2o.get_model(aml.leaderboard.as_data_frame()['model_id'][14]) # adjust index number as needed
print(h2o.get_model(glm))
glm.std_coef_plot()
# Get model ids for all models in the AutoML Leaderboard

model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])



# Get the "All Models" Stacked Ensemble model

se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])



# Get the Stacked Ensemble metalearner model

metalearner = h2o.get_model(se.metalearner()['name'])
# check the contributions of base models in the ensemble to the GLM metalearner

metalearner.std_coef_plot()
import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results written to the current directory are saved as output