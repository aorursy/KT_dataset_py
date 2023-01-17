import h2o

import os



from h2o.automl import H2OAutoML
h2o.init()
train = h2o.import_file("../input/petfinder-adoption-prediction/train/train.csv")

test = h2o.import_file("../input/petfinder-adoption-prediction/test/test.csv")
train.describe()
train["AdoptionSpeed"] = train["AdoptionSpeed"].asfactor()

train["desc_length"] = train["Description"].nchar()

train["name_length"] = train["Name"].nchar()

train["PureBreed"] = (train["Breed2"] == 0).ifelse(1,0)

train["SingleColor"] = (train["Color2"] == 0).ifelse(1,0)

train["Fee"] = (train["Fee"] == 0).ifelse(1,0)

train["VideoAmt"] = (train["VideoAmt"] == 0).ifelse(1,0)

test["desc_length"] = test["Description"].nchar()

test["name_length"] = test["Name"].nchar()

test["PureBreed"] = (test["Breed2"] == 0).ifelse(1,0)

test["SingleColor"] = (test["Color2"] == 0).ifelse(1,0)

test["Fee"] = (test["Fee"] == 0).ifelse(1,0)

test["VideoAmt"] = (test["VideoAmt"] == 0).ifelse(1,0)
train.head()
y="AdoptionSpeed"

x=train.columns

x.remove(y)

x.remove("PetID")

x.remove("RescuerID")

x.remove("Description")

x.remove("Name")
mA = H2OAutoML(max_models = 20, seed = 1, sort_metric = "RMSE")

mA.train(x = x, y = y ,training_frame = train)
mA.leaderboard
pred = mA.predict(test)

pred.head()
p2 = mA.leader.predict(test)
p2["predict"]
sub = test["PetID"]

pred["predict"]
test["PetID"] = sub["PetID"]

test["AdoptionSpeed"] = p2["predict"]
submission = test[["PetID","AdoptionSpeed"]]

h2o.export_file(submission,'submission.csv')