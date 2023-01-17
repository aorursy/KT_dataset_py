#For basic datasets, We don't need numpy and pandas explicitly. 

import h2o 

from h2o.automl import H2OAutoML

#Initializing H2O

h2o.init()
data = h2o.import_file("/kaggle/input/bank-note-authentication-uci-data/BankNote_Authentication.csv")
data.summary()
#class is our target in this query

#This is a Classification Problem and In Summary, We can spot class as integer type. 

#So, We will change it to enum type.

data['class'] = data['class'].asfactor()
#Splitting Test & Train in 80 & 20. 

train, test = data.split_frame(ratios=[0.8])
print("{} Rows in training frame & {} Rows in testing frame".format(train.nrows,test.nrows))
#Storing name of colums. 

x = train.columns #x will contain indepedent variable. 

y = "class" #y will contain dependent variable. 

x.remove(y)
# We will run AutoML for 300sec as it is a small dataset.

# Seed will help reproduce same models

aml = H2OAutoML(max_runtime_secs = 300, seed = 127)

aml.train(x=x, y=y, training_frame=train)
lb = aml.leaderboard

lb.head() #This will give us top 10 models. 
# The leader model is stored here

aml.leader

# Let's make prediction on leader

preds = aml.leader.predict(test)
preds
#We will see performance of Leader Model on test set

aml.leader.model_performance(test)
lead_id = aml.leader.model_id

print("Model ID of leader is {}".format(lead_id))
#Hyper-parameters of leader model

out = h2o.get_model(lead_id)

out.params
#Stopping Cluster

h2o.cluster().shutdown()