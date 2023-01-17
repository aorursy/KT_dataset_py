

train.head()
pc_test = patient_cancer[patient_cancer.patient > 38].reset_index(drop = True)
test = test.reset_index(drop = True)
test = pd.concat([pc_test,test],axis = 1)
train.head()
sample = train.iloc[:,2:].sample(n = 100, axis = 1) ##sample is random sample in dataset

sample["cancer"] = train.cancer 
sample
sample.describe().round()
from sklearn import preprocessing

sample = sample.drop("cancer",axis = 1)

sample.plot(kind = "hist",legend = None, bins = 20,color = 'k')

sample.plot(kind = "kde",legend = None)