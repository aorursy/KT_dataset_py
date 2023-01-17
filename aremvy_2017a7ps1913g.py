import numpy as np

import pandas as pd



from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import MultiTaskElasticNet

from sklearn.linear_model import BayesianRidge

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor



import pickle

import matplotlib.pyplot as plt 


class Dataset:

  

  def __init__(self, name, df, use_all=False):

    """

    Creates X_train, y_train, means and std_dev using the main dataframe

    """

    self.name = name

    

    if name!='all':

      self.df = df[df[name]==1.0]

    else:

      self.df = df

    

    self.X_train_raw = np.asarray(self.df)[:,1:96]

    self.y_train_raw = np.asarray(self.df['label'])



    self.X_train = self.X_train_raw

    self.y_train = self.y_train_raw

    

    self.means = np.asarray(self.df.mean())[1:96]

    self.means[-1] = 0

    self.std_dev = np.asarray(self.df.std())[1:96]

    self.std_dev[-1] = 1

    

    self.X_val, self.y_val = None, None



    self.scaler, self.pca = None, None



    print("Agent name: ", name)

    print("Shapes", self.X_train.shape, self.y_train.shape)

    print("Means and std_dev shapes", self.means.shape, self.std_dev.shape)



  def train_val_split():

    pass



  def mean_center(self):



    self.X_train = self.X_train-self.means

    #print(self.name, np.mean(self.X_train, axis=0))



  def standardize(self):

    self.scaler = StandardScaler(copy=False)

    self.scaler.fit_transform(self.X_train)



    #Test data can be scaled using this 'scaler' object using scaler.transform

    #Foe inverse transform, us scaler.inverse_transform(x)



  def PCA(self, num_components):

    self.pca = PCA(n_components=num_components)

    self.X_train = self.pca.fit_transform(self.X_train)

    #print(self.X_train.shape)



    #This pca object can now be used to transform test sets with the same PCA
from google.colab import drive

drive.mount('/content/drive')
#!ls drive/My\ Drive/Acads/ML\ minor\ project
#filename = 'train.csv'

filename = "/content/drive/My Drive/Acads/ML minor project/train.csv"

df = pd.read_csv(filename)



df.head(10)
for key, val in df.iteritems():

  df[key].clip(upper = df[key].mean() + df[key].std()*5)

  df[key].clip(lower = df[key].mean() - df[key].std()*5)
"""

Make a dictionary of 7 datasets for 7 agents

"""



datasets = {}



for i in range(7):

  name = 'a'+str(i)

  datasets[name] = Dataset(name, df)
#CHECKS



datasets['a1'].std_dev
"""

Mean - center

"""

for ai in datasets:

  datasets[ai].mean_center()
df['label'].std()
"""

count = max number of zeros found in std_dev.

These features can definitely be eliminated

"""



count = min([sum([(std == 0) for std in datasets[ai].std_dev]) for ai in datasets])

count
"""

PCA

"""

#HYPERPARAMETER 1: NUM_ELIM

NUM_ELIM = 30



min_components = datasets['a0'].X_train.shape[1] - count



num_components = min_components - NUM_ELIM



for ai in datasets:

  datasets[ai].PCA(num_components)
#CHECKS

#ANALYSIS



for ai in datasets:

  print("Std Dev "+ai+": ", np.std(datasets[ai].X_train, axis=0))
for ai in datasets:

  datasets[ai].standardize()



#CHECKS



datasets['a3'].X_train
class Trainer:



  def __init__(self, X_train, y_train):

    self.X_train = X_train

    self.y_train = y_train



    self.X_val, self.y_val = None, None



    self.standardization_weights = None 

    #These are separate from the dataset standardization scaler params

    #These will also be used with a transform (not fit) on the test data



    self.num_features = X_train.shape[1]-1

    self.shapeX = X_train.shape

    self.shapeY = y_train.shape

    self.max_power = 1

    self.products = False

    self.reciprocals = False

    self.exponentials = False



    self.model = None

    



  def train_val_split(self, split = 0.2):

    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=split, random_state=42)

    

    self.shapeX = self.X_train.shape

    self.shapeY = self.y_train.shape



  """

  Dataset modification functions

  """

  def add_powers(self, power):

    num_features = self.X_train.shape[1]-1



    X = np.zeros([self.X_train.shape[0], num_features*power+1])

    

    for i in range(power):

      X[:, i*num_features:(i+1)*num_features] = np.power(self.X_train[:,:num_features],i+1)



    X[:,-1] = self.X_train[:,-1]

    self.X_train = X



    self.shapeX = self.X_train.shape

    self.shapeY = self.y_train.shape

    self.max_power = power



    print(self.shapeX)

  

  

  

  def add_products(self, start, end):

    num_columns = end - start

    num_products = (num_columns*(num_columns-1))/2

    

    X = np.zeros([self.X_train.shape[0], self.X_train.shape[1]+int(num_products)])

    X[:, :self.X_train.shape[1]] = self.X_train



    count =0

    for i in range(num_columns-1):

      for j in range(i+1, num_columns):

        X[:,self.X_train.shape[1]+count] = np.multiply(X[:,start+i], X[:,start+j])

        count = count+1

    self.X_train = X



    self.shapeX = self.X_train.shape

    self.shapeY = self.y_train.shape

    self.products = True







  def add_reciprocals(self, start, end):

    num_columns = end-start



    X=np.zeros([self.X_train.shape[0], self.X_train.shape[1]+num_columns])

    X[:, :self.X_train.shape[1]] = self.X_train



    for i in range(num_columns):

      X[:,self.X_train.shape[1]+i] = 1/(abs(self.X_train[:,start+i])+0.001)



    self.X_train = X



    self.shapeX = self.X_train.shape

    self.shapeY = self.y_train.shape

    self.reciprocals = True



  def add_exp(self, start, end):

    num_columns = end-start



    X=np.zeros([self.X_train.shape[0], self.X_train.shape[1]+num_columns])

    X[:, :self.X_train.shape[1]] = self.X_train



    for i in range(num_columns):

      X[:,self.X_train.shape[1]+i] = np.exp(self.X_train[:,start+i])



    self.X_train = X

    

    self.shapeX = self.X_train.shape

    self.shapeY = self.y_train.shape

    self.exponentials = True





  """

  Preproc after adding features:

  """

  def standardize(self):

    self.standardization_weights = StandardScaler(copy=False)

    self.standardization_weights.fit_transform(self.X_train)



    #Test data can be scaled using this 'scaler' object using scaler.transform

    #Foe inverse transform, us scaler.inverse_transform(x)



  """

  Model

  """

  def L2(self, lambd= 1.0, max_iter=1000, verbose = False):

    self.model = Ridge(alpha = lambd, max_iter=max_iter)



  def L1(self, lambd =1.0,max_iter=1000, verbose = False):

    self.model = Lasso(alpha=lambd, max_iter=max_iter)



  def Elastic(self, lambd=0.1, verbose = False):

    self.model = MultiTaskElasticNet(alpha = lambd)

  """

  Train

  """



  def train(self):

    self.model.fit(self.X_train, self.y_train)



  """

  Evaluate

  """

  def predict(self, X):

    y = self.model.predict(X)

    return y



  def rmse(self, y_true, y_pred):

    return mean_squared_error(y_true, y_pred)





  def evaluate(self):

    train_pred = self.model.predict(self.X_train)

    val_pred = self.model.predict(self.X_val)



    print("Training RMSE = ", self.rmse(self.y_train, train_pred))

    print("Validation RMSE = ", self.rmse(self.y_val, val_pred))



#TEST TRAINER CLASS



test_arr = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])



test_trainer = Trainer(test_arr, np.array([1,2]).T)



print(test_trainer.X_train)



#

test_trainer.add_exp(0,2)



print(test_trainer.X_train)
#L1 model for a0 (trial)



trainer_l1_a0 = Trainer(datasets['a0'].X_train, datasets['a0'].y_train)



trainer_l1_a0.add_powers(3)

trainer_l1_a0.add_products(0, trainer_l1_a0.num_features)

#trainer_l1_a0.add_reciprocals(0, trainer_l1_a0.num_features)

#trainer_l1_a0.add_exp(0, trainer_l1_a0.num_features)



trainer_l1_a0.standardize()



trainer_l1_a0.shapeX
trainer_l1_a0.L1(0.02)

trainer_l1_a0.train_val_split(0.2)

trainer_l1_a0.train()



y_pred = trainer_l1_a0.predict(trainer_l1_a0.X_val)

trainer_l1_a0.evaluate()
trainer_l2_a0 = Trainer(datasets['a0'].X_train, datasets['a0'].y_train)

#trainer_l2_a0.add_powers(2)

#trainer_l2_a0.add_products(0, trainer_l2_a0.num_features)

#trainer_l2_a0.add_reciprocals(0, trainer_l2_a0.num_features)

#trainer_l2_a0.add_exp(0, trainer_l2_a0.num_features)



trainer_l2_a0.standardize()



print(trainer_l2_a0.shapeX)



trainer_l2_a0.L2(3.0)

trainer_l2_a0.train_val_split(0.2)

trainer_l2_a0.train()



y_pred = trainer_l2_a0.predict(trainer_l2_a0.X_val)

trainer_l2_a0.evaluate()
#L2

#data = Dataset('all', df)

#data.standardize()

#trainer_l2_all = Trainer(data.X_train, data.y_train)



#Uncommented:

trainer_l2_all = Trainer(datasets['a1'].X_train, datasets['a1'].y_train)



#

trainer_l2_all.add_powers(2)

trainer_l2_all.add_products(0, trainer_l2_all.num_features)

trainer_l2_all.add_reciprocals(0, trainer_l2_all.num_features)

trainer_l2_all.add_exp(0, trainer_l2_all.num_features)



trainer_l2_all.standardize()



print(trainer_l2_all.shapeX)



trainer_l2_all.L1(0.1, 8000)

#trainer_l2_all.model = RandomForestRegressor()



trainer_l2_all.train_val_split(0.2)

trainer_l2_all.train()



y_pred = trainer_l2_all.predict(trainer_l2_all.X_val)

trainer_l2_all.evaluate()

for a in datasets:

  print("Agent: ",a)



  trainer = Trainer(datasets[a].X_train, datasets[a].y_train)

  #

  #trainer.add_powers(2)

  #trainer.add_products(0, trainer.num_features)

  #trainer.add_reciprocals(0, trainer.num_features)

  #trainer.add_exp(0, trainer.num_features)



  #trainer.standardize()



  print(trainer.shapeX)



  #trainer.L1(0.1, 8000)

  trainer.model = RandomForestRegressor(n_estimators=100, verbose=1)

  #trainer.model = GradientBoostingRegressor(learning_rate = 0.18, n_estimators=250, verbose=1, n_iter_no_change=1)



  trainer.train_val_split(0.2)

  trainer.train()



  

  y_pred = trainer.predict(trainer.X_val)

  trainer.evaluate()



  modelname = 'RF_minimal_' + a +'.sav'

  pickle.dump(trainer.model, open(modelname, 'wb'))

  print()

!mkdir drive/My\ Drive/Acads/ML\ minor\ project/minimal_RF
!mv RF_min* drive/My\ Drive/Acads/ML\ minor\ project/minimal_RF
!rm GBR*
filename = "/content/drive/My Drive/Acads/ML minor project/test.csv"

dft = pd.read_csv(filename)

dft.head()


for key, val in dft.iteritems():

  dft[key].clip(upper = df[key].mean() + df[key].std()*5)

  dft[key].clip(lower = df[key].mean() - df[key].std()*5)
testset = {}

X_test ={}





for i in range(7):

  name = 'a'+str(i)

  testset[name] = pd.DataFrame(dft[dft[name]==1.0])

  X_test[name] = np.asarray(testset[name])[:,1:96]

  print(name, X_test[name].shape)
models={}



for i in range(7):

  name = 'a'+str(i)

  modelname = 'drive/My Drive/Acads/ML minor project/minimal_RF/RF_minimal_' + a +'.sav'

  model = pickle.load(open(modelname, 'rb'))

  models[name] = model
models
#lol
predictions = {}



for i in range(7):

  name = 'a'+str(i)

  predictions[name] = models[name].predict(X_test[name])



predictions
testset['a1']['id'].shape, predictions['a1'].shape
results = {}



for i in range(7):

  name = 'a'+str(i)



  results[name] = pd.DataFrame({'id':testset[name]['id'], 'label':predictions[name]})
results['a0']
final_df = pd.DataFrame(results['a0'])



for a in range(1,7):

  name = 'a'+str(a)

  final_df = final_df.append(results[name])



final_df=final_df.sort_values('id')
final_df.to_csv('Sub1.csv',index=False)