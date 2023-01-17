##importing relevant libraries
import numpy as np
from sklearn.model_selection import train_test_split
##generating some data
a = np.arange(1,101)
a
b = np.arange(401,501)
b
##spliting the database
a_train,a_test,b_train,b_test = train_test_split(a,b,test_size=0.2,random_state=365)
##now exploring the dataset
a_train.shape,a_test.shape
a_train
a_test
b_train
b_test
