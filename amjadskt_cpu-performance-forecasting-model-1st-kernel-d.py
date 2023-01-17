# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra


#Instance based CPU performance prediction model independant of vendor name
# Data Source Information
#    -- Creators: Phillip Ein-Dor and Jacob Feldmesser
#      -- Ein-Dor: Faculty of Management; Tel Aviv University; Ramat-Aviv; 
#         Tel Aviv, 69978; Israel
#    -- Donor: David W. Aha (aha@ics.uci.edu) (714) 856-8779   
#    -- Date: October, 1987
 
#  Past Usage:
#     1. Ein-Dor and Feldmesser (CACM 4/87, pp 308-317)
#        -- Results: 
#           -- linear regression prediction of relative cpu performance
#           -- Recorded 34% average deviation from actual values 
#     2. Kibler,D. & Aha,D. (1988).  Instance-Based Prediction of
#        Real-Valued Attributes.  In Proceedings of the CSCSI (Canadian
#        AI) Conference.
#        -- Results:
#           -- instance-based prediction of relative cpu performance
#           -- similar results; no transformations required
#     - Predicted attribute: cpu relative performance (numeric)

#  Relevant Information:
#    -- The estimated relative performance values were estimated by the authors
#       using a linear regression method.  See their article (pp 308-313) for
#       more details on how the relative performance values were set.

#  Number of Instances: 209 

# 

#  Attribute Information:
#    
#     
#     MYCT: machine cycle time in nanoseconds (integer)
#     MMIN: minimum main memory in kilobytes (integer)
#     MMAX: maximum main memory in kilobytes (integer)
#     CACH: cache memory in kilobytes (integer)
#     CHMIN: minimum channels in units (integer)
#     CHMAX: maximum channels in units (integer)
#     PRP: published relative performance (integer)
#     ERP: estimated relative performance from the original article (integer)

#  Missing Attribute Values: None

#  Class Distribution: the class value (PRP) is continuously valued.
#    PRP Value Range:   Number of Instances in Range:
#    0-20               31
#    21-100             121
#    101-200            27
#    201-300            13
#    301-400            7
#    401-500            4
#    501-600            2
#    above 600          4

# Summary Statistics:
# 	   Min  Max   Mean    SD      PRP Correlation
#    MCYT:   17   1500  203.8   260.3   -0.3071
#    MMIN:   64   32000 2868.0  3878.7   0.7949
#    MMAX:   64   64000 11796.1 11726.6  0.8630
#    CACH:   0    256   25.2    40.6     0.6626
#    CHMIN:  0    52    4.7     6.8      0.6089
#    CHMAX:  0    176   18.2    26.0     0.6052
#    PRP:    6    1150  105.6   160.8    1.0000
#    ERP:   15    1238  99.3    154.8    0.9665








from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE #Visualization


with open('../input/machine.data') as f :

	_data_space = f.readlines()


data_space = []
temp = []

for str_dataset in _data_space :

	dataset = str_dataset.split(',')
	data_space.append(dataset)

			



def make_brand_independant(dataset) :

	return dataset[2:]  


def filter_and_convert(dataset) :

	temp = []

	for attrib in dataset :

		if '\n' not in attrib :

			temp.append(int(attrib))

		else :

			temp.append(int(attrib[:-1]))


	return temp


def split_dimensions(dataset) :

	y_val = dataset[-1]
	x_val = dataset[:-1]

	return x_val, y_val


__data_space = []
_temp = []
x_vals_train = []
y_vals_train = []

for dataset in data_space :

	temp.append(make_brand_independant(dataset))

for dataset in temp :

	_temp.append(filter_and_convert(dataset))

for dataset in _temp :

	x_val_temp, y_val_temp = split_dimensions(dataset)

	x_vals_train.append(x_val_temp)
	y_vals_train.append(y_val_temp)


#print x_vals_train

prediction_model = LinearRegression()
prediction_model.fit(x_vals_train, y_vals_train)

print ('Slopes of model : ', prediction_model.coef_)
print ('Intercept of model : ', prediction_model.intercept_)



# Any results you write to the current directory are saved as output.
