from tpot import TPOTClassifier

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

import numpy as np
#load the data

digits = load_digits();

data = digits.data;

print(digits.target);

x_train, x_test, y_train, y_test = train_test_split(digits.data.astype('float64'), digits.target.astype('float64'), train_size=0.75, test_size=0.25);



tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2);

tpot.fit(x_train, y_train);

print(tpot.score(x_test, y_test));

tpot.export('tpot_mnist_pipline.py');