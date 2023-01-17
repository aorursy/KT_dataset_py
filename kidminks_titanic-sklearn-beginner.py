# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LogisticRegression
import random
def get_data_frame(file_name):
	data_frame = pd.read_csv(file_name)
	data_frame["Age"] = data_frame["Age"].fillna(data_frame["Age"].mean())
	data_frame["Age"] = data_frame["Age"] // 5
	data_frame["Age"] = data_frame["Age"]/25
	mapping = {'male': 1, 'female': 2}
	data_frame = data_frame.replace({'Sex': mapping})
	embarked = ['C', 'S', 'Q']
	data_frame["Embarked"] = data_frame["Embarked"].fillna(random.choice(embarked))
	mapping = {'C': 1, 'Q': 2, 'S': 3}
	data_frame = data_frame.replace({'Embarked': mapping})
	return data_frame

def output(data_frame):
	return data_frame.as_matrix(columns=["Survived"])

def input_data(data_frame, is_Train):
	p_id = data_frame["PassengerId"]
	del data_frame["PassengerId"]
	del data_frame["Name"]
	del data_frame["Ticket"]
	del data_frame["Fare"]
	del data_frame["Cabin"]
	if is_Train:
		del data_frame["Survived"]
		return data_frame.as_matrix()
	return data_frame.as_matrix(), p_id
data_frame_train = get_data_frame('../input/train.csv')
train_output = output(data_frame_train)
train_input = input_data(data_frame_train, True)
data_frame_test = get_data_frame('../input/test.csv')
test_input, p_id = input_data(data_frame_test, False)
train_output = np.squeeze(train_output)
print(train_input.shape)
print(train_output.shape)
print(test_input.shape)
model = LogisticRegression()
model.fit(train_input, train_output)
model.score(train_input,train_output)
prediction = model.predict(test_input)
prediction.shape
def write_data(test_prediction, p_id):
	test_prediction = pd.DataFrame(test_prediction,columns=['Survived'])
	test_prediction = test_prediction.join(p_id)
	return test_prediction
write_data(prediction, p_id).to_csv('output.csv', index = False)
