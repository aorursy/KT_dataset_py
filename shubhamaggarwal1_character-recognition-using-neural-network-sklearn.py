# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import binarize
from sklearn.model_selection import GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

training_df = pd.read_csv('../input/train.csv')
#Image Size = 28*28
X = training_df.drop('label', axis=1)
y = training_df['label']

X.describe()
X_scaled = X / 255.0
X_scaled.describe()
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X_scaled, y, test_size=0.3, random_state=1, stratify=y)
model = MLPClassifier(solver='lbfgs', activation='relu', learning_rate_init = 0.01, max_iter=400, alpha=1e-3, hidden_layer_sizes=(64,28), random_state=1)
model.fit(train_X, train_y)
y.value_counts().sort_index()
train_y_pred = model.predict(train_X)
test_y_pred = model.predict(test_X)

#Training Prediction Accuracy
print(accuracy_score(train_y.values, train_y_pred))
#Test Prediction Accuracy
print(accuracy_score(test_y.values, test_y_pred))

#Classification Report
print(classification_report(test_y.values, test_y_pred))
#from sklearn.model_selection import GridSearchCV

#parameter_options = {
#    "learning_rate_init" : (0.1, 0.01, 0.005, 0.001),
#    "alpha" : (1e-2, 1e-3, 1e-4, 1e-5)
#}

#tuning_model = MLPClassifier(solver='lbfgs', activation='relu', max_iter=400, hidden_layer_sizes=(64,28), random_state=1)

#clf = GridSearchCV(tuning_model, parameter_options, cv=5)
#clf.fit(X_scaled, y)
hyper_parameter_json = '{"mean_fit_time":{"0":62.2645419121,"1":61.8836660862,"2":61.3482870579,"3":61.9847536564,"4":64.099307251,"5":64.139607954,"6":64.1262664795,"7":64.1702826977,"8":62.9692382336,"9":62.8342501163,"10":62.9287622452,"11":62.7448991299,"12":63.1923060417,"13":63.5381466866,"14":63.3286220551,"15":62.867953968},"std_fit_time":{"0":1.2380253246,"1":0.7392680755,"2":0.8371015025,"3":1.2630434989,"4":3.1459295475,"5":3.4569018506,"6":3.3554040279,"7":3.3614496645,"8":3.7456630793,"9":3.5959209416,"10":3.6718319373,"11":3.6167472564,"12":3.345812199,"13":3.5025556031,"14":3.47975575,"15":3.2289069236},"mean_score_time":{"0":0.0544639587,"1":0.0534081936,"2":0.0537557602,"3":0.0544393063,"4":0.0532652855,"5":0.0528447628,"6":0.0531326294,"7":0.0537816048,"8":0.0536803246,"9":0.0548650265,"10":0.0528648376,"11":0.0532189846,"12":0.0533833981,"13":0.0529069901,"14":0.0531209946,"15":0.0529269218},"std_score_time":{"0":0.0024103018,"1":0.0027094884,"2":0.0028988523,"3":0.0025701715,"4":0.0004388501,"5":0.0005909962,"6":0.0008000669,"7":0.0013918448,"8":0.0017466922,"9":0.0016980453,"10":0.0007532973,"11":0.0010817515,"12":0.0016575218,"13":0.0006490277,"14":0.0010582231,"15":0.0004445466},"param_alpha":{"0":0.01,"1":0.01,"2":0.01,"3":0.01,"4":0.001,"5":0.001,"6":0.001,"7":0.001,"8":0.0001,"9":0.0001,"10":0.0001,"11":0.0001,"12":0.00001,"13":0.00001,"14":0.00001,"15":0.00001},"param_learning_rate_init":{"0":0.1,"1":0.01,"2":0.005,"3":0.001,"4":0.1,"5":0.01,"6":0.005,"7":0.001,"8":0.1,"9":0.01,"10":0.005,"11":0.001,"12":0.1,"13":0.01,"14":0.005,"15":0.001},"params":{"0":{"alpha":0.01,"learning_rate_init":0.1},"1":{"alpha":0.01,"learning_rate_init":0.01},"2":{"alpha":0.01,"learning_rate_init":0.005},"3":{"alpha":0.01,"learning_rate_init":0.001},"4":{"alpha":0.001,"learning_rate_init":0.1},"5":{"alpha":0.001,"learning_rate_init":0.01},"6":{"alpha":0.001,"learning_rate_init":0.005},"7":{"alpha":0.001,"learning_rate_init":0.001},"8":{"alpha":0.0001,"learning_rate_init":0.1},"9":{"alpha":0.0001,"learning_rate_init":0.01},"10":{"alpha":0.0001,"learning_rate_init":0.005},"11":{"alpha":0.0001,"learning_rate_init":0.001},"12":{"alpha":0.00001,"learning_rate_init":0.1},"13":{"alpha":0.00001,"learning_rate_init":0.01},"14":{"alpha":0.00001,"learning_rate_init":0.005},"15":{"alpha":0.00001,"learning_rate_init":0.001}},"split0_test_score":{"0":0.9650208209,"1":0.9650208209,"2":0.9650208209,"3":0.9650208209,"4":0.9654967281,"5":0.9654967281,"6":0.9654967281,"7":0.9654967281,"8":0.9653777513,"9":0.9653777513,"10":0.9653777513,"11":0.9653777513,"12":0.9657346817,"13":0.9657346817,"14":0.9657346817,"15":0.9657346817},"split1_test_score":{"0":0.9656075211,"1":0.9656075211,"2":0.9656075211,"3":0.9656075211,"4":0.9659645365,"5":0.9659645365,"6":0.9659645365,"7":0.9659645365,"8":0.9670355825,"9":0.9670355825,"10":0.9670355825,"11":0.9670355825,"12":0.9659645365,"13":0.9659645365,"14":0.9659645365,"15":0.9659645365},"split2_test_score":{"0":0.9645195857,"1":0.9645195857,"2":0.9645195857,"3":0.9645195857,"4":0.9651148946,"5":0.9651148946,"6":0.9651148946,"7":0.9651148946,"8":0.9651148946,"9":0.9651148946,"10":0.9651148946,"11":0.9651148946,"12":0.9649958328,"13":0.9649958328,"14":0.9649958328,"15":0.9649958328},"split3_test_score":{"0":0.9623675122,"1":0.9623675122,"2":0.9623675122,"3":0.9623675122,"4":0.9632011433,"5":0.9632011433,"6":0.9632011433,"7":0.9632011433,"8":0.9617720615,"9":0.9617720615,"10":0.9617720615,"11":0.9617720615,"12":0.9620102418,"13":0.9620102418,"14":0.9620102418,"15":0.9620102418},"split4_test_score":{"0":0.9667698904,"1":0.9667698904,"2":0.9667698904,"3":0.9667698904,"4":0.9664125774,"5":0.9664125774,"6":0.9664125774,"7":0.9664125774,"8":0.9677227251,"9":0.9677227251,"10":0.9677227251,"11":0.9677227251,"12":0.9665316818,"13":0.9665316818,"14":0.9665316818,"15":0.9665316818},"mean_test_score":{"0":0.9648571429,"1":0.9648571429,"2":0.9648571429,"3":0.9648571429,"4":0.9652380952,"5":0.9652380952,"6":0.9652380952,"7":0.9652380952,"8":0.9654047619,"9":0.9654047619,"10":0.9654047619,"11":0.9654047619,"12":0.965047619,"13":0.965047619,"14":0.965047619,"15":0.965047619},"std_test_score":{"0":0.0014530598,"1":0.0014530598,"2":0.0014530598,"3":0.0014530598,"4":0.0011078317,"5":0.0011078317,"6":0.0011078317,"7":0.0011078317,"8":0.0020643383,"9":0.0020643383,"10":0.0020643383,"11":0.0020643383,"12":0.001596234,"13":0.001596234,"14":0.001596234,"15":0.001596234},"rank_test_score":{"0":13,"1":13,"2":13,"3":13,"4":5,"5":5,"6":5,"7":5,"8":1,"9":1,"10":1,"11":1,"12":9,"13":9,"14":9,"15":9},"split0_train_score":{"0":1.0,"1":1.0,"2":1.0,"3":1.0,"4":1.0,"5":1.0,"6":1.0,"7":1.0,"8":1.0,"9":1.0,"10":1.0,"11":1.0,"12":1.0,"13":1.0,"14":1.0,"15":1.0},"split1_train_score":{"0":1.0,"1":1.0,"2":1.0,"3":1.0,"4":1.0,"5":1.0,"6":1.0,"7":1.0,"8":1.0,"9":1.0,"10":1.0,"11":1.0,"12":1.0,"13":1.0,"14":1.0,"15":1.0},"split2_train_score":{"0":1.0,"1":1.0,"2":1.0,"3":1.0,"4":1.0,"5":1.0,"6":1.0,"7":1.0,"8":1.0,"9":1.0,"10":1.0,"11":1.0,"12":1.0,"13":1.0,"14":1.0,"15":1.0},"split3_train_score":{"0":1.0,"1":1.0,"2":1.0,"3":1.0,"4":1.0,"5":1.0,"6":1.0,"7":1.0,"8":1.0,"9":1.0,"10":1.0,"11":1.0,"12":1.0,"13":1.0,"14":1.0,"15":1.0},"split4_train_score":{"0":1.0,"1":1.0,"2":1.0,"3":1.0,"4":1.0,"5":1.0,"6":1.0,"7":1.0,"8":1.0,"9":1.0,"10":1.0,"11":1.0,"12":1.0,"13":1.0,"14":1.0,"15":1.0},"mean_train_score":{"0":1.0,"1":1.0,"2":1.0,"3":1.0,"4":1.0,"5":1.0,"6":1.0,"7":1.0,"8":1.0,"9":1.0,"10":1.0,"11":1.0,"12":1.0,"13":1.0,"14":1.0,"15":1.0},"std_train_score":{"0":0.0,"1":0.0,"2":0.0,"3":0.0,"4":0.0,"5":0.0,"6":0.0,"7":0.0,"8":0.0,"9":0.0,"10":0.0,"11":0.0,"12":0.0,"13":0.0,"14":0.0,"15":0.0}}'
hyper_parameter_df = pd.read_json(hyper_parameter_json)
hyper_parameter_df.sort_values('rank_test_score', inplace=True)
hyper_parameter_df[["param_alpha", "param_learning_rate_init", "mean_test_score", "std_test_score", "rank_test_score"]]
nn_model = MLPClassifier(solver='lbfgs', activation='relu', learning_rate_init = 0.1, max_iter=400, alpha=0.001, hidden_layer_sizes=(64,28), random_state=1)
#Fitting on Complete Scaled Data
nn_model.fit(X_scaled, y)
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
#Scaling Test Data
test_scaled_data = test_data / 255.0
#Predicting Test Data
test_predictions = nn_model.predict(test_scaled_data)
result = pd.DataFrame(test_predictions, columns=['Label'])
result.reset_index(inplace=True)
result.rename(columns={'index': 'ImageId'}, inplace=True)
result['ImageId'] = result['ImageId']+1
result.to_csv('output.csv', index=False)
result.head()