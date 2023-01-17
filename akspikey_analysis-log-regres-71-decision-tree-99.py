import pandas as pd

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn import tree

%matplotlib inline
data_set = pd.read_csv('../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')
data_set = data_set.dropna()
answer_data = data_set.iloc[:,0:50]
answer_data['country'] = data_set['country']
for col in answer_data.columns:

    answer_data[col] = answer_data[col].astype('category').cat.codes
corr_data = pd.DataFrame(answer_data.corr()['country'][:])
corr_data = corr_data.reset_index()
top_correlation = corr_data.sort_values('country', ascending=False).head(10)['index'].to_list()
least_correlation = corr_data.sort_values('country', ascending=False).tail(5)['index'].to_list()
correlation_data = answer_data[top_correlation+least_correlation]
target_data = answer_data['country']
var_train, var_test, res_train, res_test = train_test_split(correlation_data, target_data, test_size = 0.3)
logistic_reg = LogisticRegression(random_state=0).fit(var_train, res_train)
prediction = logistic_reg.predict(var_test)
accuracy_score(res_test, prediction)
decision_tree = tree.DecisionTreeClassifier()

decision_tree = decision_tree.fit(var_train, res_train)
decision_prediction = decision_tree.predict(var_test)
accuracy_score(res_test, decision_prediction)