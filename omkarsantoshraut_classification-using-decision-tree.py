import pandas as pd

from sklearn import tree



df = pd.read_excel('../input/salaries/salary.xlsx')

df
new_company = {'google':0, 'abc pharma':1, 'facebook':2}

new_job = {'sales executive':0, 'business manager':1, 'computer programmer':2}

new_degree = {'bachelors':0, 'masters':1}



df['company'] = df['company'].map(new_company)

df['job'] = df['job'].map(new_job)

df['degree'] = df['degree'].map(new_degree)

df
input = df.iloc[:, :3]

input
target = df.iloc[:, -1]

target
model_obj = tree.DecisionTreeClassifier()
model = model_obj.fit(input, target)
model_obj.predict([[2, 2, 1]])
test_input = [[1,0,1],[2,2,1],[2,2,0]]

test_target = [0, 1,1]

model_obj.score(test_input, test_target)
import matplotlib.pyplot as plt

plt.figure(figsize = (15,10))

tree.plot_tree(model, filled = True)