import pandas as pd
S = pd.DataFrame({'Grade':['Ex','Gd','Ag','Po']})
print('The original feature Grade:\n',S)
S['New_Grade'] = S['Grade'] .map({'Ex':4,'Gd':3,'Ag':2,'Po':1})
print('The feature Grade after transformation:\n',S)
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
S['Grade_by_lab'] = lab.fit_transform(S['Grade'])
print('The feature Grade after transformation:\n',S)
S['Grade_factorized'] = S['Grade'].factorize(sort=True)[0]
print(S[['Grade','Grade_factorized']])
S['Grade_ordered'] = S['Grade'].astype('category', ordered=True, categories=['Po', 'Ag','Gd','Ex'])
print(S['Grade_ordered'])
print('-----------------------------')
print(S['Grade_ordered'].factorize(sort=True))
S['Grade_ordered_factorize'] = S['Grade_ordered'].factorize(sort=True)[0]
print('-----------------------------')
print(S[['Grade','Grade_ordered_factorize']])
