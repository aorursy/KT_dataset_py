import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
credit_card = pd.read_csv('../input/creditcard.csv')
credit_card.head()
credit_card.shape
credit_card.describe()
credit_card.Class.unique()
total_fraud_count = sum(credit_card.Class[credit_card.Class == 1])
print('The total fraud happened is ', total_fraud_count, ' which is', round(total_fraud_count/credit_card.shape[0] * 100, 4) , '%')
credit_card[credit_card.Class == 1].head()
non_fradaulent = credit_card[credit_card.Class == 0].sample(98, random_state=1)
fradaulent = credit_card[credit_card.Class == 1].sample(2, random_state=1)
new_credit_card = pd.concat([non_fradaulent, fradaulent]).sample(100, random_state=1)
new_credit_card.head()
fig, ax = plt.subplots(figsize=(20,15))
corr = new_credit_card.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, linewidths=.5, ax=ax, vmin=0, vmax=1)
def cosineSimilarity(vi, vj):
    return np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
class Similarity:
    def __init__(self, inner_id_, class_, innerClass_, similarity_):
        self.inner_id_ = inner_id_
        self.class_ = class_
        self.similarity_ = similarity_
        self.innerClass_ = innerClass_
    
    def __str__(self):
        return 'Class '+ str(int(self.class_))+' matching with id : '+ str(self.inner_id_) +' class '+ str(int(self.innerClass_)) +'. The similarity percent is '+ str(self.similarity_)
    
similarity_map = {}

for id_, row in new_credit_card.iterrows():
    similarity_list = []
    for inner_id_, inner_row in new_credit_card.iterrows():
        similarity_percent = cosineSimilarity(row.values[:-1], inner_row[:-1])
        similarity_list.append(Similarity(inner_id_, row.Class, inner_row.Class, similarity_percent))
    similarity_map[id_] = similarity_list
        
for key, val in similarity_map.items():
    print('Given transaction id : ', key)
    val.sort(key = lambda x : x.similarity_, reverse=False)
    for similar_ in val[:10]:
        print(similar_)
for key, val in similarity_map.items():
    print('Given transaction id : ', key)
    val.sort(key = lambda x : x.similarity_, reverse=True)
    for similar_ in val[:10]:
        print(similar_)
