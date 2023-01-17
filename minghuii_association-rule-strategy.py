import numpy as np
import os
import pandas as pd
print(os.listdir("../input"))


rule = pd.read_csv('../input/association-rules/association_rules.csv')
rule
def get_association (pred):
    # pred is an instance of test['prediction']
    for idx, row in rule.iterrows():
        if type(pred) != str:
            pred = '66'
        row_set = set(row.antecedent_label.split())
        pred_set = set(pred.split())
        if row_set <= pred_set:
            pred = ' '.join(set(row.consequent_label.split()) | pred_set)
    return pred
test = pd.read_csv('../input/prediction/prediction.csv')
test
output = test.copy()
output['prediction'] = test['prediction'].apply(get_association)
output