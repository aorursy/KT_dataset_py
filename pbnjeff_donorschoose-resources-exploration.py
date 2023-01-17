import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))
resources = pd.read_csv('../input/resources.csv')
data_train = pd.read_csv('../input/train.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
data_train.head()
# SELECT id, COUNT(description)
# FROM data_train
# INNER JOIN resources ON data_train.id == resources.id

projects_summed = resources.groupby(['id'])[['quantity','price']].sum()
import matplotlib.pyplot as plt
plt.loglog(projects_summed['quantity'],projects_summed['price'],'o')
plt.xlabel('Quantity of items')
plt.ylabel('Total price of items')
projects_summed['price_per_item'] = projects_summed.price / projects_summed.quantity
plt.loglog(projects_summed['quantity'],projects_summed['price_per_item'],'o')
plt.xlabel('Quantity of items')
plt.ylabel('Price per item')
