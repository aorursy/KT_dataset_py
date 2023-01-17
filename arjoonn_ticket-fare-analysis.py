%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
data.info()
len(data.Ticket.unique())
counts = []
for tk in data.Ticket.unique():
    this = data.loc[data.Ticket == tk, 'Fare']
    l = len(this.unique())
    counts.append(l)
    if l > 1:
        print(data.loc[data.Ticket == tk])
print(set(counts))
data.Fare.hist(figsize=(15, 10), bins=50)
for tk in data.Ticket.unique():
    this = data.loc[data.Ticket == tk, 'Fare']
    no_of_people = this.count()
    distribute_evenly = this / no_of_people
    data.loc[data.Ticket == tk, 'Fare'] = distribute_evenly
data.Fare.hist(bins=50, alpha=0.5)