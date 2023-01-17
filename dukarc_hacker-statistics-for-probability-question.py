import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
sns.set()

# I should fix this ... but meh for now
import warnings
warnings.filterwarnings('ignore')
table_count = 10 # Number of Tables
trials = 10000 # Trials to Simulate Results
def trip_trial(n):
    trips = 0
    values = set()
    while len(values) < n:
        values.add(np.random.randint(n))
        trips += 1
    return trips
trips = np.array([])
for i in range(trials):
    trips = np.append(trips, trip_trial(table_count))
sns.distplot(trips)
_ = plt.title('Histogram of Trials where all 10 Tables were assigned')
def ecdf(data):
    n = float(len(data))
    x = np.sort(data)
    y = np.arange(1, n+1)/n
    return x, y
x, y = ecdf(trips)
_ = plt.cla()
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.figure(figsize=(30,30))
_ = plt.show()
