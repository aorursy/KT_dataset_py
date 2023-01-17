import numpy as np
from collections import Counter

num_kagglers = 65
num_trials   = 100_000

num_shared_birth_dates         = np.zeros(num_trials)
num_kagglers_sharing_birthdays = np.zeros(num_trials)
max_kagglers_born_on_same_day  = np.zeros(num_trials)

for trial in range(num_trials):
    birthdays = np.random.randint(1, 365, num_kagglers)
    counts = Counter(birthdays)
    values = counts.values()
    num_shared_birth_dates[trial]         = sum(1 for x in values if x>1)
    num_kagglers_sharing_birthdays[trial] = sum(x for x in values if x>1)
    max_kagglers_born_on_same_day[trial]  = max(values)
print("Average shared birth dates %0.2f" % np.mean(num_shared_birth_dates))
print("Median shared birth dates %0.0f" % np.median(num_shared_birth_dates))
print("Average number Kagglers sharing birthdays %0.2f" % np.mean(num_kagglers_sharing_birthdays))
print("Median number of Kagglers sharing birthdays %0.0f" % np.median(num_kagglers_sharing_birthdays))
print("Average maximum number of Kagglers born on same day %0.2f" % np.mean(max_kagglers_born_on_same_day))
print("Chance at least 3 Kagglers were born on same day %0.2f%%" % (100.0*sum(1 for x in max_kagglers_born_on_same_day if x>2)/num_trials))
print("Chance at least 4 Kagglers were born on same day %0.2f%%" % (100.0*sum(1 for x in max_kagglers_born_on_same_day if x>3)/num_trials))
print("Chance at least 5 Kagglers were born on same day %0.2f%%" % (100.0*sum(1 for x in max_kagglers_born_on_same_day if x>4)/num_trials))
