import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
data = pd.read_csv('../input/train.csv')
previously_posted = data[['teacher_number_of_previously_posted_projects','project_is_approved']]
import seaborn as sns
import matplotlib.pyplot as plt

sns.stripplot(x='project_is_approved',
              y='teacher_number_of_previously_posted_projects',
              data=previously_posted,
              jitter=True,
              alpha=0.4)
plt.xlabel('Project approved')
plt.ylabel('Number of previous submissions')
previously_posted_approved = previously_posted[previously_posted['project_is_approved'] == 1].drop('project_is_approved',axis=1)
previously_posted_rejected = previously_posted[previously_posted['project_is_approved'] == 0].drop('project_is_approved',axis=1)
(num_approved,bins_approved,patches_approved)= plt.hist(previously_posted_approved['teacher_number_of_previously_posted_projects'],
                                                        bins=20, color='r',
                                                        alpha=0.4, label='Approved',
                                                       log=True)
(num_rejected,bins_rejected,patches_rejected) = plt.hist(previously_posted_rejected['teacher_number_of_previously_posted_projects'],
                                                        bins=20, color='b',
                                                        alpha=0.4, label='Rejected',
                                                        log=True)

plt.xlabel('Number of previous attempts')
plt.ylabel('Number of projects approved')

plt.xticks(bins, rotation='vertical')

plt.legend()

plt.show()
num_total = num_approved + num_rejected
num_total[num_total == 0] = 1
probability_approval = num_approved / num_total
probability_approval_binned = pd.DataFrame()
bins_approved = bins_approved[1:]
probability_approval_binned['num_prev_attempt'] = bins_approved
probability_approval_binned['probability'] = probability_approval

plt.plot(probability_approval_binned['num_prev_attempt'],
        probability_approval_binned['probability'])

plt.xlabel('Number of previous attempts')
plt.ylabel('Probability approved')