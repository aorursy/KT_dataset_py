import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
data = pd.read_csv('../input/train.csv')
data_approved = data[data['project_is_approved'] == 1]
data_rejected = data[data['project_is_approved'] == 0]
data_approved = data_approved.drop('project_is_approved', axis=1)
data_rejected = data_rejected.drop('project_is_approved', axis=1)
approved_counts = data_approved.groupby('teacher_prefix')['teacher_prefix'].count()
print(approved_counts)
rejected_counts = data_rejected.groupby('teacher_prefix')['teacher_prefix'].count()
print(rejected_counts)
total_applicants = approved_counts.sum() + rejected_counts.sum()
print(total_applicants)
total_known_men = approved_counts['Mr.'] + rejected_counts['Mr.']
print(total_known_men)
total_known_women = approved_counts['Ms.'] + approved_counts['Mrs.'] + \
                        rejected_counts['Ms.'] + rejected_counts['Mrs.']
print(total_known_women)
approval_rate_men = approved_counts['Mr.'] / total_known_men
print(approval_rate_men)
approval_rate_women = (approved_counts['Ms.'] + approved_counts['Mrs.']) / total_known_women
print(approval_rate_women)
total_num_dr = approved_counts['Dr.'] + rejected_counts['Dr.']
approval_rate_dr = approved_counts['Dr.'] / total_num_dr
print(approval_rate_dr)
total_known_teachers = approved_counts['Teacher'] + rejected_counts['Teacher']
approved_rate_teacher = approved_counts['Teacher'] / total_known_teachers
print(approved_rate_teacher)
total_num_non_dr = total_known_men + total_known_women + total_known_teachers
approval_rate_non_dr = (approved_counts['Mr.'] + approved_counts['Mrs.'] + approved_counts['Ms.'] + \
                           approved_counts['Teacher']) / total_num_non_dr
print(approval_rate_non_dr)
print('Total number of doctors: ' + str(total_num_dr))
print('Total number of non-doctors: ' + str(total_num_non_dr))