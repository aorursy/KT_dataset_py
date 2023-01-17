# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/4054689"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
attack_annotated_comments_df = pd.read_csv("../input/4054689/attack_annotated_comments.tsv", delimiter="\t")

attack_annotations_df = pd.read_csv("../input/4054689/attack_annotations.tsv", delimiter='\t')
attack_annotated_comments_df.head(10)
attack_annotated_comments_df['rev_id'].size == attack_annotated_comments_df['rev_id'].unique().size
print(attack_annotated_comments_df[attack_annotated_comments_df['rev_id'] == 89320]['comment'][3].replace('NEWLINE_TOKEN', '\n'))
attack_annotations_df.head(10)
attack_annotations_df[attack_annotations_df['rev_id'] ==  89320]
attack_worker_demographics_df = pd.read_csv("../input/4054689/attack_worker_demographics.tsv", delimiter='\t')
attack_worker_demographics_df.head(10)
worker_ids = attack_annotations_df[attack_annotations_df['rev_id'] ==  89320]['worker_id']

attack_worker_demographics_df[attack_worker_demographics_df['worker_id'].isin(worker_ids)]
attack_worker_demographics_df['worker_id'].size == attack_worker_demographics_df['worker_id'].unique().size
sns.countplot(x = "education", hue = "gender", data = attack_worker_demographics_df)
sns.countplot(x = 'gender', data = attack_worker_demographics_df)
sns.countplot(x = "gender", hue = "english_first_language", data = attack_worker_demographics_df)
sns.countplot(x = "education", hue = "english_first_language", data = attack_worker_demographics_df)