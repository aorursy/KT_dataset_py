# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install pydriller
import os
import sys 
import psutil


def print_dataset(commit):
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
        commit.hash,
        commit.author.name,
        commit.author.email,
        commit.author_date,
        commit.committer.name,
        commit.committer.email,
        commit.committer_date,
        commit.msg,
        m.filename,
        m.change_type
        )
    )

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
from pydriller import RepositoryMining

i=0
print('sha\tAuthor Name\tAuthor Email\tAuthor Date\tCommit Date\tCommit Email\tCommit Date\tMessage\tFilename\tFile Status\n')
for commit in RepositoryMining('https://github.com/FelipeCortez/Calendala.git').traverse_commits():
    for m in commit.modifications:
        if i>=10:
            break
        print_dataset(commit)
        i=i+1
    if i>=10:
        break
    print("\n")

print("Program End")
process = psutil.Process(os.getpid())
print(process.memory_info().rss /1024)  # in KB
import os
import sys 
import psutil

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
from pydriller import RepositoryMining
i=0
print('sha\tAuthor Name\tAuthor Email\tAuthor Date\tCommit Date\tCommit Email\tCommit Date\tMessage\tFilename\tFile Status\n')
for commit in RepositoryMining('https://github.com/FelipeCortez/Calendala.git',only_modifications_with_file_types=['uml']).traverse_commits():
    for m in commit.modifications:
        print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            commit.hash,
            commit.author.name,
            commit.author.email,
            commit.author_date,
            commit.committer.name,
            commit.committer.email,
            commit.committer_date,
            commit.msg,
            m.filename,
            m.change_type
        )
             )

print("\n")

print("Program End")
process = psutil.Process(os.getpid())
print(process.memory_info().rss /1024)  # in KB