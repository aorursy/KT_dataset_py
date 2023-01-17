!pip install pydriller
from pydriller import RepositoryMining



project_url = 'https://github.com/ishepard/pydriller.git'



for commit in RepositoryMining(project_url).traverse_commits():

    print('Hash {}, author {}'.format(commit.hash, commit.author.name))

    
urls = [ "https://github.com/ishepard/pydriller.git", "https://github.com/TheAlgorithms/Java.git", "https://github.com/hmkcode/Java.git"]

for commit in RepositoryMining(path_to_repo=urls).traverse_commits():

    print("commit {}, Author {}, date {}".format(commit.hash, commit.author.name, commit.author_date))
for commit in RepositoryMining('https://github.com/TheAlgorithms/Java.git').traverse_commits():

    for m in commit.modifications:

        print(

            "Author {}".format(commit.author.name),

            " modified {}".format(m.filename),

            " with a change type of {}".format(m.change_type.name),

            " and the complexity is {}".format(m.complexity)

        )
# analyze only 1 local repository

url = "repos/pydriller/"



# analyze 2 local repositories

url = ["repos/pydriller/", "repos/anotherrepo/"]



# analyze both local and remote

url = ["repos/pydriller/", "https://github.com/apache/hadoop.git", "repos/anotherrepo"]



# analyze 1 remote repository

url = "https://github.com/ishepard/pydriller.git"
url = "https://github.com/ishepard/pydriller.git"

for commit in RepositoryMining(url, single='05526fad873c3fc83e40bcbc424bd1b3e5393dd5').traverse_commits():

    print('Hash {}, author {}'.format(commit.hash, commit.author.name))

from datetime import datetime

import socket

for commit in RepositoryMining(url, since=datetime(2020, 1, 1, 1, 0, 0)).traverse_commits():

    print('Hash {}, author {} Date {}'.format(commit.hash, commit.author.name, commit.author_date))

dt1 = datetime(2020, 1, 1, 1, 0, 0)

dt2 = datetime(2020, 1, 15, 17, 59, 0)

for commit in RepositoryMining(url, since=dt1, to=dt2).traverse_commits():

    print('Hash {}, author {} Date {}'.format(commit.hash, commit.author.name, commit.author_date))
dt1 = datetime(2018, 4, 1, 0, 0, 0)

for commit in RepositoryMining(url, to=dt1).traverse_commits():

    print('Hash {}, author {} Date {}'.format(commit.hash, commit.author.name, commit.author_date))
# Only commits in branch1

RepositoryMining('path/to/the/repo', only_in_branch='branch1').traverse_commits()



# Only commits in branch1 and no merges

RepositoryMining('path/to/the/repo', only_in_branch='branch1', only_no_merge=True).traverse_commits()



# Only commits of author "ishepard" (yeah, that's me)

RepositoryMining('path/to/the/repo', only_authors=['ishepard']).traverse_commits()



# Only these 3 commits

RepositoryMining('path/to/the/repo', only_commits=['hash1', 'hash2', 'hash3']).traverse_commits()



# Only commit that modified "Matricula.javax"

RepositoryMining('path/to/the/repo', filepath='Matricula.javax').traverse_commits()



# Only commits that modified a java file

RepositoryMining('path/to/the/repo', only_modifications_with_file_types=['.java']).traverse_commits()