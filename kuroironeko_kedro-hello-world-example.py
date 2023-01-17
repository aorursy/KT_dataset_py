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
!pip install kedro
config_file = './config.yml'

output_dir = 'iris'

repo_name = 'repo_iris'

options = [

    f"output_dir: {output_dir}",

    'project_name: project_iris',

    f"repo_name: {repo_name}" ,

    'python_package: package_iris' ,

    'include_example: true',

]

with open(config_file, 'w') as f:

    f.write('\n'.join(options))

!cat ./config.yml

os.mkdir(output_dir)

!kedro new -c {config_file}
!cd "{output_dir}/{repo_name}" && kedro run