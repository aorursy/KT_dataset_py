# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



kickstarter = pd.read_csv('../input/most_backed.csv')

categories = list(kickstarter.category.unique())

len(categories) #115



kickstarter["category"].value_counts().head(10) #Top 10 most used categories



kickstarter.sort_values(by='amt.pledged', ascending=0) #Top 5 most funded projects



pledged = kickstarter['amt.pledged'].groupby(kickstarter['category'])

pledged.sum().sort_values(ascending=0)

# Any results you write to the current directory are saved as output.
